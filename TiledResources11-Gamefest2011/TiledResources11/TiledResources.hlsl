//--------------------------------------------------------------------------------------
// File: TiledResources.hlsl
//
// The HLSL file containing tiled resource measurement and management shaders for the 
// TiledResources11 sample.  Some shaders here are used to produce residency sample views,
// which feed back residency information into the title residency manager for page streaming.
// Other shaders are used to measure mip level partial residency on a single tiled texture,
// for generating the sampling quality maps.
// 
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "TiledResourceEmulationLib.hlsl"

//--------------------------------------------------------------------------------------
// Constant buffer for the residency sample vertex shader:
//--------------------------------------------------------------------------------------
cbuffer cbResidencyVS : register( b0 )
{
    matrix  g_matWVP  : packoffset( c0 );
}

//--------------------------------------------------------------------------------------
// Vertex shader for simple residency sample rendering.  Transforms position and passes
// through the texture coordinates.
//--------------------------------------------------------------------------------------
void VSResidencyTransform( in float4 InPosition : POSITION0, in float3 InTex0 : TEXCOORD0, out float3 OutTex0 : TEXCOORD0, out float4 OutPosition : POSITION0 )
{
    OutPosition = mul( float4( InPosition.xyz, 1 ), g_matWVP );
    OutTex0 = InTex0;
}

//--------------------------------------------------------------------------------------
// Constant buffer for the residency sample pixel shader:
//--------------------------------------------------------------------------------------
cbuffer cbResidencyPS : register( b0 )
{
    float4 g_ResidencyConstant : packoffset(c0);
}

//--------------------------------------------------------------------------------------
// Residency sample pixel shader.  This pixel shader encodes several pieces of data into
// 7 channels across 2 rendertargets.
//
// Rendertarget 0 contains:
//   R : Fractional U texture coordinate at this pixel
//   G : Fractional V texture coordinate at this pixel
//   B : Encoded minimum texture coordinate derivative at this pixel
//   A : Resource set ID of the object at this pixel
// 
// Rendertarget 1 contains:
//   R : Encoded whole number component of U texture coordinate at this pixel
//   G : Encoded whole number component of V texture coordinate at this pixel
//   B : Encoded array slice index at this pixel
//   A : Zero (unused)
//--------------------------------------------------------------------------------------
void PSResidencySampleTex0( float3 TexCoord0 : TEXCOORD0, out float4 UVGradientID : SV_TARGET0, out float4 ExtendedUVSlice : SV_TARGET1 )
{
    // Compute the fractional component of the UV coordinates, and output them to the
    // RG channels of the first rendertarget:
    UVGradientID.xy = frac( TexCoord0.xy );

    // Encode the whole number component of the UV coordinates, and output them to the
    // RG channels of the second rendertarget:
    ExtendedUVSlice.xy = ( floor( TexCoord0.xy ) + 128.0 ) / 255.0;
    
    // Compute both partial derivatives of the texture coordinate at this pixel:
    float4 Deriv;
    Deriv.xy = ddx( TexCoord0 );
    Deriv.zw = ddy( TexCoord0 );
    
    // Compute the squared magnitude of the U derivative at this pixel:
    float DerivUMagSq = dot( Deriv.xz, Deriv.xz );

    // Compute the squared magnitude of the V derivative at this pixel:
    float DerivVMagSq = dot( Deriv.yw, Deriv.yw );
    
    // Compute the minimum of the U and V magnitudes, to get the minimum
    // derivative:
    float MinDeriv = sqrt( max( DerivUMagSq, DerivVMagSq ) );

    // Encode the minimum derivative into the B channel of the first rendertarget.
    // The derivative will vary over a wide numeric range, so we use a logarithm
    // to better utilize our very limited 8 bits of output precision:
    UVGradientID.z = saturate( log( MinDeriv ) * g_ResidencyConstant.y + g_ResidencyConstant.z );
    
    // Output the resource set ID to the A channel of the first rendertarget:
    UVGradientID.w = g_ResidencyConstant.x;

    // Encode the array slice index into the B channel of the second rendertarget:
    ExtendedUVSlice.z = TexCoord0.z / 255.0;

    // A channel of second rendertarget is unused:
    ExtendedUVSlice.w = 0;
}

//--------------------------------------------------------------------------------------
// Pass-through vertex shader for rendering the sampling quality map.
//--------------------------------------------------------------------------------------
void VSQualityPassThru( in float2 InPosition : POSITION0, in float2 InTex0 : TEXCOORD0, out float2 OutTex0 : TEXCOORD0, out float4 OutPosition : POSITION0 )
{
    OutPosition = float4( InPosition, 0, 1 );
    OutTex0 = InTex0;
}

//--------------------------------------------------------------------------------------
// Textures and samplers for rendering the sampling quality map.
//--------------------------------------------------------------------------------------
Texture2D g_QualityMap : register(t0);
Texture2DArray g_QualityMapArray : register(t0);
SamplerState g_QualityMapSamplerState : register(s0);

//--------------------------------------------------------------------------------------
// Pixel shader constants for the sampling quality map pixel shader:
//--------------------------------------------------------------------------------------
cbuffer CB_QualitySampleRender : register(b0)
{
    float4 g_LODConstant : packoffset(c0);
};

//--------------------------------------------------------------------------------------
// A constant for encoding LODs from 0-16 into a 0-1 value.
//--------------------------------------------------------------------------------------
static const float g_LODEncode = 1.0 / 16.0;

//--------------------------------------------------------------------------------------
// Pixel shader for sampling quality map rendering of a tiled texture 2D.
// This pixel shader updates a state machine for each pixel. Each pixel maps to one page 
// in the base level of the tiled texture.
// The goal is to steadily decrease the LOD value at each pixel, until we hit a mip LOD 
// value that is not resident in the tiled texture at that pixel.  If the current value
// becomes too low all of the sudden, when a page is unmapped from that area of the
// tiled texture, then the value quickly jumps up to the next resident mip LOD value.
//
// Note that this shader is only executed when the sampling quality manager believes that
// pages have been mapped or unmapped on the tiled texture very recently.
//--------------------------------------------------------------------------------------
float4 PSQualitySample( float2 TexCoord0 : TEXCOORD0 ) : SV_TARGET
{
    // Get the current LOD value from the sampling quality map at this pixel:
    float CurrentLOD = g_QualityMap.Sample( g_QualityMapSamplerState, TexCoord0 ).r / g_LODEncode;
    
    // The shader constant dictates how fast we want to push the LOD value lower.
    // This value is updated each frame based on the delta time.
    float LODTransitionRate = g_LODConstant.x;
    
    // Push the LOD value downwards by the LOD transition rate:
    CurrentLOD = max( 0, CurrentLOD - LODTransitionRate );
    
    bool Resident = false;

    // The integer LoopCount variable is here to make the shader compiler happy:
    int LoopCount = 0;

    // Loop until the CurrentLOD value represents a resident mip level at these texcoords:
    while( !Resident && LoopCount < 9 )
    {
        // Compute the highest whole number less than CurrentLOD:
        float TestLOD = floor( CurrentLOD );

        // Run a test sample of the tiled texture, using the texcoords and the test LOD:
        float4 UnusedColorSample = TiledTex2D_Point_FixedLOD( 0, TexCoord0, TexCoord0, TestLOD );

        // Check if the test sample hit resident pages:
        Resident = GetResidencyStatus();

        // If we hit nonresident pages, we need to increase CurrentLOD to the next whole number:
        if( !Resident )
        {
            // Jump up to the next highest integer:
            CurrentLOD = ceil( CurrentLOD ) + 0.001;            

            // Clamp the CurrentLOD value to the tiled texture's mip level count, and loop no further:
            if( CurrentLOD >= g_LODConstant.y )
            {
                Resident = true;
            }
        }

        ++LoopCount;
    }

    // Clamp the result to the tiled texture's mip level count:
    CurrentLOD = min( CurrentLOD, g_LODConstant.y );

    // Encode the LOD value to 0..1 range and write it to the rendertarget:
    float EncodedLOD = CurrentLOD * g_LODEncode;
    return float4( EncodedLOD, EncodedLOD, EncodedLOD, EncodedLOD );
}

//--------------------------------------------------------------------------------------
// Pixel shader for sampling quality map rendering of a tiled texture 2D array.
// This pixel shader updates a state machine for each pixel. Each pixel maps to one page 
// in the base level of the tiled texture.  The array slice index is passed in as a
// shader constant, and one pass is made for each array slice.
// The goal is to steadily decrease the LOD value at each pixel, until we hit a mip LOD 
// value that is not resident in the tiled texture at that pixel.  If the current value
// becomes too low all of the sudden, when a page is unmapped from that area of the
// tiled texture, then the value quickly jumps up to the next resident mip LOD value.
//
// Note that this shader is only executed when the sampling quality manager believes that
// pages have been mapped or unmapped on the tiled texture array very recently.
//--------------------------------------------------------------------------------------
float4 PSQualitySampleArray( float2 TexCoord0 : TEXCOORD0 ) : SV_TARGET
{
    float3 TexCoord3D = float3( TexCoord0.xy, g_LODConstant.z );

    // Get the current LOD value from the sampling quality map at this pixel:
    float CurrentLOD = g_QualityMapArray.Sample( g_QualityMapSamplerState, TexCoord3D ).r / g_LODEncode;
    
    // The shader constant dictates how fast we want to push the LOD value lower.
    // This value is updated each frame based on the delta time.
    float LODTransitionRate = g_LODConstant.x;
    
    // Push the LOD value downwards by the LOD transition rate:
    CurrentLOD = max( 0, CurrentLOD - LODTransitionRate );
    
    bool Resident = false;

    // The integer LoopCount variable is here to make the shader compiler happy:
    int LoopCount = 0;

    // Loop until the CurrentLOD value represents a resident mip level at these texcoords:
    while( !Resident && LoopCount < 9 )
    {
        // Compute the highest whole number less than CurrentLOD:
        float TestLOD = floor( CurrentLOD );

        // Run a test sample of the tiled texture, using the texcoords and the test LOD:
        float4 UnusedColorSample = TiledTex3D_Point_FixedLOD( 0, TexCoord3D, TexCoord3D.xy, TestLOD );

        // Check if the test sample hit resident pages:
        Resident = GetResidencyStatus();

        // If we hit nonresident pages, we need to increase CurrentLOD to the next whole number:
        if( !Resident )
        {
            // Jump up to the next highest integer
            CurrentLOD = ceil( CurrentLOD ) + 0.001;            

            // Clamp the CurrentLOD value to the tiled texture's mip level count, and loop no further:
            if( CurrentLOD >= g_LODConstant.y )
            {
                Resident = true;
            }
        }

        ++LoopCount;
    }

    // Clamp the result to the tiled texture's mip level count:
    CurrentLOD = min( CurrentLOD, g_LODConstant.y );

    // Encode the LOD value to 0..1 range and write it to the rendertarget:
    float EncodedLOD = CurrentLOD * g_LODEncode;    
    return float4( EncodedLOD, EncodedLOD, EncodedLOD, EncodedLOD );
}

//--------------------------------------------------------------------------------------
// Vertex shader for page debug render:
//--------------------------------------------------------------------------------------
void VSDebugRenderPages( in float2 InPosition : POSITION0, in float4 InColor : COLOR0, out float4 OutColor : COLOR0, out float4 OutPosition : POSITION0 )
{
    OutPosition = mul( float4( InPosition.xy, 0, 1 ), g_matWVP );
    OutColor = InColor;
}

//--------------------------------------------------------------------------------------
// Pixel shader for page debug render:
//--------------------------------------------------------------------------------------
void PSDebugRenderPages( in float4 Color : COLOR0, out float4 Target0 : SV_TARGET0 )
{
    Target0 = Color;
}

//--------------------------------------------------------------------------------------
// Textures and samplers for onscreen debug texture views:
//--------------------------------------------------------------------------------------
Texture2D g_DebugTexture : register(t0);
SamplerState g_DebugSamplerState : register(s0);

//--------------------------------------------------------------------------------------
// Vertex shader for onscreen debug texture views:
//--------------------------------------------------------------------------------------
void VSDebugRenderTexture( in float2 InPosition : POSITION0, in float2 InTex : TEXCOORD0, out float2 OutTex : TEXCOORD0, out float4 OutPosition : POSITION0 )
{
    OutPosition = mul( float4( InPosition.xy, 0, 1 ), g_matWVP );
    OutTex = InTex;
}

//--------------------------------------------------------------------------------------
// Pixel shader for onscreen debug texture views:
//--------------------------------------------------------------------------------------
void PSDebugRenderTexture( in float2 Tex : TEXCOORD0, out float4 Target0 : SV_TARGET0 )
{
    Target0 = g_DebugTexture.Sample( g_DebugSamplerState, Tex );
}
