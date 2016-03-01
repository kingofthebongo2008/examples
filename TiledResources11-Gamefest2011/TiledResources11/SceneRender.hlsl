//--------------------------------------------------------------------------------------
// File: SceneRender.hlsl
//
// The HLSL file containing scene rendering shaders for the TiledResources11 sample.
// 
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "TiledResources.hlsl"

//--------------------------------------------------------------------------------------
// Constant Buffers
//--------------------------------------------------------------------------------------
cbuffer cbPerObject : register( b0 )
{
    matrix  g_mWorldViewProjection  : packoffset( c0 );
}

//-----------------------------------------------------------------------------------------
// Textures and Samplers
//-----------------------------------------------------------------------------------------
Texture2D       g_texSamplingQualityMap         : register( t1 );
Texture2D       g_texSamplingQualityMap2        : register( t2 );
Texture2DArray  g_texSamplingQualityMapArray    : register( t2 );
SamplerState    g_samSamplingQualityMap         : register( s1 );

//--------------------------------------------------------------------------------------
// shader input/output structure
//--------------------------------------------------------------------------------------
struct VS_INPUT
{
    float4 Position     : POSITION; // vertex position 
    float3 TextureUVW   : TEXCOORD0;// vertex texture coords 
};

struct VS_OUTPUT
{
    float3 TextureUVW   : TEXCOORD0;   // vertex texture coords 
    float4 Position     : SV_POSITION; // vertex position 
};

//--------------------------------------------------------------------------------------
// This vertex shader transforms position and passes through texture coordinates.
//--------------------------------------------------------------------------------------
VS_OUTPUT VSTransform( VS_INPUT input )
{
    VS_OUTPUT Output;

    // Transform the position from object space to homogeneous projection space
    Output.Position = mul( input.Position, g_mWorldViewProjection );

    // Just copy the texture coordinate through
    Output.TextureUVW = input.TextureUVW; 
    
    return Output;    
}

//--------------------------------------------------------------------------------------
// This shader outputs the pixel's color by sampling from a single tiled texture 2D.
//--------------------------------------------------------------------------------------
float4 PSSceneRender( VS_OUTPUT In ) : SV_TARGET
{ 
    float2 RoundedUV = In.TextureUVW.xy;
#ifdef RoundUV
    float Width, Height, NumLevels;
    g_texSamplingQualityMap.GetDimensions( 0, Width, Height, NumLevels );
    RoundedUV.x = floor( RoundedUV.x * Width ) / Width;
    RoundedUV.y = floor( RoundedUV.y * Height ) / Height;
#endif

    float EncodedLOD = g_texSamplingQualityMap.Sample( g_samSamplingQualityMap, RoundedUV ).r;
    float MinLOD = EncodedLOD / g_LODEncode;
    float4 Sample = TiledTex2D_Trilinear_MinLOD( 0, In.TextureUVW.xy, RoundedUV, MinLOD );
    if( GetResidencyStatus() == false )
    {
        Sample = float4( 1, 0, 1, 1 );
    }
    return Sample;
}

//--------------------------------------------------------------------------------------
// This shader outputs the pixel's color by sampling from a single tiled texture 2D array.
//--------------------------------------------------------------------------------------
float4 PSSceneRenderArray( VS_OUTPUT In ) : COLOR0
{
    float2 RoundedUV = In.TextureUVW.xy;
#ifdef RoundUV
    float Width, Height, NumLevels, NumSlices;
    g_texSamplingQualityMapArray.GetDimensions( 0, Width, Height, NumSlices, NumLevels );
    RoundedUV.x = floor( RoundedUV.x * Width ) / Width;
    RoundedUV.y = floor( RoundedUV.y * Height ) / Height;
#endif

    float EncodedLOD = g_texSamplingQualityMapArray.Sample( g_samSamplingQualityMap, float3( RoundedUV.xy, In.TextureUVW.z ) ).r;
    float MinLOD = EncodedLOD / g_LODEncode;
    float4 Sample = TiledTex3D_Trilinear_MinLOD( 0, In.TextureUVW, RoundedUV, MinLOD );
    if( GetResidencyStatus() == false )
    {
        Sample = float4( 1, 0, 1, 1 );
    }
    return Sample;
}

//--------------------------------------------------------------------------------------
// This shader outputs the pixel's color by sampling from a single tiled texture 2D
// array, accessed as a texture quilt.
//--------------------------------------------------------------------------------------
float4 PSSceneRenderQuilt( VS_OUTPUT In ) : COLOR0
{
    // need to compute the texture LOD before we convert 2D quilt texcoords into 3D array slice texcoords
    // if we don't do this, the texture LOD will be improperly computed along quilt boundaries, since the
    // gradients along the boundary will be invalid due to the 2D to 3D conversion
    float ComputedLOD = TiledTex3D_ComputeLOD( 0, In.TextureUVW );
    
    // convert 2D quilt texcoords into 3D array slice texcoords
    float3 ArrayUVW = Quilt2DToTex3D( 0, In.TextureUVW.xy );

    float2 RoundedUV = ArrayUVW.xy;
#ifdef RoundUV
    float Width, Height, NumLevels, NumSlices;
    g_texSamplingQualityMapArray.GetDimensions( 0, Width, Height, NumSlices, NumLevels );
    RoundedUV.x = floor( RoundedUV.x * Width ) / Width;
    RoundedUV.y = floor( RoundedUV.y * Height ) / Height;
#endif

    float EncodedLOD = g_texSamplingQualityMapArray.Sample( g_samSamplingQualityMap, float3( RoundedUV.xy, ArrayUVW.z ) ).r;
    float MinLOD = EncodedLOD / g_LODEncode;
    float4 Sample = TiledTex3D_Trilinear_FixedLOD( 0, ArrayUVW, RoundedUV, max( MinLOD, ComputedLOD ) );
    if( GetResidencyStatus() == false )
    {
        Sample = float4( 1, 0, 1, 1 );
    }
    return Sample;
}

//--------------------------------------------------------------------------------------
// Constant Buffer for the terrain vertex shader
//--------------------------------------------------------------------------------------
cbuffer CBTerrainVS : register(b0)
{
    float4x4 g_matWorldViewProjection;
    float4   g_PositionTexCoordScaleOffset;
    float4   g_HeightmapConstants;
};

//--------------------------------------------------------------------------------------
// This vertex shader samples from a tiled texture 2D heightmap, offsets the vertex Y
// coordinate by the height value, transforms the vertex, and then passes through texture
// coordinates.
//--------------------------------------------------------------------------------------
void VSTerrain( in float2 InPosition : POSITION0, in float2 InTexCoord : TEXCOORD0, out float3 OutTexCoord : TEXCOORD0, out float4 OutPosition : SV_POSITION )
{
    // The terrain mesh is made up of a grid of smaller meshes; each of the smaller meshes has 0..1 coordinates.
    // Convert 0..1 relative positions and texcoords into terrain-absolute coordinates:
    InPosition.xy = InPosition.xy * g_PositionTexCoordScaleOffset.xy + g_PositionTexCoordScaleOffset.zw;
    InTexCoord.xy = InTexCoord.xy * g_PositionTexCoordScaleOffset.xy + g_PositionTexCoordScaleOffset.zw;

    // Do not allow wrapping on the texture coordinates:
    InTexCoord.xy = clamp( InTexCoord.xy, 0, 0.9999f );

    float2 RoundedUV = InTexCoord;
#ifdef RoundUV
    float Width, Height, NumLevels;
    g_texSamplingQualityMap.GetDimensions( 0, Width, Height, NumLevels );
    RoundedUV.x = floor( RoundedUV.x * Width ) / Width;
    RoundedUV.y = floor( RoundedUV.y * Height ) / Height;
#endif

    // Perform max reduction filtering on the sampling quality map.
    // Reduction filtering is where we gather multiple samples like a bilinear sample, but
    // instead of performing a bilinear blend, max reduction filtering returns the max value
    // of the samples.
    // We want to use max reduction filtering so that we always get the maximum value of the
    // sampling quality map at a given sample, not a blended result.  With blending, we might
    // get intermediate values that are not valid.  For example, blending between values of 1.0 and
    // 5.0 would return 3.0, but there may not be a mip LOD 3 available at the sampled location in
    // the tiled texture.

    float4 EncodedSamples = 0;
#ifdef vs_4_1
    // Shader model 4.1 allows us to gather 4 samples in one operation:
    EncodedSamples = g_texSamplingQualityMap.Gather( g_samSamplingQualityMap, InTexCoord.xy );
#else
    // Perform 5 separate samples for shader model 4.0:
    EncodedSamples.x = g_texSamplingQualityMap.SampleLevel( g_samSamplingQualityMap, InTexCoord.xy, 0, int2(-1,-1) ).r;
    EncodedSamples.y = g_texSamplingQualityMap.SampleLevel( g_samSamplingQualityMap, InTexCoord.xy, 0, int2(1,-1) ).r;
    EncodedSamples.z = g_texSamplingQualityMap.SampleLevel( g_samSamplingQualityMap, InTexCoord.xy, 0, int2(-1,1) ).r;
    EncodedSamples.w = g_texSamplingQualityMap.SampleLevel( g_samSamplingQualityMap, InTexCoord.xy, 0, int2(1,1) ).r;
    float CenterSample = g_texSamplingQualityMap.SampleLevel( g_samSamplingQualityMap, InTexCoord.xy, 0, int2(0,0) ).r;
    EncodedSamples.x = max( CenterSample, EncodedSamples.x );
#endif
    // Ensure that the max value is in the X component:
    EncodedSamples.xy = max( EncodedSamples.xy, EncodedSamples.zw );
    EncodedSamples.x = max( EncodedSamples.x, EncodedSamples.y );

    // Sample the height map using the max filtered LOD value from the sampling quality map:
    float HeightMapLOD = EncodedSamples.x / g_LODEncode;
    float4 HeightMapTextureSample = TiledTex2D_Trilinear_FixedLOD( 0, InTexCoord, RoundedUV, HeightMapLOD );
    
    // Construct the 3D position from the heightmap value and the terrain absolute XZ values:
    float4 Position = float4( InPosition.x, 0, InPosition.y, 1 );
    Position.y = HeightMapTextureSample.x * g_HeightmapConstants.x;
    
    // Transform the position:
    OutPosition = mul( Position, g_matWorldViewProjection );

    // Output the terrain absolute texture coordinate:
    OutTexCoord = float3( InTexCoord, 0 );
}

//--------------------------------------------------------------------------------------
// Constant Buffer for the terrain pixel shader
//--------------------------------------------------------------------------------------
cbuffer CBTerrainPS : register(b0)
{
    float4 light_direction_world;
    float4 ambient_light;
};

//--------------------------------------------------------------------------------------
// This pixel shader samples from a pair of tiled textures to get the diffuse color and
// surface normal for the pixel.  A simple diffuse lighting computation is done, and the
// pixel color is returned.
//--------------------------------------------------------------------------------------
float4 PSTerrainRender( in float3 InTexCoord : TEXCOORD0 ) : SV_TARGET
{
    float2 RoundedUV = InTexCoord.xy;
    float2 RoundedUV2 = InTexCoord.xy;
#ifdef RoundUV
    {
        float Width, Height, NumLevels;
        g_texSamplingQualityMap.GetDimensions( 0, Width, Height, NumLevels );
        RoundedUV.x = floor( RoundedUV.x * Width ) / Width;
        RoundedUV.y = floor( RoundedUV.y * Height ) / Height;
    }
    {
        float Width, Height, NumLevels;
        g_texSamplingQualityMap2.GetDimensions( 0, Width, Height, NumLevels );
        RoundedUV2.x = floor( RoundedUV2.x * Width ) / Width;
        RoundedUV2.y = floor( RoundedUV2.y * Height ) / Height;
    }
#endif

    float EncodedLOD = g_texSamplingQualityMap.Sample( g_samSamplingQualityMap, RoundedUV ).r;
    float MinLOD = EncodedLOD / g_LODEncode;
    float4 DiffuseMapSample = TiledTex2D_Trilinear_MinLOD( 0, InTexCoord.xy, RoundedUV, MinLOD );
    if( GetResidencyStatus() == false )
    {
        DiffuseMapSample = float4( 1, 0, 1, 1 );
    }

    float EncodedLOD2 = g_texSamplingQualityMap2.Sample( g_samSamplingQualityMap, RoundedUV2 ).r;
    float MinLOD2 = EncodedLOD2 / g_LODEncode;
    float4 NormalMapSample = TiledTex2D_Trilinear_MinLOD( 1, InTexCoord.xy, RoundedUV2, MinLOD2 );
    if( GetResidencyStatus() == false )
    {
        NormalMapSample = float4( 0, 0, 1, 1 );
    }
    else
    {
        NormalMapSample.xy = NormalMapSample.xy * 2 - 1;
        NormalMapSample.z = 0.5;
        NormalMapSample.xyz = normalize( NormalMapSample.xyz );        
    }

    float DiffuseLight = saturate( dot( NormalMapSample.xyz, -light_direction_world.xyz ) );
    
    return ( DiffuseLight.xxxx + ambient_light ) * DiffuseMapSample;
}
