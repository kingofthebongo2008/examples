//--------------------------------------------------------------------------------------
// TiledResourceEmulationLib.hlsl
//
// Defines HLSL functions that emulate tiled texture fetches.
// Public functions include the TiledTex2D_ and TiledTex3D_ family of functions as 
// well as GetResidencyStatus().
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

// Optimization switches
// These defines enable functionality that trades ALU performance for compatibility.

// Support for non-power-of-2 dimension textures.  There is a correction factor applied to the raw texture UV
// that supports UV spaces that do not span the entire width and/or height of a particular mip level.
#define SUPPORT_NON_POW2_DIMENSIONS 1

// Sampler state used to sample index map textures:
SamplerState   IndexMapSamplerState             : register(s13);

// Sampler states used to sample the physical page array textures:
SamplerState   PhysicalPageSamplerStatePoint    : register(s14);
SamplerState   PhysicalPageSamplerStateBilinear : register(s15);

#ifdef TILED_VERTEX_SHADER

// Number of tiled resource slots allowed in the vertex shader:
#define TILED_RESOURCE_SLOTS 4

// Index map texture:
Texture2D       g_IndexMapTexture2D[TILED_RESOURCE_SLOTS]      : register(t7);

// Index map array texture:
Texture2DArray  g_IndexMapTexture3D[TILED_RESOURCE_SLOTS]      : register(t7);

// Physical page array texture:
Texture2DArray  g_PhysicalPageTexture[TILED_RESOURCE_SLOTS]    : register(t11);

#else

// Number of tiled resource slots allowed in the pixel shader:
#define TILED_RESOURCE_SLOTS 4

// Index map texture:
Texture2D       g_IndexMapTexture2D[TILED_RESOURCE_SLOTS]      : register(t7);

// Index map array texture:
Texture2DArray  g_IndexMapTexture3D[TILED_RESOURCE_SLOTS]      : register(t7);

// Physical page array texture:
Texture2DArray  g_PhysicalPageTexture[TILED_RESOURCE_SLOTS]    : register(t11);

#endif

//--------------------------------------------------------------------------------------
// Name: TiledResourceConstants
// Desc: Struct of shader constants that are used to sample from a tiled resource.
//--------------------------------------------------------------------------------------
struct TiledResourceConstants
{
    // Page UV size LOD constants - one shader constant per index map LOD
    // X component converts from resource 0-1 U space to one page width 0-1 U space
    // Y component converts from resource 0-1 V space to one page height 0-1 V space
    // Z component adjusts resource 0-1 U space to index map 0-1 U space
    // W component adjusts resource 0-1 V space to index map 0-1 V space
    float4 PhysicalPageUVSizePerLOD[9];    

    // Resource constant
    // X: Mip bias to apply to the index map sample
    // Y: Reciprocal of the number of array slices in the index map
    // Z: Quilt width in UV space
    // W: Quilt height in UV space
    float4 ResourceConstant;
};

//--------------------------------------------------------------------------------------
// Name: cbTiledResource
// Desc: A constant buffer that holds several slots of tiled resource shader constants.
//--------------------------------------------------------------------------------------
cbuffer cbTiledResource : register(b12)
{
    TiledResourceConstants g_TiledResourceConstants[TILED_RESOURCE_SLOTS];
};

//--------------------------------------------------------------------------------------
// Name: PagePoolConstants
// Desc: Struct of shader constants that are used to sample from the physical page array
//       texture associated with a tiled resource.
//--------------------------------------------------------------------------------------
struct PagePoolConstants
{
    // 2D transform that transforms 0..1 UV space to the rectangle within the texel border of a single page
    float4 PageBorderUVTransform;

    // array texture constants
    // X: number of array slices
    // Y: unused
    // Z: 1 / atlas width
    // W: 1 / atlas height
    float4 ArrayTexConstants;
};

//--------------------------------------------------------------------------------------
// Name: cbTiledResource
// Desc: A constant buffer that holds several slots of physical page array texture shader constants.
//--------------------------------------------------------------------------------------
cbuffer cbPagePool : register(b13)
{
    PagePoolConstants g_PagePoolConstants[TILED_RESOURCE_SLOTS];
};

// A global variable that is set to false when a tiled resource sample hits nonresident pages:
static bool g_ResidencyStatus = true;

//--------------------------------------------------------------------------------------
// Name: Get2DLOD (Texture2D version)
// Desc: A method that returns the tiled resource mip LOD of the given tiled texture at the
//       given UV coordinates.  Note that this computes the mip LOD against the index map
//       and then applies a bias at the end to convert that mip LOD to the LOD against
//       the tiled texture.
//--------------------------------------------------------------------------------------
float Get2DLOD( uniform Texture2D Tex, uniform SamplerState SS, float2 UV, uniform int IndexMapSlot )
{
#ifdef ps_4_1
    float LOD = Tex.CalculateLevelOfDetailUnclamped( SS, UV );
#else
    float4 Deriv;
    Deriv.xy = ddx( UV );
    Deriv.zw = ddy( UV );
    
    float DerivUMagSq = dot( Deriv.xz, Deriv.xz );
    float DerivVMagSq = dot( Deriv.yw, Deriv.yw );
    
    float MinDeriv = sqrt( max( DerivUMagSq, DerivVMagSq ) );
    float TexWidth, TexHeight;
    Tex.GetDimensions( TexWidth, TexHeight );
    float MinPixels = min( MinDeriv * TexWidth, MinDeriv * TexHeight );
    float LOD = log2( MinPixels );
#endif
    return LOD.x + g_TiledResourceConstants[IndexMapSlot].ResourceConstant.x;
}

//--------------------------------------------------------------------------------------
// Name: Get2DLOD (Texture2DArray version)
// Desc: A method that returns the tiled resource mip LOD of the given tiled texture at the
//       given UVW coordinates.  Note that this computes the mip LOD against the index map
//       and then applies a bias at the end to convert that mip LOD to the LOD against
//       the tiled texture.
//--------------------------------------------------------------------------------------
float Get2DLOD( uniform Texture2DArray Tex, uniform SamplerState SS, float3 UVW, uniform int IndexMapSlot )
{
#ifdef ps_4_1
    float LOD = Tex.CalculateLevelOfDetailUnclamped( SS, UVW.xy );
#else
    float3 DerivX, DerivY;
    DerivX = ddx( UVW );
    DerivY = ddy( UVW );
    
    float DerivUMagSq = dot( DerivX.xx, DerivY.xx );
    float DerivVMagSq = dot( DerivX.yy, DerivY.yy );
    
    float MinDeriv = sqrt( max( DerivUMagSq, DerivVMagSq ) );
    float TexWidth, TexHeight, TexDepth;
    Tex.GetDimensions( TexWidth, TexHeight, TexDepth );
    float MinPixels = min( MinDeriv * TexWidth, MinDeriv * TexHeight );
    float LOD = log2( MinPixels );
#endif
    return LOD.x + g_TiledResourceConstants[IndexMapSlot].ResourceConstant.x;
}

//--------------------------------------------------------------------------------------

float4 FetchFromIndexMap( uniform Texture2D Tex, uniform SamplerState SS, const float2 RoundedUV, const float LOD )
{
    float4 Sample = Tex.SampleLevel( SS, RoundedUV, LOD );
    return Sample * 255.0f;
}

//--------------------------------------------------------------------------------------

float4 FetchFromIndexMapArray( uniform Texture2DArray Tex, uniform SamplerState SS, const float3 RoundedUVW, const float LOD )
{
    float4 Sample = Tex.SampleLevel( SS, RoundedUVW, LOD );
    return Sample * 255.0f;
}

//--------------------------------------------------------------------------------------
// Name: ComputePhysicalPageUV_MinLOD
// Desc: Given a UV coord and index map slot, returns a physical page array texture 
//       location that should be sampled to produce the final texture sample.
//       A MinLOD value is provided that is used to clamp the computed LOD value.
//--------------------------------------------------------------------------------------
float3 ComputePhysicalPageUV_MinLOD( const float2 TextureUV, const float2 RoundedUV, const float MinLOD, uniform int IndexMapSlot, uniform SamplerState IndexMapSamplerState )
{
    // Compute the mip LOD of the sample against the texture:
    Texture2D IndexMapTexture = g_IndexMapTexture2D[IndexMapSlot];
    float LOD = Get2DLOD( IndexMapTexture, IndexMapSamplerState, TextureUV, IndexMapSlot );

    // Clamp and round the LOD value to a whole number lookup index:
    LOD = clamp( LOD, MinLOD, 8 );
    float IndexMapLOD = round( LOD );
    float ConstantLookupLOD = IndexMapLOD;
    
    // determine physical page size and resource UV scaling for the given LOD:
    float4 InvPageSize_UVScale = g_TiledResourceConstants[IndexMapSlot].PhysicalPageUVSizePerLOD[ConstantLookupLOD];
    
    // adjust incoming UV coordinates to map to the proper scale for the index map lookup:
    float2 IndexMapUV = RoundedUV.xy * InvPageSize_UVScale.zw;
    
    // determine physical page location:
    // X component is the atlas X index
    // Y component is the atlas Y index
    // Z component is the normalized array slice index
    // W component is 255 if the entry is valid, 0 otherwise
    float4 PhysicalPageLocation = FetchFromIndexMap( IndexMapTexture, IndexMapSamplerState, IndexMapUV, IndexMapLOD.x );

    // convert resource UV space to page UV space:
    float2 PageUV = frac( TextureUV * InvPageSize_UVScale.xy );
    if( PhysicalPageLocation.w <= 0 )
    {
        // page fault; return the page UV and -1 for the array slice index
        g_ResidencyStatus = false;
        return float3( PageUV, -1 );
    }
    else
    {
        g_ResidencyStatus = true;
    }
    
    // offset the page UV by the border offset:
    PageUV = ( PageUV * g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.xy ) + g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.zw;
    
    // compute the atlas UV by offsetting the page UV by the atlas X and Y coordinates
    float2 AtlasUV = ( PageUV + PhysicalPageLocation.xy ) * g_PagePoolConstants[IndexMapSlot].ArrayTexConstants.zw;
    
    // return the 3D texture coordinate used to look up into the physical page array texture:
    return float3( AtlasUV, PhysicalPageLocation.z );
}

//--------------------------------------------------------------------------------------
// Name: ComputePhysicalPageUV
// Desc: Given a UV coord and index map slot, returns a physical page array texture 
//       location that should be sampled to produce the final texture sample.
//--------------------------------------------------------------------------------------
float3 ComputePhysicalPageUV( const float2 TextureUV, const float2 RoundedUV, uniform int IndexMapSlot, uniform SamplerState IndexMapSamplerState )
{
    return ComputePhysicalPageUV_MinLOD( TextureUV, RoundedUV, 0, IndexMapSlot, IndexMapSamplerState );
}

//--------------------------------------------------------------------------------------
// Name: ComputePhysicalPageUV_FixedLOD
// Desc: Given a UV coord and index map slot, returns a physical page array texture 
//       location that should be sampled to produce the final texture sample.
//       A fixed LOD value is provided instead of using a computed LOD value.
//--------------------------------------------------------------------------------------
float3 ComputePhysicalPageUV_FixedLOD( const float2 TextureUV, const float2 RoundedUV, const float FixedLOD, uniform int IndexMapSlot, uniform SamplerState IndexMapSamplerState )
{
    // Compute the mip LOD of the sample against the texture:
    Texture2D IndexMapTexture = g_IndexMapTexture2D[IndexMapSlot];

    // Clamp and round the LOD value to a whole number lookup index:
    float LOD = clamp( FixedLOD, 0, 8 );
    float IndexMapLOD = round( LOD );
    float ConstantLookupLOD = IndexMapLOD;
    
    // determine physical page size and resource UV scaling for the given LOD:
    float4 InvPageSize_UVScale = g_TiledResourceConstants[IndexMapSlot].PhysicalPageUVSizePerLOD[ConstantLookupLOD];
    
    // adjust incoming UV coordinates to map to the proper scale for the index map lookup:
    float2 IndexMapUV = RoundedUV.xy * InvPageSize_UVScale.zw;
    
    // determine physical page location:
    // X component is the atlas X index
    // Y component is the atlas Y index
    // Z component is the normalized array slice index
    // W component is 255 if the entry is valid, 0 otherwise
    float4 PhysicalPageLocation = FetchFromIndexMap( IndexMapTexture, IndexMapSamplerState, IndexMapUV, IndexMapLOD.x );
    
    // convert resource UV space to page UV space:
    float2 PageUV = frac( TextureUV * InvPageSize_UVScale.xy );
    if( PhysicalPageLocation.w <= 0 )
    {
        // page fault; return the page UV and -1 for the array slice index
        g_ResidencyStatus = false;
        return float3( PageUV, -1 );
    }
    else
    {
        g_ResidencyStatus = true;
    }
    
    // offset the page UV by the border offset:
    PageUV = ( PageUV * g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.xy ) + g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.zw;
    
    // compute the atlas UV by offsetting the page UV by the atlas X and Y coordinates
    float2 AtlasUV = ( PageUV + PhysicalPageLocation.xy ) * g_PagePoolConstants[IndexMapSlot].ArrayTexConstants.zw;
    
    // return the 3D texture coordinate used to look up into the physical page array texture:
    return float3( AtlasUV, PhysicalPageLocation.z );
}

//--------------------------------------------------------------------------------------
// Name: ComputeTrilinearPhysicalPageUV_MinLOD
// Desc: Given a UV coord and index map slot, returns a pair of physical page array texture 
//       locations that should be sampled and blended to produce the final texture sample.
//       The return value is the 0..1 lerp value for the trilinear blend.
//       A MinLOD value is provided that is used to clamp the computed LOD value.
//--------------------------------------------------------------------------------------
float ComputeTrilinearPhysicalPageUV_MinLOD( const float2 TextureUV, const float2 RoundedUV, const float MinLOD, uniform int IndexMapSlot, uniform SamplerState IndexMapSamplerState, out float3 PhysicalPageA, out float3 PhysicalPageB )
{
    // Compute the mip LOD of the sample against the texture:
    Texture2D IndexMapTexture = g_IndexMapTexture2D[IndexMapSlot];
    float LOD = Get2DLOD( IndexMapTexture, IndexMapSamplerState, TextureUV, IndexMapSlot );

    // Clamp and round the LOD value to two whole number lookup indices:
    LOD = clamp( LOD, MinLOD, 8 );
    float TrilinearLerp = frac( LOD );
    float IndexMapLODA = floor( LOD );
    float IndexMapLODB = ceil( LOD );
    float ConstantLookupLODA = IndexMapLODA;
    float ConstantLookupLODB = IndexMapLODB;
    
    // determine physical page size and resource UV scaling for the given LOD:
    float4 InvPageSize_UVScaleA = g_TiledResourceConstants[IndexMapSlot].PhysicalPageUVSizePerLOD[ConstantLookupLODA];
    float4 InvPageSize_UVScaleB = g_TiledResourceConstants[IndexMapSlot].PhysicalPageUVSizePerLOD[ConstantLookupLODB];
    
    // adjust incoming UV coordinates to map to the proper scale for the index map lookup:
    float2 IndexMapUVA = RoundedUV.xy * InvPageSize_UVScaleA.zw;
    float2 IndexMapUVB = RoundedUV.xy * InvPageSize_UVScaleB.zw;
    
    // determine physical page location:
    // X component is the atlas X index
    // Y component is the atlas Y index
    // Z component is the normalized array slice index
    // W component is 255 if the entry is valid, 0 otherwise
    float4 PhysicalPageLocationA = FetchFromIndexMap( IndexMapTexture, IndexMapSamplerState, IndexMapUVA, IndexMapLODA.x );
    float4 PhysicalPageLocationB = FetchFromIndexMap( IndexMapTexture, IndexMapSamplerState, IndexMapUVB, IndexMapLODB.x );
    
    // convert resource UV space to page UV space:
    float2 PageUVA = frac( TextureUV * InvPageSize_UVScaleA.xy );
    float2 PageUVB = frac( TextureUV * InvPageSize_UVScaleB.xy );

    bool ResidencyStatus = true;
    
    if( PhysicalPageLocationA.w <= 0 )
    {
        // page fault; return the page UV and -1 for the array slice index
        PhysicalPageA = float3( PageUVA, -1 );
        ResidencyStatus = false;
    }
    else
    {
        // offset the page UV by the border offset:
        PageUVA = ( PageUVA * g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.xy ) + g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.zw;
        
        // compute the atlas UV by offsetting the page UV by the atlas X and Y coordinates
        float2 AtlasUV = ( PageUVA + PhysicalPageLocationA.xy ) * g_PagePoolConstants[IndexMapSlot].ArrayTexConstants.zw;
        
        // return the 3D texture coordinate used to look up into the physical page array texture:
        PhysicalPageA = float3( AtlasUV, PhysicalPageLocationA.z );
    }
    
    if( PhysicalPageLocationB.w <= 0 )
    {
        // page fault; return the page UV and -1 for the array slice index
        PhysicalPageB = float3( PageUVB, -1 );
        
        ResidencyStatus = false;
    }
    else
    {
        // offset the page UV by the border offset:
        PageUVB = ( PageUVB * g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.xy ) + g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.zw;
        
        // compute the atlas UV by offsetting the page UV by the atlas X and Y coordinates
        float2 AtlasUV = ( PageUVB + PhysicalPageLocationB.xy ) * g_PagePoolConstants[IndexMapSlot].ArrayTexConstants.zw;
        
        // return the 3D texture coordinate used to look up into the physical page array texture:
        PhysicalPageB = float3( AtlasUV, PhysicalPageLocationB.z );
    }
    
    g_ResidencyStatus = ResidencyStatus;

    return TrilinearLerp;
}

//--------------------------------------------------------------------------------------
// Name: ComputeTrilinearPhysicalPageUV_MinLOD
// Desc: Given a UV coord and index map slot, returns a pair of physical page array texture 
//       locations that should be sampled and blended to produce the final texture sample.
//       The return value is the 0..1 lerp value for the trilinear blend.
//       A fixed LOD value is provided that is used instead of a computed LOD value.
//--------------------------------------------------------------------------------------
float ComputeTrilinearPhysicalPageUV_FixedLOD( const float2 TextureUV, const float2 RoundedUV, const float FixedLOD, uniform int IndexMapSlot, uniform SamplerState IndexMapSamplerState, out float3 PhysicalPageA, out float3 PhysicalPageB )
{
    Texture2D IndexMapTexture = g_IndexMapTexture2D[IndexMapSlot];

    // Clamp and round the LOD value to two whole number lookup indices:
    float LOD = clamp( FixedLOD, 0, 8 );
    float TrilinearLerp = frac( LOD );
    float IndexMapLODA = floor( LOD );
    float IndexMapLODB = ceil( LOD );
    float ConstantLookupLODA = IndexMapLODA;
    float ConstantLookupLODB = IndexMapLODB;
    
    // determine physical page size and resource UV scaling for the given LOD:
    float4 InvPageSize_UVScaleA = g_TiledResourceConstants[IndexMapSlot].PhysicalPageUVSizePerLOD[ConstantLookupLODA];
    float4 InvPageSize_UVScaleB = g_TiledResourceConstants[IndexMapSlot].PhysicalPageUVSizePerLOD[ConstantLookupLODB];
    
    // adjust incoming UV coordinates to map to the proper scale for the index map lookup:
    float2 IndexMapUVA = RoundedUV.xy * InvPageSize_UVScaleA.zw;
    float2 IndexMapUVB = RoundedUV.xy * InvPageSize_UVScaleB.zw;
    
    // determine physical page location:
    // X component is the atlas X index
    // Y component is the atlas Y index
    // Z component is the normalized array slice index
    // W component is 255 if the entry is valid, 0 otherwise
    float4 PhysicalPageLocationA = FetchFromIndexMap( IndexMapTexture, IndexMapSamplerState, IndexMapUVA, IndexMapLODA.x );
    float4 PhysicalPageLocationB = FetchFromIndexMap( IndexMapTexture, IndexMapSamplerState, IndexMapUVB, IndexMapLODB.x );
    
    // convert resource UV space to page UV space:
    float2 PageUVA = frac( TextureUV * InvPageSize_UVScaleA.xy );
    float2 PageUVB = frac( TextureUV * InvPageSize_UVScaleB.xy );

    bool ResidencyStatus = true;
    
    if( PhysicalPageLocationA.w <= 0 )
    {
        // page fault; return the page UV and -1 for the array slice index
        PhysicalPageA = float3( PageUVA, -1 );
        ResidencyStatus = false;
    }
    else
    {
        // offset the page UV by the border offset:
        PageUVA = ( PageUVA * g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.xy ) + g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.zw;
        
        // compute the atlas UV by offsetting the page UV by the atlas X and Y coordinates
        float2 AtlasUV = ( PageUVA + PhysicalPageLocationA.xy ) * g_PagePoolConstants[IndexMapSlot].ArrayTexConstants.zw;
        
        // return the 3D texture coordinate used to look up into the physical page array texture:
        PhysicalPageA = float3( AtlasUV, PhysicalPageLocationA.z );
    }
    
    if( PhysicalPageLocationB.w <= 0 )
    {
        // page fault; return the page UV and -1 for the array slice index
        PhysicalPageB = float3( PageUVB, -1 );
        
        ResidencyStatus = false;
    }
    else
    {
        // offset the page UV by the border offset:
        PageUVB = ( PageUVB * g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.xy ) + g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.zw;
        
        // compute the atlas UV by offsetting the page UV by the atlas X and Y coordinates
        float2 AtlasUV = ( PageUVB + PhysicalPageLocationB.xy ) * g_PagePoolConstants[IndexMapSlot].ArrayTexConstants.zw;
        
        // return the 3D texture coordinate used to look up into the physical page array texture:
        PhysicalPageB = float3( AtlasUV, PhysicalPageLocationB.z );
    }
    
    g_ResidencyStatus = ResidencyStatus;
    
    return TrilinearLerp;
}

//--------------------------------------------------------------------------------------
// Name: ComputeTrilinearPhysicalPageUV
// Desc: Given a UV coord and index map slot, returns a pair of physical page array texture 
//       locations that should be sampled and blended to produce the final texture sample.
//       The return value is the 0..1 lerp value for the trilinear blend.
//--------------------------------------------------------------------------------------
float ComputeTrilinearPhysicalPageUV( const float2 TextureUV, const float2 RoundedUV, uniform int IndexMapSlot, uniform SamplerState IndexMapSamplerState, out float3 PhysicalPageA, out float3 PhysicalPageB )
{
    return ComputeTrilinearPhysicalPageUV_MinLOD( TextureUV, RoundedUV, 0, IndexMapSlot, IndexMapSamplerState, PhysicalPageA, PhysicalPageB );
}

//--------------------------------------------------------------------------------------
// Name: ComputePhysicalPageUVW_MinLOD
// Desc: Given a UVW coord and index map slot, returns a physical page array texture 
//       location that should be sampled to produce the final texture sample.
//       A MinLOD value is provided that is used to clamp the computed LOD value.
//--------------------------------------------------------------------------------------
float3 ComputePhysicalPageUVW_MinLOD( const float3 TextureUVW, const float2 RoundedUV, const float MinLOD, uniform int IndexMapSlot, uniform SamplerState IndexMapSamplerState )
{
    // Compute the mip LOD of the sample against the texture:
    Texture2DArray IndexMapTexture = g_IndexMapTexture3D[IndexMapSlot];
    float LOD = Get2DLOD( IndexMapTexture, IndexMapSamplerState, TextureUVW, IndexMapSlot );

    // Clamp and round the LOD value to a whole number lookup index:
    LOD = clamp( LOD, MinLOD, 8 );
    float IndexMapLOD = round( LOD );
    float ConstantLookupLOD = IndexMapLOD;
    
    // determine physical page size and resource UV scaling for the given LOD:
    float4 InvPageSize_UVScale = g_TiledResourceConstants[IndexMapSlot].PhysicalPageUVSizePerLOD[ConstantLookupLOD];
    
    // adjust incoming UV coordinates to map to the proper scale for the index map lookup:
#ifdef SUPPORT_NON_POW2_DIMENSIONS
    float3 IndexMapUVW = float3( RoundedUV.xy * InvPageSize_UVScale.zw, TextureUVW.z );
#else
    float3 IndexMapUVW = float3( RoundedUV.xy, TextureUVW.z );
#endif
    
    // determine physical page location:
    // X component is the atlas X index
    // Y component is the atlas Y index
    // Z component is the normalized array slice index
    // W component is 255 if the entry is valid, 0 otherwise
    float4 PhysicalPageLocation = FetchFromIndexMapArray( IndexMapTexture, IndexMapSamplerState, IndexMapUVW, IndexMapLOD.x );
    
    // convert resource UV space to page UV space:
    float2 PageUV = frac( TextureUVW.xy * InvPageSize_UVScale.xy );
    if( PhysicalPageLocation.w <= 0 )
    {
        // page fault; return the page UV and -1 for the array slice index
        g_ResidencyStatus = false;
        return float3( PageUV, -1 );
    }
    else
    {
        g_ResidencyStatus = true;
    }
    
    // offset the page UV by the border offset:
    PageUV = ( PageUV * g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.xy ) + g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.zw;
    
    // compute the atlas UV by offsetting the page UV by the atlas X and Y coordinates
    float2 AtlasUV = ( PageUV + PhysicalPageLocation.xy ) * g_PagePoolConstants[IndexMapSlot].ArrayTexConstants.zw;
    
    // return the 3D texture coordinate used to look up into the physical page array texture:
    return float3( AtlasUV, PhysicalPageLocation.z );
}

//--------------------------------------------------------------------------------------
// Name: ComputePhysicalPageUVW
// Desc: Given a UVW coord and index map slot, returns a physical page array texture 
//       location that should be sampled to produce the final texture sample.
//--------------------------------------------------------------------------------------
float3 ComputePhysicalPageUVW( const float3 TextureUVW, const float2 RoundedUV, uniform int IndexMapSlot, uniform SamplerState IndexMapSamplerState )
{
    return ComputePhysicalPageUVW_MinLOD( TextureUVW, RoundedUV, 0, IndexMapSlot, IndexMapSamplerState );
}

//--------------------------------------------------------------------------------------
// Name: ComputePhysicalPageUVW_FixedLOD
// Desc: Given a UVW coord and index map slot, returns a physical page array texture 
//       location that should be sampled to produce the final texture sample.
//       A fixed LOD value is provided that is used instead of a computed LOD value.
//--------------------------------------------------------------------------------------
float3 ComputePhysicalPageUVW_FixedLOD( const float3 TextureUVW, const float2 RoundedUV, const float FixedLOD, uniform int IndexMapSlot, uniform SamplerState IndexMapSamplerState )
{
    // Compute the mip LOD of the sample against the texture:
    Texture2DArray IndexMapTexture = g_IndexMapTexture3D[IndexMapSlot];
    float LOD = clamp( FixedLOD, 0, 8 );

    // Clamp and round the LOD value to a whole number lookup index:
    float IndexMapLOD = round( LOD );
    float ConstantLookupLOD = IndexMapLOD;
    
    // determine physical page size and resource UV scaling for the given LOD:
    float4 InvPageSize_UVScale = g_TiledResourceConstants[IndexMapSlot].PhysicalPageUVSizePerLOD[ConstantLookupLOD];
    
    // adjust incoming UV coordinates to map to the proper scale for the index map lookup:
#ifdef SUPPORT_NON_POW2_DIMENSIONS
    float3 IndexMapUVW = float3( RoundedUV.xy * InvPageSize_UVScale.zw, TextureUVW.z );
#else
    float3 IndexMapUVW = float3( RoundedUV.xy, TextureUVW.z );
#endif

    // determine physical page location:
    // X component is the atlas X index
    // Y component is the atlas Y index
    // Z component is the normalized array slice index
    // W component is 255 if the entry is valid, 0 otherwise
    float4 PhysicalPageLocation = FetchFromIndexMapArray( IndexMapTexture, IndexMapSamplerState, IndexMapUVW, IndexMapLOD.x );
    
    // convert resource UV space to page UV space:
    float2 PageUV = frac( TextureUVW.xy * InvPageSize_UVScale.xy );
    if( PhysicalPageLocation.w <= 0 )
    {
        // page fault; return the page UV and -1 for the array slice index
        g_ResidencyStatus = false;
        return float3( PageUV, -1 );
    }
    else
    {
        g_ResidencyStatus = true;
    }
    
    // offset the page UV by the border offset:
    PageUV = ( PageUV * g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.xy ) + g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.zw;
    
    // compute the atlas UV by offsetting the page UV by the atlas X and Y coordinates
    float2 AtlasUV = ( PageUV + PhysicalPageLocation.xy ) * g_PagePoolConstants[IndexMapSlot].ArrayTexConstants.zw;
    
    // return the 3D texture coordinate used to look up into the physical page array texture:
    return float3( AtlasUV, PhysicalPageLocation.z );
}

//--------------------------------------------------------------------------------------
// Name: ComputeTrilinearPhysicalPageUVW_MinLOD
// Desc: Given a UVW coord and index map slot, returns a pair of physical page array texture 
//       locations that should be sampled and blended to produce the final texture sample.
//       The return value is the 0..1 lerp value for the trilinear blend.
//       A MinLOD value is provided that is used to clamp the computed LOD value.
//--------------------------------------------------------------------------------------
float ComputeTrilinearPhysicalPageUVW_MinLOD( const float3 TextureUVW, const float2 RoundedUV, const float MinLOD, uniform int IndexMapSlot, uniform SamplerState IndexMapSamplerState, out float3 PhysicalPageA, out float3 PhysicalPageB )
{
    // Compute the mip LOD of the sample against the texture:
    Texture2DArray IndexMapTexture = g_IndexMapTexture3D[IndexMapSlot];
    float LOD = Get2DLOD( IndexMapTexture, IndexMapSamplerState, TextureUVW, IndexMapSlot );

    // Clamp and round the LOD value to two whole number lookup indices:
    LOD = clamp( LOD, MinLOD, 8 );
    float TrilinearLerp = frac( LOD );
    float IndexMapLODA = floor( LOD );
    float IndexMapLODB = ceil( LOD );
    float ConstantLookupLODA = IndexMapLODA;
    float ConstantLookupLODB = IndexMapLODB;
    
    // determine physical page size and resource UV scaling for the given LOD:
    float4 InvPageSize_UVScaleA = g_TiledResourceConstants[IndexMapSlot].PhysicalPageUVSizePerLOD[ConstantLookupLODA];
    float4 InvPageSize_UVScaleB = g_TiledResourceConstants[IndexMapSlot].PhysicalPageUVSizePerLOD[ConstantLookupLODB];
    
    // adjust incoming UV coordinates to map to the proper scale for the index map lookup:
#ifdef SUPPORT_NON_POW2_DIMENSIONS
    float3 IndexMapUVWA = float3( RoundedUV.xy * InvPageSize_UVScaleA.zw, TextureUVW.z );
    float3 IndexMapUVWB = float3( RoundedUV.xy * InvPageSize_UVScaleB.zw, TextureUVW.z );
#else
    float3 IndexMapUVWA = float3( RoundedUV.xy, TextureUVW.z );
    float3 IndexMapUVWB = float3( RoundedUV.xy, TextureUVW.z );
#endif
    
    // determine physical page location:
    // X component is the atlas X index
    // Y component is the atlas Y index
    // Z component is the normalized array slice index
    // W component is 255 if the entry is valid, 0 otherwise
    float4 PhysicalPageLocationA = FetchFromIndexMapArray( IndexMapTexture, IndexMapSamplerState, IndexMapUVWA, IndexMapLODA.x );
    float4 PhysicalPageLocationB = FetchFromIndexMapArray( IndexMapTexture, IndexMapSamplerState, IndexMapUVWB, IndexMapLODB.x );
    
    // convert resource UV space to page UV space:
    float2 PageUVA = frac( TextureUVW.xy * InvPageSize_UVScaleA.xy );
    float2 PageUVB = frac( TextureUVW.xy * InvPageSize_UVScaleB.xy );
    
    bool ResidencyStatus = true;
    
    if( PhysicalPageLocationA.w <= 0 )
    {
        // page fault; return the page UV and -1 for the array slice index
        PhysicalPageA = float3( PageUVA, -1 );
        ResidencyStatus = false;
    }
    else
    {
        // offset the page UV by the border offset:
        PageUVA = ( PageUVA * g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.xy ) + g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.zw;
        
        // compute the atlas UV by offsetting the page UV by the atlas X and Y coordinates
        float2 AtlasUV = ( PageUVA + PhysicalPageLocationA.xy ) * g_PagePoolConstants[IndexMapSlot].ArrayTexConstants.zw;
        
        // return the 3D texture coordinate used to look up into the physical page array texture:
        PhysicalPageA = float3( AtlasUV, PhysicalPageLocationA.z );
    }
    
    if( PhysicalPageLocationB.w <= 0 )
    {
        // page fault; return the page UV and -1 for the array slice index
        PhysicalPageB = float3( PageUVB, -1 );
        
        ResidencyStatus = false;
    }
    else
    {
        // offset the page UV by the border offset:
        PageUVB = ( PageUVB * g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.xy ) + g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.zw;
        
        // compute the atlas UV by offsetting the page UV by the atlas X and Y coordinates
        float2 AtlasUV = ( PageUVB + PhysicalPageLocationB.xy ) * g_PagePoolConstants[IndexMapSlot].ArrayTexConstants.zw;
        
        // return the 3D texture coordinate used to look up into the physical page array texture:
        PhysicalPageB = float3( AtlasUV, PhysicalPageLocationB.z );
    }
    
    g_ResidencyStatus = ResidencyStatus;
    
    return TrilinearLerp;
}

//--------------------------------------------------------------------------------------
// Name: ComputeTrilinearPhysicalPageUVW
// Desc: Given a UVW coord and index map slot, returns a pair of physical page array texture 
//       locations that should be sampled and blended to produce the final texture sample.
//       The return value is the 0..1 lerp value for the trilinear blend.
//--------------------------------------------------------------------------------------
float ComputeTrilinearPhysicalPageUVW( const float3 TextureUVW, const float2 RoundedUV, uniform int IndexMapSlot, uniform SamplerState IndexMapSamplerState, out float3 PhysicalPageA, out float3 PhysicalPageB )
{
    return ComputeTrilinearPhysicalPageUVW_MinLOD( TextureUVW, RoundedUV, 0, IndexMapSlot, IndexMapSamplerState, PhysicalPageA, PhysicalPageB );
}

//--------------------------------------------------------------------------------------
// Name: ComputeTrilinearPhysicalPageUVW_FixedLOD
// Desc: Given a UVW coord and index map slot, returns a pair of physical page array texture 
//       locations that should be sampled and blended to produce the final texture sample.
//       The return value is the 0..1 lerp value for the trilinear blend.
//       A fixed LOD value is provided that is used instead of a computed LOD value.
//--------------------------------------------------------------------------------------
float ComputeTrilinearPhysicalPageUVW_FixedLOD( const float3 TextureUVW, const float2 RoundedUV, const float FixedLOD, uniform int IndexMapSlot, uniform SamplerState IndexMapSamplerState, out float3 PhysicalPageA, out float3 PhysicalPageB )
{
    Texture2DArray IndexMapTexture = g_IndexMapTexture3D[IndexMapSlot];

    // Clamp and round the LOD value to two whole number lookup indices:
    float LOD = clamp( FixedLOD, 0, 8 );
    float TrilinearLerp = frac( LOD );
    float IndexMapLODA = floor( LOD );
    float IndexMapLODB = ceil( LOD );
    float ConstantLookupLODA = IndexMapLODA;
    float ConstantLookupLODB = IndexMapLODB;
    
    // determine physical page size and resource UV scaling for the given LOD:
    float4 InvPageSize_UVScaleA = g_TiledResourceConstants[IndexMapSlot].PhysicalPageUVSizePerLOD[ConstantLookupLODA];
    float4 InvPageSize_UVScaleB = g_TiledResourceConstants[IndexMapSlot].PhysicalPageUVSizePerLOD[ConstantLookupLODB];
    
    // adjust incoming UV coordinates to map to the proper scale for the index map lookup:
#ifdef SUPPORT_NON_POW2_DIMENSIONS
    float3 IndexMapUVWA = float3( RoundedUV.xy * InvPageSize_UVScaleA.zw, TextureUVW.z );
    float3 IndexMapUVWB = float3( RoundedUV.xy * InvPageSize_UVScaleB.zw, TextureUVW.z );
#else
    float3 IndexMapUVWA = float3( RoundedUV.xy, TextureUVW.z );
    float3 IndexMapUVWB = float3( RoundedUV.xy, TextureUVW.z );
#endif
    
    // determine physical page location:
    // X component is the atlas X index
    // Y component is the atlas Y index
    // Z component is the normalized array slice index
    // W component is 255 if the entry is valid, 0 otherwise
    float4 PhysicalPageLocationA = FetchFromIndexMapArray( IndexMapTexture, IndexMapSamplerState, IndexMapUVWA, IndexMapLODA.x );
    float4 PhysicalPageLocationB = FetchFromIndexMapArray( IndexMapTexture, IndexMapSamplerState, IndexMapUVWB, IndexMapLODB.x );
    
    // convert resource UV space to page UV space:
    float2 PageUVA = frac( TextureUVW.xy * InvPageSize_UVScaleA.xy );
    float2 PageUVB = frac( TextureUVW.xy * InvPageSize_UVScaleB.xy );
    
    bool ResidencyStatus = true;
    
    if( PhysicalPageLocationA.w <= 0 )
    {
        // page fault; return the page UV and -1 for the array slice index
        PhysicalPageA = float3( PageUVA, -1 );
        ResidencyStatus = false;
    }
    else
    {
        // offset the page UV by the border offset:
        PageUVA = ( PageUVA * g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.xy ) + g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.zw;
        
        // compute the atlas UV by offsetting the page UV by the atlas X and Y coordinates
        float2 AtlasUV = ( PageUVA + PhysicalPageLocationA.xy ) * g_PagePoolConstants[IndexMapSlot].ArrayTexConstants.zw;
        
        // return the 3D texture coordinate used to look up into the physical page array texture:
        PhysicalPageA = float3( AtlasUV, PhysicalPageLocationA.z );
    }
    
    if( PhysicalPageLocationB.w <= 0 )
    {
        // page fault; return the page UV and -1 for the array slice index
        PhysicalPageB = float3( PageUVB, -1 );
        
        ResidencyStatus = false;
    }
    else
    {
        // offset the page UV by the border offset:
        PageUVB = ( PageUVB * g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.xy ) + g_PagePoolConstants[IndexMapSlot].PageBorderUVTransform.zw;
        
        // compute the atlas UV by offsetting the page UV by the atlas X and Y coordinates
        float2 AtlasUV = ( PageUVB + PhysicalPageLocationB.xy ) * g_PagePoolConstants[IndexMapSlot].ArrayTexConstants.zw;
        
        // return the 3D texture coordinate used to look up into the physical page array texture:
        PhysicalPageB = float3( AtlasUV, PhysicalPageLocationB.z );
    }
    
    g_ResidencyStatus = ResidencyStatus;
    
    return TrilinearLerp;
}

//--------------------------------------------------------------------------------------
//
// BEGIN PUBLIC FUNCTIONS
//
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
// Name: GetResidencyStatus
// Desc: Returns true if the last tiled resource sample fell completely within resident
//       pages; returns false otherwise.
//--------------------------------------------------------------------------------------
bool GetResidencyStatus()
{
    return g_ResidencyStatus;
}

//--------------------------------------------------------------------------------------
// Name: GetQuiltDimensions
// Desc: Returns the quilt dimensions of the given tiled resource slot.
//--------------------------------------------------------------------------------------
void GetQuiltDimensions( uniform int IndexMapSlot, out float QuiltWidth, out float QuiltHeight )
{
    const float4 QuiltConstants = g_TiledResourceConstants[IndexMapSlot].ResourceConstant;
    QuiltWidth = QuiltConstants.z;
    QuiltHeight = QuiltConstants.w;
}

//--------------------------------------------------------------------------------------
// Name: Quilt2DToTex3D
// Desc: Converts an extended UV address formatted for quilting into a normalized UVW
//       address that indexes into an array texture.
//--------------------------------------------------------------------------------------
float3 Quilt2DToTex3D( uniform int IndexMapSlot, const float2 UV )
{
    const float4 QuiltConstants = g_TiledResourceConstants[IndexMapSlot].ResourceConstant;
    
    int RowIndex = clamp( (int)UV.y, 0, QuiltConstants.w );
    int ColumnIndex = clamp( (int)UV.x, 0, QuiltConstants.z );
    int SliceIndex = RowIndex * QuiltConstants.z + ColumnIndex;
    
    float SliceAddress = (float)SliceIndex;
    
    return float3( frac( UV ), SliceAddress );
}

//--------------------------------------------------------------------------------------
//
// BEGIN PUBLIC TEXTURE SAMPLING FUNCTIONS
//
// Note the "RoundedUV" parameter - this parameter must be prepared in a special way on
// AMD Radeon hardware.  The incoming UV coordinates must be rounded to the nearest
// texel center on the base level of the tiled resource's sampling quality map.  This
// works around a texture filtering problem.  On all other hardware, the RoundedUV
// parameter can simply be the incoming 2D texture coordinates, without modification.
//
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
// Name: TiledTex2D_Point
// Desc: Samples a tiled texture 2D using point sampling.
//--------------------------------------------------------------------------------------
float4 TiledTex2D_Point( uniform int IndexMapSlot, const float2 UV, const float2 RoundedUV )
{
    float3 PhysicalCoords = ComputePhysicalPageUV( UV, RoundedUV, IndexMapSlot, IndexMapSamplerState );
    if( PhysicalCoords.z < 0 )
    {
        return 0;
        return float4( PhysicalCoords.xy, 0, 1 );
    }
    Texture2DArray PhysicalPageTexture = g_PhysicalPageTexture[IndexMapSlot];
    float4 Sample = PhysicalPageTexture.Sample( PhysicalPageSamplerStatePoint, PhysicalCoords );
    return Sample;
}

//--------------------------------------------------------------------------------------
// Name: TiledTex2D_Point_MinLOD
// Desc: Samples a tiled texture 2D using point sampling, using a minimum LOD clamp value.
//--------------------------------------------------------------------------------------
float4 TiledTex2D_Point_MinLOD( uniform int IndexMapSlot, const float2 UV, const float2 RoundedUV, const float MinLOD )
{
    float3 PhysicalCoords = ComputePhysicalPageUV_MinLOD( UV, RoundedUV, MinLOD, IndexMapSlot, IndexMapSamplerState );
    if( PhysicalCoords.z < 0 )
    {
        return 0;
        return float4( PhysicalCoords.xy, 0, 1 );
    }
    Texture2DArray PhysicalPageTexture = g_PhysicalPageTexture[IndexMapSlot];
    float4 Sample = PhysicalPageTexture.Sample( PhysicalPageSamplerStatePoint, PhysicalCoords );
    return Sample;
}

//--------------------------------------------------------------------------------------
// Name: TiledTex2D_Point_FixedLOD
// Desc: Samples a tiled texture 2D using point sampling, using a fixed LOD value.
//--------------------------------------------------------------------------------------
float4 TiledTex2D_Point_FixedLOD( uniform int IndexMapSlot, const float2 UV, const float2 RoundedUV, const float FixedLOD )
{
    float3 PhysicalCoords = ComputePhysicalPageUV_FixedLOD( UV, RoundedUV, FixedLOD, IndexMapSlot, IndexMapSamplerState );
    if( PhysicalCoords.z < 0 )
    {
        return 0;
        return float4( PhysicalCoords.xy, 0, 1 );
    }
    Texture2DArray PhysicalPageTexture = g_PhysicalPageTexture[IndexMapSlot];
    float4 Sample = PhysicalPageTexture.SampleLevel( PhysicalPageSamplerStatePoint, PhysicalCoords, 0 );
    return Sample;
}

//--------------------------------------------------------------------------------------
// Name: TiledTex2D_Bilinear
// Desc: Samples a tiled texture 2D using bilinear sampling.
//--------------------------------------------------------------------------------------
float4 TiledTex2D_Bilinear( uniform int IndexMapSlot, const float2 UV, const float2 RoundedUV )
{
    float3 PhysicalCoords = ComputePhysicalPageUV( UV, RoundedUV, IndexMapSlot, IndexMapSamplerState );
    if( PhysicalCoords.z < 0 )
    {
        return 0;
        return float4( PhysicalCoords.xy, 0, 1 );
    }
    Texture2DArray PhysicalPageTexture = g_PhysicalPageTexture[IndexMapSlot];
    float4 Sample = PhysicalPageTexture.Sample( PhysicalPageSamplerStateBilinear, PhysicalCoords );
    return Sample;
}

//--------------------------------------------------------------------------------------
// Name: TiledTex2D_Bilinear_MinLOD
// Desc: Samples a tiled texture 2D using bilinear sampling, using a minimum LOD clamp value.
//--------------------------------------------------------------------------------------
float4 TiledTex2D_Bilinear_MinLOD( uniform int IndexMapSlot, const float2 UV, const float2 RoundedUV, const float MinLOD )
{
    float3 PhysicalCoords = ComputePhysicalPageUV_MinLOD( UV, RoundedUV, MinLOD, IndexMapSlot, IndexMapSamplerState );
    if( PhysicalCoords.z < 0 )
    {
        return 0;
        return float4( PhysicalCoords.xy, 0, 1 );
    }
    Texture2DArray PhysicalPageTexture = g_PhysicalPageTexture[IndexMapSlot];
    float4 Sample = PhysicalPageTexture.Sample( PhysicalPageSamplerStateBilinear, PhysicalCoords );
    return Sample;
}

//--------------------------------------------------------------------------------------
// Name: TiledTex2D_Trilinear
// Desc: Samples a tiled texture 2D using trilinear sampling.  Note that the trilinear
//       sample is emulated with two bilinear samples and a lerp.
//--------------------------------------------------------------------------------------
float4 TiledTex2D_Trilinear( uniform int IndexMapSlot, const float2 UV, const float2 RoundedUV )
{
    float3 PhysicalCoordsA;
    float3 PhysicalCoordsB;
    float TrilinearLerp = ComputeTrilinearPhysicalPageUV( UV, RoundedUV, IndexMapSlot, IndexMapSamplerState, PhysicalCoordsA, PhysicalCoordsB );

    Texture2DArray PhysicalPageTexture = g_PhysicalPageTexture[IndexMapSlot];
    float4 SampleA;
    if( PhysicalCoordsA.z >= 0 )
    {
        SampleA = PhysicalPageTexture.Sample( PhysicalPageSamplerStateBilinear, PhysicalCoordsA );
    }
    else
    {
        SampleA = float4( PhysicalCoordsA.xy, 0, 1 );
        SampleA = 0;
    }

    float4 SampleB;
    if( PhysicalCoordsB.z >= 0 )
    {
        SampleB = PhysicalPageTexture.Sample( PhysicalPageSamplerStateBilinear, PhysicalCoordsB );
    }
    else
    {
        SampleB = float4( PhysicalCoordsB.xy, 0, 1 );
        SampleB = 0;
    }

    return lerp( SampleA, SampleB, TrilinearLerp );
}

//--------------------------------------------------------------------------------------
// Name: TiledTex2D_Trilinear_MinLOD
// Desc: Samples a tiled texture 2D using trilinear sampling and a minimum LOD clamp value.  
//       Note that the trilinear sample is emulated with two bilinear samples and a lerp.
//--------------------------------------------------------------------------------------
float4 TiledTex2D_Trilinear_MinLOD( uniform int IndexMapSlot, const float2 UV, const float2 RoundedUV, const float MinLOD )
{
    float3 PhysicalCoordsA;
    float3 PhysicalCoordsB;
    float TrilinearLerp = ComputeTrilinearPhysicalPageUV_MinLOD( UV, RoundedUV, MinLOD, IndexMapSlot, IndexMapSamplerState, PhysicalCoordsA, PhysicalCoordsB );

    Texture2DArray PhysicalPageTexture = g_PhysicalPageTexture[IndexMapSlot];
    float4 SampleA;
    if( PhysicalCoordsA.z >= 0 )
    {
        SampleA = PhysicalPageTexture.Sample( PhysicalPageSamplerStateBilinear, PhysicalCoordsA );
    }
    else
    {
        SampleA = float4( PhysicalCoordsA.xy, 0, 1 );
        SampleA = 0;
    }

    float4 SampleB;
    if( PhysicalCoordsB.z >= 0 )
    {
        SampleB = PhysicalPageTexture.Sample( PhysicalPageSamplerStateBilinear, PhysicalCoordsB );
    }
    else
    {
        SampleB = float4( PhysicalCoordsB.xy, 0, 1 );
        SampleB = 0;
    }

    return lerp( SampleA, SampleB, TrilinearLerp );
}

//--------------------------------------------------------------------------------------
// Name: TiledTex2D_Trilinear_FixedLOD
// Desc: Samples a tiled texture 2D using trilinear sampling and a fixed LOD value.  
//       Note that the trilinear sample is emulated with two bilinear samples and a lerp.
//--------------------------------------------------------------------------------------
float4 TiledTex2D_Trilinear_FixedLOD( uniform int IndexMapSlot, const float2 UV, const float2 RoundedUV, const float FixedLOD )
{
    float3 PhysicalCoordsA;
    float3 PhysicalCoordsB;
    float TrilinearLerp = ComputeTrilinearPhysicalPageUV_FixedLOD( UV, RoundedUV, FixedLOD, IndexMapSlot, IndexMapSamplerState, PhysicalCoordsA, PhysicalCoordsB );

    Texture2DArray PhysicalPageTexture = g_PhysicalPageTexture[IndexMapSlot];
    float4 SampleA;
    if( PhysicalCoordsA.z >= 0 )
    {
        SampleA = PhysicalPageTexture.SampleLevel( PhysicalPageSamplerStateBilinear, PhysicalCoordsA, 0 );
    }
    else
    {
        SampleA = float4( PhysicalCoordsA.xy, 0, 1 );
        SampleA = 0;
    }

    float4 SampleB;
    if( PhysicalCoordsB.z >= 0 )
    {
        SampleB = PhysicalPageTexture.SampleLevel( PhysicalPageSamplerStateBilinear, PhysicalCoordsB, 0 );
    }
    else
    {
        SampleB = float4( PhysicalCoordsB.xy, 0, 1 );
        SampleB = 0;
    }

    return lerp( SampleA, SampleB, TrilinearLerp );
}

//--------------------------------------------------------------------------------------
// Name: TiledTex3D_Point
// Desc: Samples a tiled texture 2D array using point sampling.  
//--------------------------------------------------------------------------------------
float4 TiledTex3D_Point( uniform int IndexMapSlot, const float3 UVW, const float2 RoundedUV )
{
    float3 PhysicalCoords = ComputePhysicalPageUVW( UVW, RoundedUV, IndexMapSlot, IndexMapSamplerState );
    if( PhysicalCoords.z < 0 )
    {
        return 0;
        return float4( PhysicalCoords.xy, 0, 1 );
    }
    Texture2DArray PhysicalPageTexture = g_PhysicalPageTexture[IndexMapSlot];
    float4 Sample = PhysicalPageTexture.Sample( PhysicalPageSamplerStatePoint, PhysicalCoords );
    return Sample;
}

//--------------------------------------------------------------------------------------
// Name: TiledTex3D_Point_MinLOD
// Desc: Samples a tiled texture 2D array using point sampling and a minimum LOD clamp value. 
//--------------------------------------------------------------------------------------
float4 TiledTex3D_Point_MinLOD( uniform int IndexMapSlot, const float3 UVW, const float2 RoundedUV, const float MinLOD )
{
    float3 PhysicalCoords = ComputePhysicalPageUVW_MinLOD( UVW, RoundedUV, MinLOD, IndexMapSlot, IndexMapSamplerState );
    if( PhysicalCoords.z < 0 )
    {
        return 0;
        return float4( PhysicalCoords.xy, 0, 1 );
    }
    Texture2DArray PhysicalPageTexture = g_PhysicalPageTexture[IndexMapSlot];
    float4 Sample = PhysicalPageTexture.Sample( PhysicalPageSamplerStatePoint, PhysicalCoords );
    return Sample;
}

//--------------------------------------------------------------------------------------
// Name: TiledTex3D_Point_FixedLOD
// Desc: Samples a tiled texture 2D array using point sampling and a fixed LOD value. 
//--------------------------------------------------------------------------------------
float4 TiledTex3D_Point_FixedLOD( uniform int IndexMapSlot, const float3 UVW, const float2 RoundedUV, const float FixedLOD )
{
    float3 PhysicalCoords = ComputePhysicalPageUVW_FixedLOD( UVW, RoundedUV, FixedLOD, IndexMapSlot, IndexMapSamplerState );
    if( PhysicalCoords.z < 0 )
    {
        return 0;
        return float4( PhysicalCoords.xy, 0, 1 );
    }
    Texture2DArray PhysicalPageTexture = g_PhysicalPageTexture[IndexMapSlot];
    float4 Sample = PhysicalPageTexture.Sample( PhysicalPageSamplerStatePoint, PhysicalCoords );
    return Sample;
}

//--------------------------------------------------------------------------------------
// Name: TiledTex3D_Bilinear
// Desc: Samples a tiled texture 2D array using bilinear sampling. 
//--------------------------------------------------------------------------------------
float4 TiledTex3D_Bilinear( uniform int IndexMapSlot, const float3 UVW, const float2 RoundedUV )
{
    float3 PhysicalCoords = ComputePhysicalPageUVW( UVW, RoundedUV, IndexMapSlot, IndexMapSamplerState );
    if( PhysicalCoords.z < 0 )
    {
        return 0;
        return float4( PhysicalCoords.xy, 0, 1 );
    }
    Texture2DArray PhysicalPageTexture = g_PhysicalPageTexture[IndexMapSlot];
    float4 Sample = PhysicalPageTexture.Sample( PhysicalPageSamplerStateBilinear, PhysicalCoords );
    return Sample;
}

//--------------------------------------------------------------------------------------
// Name: TiledTex3D_Bilinear_MinLOD
// Desc: Samples a tiled texture 2D array using bilinear sampling and a minimum LOD clamp value. 
//--------------------------------------------------------------------------------------
float4 TiledTex3D_Bilinear_MinLOD( uniform int IndexMapSlot, const float3 UVW, const float2 RoundedUV, const float MinLOD )
{
    float3 PhysicalCoords = ComputePhysicalPageUVW_MinLOD( UVW, RoundedUV, MinLOD, IndexMapSlot, IndexMapSamplerState );
    if( PhysicalCoords.z < 0 )
    {
        return 0;
        return float4( PhysicalCoords.xy, 0, 1 );
    }
    Texture2DArray PhysicalPageTexture = g_PhysicalPageTexture[IndexMapSlot];
    float4 Sample = PhysicalPageTexture.Sample( PhysicalPageSamplerStateBilinear, PhysicalCoords );
    return Sample;
}

//--------------------------------------------------------------------------------------
// Name: TiledTex3D_Trilinear
// Desc: Samples a tiled texture 2D array using trilinear sampling. 
//       Note that the trilinear sample is emulated with two bilinear samples and a lerp.
//--------------------------------------------------------------------------------------
float4 TiledTex3D_Trilinear( uniform int IndexMapSlot, const float3 UVW, const float2 RoundedUV )
{
    float3 PhysicalCoordsA;
    float3 PhysicalCoordsB;
    float TrilinearLerp = ComputeTrilinearPhysicalPageUVW( UVW, RoundedUV, IndexMapSlot, IndexMapSamplerState, PhysicalCoordsA, PhysicalCoordsB );

    Texture2DArray PhysicalPageTexture = g_PhysicalPageTexture[IndexMapSlot];
    float4 SampleA;
    if( PhysicalCoordsA.z >= 0 )
    {
        SampleA = PhysicalPageTexture.Sample( PhysicalPageSamplerStateBilinear, PhysicalCoordsA );
    }
    else
    {
        SampleA = float4( PhysicalCoordsA.xy, 0, 1 );
        SampleA = 0;
    }

    float4 SampleB;
    if( PhysicalCoordsB.z >= 0 )
    {
        SampleB = PhysicalPageTexture.Sample( PhysicalPageSamplerStateBilinear, PhysicalCoordsB );
    }
    else
    {
        SampleB = float4( PhysicalCoordsB.xy, 0, 1 );
        SampleB = 0;
    }

    return lerp( SampleA, SampleB, TrilinearLerp );
}

//--------------------------------------------------------------------------------------
// Name: TiledTex3D_Trilinear_MinLOD
// Desc: Samples a tiled texture 2D array using trilinear sampling and a minimum LOD clamp value.
//       Note that the trilinear sample is emulated with two bilinear samples and a lerp.
//--------------------------------------------------------------------------------------
float4 TiledTex3D_Trilinear_MinLOD( uniform int IndexMapSlot, const float3 UVW, const float2 RoundedUV, const float MinLOD )
{
    float3 PhysicalCoordsA;
    float3 PhysicalCoordsB;
    float TrilinearLerp = ComputeTrilinearPhysicalPageUVW_MinLOD( UVW, RoundedUV, MinLOD, IndexMapSlot, IndexMapSamplerState, PhysicalCoordsA, PhysicalCoordsB );

    Texture2DArray PhysicalPageTexture = g_PhysicalPageTexture[IndexMapSlot];
    float4 SampleA;
    if( PhysicalCoordsA.z >= 0 )
    {
        SampleA = PhysicalPageTexture.Sample( PhysicalPageSamplerStateBilinear, PhysicalCoordsA );
    }
    else
    {
        SampleA = float4( PhysicalCoordsA.xy, 0, 1 );
        SampleA = 0;
    }

    float4 SampleB;
    if( PhysicalCoordsB.z >= 0 )
    {
        SampleB = PhysicalPageTexture.Sample( PhysicalPageSamplerStateBilinear, PhysicalCoordsB );
    }
    else
    {
        SampleB = float4( PhysicalCoordsB.xy, 0, 1 );
        SampleB = 0;
    }

    return lerp( SampleA, SampleB, TrilinearLerp );
}

//--------------------------------------------------------------------------------------
// Name: TiledTex3D_Trilinear_FixedLOD
// Desc: Samples a tiled texture 2D array using trilinear sampling and a fixed LOD value.
//       Note that the trilinear sample is emulated with two bilinear samples and a lerp.
//--------------------------------------------------------------------------------------
float4 TiledTex3D_Trilinear_FixedLOD( uniform int IndexMapSlot, const float3 UVW, const float2 RoundedUV, const float FixedLOD )
{
    float3 PhysicalCoordsA;
    float3 PhysicalCoordsB;
    float TrilinearLerp = ComputeTrilinearPhysicalPageUVW_FixedLOD( UVW, RoundedUV, FixedLOD, IndexMapSlot, IndexMapSamplerState, PhysicalCoordsA, PhysicalCoordsB );

    Texture2DArray PhysicalPageTexture = g_PhysicalPageTexture[IndexMapSlot];
    float4 SampleA;
    if( PhysicalCoordsA.z >= 0 )
    {
        SampleA = PhysicalPageTexture.Sample( PhysicalPageSamplerStateBilinear, PhysicalCoordsA );
    }
    else
    {
        SampleA = float4( PhysicalCoordsA.xy, 0, 1 );
        SampleA = 0;
    }

    float4 SampleB;
    if( PhysicalCoordsB.z >= 0 )
    {
        SampleB = PhysicalPageTexture.Sample( PhysicalPageSamplerStateBilinear, PhysicalCoordsB );
    }
    else
    {
        SampleB = float4( PhysicalCoordsB.xy, 0, 1 );
        SampleB = 0;
    }

    return lerp( SampleA, SampleB, TrilinearLerp );
}

//--------------------------------------------------------------------------------------
// Name: TiledTex3D_ComputeLOD
// Desc: Computes the mipmap LOD of the given 3D texture coordinates on the given tiled
//       texture 2D array.
//--------------------------------------------------------------------------------------
float TiledTex3D_ComputeLOD( uniform int IndexMapSlot, const float3 UVW )
{
    float LOD = Get2DLOD( g_IndexMapTexture3D[IndexMapSlot], IndexMapSamplerState, UVW, IndexMapSlot );
    return LOD;
}

//--------------------------------------------------------------------------------------
// Name: TiledTex2D_ComputeLOD
// Desc: Computes the mipmap LOD of the given texture coordinates on the given tiled
//       texture 2D.
//--------------------------------------------------------------------------------------
float TiledTex2D_ComputeLOD( uniform int IndexMapSlot, const float2 UV )
{
    float LOD = Get2DLOD( g_IndexMapTexture2D[IndexMapSlot], IndexMapSamplerState, UV, IndexMapSlot );
    return LOD;
}

