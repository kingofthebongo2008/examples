//--------------------------------------------------------------------------------------
// TiledResourceD3D11.h
//
// Utility methods and defines for tiled resources, specific to the D3D11 implementation.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#ifdef _XBOX
#error This file is for Windows D3D11 only.
#endif

// These D3DCOLOR macros are only included in D3D9:
#define D3DCOLOR_GETALPHA(argb)      (((argb) >> 24) & 0xff)
#define D3DCOLOR_GETRED(argb)        (((argb) >> 16) & 0xff)
#define D3DCOLOR_GETGREEN(argb)      (((argb) >> 8) & 0xff)
#define D3DCOLOR_GETBLUE(argb)       ((argb) & 0xff)

// PIX named events on D3D11 go through a DXUT utility function:
#define PIXBeginNamedEvent(c,s) DXUT_BeginPerfEvent(c,TEXT(s))
#define PIXEndNamedEvent DXUT_EndPerfEvent

// DXGI_FORMAT_MAX gives us a maximum index so we can loop over DXGI formats, 
// or create arrays sized to the DXGI format list:
#define DXGI_FORMAT_MAX (DXGI_FORMAT_BC7_UNORM_SRGB + 1)

//--------------------------------------------------------------------------------------
// Name: XGNextMultiple
// Desc: Computes the next multiple value for the given value and multiple.  For example,
//       the next multiple of 5 with a value of 8 is 10.
//--------------------------------------------------------------------------------------
__inline UINT XGNextMultiple(
    UINT Value,
    UINT Multiple
    )
{
    return ((Value + Multiple - 1) / Multiple) * Multiple;
}

namespace TiledRuntime
{
    // MAX_ARRAY_SLICES is the maximum size of an array texture in D3D11:
    static const UINT MAX_ARRAY_SLICES = 1024;

    // We use 32bpp RGBA unorm for the GPU index map texture, because 565 BGR is not commonly available:
    static const DXGI_FORMAT DXGI_FORMAT_INDEXMAP = DXGI_FORMAT_R8G8B8A8_UNORM;

    //--------------------------------------------------------------------------------------
    // Name: CreateZeroedArrayTexture
    // Desc: Creates a staging or default usage array texture with the given parameters.
    //--------------------------------------------------------------------------------------
    inline VOID CreateZeroedArrayTexture( ID3D11Device* pd3dDevice, UINT Width, UINT Height, UINT SliceCount, UINT LevelCount, DXGI_FORMAT Format, BOOL StagingTexture, ID3D11Texture2D** ppArrayTexture )
    {
        D3D11_TEXTURE2D_DESC TexDesc = { 0 };
        TexDesc.ArraySize = SliceCount;
        TexDesc.Format = Format;
        TexDesc.Width = Width;
        TexDesc.Height = Height;
        TexDesc.MipLevels = LevelCount;
        TexDesc.MiscFlags = 0;
        TexDesc.SampleDesc.Count = 1;
        TexDesc.SampleDesc.Quality = 0;

        if( StagingTexture )
        {
            TexDesc.Usage = D3D11_USAGE_STAGING;
            TexDesc.BindFlags = 0;
            TexDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
        }
        else
        {
            TexDesc.Usage = D3D11_USAGE_DEFAULT;
            TexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            TexDesc.CPUAccessFlags = 0;
        }

        ASSERT( ppArrayTexture != NULL );
        HRESULT hr = pd3dDevice->CreateTexture2D( &TexDesc, NULL, ppArrayTexture );
        ASSERT( SUCCEEDED(hr) && *ppArrayTexture != NULL );
    }

    //--------------------------------------------------------------------------------------
    // Name: CreateZeroedTexture2D
    // Desc: Creates a staging or default usage texture2D with the given parameters.
    //--------------------------------------------------------------------------------------
    inline VOID CreateZeroedTexture2D( ID3D11Device* pd3dDevice, UINT Width, UINT Height, UINT LevelCount, DXGI_FORMAT Format, BOOL StagingTexture, ID3D11Texture2D** ppTexture )
    {
        CreateZeroedArrayTexture( pd3dDevice, Width, Height, 1, LevelCount, Format, StagingTexture, ppTexture );
    }

    //--------------------------------------------------------------------------------------
    // Name: GetMipLevelSize
    // Desc: Computes the size of the given mip level of the given texture2D.
    //--------------------------------------------------------------------------------------
    inline SIZE GetMipLevelSize( ID3D11Texture2D* pTex2D, UINT MipLevel )
    {
        D3D11_TEXTURE2D_DESC TexDesc;
        pTex2D->GetDesc( &TexDesc );

        ASSERT( MipLevel < TexDesc.MipLevels );

        SIZE MipSize = { TexDesc.Width, TexDesc.Height };
        while( MipLevel > 0 )
        {
            MipSize.cx = max( 1, MipSize.cx / 2 );
            MipSize.cy = max( 1, MipSize.cy / 2 );
            --MipLevel;
        }

        return MipSize;
    }

    //--------------------------------------------------------------------------------------
    // Name: GetBytesPerTexel
    // Desc: Returns the number of bytes per texel of the given DXGI format.  Note that block
    //       compressed formats return 0; their memory consumption is computed in separate
    //       codepaths.
    //--------------------------------------------------------------------------------------
    inline UINT GetBytesPerTexel( const DXGI_FORMAT DataFormat )
    {
        switch( DataFormat )
        {
        case DXGI_FORMAT_R8_TYPELESS : 
        case DXGI_FORMAT_R8_UNORM: 
        case DXGI_FORMAT_R8_UINT : 
        case DXGI_FORMAT_R8_SNORM: 
        case DXGI_FORMAT_R8_SINT : 
        case DXGI_FORMAT_A8_UNORM: 
            return 1;
        case DXGI_FORMAT_R8G8_TYPELESS : 
        case DXGI_FORMAT_R8G8_UNORM: 
        case DXGI_FORMAT_R8G8_UINT : 
        case DXGI_FORMAT_R8G8_SNORM: 
        case DXGI_FORMAT_R8G8_SINT : 
        case DXGI_FORMAT_R16_TYPELESS: 
        case DXGI_FORMAT_R16_FLOAT : 
        case DXGI_FORMAT_D16_UNORM : 
        case DXGI_FORMAT_R16_UNORM : 
        case DXGI_FORMAT_R16_UINT: 
        case DXGI_FORMAT_R16_SNORM : 
        case DXGI_FORMAT_R16_SINT: 
        case DXGI_FORMAT_B5G6R5_UNORM: 
        case DXGI_FORMAT_B5G5R5A1_UNORM: 
            return 2;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: 
        case DXGI_FORMAT_R10G10B10A2_UNORM : 
        case DXGI_FORMAT_R10G10B10A2_UINT: 
        case DXGI_FORMAT_R11G11B10_FLOAT : 
        case DXGI_FORMAT_R8G8B8A8_TYPELESS : 
        case DXGI_FORMAT_R8G8B8A8_UNORM: 
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : 
        case DXGI_FORMAT_R8G8B8A8_UINT : 
        case DXGI_FORMAT_R8G8B8A8_SNORM: 
        case DXGI_FORMAT_R8G8B8A8_SINT : 
        case DXGI_FORMAT_R16G16_TYPELESS : 
        case DXGI_FORMAT_R16G16_FLOAT: 
        case DXGI_FORMAT_R16G16_UNORM: 
        case DXGI_FORMAT_R16G16_UINT : 
        case DXGI_FORMAT_R16G16_SNORM: 
        case DXGI_FORMAT_R16G16_SINT : 
        case DXGI_FORMAT_R32_TYPELESS: 
        case DXGI_FORMAT_D32_FLOAT : 
        case DXGI_FORMAT_R32_FLOAT : 
        case DXGI_FORMAT_R32_UINT: 
        case DXGI_FORMAT_R32_SINT: 
        case DXGI_FORMAT_R9G9B9E5_SHAREDEXP: 
        case DXGI_FORMAT_B8G8R8A8_UNORM: 
        case DXGI_FORMAT_B8G8R8X8_UNORM: 
        case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM: 
        case DXGI_FORMAT_B8G8R8A8_TYPELESS : 
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB : 
        case DXGI_FORMAT_B8G8R8X8_TYPELESS : 
        case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB : 
            return 4;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_UNORM:
        case DXGI_FORMAT_R16G16B16A16_UINT : 
        case DXGI_FORMAT_R16G16B16A16_SNORM: 
        case DXGI_FORMAT_R16G16B16A16_SINT : 
        case DXGI_FORMAT_R32G32_TYPELESS : 
        case DXGI_FORMAT_R32G32_FLOAT: 
        case DXGI_FORMAT_R32G32_UINT : 
        case DXGI_FORMAT_R32G32_SINT : 
            return 8;
        case DXGI_FORMAT_R32G32B32A32_TYPELESS:
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
        case DXGI_FORMAT_R32G32B32A32_UINT:
        case DXGI_FORMAT_R32G32B32A32_SINT:
            return 16;
        case DXGI_FORMAT_BC1_TYPELESS: 
        case DXGI_FORMAT_BC1_UNORM : 
        case DXGI_FORMAT_BC1_UNORM_SRGB: 
        case DXGI_FORMAT_BC4_TYPELESS: 
        case DXGI_FORMAT_BC4_UNORM : 
        case DXGI_FORMAT_BC4_SNORM : 
            // BC textures return 0:
            return 0;
        case DXGI_FORMAT_BC2_TYPELESS: 
        case DXGI_FORMAT_BC2_UNORM : 
        case DXGI_FORMAT_BC2_UNORM_SRGB: 
        case DXGI_FORMAT_BC3_TYPELESS: 
        case DXGI_FORMAT_BC3_UNORM : 
        case DXGI_FORMAT_BC3_UNORM_SRGB: 
        case DXGI_FORMAT_BC5_TYPELESS: 
        case DXGI_FORMAT_BC5_UNORM : 
        case DXGI_FORMAT_BC5_SNORM : 
        case DXGI_FORMAT_BC6H_TYPELESS : 
        case DXGI_FORMAT_BC6H_UF16 : 
        case DXGI_FORMAT_BC6H_SF16 : 
        case DXGI_FORMAT_BC7_TYPELESS: 
        case DXGI_FORMAT_BC7_UNORM : 
        case DXGI_FORMAT_BC7_UNORM_SRGB: 
            // BC textures return 0:
            return 0;
        case DXGI_FORMAT_R32G32B32_TYPELESS:
        case DXGI_FORMAT_R32G32B32_FLOAT:
        case DXGI_FORMAT_R32G32B32_UINT:
        case DXGI_FORMAT_R32G32B32_SINT:
        case DXGI_FORMAT_R32G8X24_TYPELESS : 
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: 
        case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: 
        case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT : 
        case DXGI_FORMAT_R24G8_TYPELESS: 
        case DXGI_FORMAT_D24_UNORM_S8_UINT : 
        case DXGI_FORMAT_R24_UNORM_X8_TYPELESS : 
        case DXGI_FORMAT_X24_TYPELESS_G8_UINT: 
        case DXGI_FORMAT_R1_UNORM: 
        case DXGI_FORMAT_R8G8_B8G8_UNORM : 
        case DXGI_FORMAT_G8R8_G8B8_UNORM : 
            // Unsupported formats!
            RIP;
            return 0;
        default:
            return 0;
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: GetPageSizeTexels
    // Desc: Gets a single page's size in texels, for the given format.
    //--------------------------------------------------------------------------------------
    inline SIZE GetPageSizeTexels( const DXGI_FORMAT DataFormat )
    {
        SIZE TexelSize = { 0, 0 };

        switch( DataFormat )
        {
        case DXGI_FORMAT_R8_TYPELESS : 
        case DXGI_FORMAT_R8_UNORM: 
        case DXGI_FORMAT_R8_UINT : 
        case DXGI_FORMAT_R8_SNORM: 
        case DXGI_FORMAT_R8_SINT : 
        case DXGI_FORMAT_A8_UNORM: 
            TexelSize.cx = 256;
            TexelSize.cy = 256;
            break;
        case DXGI_FORMAT_R8G8_TYPELESS : 
        case DXGI_FORMAT_R8G8_UNORM: 
        case DXGI_FORMAT_R8G8_UINT : 
        case DXGI_FORMAT_R8G8_SNORM: 
        case DXGI_FORMAT_R8G8_SINT : 
        case DXGI_FORMAT_R16_TYPELESS: 
        case DXGI_FORMAT_R16_FLOAT : 
        case DXGI_FORMAT_D16_UNORM : 
        case DXGI_FORMAT_R16_UNORM : 
        case DXGI_FORMAT_R16_UINT: 
        case DXGI_FORMAT_R16_SNORM : 
        case DXGI_FORMAT_R16_SINT: 
        case DXGI_FORMAT_B5G6R5_UNORM: 
        case DXGI_FORMAT_B5G5R5A1_UNORM: 
            TexelSize.cx = 256;
            TexelSize.cy = 128;
            break;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: 
        case DXGI_FORMAT_R10G10B10A2_UNORM : 
        case DXGI_FORMAT_R10G10B10A2_UINT: 
        case DXGI_FORMAT_R11G11B10_FLOAT : 
        case DXGI_FORMAT_R8G8B8A8_TYPELESS : 
        case DXGI_FORMAT_R8G8B8A8_UNORM: 
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : 
        case DXGI_FORMAT_R8G8B8A8_UINT : 
        case DXGI_FORMAT_R8G8B8A8_SNORM: 
        case DXGI_FORMAT_R8G8B8A8_SINT : 
        case DXGI_FORMAT_R16G16_TYPELESS : 
        case DXGI_FORMAT_R16G16_FLOAT: 
        case DXGI_FORMAT_R16G16_UNORM: 
        case DXGI_FORMAT_R16G16_UINT : 
        case DXGI_FORMAT_R16G16_SNORM: 
        case DXGI_FORMAT_R16G16_SINT : 
        case DXGI_FORMAT_R32_TYPELESS: 
        case DXGI_FORMAT_D32_FLOAT : 
        case DXGI_FORMAT_R32_FLOAT : 
        case DXGI_FORMAT_R32_UINT: 
        case DXGI_FORMAT_R32_SINT: 
        case DXGI_FORMAT_R9G9B9E5_SHAREDEXP: 
        case DXGI_FORMAT_B8G8R8A8_UNORM: 
        case DXGI_FORMAT_B8G8R8X8_UNORM: 
        case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM: 
        case DXGI_FORMAT_B8G8R8A8_TYPELESS : 
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB : 
        case DXGI_FORMAT_B8G8R8X8_TYPELESS : 
        case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB : 
            TexelSize.cx = 128;
            TexelSize.cy = 128;
            break;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_UNORM:
        case DXGI_FORMAT_R16G16B16A16_UINT : 
        case DXGI_FORMAT_R16G16B16A16_SNORM: 
        case DXGI_FORMAT_R16G16B16A16_SINT : 
        case DXGI_FORMAT_R32G32_TYPELESS : 
        case DXGI_FORMAT_R32G32_FLOAT: 
        case DXGI_FORMAT_R32G32_UINT : 
        case DXGI_FORMAT_R32G32_SINT : 
            TexelSize.cx = 128;
            TexelSize.cy = 64;
            break;
        case DXGI_FORMAT_R32G32B32A32_TYPELESS:
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
        case DXGI_FORMAT_R32G32B32A32_UINT:
        case DXGI_FORMAT_R32G32B32A32_SINT:
            TexelSize.cx = 64;
            TexelSize.cy = 64;
            break;
        case DXGI_FORMAT_BC1_TYPELESS: 
        case DXGI_FORMAT_BC1_UNORM : 
        case DXGI_FORMAT_BC1_UNORM_SRGB: 
        case DXGI_FORMAT_BC4_TYPELESS: 
        case DXGI_FORMAT_BC4_UNORM : 
        case DXGI_FORMAT_BC4_SNORM : 
            TexelSize.cx = 512;
            TexelSize.cy = 256;
            break;
        case DXGI_FORMAT_BC2_TYPELESS: 
        case DXGI_FORMAT_BC2_UNORM : 
        case DXGI_FORMAT_BC2_UNORM_SRGB: 
        case DXGI_FORMAT_BC3_TYPELESS: 
        case DXGI_FORMAT_BC3_UNORM : 
        case DXGI_FORMAT_BC3_UNORM_SRGB: 
        case DXGI_FORMAT_BC5_TYPELESS: 
        case DXGI_FORMAT_BC5_UNORM : 
        case DXGI_FORMAT_BC5_SNORM : 
        case DXGI_FORMAT_BC6H_SF16:
        case DXGI_FORMAT_BC6H_UF16:
        case DXGI_FORMAT_BC6H_TYPELESS:
        case DXGI_FORMAT_BC7_TYPELESS:
        case DXGI_FORMAT_BC7_UNORM:
        case DXGI_FORMAT_BC7_UNORM_SRGB:
            TexelSize.cx = 256;
            TexelSize.cy = 256;
            break;
        default:
            // Many formats are unsupported for tiled resources:
            RIP;
            break;
        }

        return TexelSize;
    }

    //--------------------------------------------------------------------------------------
    // Name: GetPageBorderTexelCount
    // Desc: Returns the width of the per-page texel border in the physical page array texture
    //       atlas.  Compressed formats have a one block border (4 texels), while
    //       uncompressed formats have a one texel border.
    //--------------------------------------------------------------------------------------
    inline UINT GetPageBorderTexelCount( const DXGI_FORMAT DataFormat )
    {
        switch( DataFormat )
        {
        case DXGI_FORMAT_BC1_TYPELESS: 
        case DXGI_FORMAT_BC1_UNORM : 
        case DXGI_FORMAT_BC1_UNORM_SRGB: 
        case DXGI_FORMAT_BC4_TYPELESS: 
        case DXGI_FORMAT_BC4_UNORM : 
        case DXGI_FORMAT_BC4_SNORM : 
        case DXGI_FORMAT_BC2_TYPELESS: 
        case DXGI_FORMAT_BC2_UNORM : 
        case DXGI_FORMAT_BC2_UNORM_SRGB: 
        case DXGI_FORMAT_BC3_TYPELESS: 
        case DXGI_FORMAT_BC3_UNORM : 
        case DXGI_FORMAT_BC3_UNORM_SRGB: 
        case DXGI_FORMAT_BC5_TYPELESS: 
        case DXGI_FORMAT_BC5_UNORM : 
        case DXGI_FORMAT_BC5_SNORM : 
        case DXGI_FORMAT_BC6H_TYPELESS : 
        case DXGI_FORMAT_BC6H_UF16 : 
        case DXGI_FORMAT_BC6H_SF16 : 
        case DXGI_FORMAT_BC7_TYPELESS: 
        case DXGI_FORMAT_BC7_UNORM : 
        case DXGI_FORMAT_BC7_UNORM_SRGB: 
            return 4;
        case DXGI_FORMAT_UNKNOWN:
        case DXGI_FORMAT_R32G32B32_TYPELESS:
        case DXGI_FORMAT_R32G32B32_FLOAT:
        case DXGI_FORMAT_R32G32B32_UINT:
        case DXGI_FORMAT_R32G32B32_SINT:
        case DXGI_FORMAT_R32G8X24_TYPELESS : 
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: 
        case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: 
        case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT : 
        case DXGI_FORMAT_R24G8_TYPELESS: 
        case DXGI_FORMAT_D24_UNORM_S8_UINT : 
        case DXGI_FORMAT_R24_UNORM_X8_TYPELESS : 
        case DXGI_FORMAT_X24_TYPELESS_G8_UINT: 
        case DXGI_FORMAT_R1_UNORM: 
        case DXGI_FORMAT_R8G8_B8G8_UNORM : 
        case DXGI_FORMAT_G8R8_G8B8_UNORM : 
            // Unsupported formats!
            RIP;
            return 0;
        default:
            return 1;
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: GetTextureSizeBytes
    // Desc: Returns an estimate of the number of video memory bytes used by a D3D11 texture.
    //--------------------------------------------------------------------------------------
    inline UINT64 GetTextureSizeBytes( ID3D11Texture2D* pTex2D )
    {
        // Get the base dimensions of the texture:
        D3D11_TEXTURE2D_DESC TexDesc;
        pTex2D->GetDesc( &TexDesc );

        UINT64 TexelCount = 0;

        // Add up the texel count in all of the mip levels:
        while( TexDesc.MipLevels > 0 )
        {
            TexelCount += ( TexDesc.Width * TexDesc.Height );
            TexDesc.Width = max( 1, TexDesc.Width / 2 );
            TexDesc.Height = max( 1, TexDesc.Height / 2 );
            --TexDesc.MipLevels;
        }

        // Multiply the texel count by the array size:
        TexelCount *= TexDesc.ArraySize;

        // Get the bytes per pixel:
        UINT BytesPerPixel = GetBytesPerTexel( TexDesc.Format );

        // Adjust bytes per pixel and texel count for compressed formats:
        switch( TexDesc.Format )
        {
        case DXGI_FORMAT_BC1_TYPELESS: 
        case DXGI_FORMAT_BC1_UNORM : 
        case DXGI_FORMAT_BC1_UNORM_SRGB: 
        case DXGI_FORMAT_BC4_TYPELESS: 
        case DXGI_FORMAT_BC4_UNORM : 
        case DXGI_FORMAT_BC4_SNORM : 
            BytesPerPixel = 8;
            TexelCount /= 16;
            break;
        case DXGI_FORMAT_BC2_TYPELESS: 
        case DXGI_FORMAT_BC2_UNORM : 
        case DXGI_FORMAT_BC2_UNORM_SRGB: 
        case DXGI_FORMAT_BC3_TYPELESS: 
        case DXGI_FORMAT_BC3_UNORM : 
        case DXGI_FORMAT_BC3_UNORM_SRGB: 
        case DXGI_FORMAT_BC5_TYPELESS: 
        case DXGI_FORMAT_BC5_UNORM : 
        case DXGI_FORMAT_BC5_SNORM : 
        case DXGI_FORMAT_BC6H_TYPELESS : 
        case DXGI_FORMAT_BC6H_UF16 : 
        case DXGI_FORMAT_BC6H_SF16 : 
        case DXGI_FORMAT_BC7_TYPELESS: 
        case DXGI_FORMAT_BC7_UNORM : 
        case DXGI_FORMAT_BC7_UNORM_SRGB: 
            BytesPerPixel = 16;
            TexelCount /= 16;
            break;
        }

        // Compute total memory consumption:
        return TexelCount * (UINT64)BytesPerPixel;
    }

    //--------------------------------------------------------------------------------------
    // Name: GetPagePoolArrayFormat
    // Desc: Returns the aliased page pool array format for a given DXGI format.  On D3D11,
    //       we cannot perform any format aliasing, so this method returns the input parameter.
    //--------------------------------------------------------------------------------------
    inline DXGI_FORMAT GetPagePoolArrayTextureFormat( DXGI_FORMAT DataFormat )
    {
        return DataFormat;
    }
}

