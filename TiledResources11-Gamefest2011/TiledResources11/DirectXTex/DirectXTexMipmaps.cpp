//-------------------------------------------------------------------------------------
// DirectXTexMipMaps.cpp
//  
// DirectX Texture Library - Mip-map generation
//
// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//-------------------------------------------------------------------------------------

#include <windows.h>

#include "directxtexp.h"

#include "scoped.h"

#include <vector>

namespace DirectX
{

static const XMVECTORF32 s_boxScale = { 0.25f, 0.25f, 0.25f, 0.25f };
static const XMVECTORF32 s_boxScale3D = { 0.125f, 0.125f, 0.125f, 0.125f };

//-------------------------------------------------------------------------------------
// Mipmap helper functions
//-------------------------------------------------------------------------------------
inline static bool ispow2(size_t x)
{
    return ((x != 0) && !(x & (x - 1)));
}

static size_t _CountMips(size_t width, size_t height)
{
    size_t mipLevels = 1;

    while ( height > 1 || width > 1 )
    {
        if ( height > 1 )
            height >>= 1;

        if ( width > 1 )
            width >>= 1;

        ++mipLevels;
    }
    
    return mipLevels;
}

bool _CalculateMipLevels( _In_ size_t width, _In_ size_t height, _Inout_ size_t& mipLevels )
{
    if ( mipLevels > 1 )
    {
        if ( !ispow2(width) || !ispow2(height) )
            return false;

        size_t maxMips = _CountMips(width,height);
        if ( mipLevels > maxMips )
            return false;
    }
    else if ( mipLevels == 0 && ispow2(width) && ispow2(height) )
    {
        mipLevels = _CountMips(width,height);
    }
    else
    {
        mipLevels = 1;
    }
    return true;
}

static size_t _CountMips3D(size_t width, size_t height, size_t depth)
{
    size_t mipLevels = 1;

    while ( height > 1 || width > 1 || depth > 1 )
    {
        if ( height > 1 )
            height >>= 1;

        if ( width > 1 )
            width >>= 1;

        if ( depth > 1 )
            depth >>= 1;

        ++mipLevels;
    }
    
    return mipLevels;
}

bool _CalculateMipLevels3D( _In_ size_t width, _In_ size_t height, _In_ size_t depth, _Inout_ size_t& mipLevels )
{
    if ( mipLevels > 1 )
    {
        if ( !ispow2(width) || !ispow2(height) || !ispow2(depth) )
            return false;

        size_t maxMips = _CountMips3D(width,height,depth);
        if ( mipLevels > maxMips )
            return false;
    }
    else if ( mipLevels == 0 && ispow2(width) && ispow2(height) && ispow2(depth) )
    {
        mipLevels = _CountMips3D(width,height,depth);
    }
    else
    {
        mipLevels = 1;
    }
    return true;
}


//-------------------------------------------------------------------------------------
// Generate a (2D) mip-map chain from a base image using WIC's image scaler
//-------------------------------------------------------------------------------------
static HRESULT _GenerateMipMapsUsingWIC( _In_ const Image& baseImage, _In_ DWORD filter, _In_ size_t levels,
                                         _In_ const WICPixelFormatGUID& pfGUID, _In_ const ScratchImage& mipChain, _In_ size_t item )
{
    assert( levels > 1 );

    if ( !baseImage.pixels || !mipChain.GetPixels() )
        return E_POINTER;

    IWICImagingFactory* pWIC = _GetWIC();
    if ( !pWIC )
        return E_NOINTERFACE;

    size_t width = baseImage.width;
    size_t height = baseImage.height;

    assert( ispow2(width) && ispow2(height) );

    ScopedObject<IWICBitmap> source;
    HRESULT hr = pWIC->CreateBitmapFromMemory( static_cast<UINT>( width ), static_cast<UINT>( height ), pfGUID,
                                               static_cast<UINT>( baseImage.rowPitch ), static_cast<UINT>( baseImage.slicePitch ),
                                               baseImage.pixels, &source );
    if ( FAILED(hr) )
        return hr;

    // Copy base image to top miplevel
    const Image *img0 = mipChain.GetImage( 0, item, 0 );
    if ( !img0 )
        return E_POINTER;

    uint8_t* pDest = img0->pixels;
    if ( !pDest )
        return E_POINTER;

    const uint8_t *pSrc = baseImage.pixels;
    for( size_t h=0; h < height; ++h )
    {
        size_t msize = min( img0->rowPitch, baseImage.rowPitch );
        memcpy_s( pDest, img0->rowPitch, pSrc, msize );  
        pSrc += baseImage.rowPitch;
        pDest += img0->rowPitch;
    }

    // Resize base image to each target mip level
    for( size_t level = 1; level < levels; ++level )
    {
        const Image *img = mipChain.GetImage( level, item, 0 );
        if ( !img )
            return E_POINTER;

        ScopedObject<IWICBitmapScaler> scaler;
        hr = pWIC->CreateBitmapScaler( &scaler );
        if ( FAILED(hr) )
            return hr;

        if ( height > 1 )
            height >>= 1;

        if ( width > 1 )
            width >>= 1;

        assert( img->width == width && img->height == height && img->format == baseImage.format );

        hr = scaler->Initialize( source.Get(), static_cast<UINT>( width ), static_cast<UINT>( height ), _GetWICInterp( filter ) );
        if ( FAILED(hr) )
            return hr;

        WICPixelFormatGUID pfScaler;
        hr = scaler->GetPixelFormat( &pfScaler );
        if ( FAILED(hr) )
            return hr;

        if ( memcmp( &pfScaler, &pfGUID, sizeof(WICPixelFormatGUID) ) == 0 )
        {
            hr = scaler->CopyPixels( 0, static_cast<UINT>( img->rowPitch ), static_cast<UINT>( img->slicePitch ), img->pixels );
            if ( FAILED(hr) )
                return hr;
        }
        else
        {
            // The WIC bitmap scaler is free to return a different pixel format than the source image, so here we
            // convert it back
            ScopedObject<IWICFormatConverter> FC;
            hr = pWIC->CreateFormatConverter( &FC );
            if ( FAILED(hr) )
                return hr;

            hr = FC->Initialize( scaler.Get(), pfGUID, _GetWICDither( filter ), 0, 0, WICBitmapPaletteTypeCustom );
            if ( FAILED(hr) )
                return hr;

            hr = FC->CopyPixels( 0, static_cast<UINT>( img->rowPitch ), static_cast<UINT>( img->slicePitch ), img->pixels );  
            if ( FAILED(hr) )
                return hr;
        }
    }

    return S_OK;
}


//-------------------------------------------------------------------------------------
// Generate volume mip-map helpers
//-------------------------------------------------------------------------------------
static HRESULT _Setup3DMips( _In_count_(depth) const Image* baseImages, _In_ size_t depth, size_t levels,
                             _Out_ ScratchImage& mipChain )
{
    if ( !baseImages || !depth )
        return E_INVALIDARG;

    assert( levels > 1 );

    size_t width = baseImages[0].width;
    size_t height = baseImages[0].height;

    HRESULT hr = mipChain.Initialize3D( baseImages[0].format, width, height, depth, levels );
    if ( FAILED(hr) )
        return hr;

    // Copy base images to top slice
    for( size_t slice=0; slice < depth; ++slice )
    {
        const Image& src = baseImages[slice];

        const Image *dest = mipChain.GetImage( 0, 0, slice );
        if ( !dest )
        {
            mipChain.Release();
            return E_POINTER;
        }

        assert( src.format == dest->format );

        uint8_t* pDest = dest->pixels;
        if ( !pDest )
        {
            mipChain.Release();
            return E_POINTER;
        }

        const uint8_t *pSrc = src.pixels;
        size_t rowPitch = src.rowPitch;
        for( size_t h=0; h < height; ++h )
        {
            size_t msize = min( dest->rowPitch, rowPitch );
            memcpy_s( pDest, dest->rowPitch, pSrc, msize );  
            pSrc += rowPitch;
            pDest += dest->rowPitch;
        }
    }

    return S_OK;
}

static HRESULT _Generate3DMipsPointFilter( _In_ size_t depth, _In_ size_t levels, _In_ const ScratchImage& mipChain )
{
    if ( !depth || !mipChain.GetImages() )
        return E_INVALIDARG;

    // This assumes that the base images are already placed into the mipChain at the top level... (see _Setup3DMips)

    assert( levels > 1 );

    size_t width = mipChain.GetMetadata().width;
    size_t height = mipChain.GetMetadata().height;

    assert( ispow2(width) && ispow2(height) && ispow2(depth) );

    // Allocate temporary space (2 scanlines)
    ScopedAlignedArray<XMVECTOR> scanline( reinterpret_cast<XMVECTOR*>( _aligned_malloc( (sizeof(XMVECTOR)*width*2), 16 ) ) );
    if ( scanline.IsNull() )
        return E_OUTOFMEMORY;

    XMVECTOR* target = scanline.Get();

    XMVECTOR* row = target + width;

    // Resize base image to each target mip level
    for( size_t level=1; level < levels; ++level )
    {
        if ( depth > 1 )
        {
            // 3D point filter
            for( size_t slice=0; slice < depth; slice += 2 )
            {
                const Image* src = mipChain.GetImage( level-1, 0, slice );
                const Image* dest = mipChain.GetImage( level, 0, slice >> 1 );

                if ( !src || !dest )
                    return E_POINTER;

                const uint8_t* pSrc = src->pixels;
                uint8_t* pDest = dest->pixels;

                size_t rowPitch = src->rowPitch;

                size_t nheight = height >> 1;

                for( size_t y = 0; y < nheight; ++y )
                {
                    if ( !_LoadScanline( row, width, pSrc, rowPitch, src->format ) )
                        return E_FAIL;
                    pSrc += rowPitch*2;

                    size_t nwidth = width >> 1;

                    for( size_t x = 0; x < nwidth; ++x )
                    {
                        target[ x ] = row[ x*2 ];
                    }

                    if ( !_StoreScanline( pDest, dest->rowPitch, dest->format, target, nwidth ) )
                        return E_FAIL;
                    pDest += dest->rowPitch;
                }
            }
        }
        else
        {
            // 2D point filter
            const Image* src = mipChain.GetImage( level-1, 0, 0 );
            const Image* dest = mipChain.GetImage( level, 0, 0 );

            if ( !src || !dest )
                return E_POINTER;

            const uint8_t* pSrc = src->pixels;
            uint8_t* pDest = dest->pixels;

            size_t rowPitch = src->rowPitch;

            size_t nheight = height >> 1;

            for( size_t y = 0; y < nheight; ++y )
            {
                if ( !_LoadScanline( row, width, pSrc, rowPitch, src->format ) )
                    return E_FAIL;
                pSrc += rowPitch*2;

                size_t nwidth = width >> 1;

                for( size_t x = 0; x < nwidth; ++x )
                {
                    target[ x ] = row[ x*2 ];
                }

                if ( !_StoreScanline( pDest, dest->rowPitch, dest->format, target, nwidth ) )
                    return E_FAIL;
                pDest += dest->rowPitch;
            }
        }

        if ( height > 1 )
            height >>= 1;

        if ( width > 1 )
            width >>= 1;

        if ( depth > 1 )
            depth >>= 1;
    }

    assert( height == 1 && width == 1 && depth == 1 );

    return S_OK;
}

static HRESULT _Generate3DMipsBoxFilter( _In_ size_t depth, _In_ size_t levels, _In_ const ScratchImage& mipChain )
{
    if ( !depth || !mipChain.GetImages() )
        return E_INVALIDARG;

    // This assumes that the base images are already placed into the mipChain at the top level... (see _Setup3DMips)

    assert( levels > 1 );

    size_t width = mipChain.GetMetadata().width;
    size_t height = mipChain.GetMetadata().height;

    assert( ispow2(width) && ispow2(height) && ispow2(depth) );

    // Allocate temporary space (5 scanlines)
    ScopedAlignedArray<XMVECTOR> scanline( reinterpret_cast<XMVECTOR*>( _aligned_malloc( (sizeof(XMVECTOR)*width*5), 16 ) ) );
    if ( scanline.IsNull() )
        return E_OUTOFMEMORY;

    XMVECTOR* target = scanline.Get();

    XMVECTOR* urow0 = target + width;
    XMVECTOR* urow1 = target + width*2;
    XMVECTOR* vrow0 = target + width*3;
    XMVECTOR* vrow1 = target + width*4;

    const XMVECTOR* urow2 = urow0 + 1;
    const XMVECTOR* urow3 = urow1 + 1;
    const XMVECTOR* vrow2 = vrow0 + 1;
    const XMVECTOR* vrow3 = vrow1 + 1;

    // Resize base image to each target mip level
    for( size_t level=1; level < levels; ++level )
    {
        if ( height == 1)
        {
            urow0 = vrow0;
            urow1 = vrow1;
        }

        if ( width == 1 )
        {
            urow2 = urow0;
            urow3 = urow1;
            vrow2 = vrow0;
            vrow3 = vrow1;
        }

        if ( depth > 1 )
        {
            // 3D box filter
            for( size_t slice=0; slice < depth; slice += 2 )
            {
                const Image* srca = mipChain.GetImage( level-1, 0, slice );
                const Image* srcb = mipChain.GetImage( level-1, 0, slice+1 );
                const Image* dest = mipChain.GetImage( level, 0, slice >> 1 );

                if ( !srca || !srcb || !dest )
                    return E_POINTER;

                const uint8_t* pSrc1 = srca->pixels;
                const uint8_t* pSrc2 = srcb->pixels;
                uint8_t* pDest = dest->pixels;

                size_t aRowPitch = srca->rowPitch;
                size_t bRowPitch = srcb->rowPitch;

                size_t nheight = height >> 1;

                for( size_t y = 0; y < nheight; ++y )
                {
                    if ( !_LoadScanline( urow0, width, pSrc1, aRowPitch, srca->format ) )
                        return E_FAIL;
                    pSrc1 += aRowPitch;

                    if ( urow0 != urow1 )
                    {
                        if ( !_LoadScanline( urow1, width, pSrc1, aRowPitch, srca->format ) )
                            return E_FAIL;
                        pSrc1 += aRowPitch;
                    }

                    if ( urow0 != vrow0 )
                    {
                        if ( !_LoadScanline( vrow0, width, pSrc2, bRowPitch, srcb->format ) )
                            return E_FAIL;
                        pSrc2 += bRowPitch;
                    }

                    if ( urow0 != vrow1 && vrow0 != vrow1 )
                    {
                        if ( !_LoadScanline( vrow1, width, pSrc2, bRowPitch, srcb->format ) )
                            return E_FAIL;
                        pSrc2 += bRowPitch;
                    }

                    size_t nwidth = width >> 1;

                    for( size_t x = 0; x < nwidth; ++x )
                    {
                        size_t x2 = x*2;

                        // Box filter: Average 2x2x2 pixels
                        XMVECTOR v = XMVectorAdd( urow0[ x2 ], urow1[ x2 ] );
                        v = XMVectorAdd( v, urow2[ x2 ] );
                        v = XMVectorAdd( v, urow3[ x2 ] );
                        v = XMVectorAdd( v, vrow0[ x2 ] );
                        v = XMVectorAdd( v, vrow1[ x2 ] );
                        v = XMVectorAdd( v, vrow2[ x2 ] );
                        v = XMVectorAdd( v, vrow3[ x2 ] );

                        target[ x ] = XMVectorMultiply( v, s_boxScale3D );
                    }

                    if ( !_StoreScanline( pDest, dest->rowPitch, dest->format, target, nwidth ) )
                        return E_FAIL;
                    pDest += dest->rowPitch;
                }
            }
        }
        else
        {
            // 2D box filter
            const Image* src = mipChain.GetImage( level-1, 0, 0 );
            const Image* dest = mipChain.GetImage( level, 0, 0 );

            if ( !src || !dest )
                return E_POINTER;

            const uint8_t* pSrc = src->pixels;
            uint8_t* pDest = dest->pixels;

            size_t rowPitch = src->rowPitch;

            size_t nheight = height >> 1;

            for( size_t y = 0; y < nheight; ++y )
            {
                if ( !_LoadScanline( urow0, width, pSrc, rowPitch, src->format ) )
                    return E_FAIL;
                pSrc += rowPitch;

                if ( urow0 != urow1 )
                {
                    if ( !_LoadScanline( urow1, width, pSrc, rowPitch, src->format ) )
                        return E_FAIL;
                    pSrc += rowPitch;
                }

                size_t nwidth = width >> 1;

                for( size_t x = 0; x < nwidth; ++x )
                {
                    size_t x2 = x*2;

                    // Box filter: Average 2x2 pixels
                    XMVECTOR v = XMVectorAdd( urow0[ x2 ], urow1[ x2 ] );
                    v = XMVectorAdd( v, urow2[ x2 ] );
                    v = XMVectorAdd( v, urow3[ x2 ] );

                    target[ x ] = XMVectorMultiply( v, s_boxScale );
                }

                if ( !_StoreScanline( pDest, dest->rowPitch, dest->format, target, nwidth ) )
                    return E_FAIL;
                pDest += dest->rowPitch;
            }
        }

        if ( height > 1 )
            height >>= 1;

        if ( width > 1 )
            width >>= 1;

        if ( depth > 1 )
            depth >>= 1;
    }

    assert( height == 1 && width == 1 && depth == 1 );

    return S_OK;
}


//=====================================================================================
// Entry-points
//=====================================================================================

//-------------------------------------------------------------------------------------
// Generate mipmap chain
//-------------------------------------------------------------------------------------
HRESULT GenerateMipMaps( const Image& baseImage, DWORD filter, size_t levels, ScratchImage& mipChain, bool allow1D )
{
    if ( !IsValid( baseImage.format ) || !ispow2(baseImage.width) || !ispow2(baseImage.height) )
        return E_INVALIDARG;

    if ( !baseImage.pixels )
        return E_POINTER;

    if ( !_CalculateMipLevels(baseImage.width, baseImage.height, levels) )
        return E_INVALIDARG;

    if ( IsCompressed( baseImage.format ) )
    {
        return HRESULT_FROM_WIN32( ERROR_NOT_SUPPORTED );
    }

    static_assert( TEX_FILTER_POINT == 0x100000, "TEX_FILTER_ flag values don't match TEX_FILTER_MASK" );
    switch(filter & TEX_FILTER_MASK)
    {
    case 0:
    case TEX_FILTER_POINT:
    case TEX_FILTER_FANT: // Equivalent to Box filter
    case TEX_FILTER_LINEAR:
    case TEX_FILTER_CUBIC:
        {
            WICPixelFormatGUID pfGUID;
            if ( _DXGIToWIC( baseImage.format, pfGUID ) )
            {
                // Case 1: Base image format is supported by Windows Imaging Component
                HRESULT hr = (baseImage.height > 1 || !allow1D)
                             ? mipChain.Initialize2D( baseImage.format, baseImage.width, baseImage.height, 1, levels )
                             : mipChain.Initialize1D( baseImage.format, baseImage.width, 1, levels ); 
                if ( FAILED(hr) )
                    return hr;

                return _GenerateMipMapsUsingWIC( baseImage, filter, levels, pfGUID, mipChain, 0 );
            }
            else
            {
                // Case 2: Base image format is not supported by WIC, so we have to convert, generate, and convert back
                assert( baseImage.format != DXGI_FORMAT_R32G32B32A32_FLOAT );
                ScratchImage temp;
                HRESULT hr = _ConvertToR32G32B32A32( baseImage, temp );
                if ( FAILED(hr) )
                    return hr;

                const Image *timg = temp.GetImage( 0, 0, 0 );
                if ( !timg )
                    return E_POINTER;

                ScratchImage tMipChain;
                hr = _GenerateMipMapsUsingWIC( *timg, filter, levels, GUID_WICPixelFormat128bppRGBAFloat, tMipChain, 0 );
                if ( FAILED(hr) )
                    return hr;

                temp.Release();

                return _ConvertFromR32G32B32A32( tMipChain.GetImages(), tMipChain.GetImageCount(), tMipChain.GetMetadata(), baseImage.format, mipChain );
            }
        }
        break;

    default:
        return HRESULT_FROM_WIN32( ERROR_NOT_SUPPORTED );
    }
}

HRESULT GenerateMipMaps( const Image* srcImages, size_t nimages, const TexMetadata& metadata,
                         DWORD filter, size_t levels, ScratchImage& mipChain )
{
    if ( !srcImages || !nimages || !IsValid(metadata.format)
         || !ispow2(metadata.width) || !ispow2(metadata.height) )
        return E_INVALIDARG;

    if ( metadata.dimension == TEX_DIMENSION_TEXTURE3D
         || IsCompressed( metadata.format ) )
        return HRESULT_FROM_WIN32( ERROR_NOT_SUPPORTED );

    if ( !_CalculateMipLevels(metadata.width, metadata.height, levels) )
        return E_INVALIDARG;

    static_assert( TEX_FILTER_POINT == 0x100000, "TEX_FILTER_ flag values don't match TEX_FILTER_MASK" );
    switch(filter & TEX_FILTER_MASK)
    {
    case 0:
    case TEX_FILTER_POINT:
    case TEX_FILTER_FANT: // Equivalent to Box filter
    case TEX_FILTER_LINEAR:
    case TEX_FILTER_CUBIC:
        {
            WICPixelFormatGUID pfGUID;
            if ( _DXGIToWIC( metadata.format, pfGUID ) )
            {
                // Case 1: Base image format is supported by Windows Imaging Component
                TexMetadata mdata2 = metadata;
                mdata2.mipLevels = levels;
                HRESULT hr = mipChain.Initialize( mdata2 ); 
                if ( FAILED(hr) )
                    return hr;

                for( size_t item = 0; item < metadata.arraySize; ++item )
                {
                    size_t index = metadata.ComputeIndex( 0, item, 0 );
                    if ( index >= nimages )
                    {
                        mipChain.Release();
                        return E_FAIL;
                    }

                    const Image& baseImage = srcImages[ index ];

                    hr = _GenerateMipMapsUsingWIC( baseImage, filter, levels, pfGUID, mipChain, item );
                    if ( FAILED(hr) )
                    {
                        mipChain.Release();
                        return hr;
                    }
                }

                return S_OK;
            }
            else
            {
                // Case 2: Base image format is not supported by WIC, so we have to convert, generate, and convert back
                assert( metadata.format != DXGI_FORMAT_R32G32B32A32_FLOAT );

                TexMetadata mdata2 = metadata;
                mdata2.mipLevels = levels;
                mdata2.format = DXGI_FORMAT_R32G32B32A32_FLOAT;
                ScratchImage tMipChain;
                HRESULT hr = tMipChain.Initialize( mdata2 ); 
                if ( FAILED(hr) )
                    return hr;

                for( size_t item = 0; item < metadata.arraySize; ++item )
                {
                    size_t index = metadata.ComputeIndex( 0, item, 0 );
                    if ( index >= nimages )
                        return E_FAIL;

                    const Image& baseImage = srcImages[ index ];

                    ScratchImage temp;
                    hr = _ConvertToR32G32B32A32( baseImage, temp );
                    if ( FAILED(hr) )
                        return hr;

                    const Image *timg = temp.GetImage( 0, 0, 0 );
                    if ( !timg )
                        return E_POINTER;

                    hr = _GenerateMipMapsUsingWIC( *timg, filter, levels, GUID_WICPixelFormat128bppRGBAFloat, tMipChain, item );
                    if ( FAILED(hr) )
                        return hr;
                }

                return _ConvertFromR32G32B32A32( tMipChain.GetImages(), tMipChain.GetImageCount(), tMipChain.GetMetadata(), metadata.format, mipChain );
            }
        }
        break;

    default:
        return HRESULT_FROM_WIN32( ERROR_NOT_SUPPORTED );;
    }
}


//-------------------------------------------------------------------------------------
// Generate mipmap chain for volume texture
//-------------------------------------------------------------------------------------
HRESULT GenerateMipMaps3D( const Image* baseImages, size_t depth, DWORD filter, size_t levels, ScratchImage& mipChain )
{
    if ( !baseImages || !depth )
        return E_INVALIDARG;

    DXGI_FORMAT format = baseImages[0].format;
    size_t width = baseImages[0].width;
    size_t height = baseImages[0].height;

    if ( !ispow2(width) || !ispow2(height) || !ispow2(depth) )
        return E_INVALIDARG;

    if ( !_CalculateMipLevels3D(width, height, depth, levels) )
        return E_INVALIDARG;

    for( size_t slice=0; slice < depth; ++slice )
    {
        if ( !baseImages[slice].pixels )
            return E_POINTER;

        if ( baseImages[slice].format != format || baseImages[slice].width != width || baseImages[slice].height != height )
        {
            // All base images must be the same format, width, and height
            return E_FAIL;
        }
    }

    if ( IsCompressed( format ) )
    {
        // We don't support generating mipmaps from compressed images, as those should be generated before compression
        return HRESULT_FROM_WIN32( ERROR_NOT_SUPPORTED );
    }

    HRESULT hr;

    static_assert( TEX_FILTER_POINT == 0x100000, "TEX_FILTER_ flag values don't match TEX_FILTER_MASK" );
    switch( filter & TEX_FILTER_MASK )
    {
    case 0:
    case TEX_FILTER_FANT:
        hr = _Setup3DMips( baseImages, depth, levels, mipChain );
        if ( FAILED(hr) )
            return hr;

        // For decimation, Fant is equivalent to a Box filter
        hr = _Generate3DMipsBoxFilter( depth, levels, mipChain );
        if ( FAILED(hr) )
            mipChain.Release();
        return hr;

    case WIC_FLAGS_FILTER_POINT:
        hr = _Setup3DMips( baseImages, depth, levels, mipChain );
        if ( FAILED(hr) )
            return hr;

        hr = _Generate3DMipsPointFilter( depth, levels, mipChain );
        if ( FAILED(hr) )
            mipChain.Release();
        return hr;

    case WIC_FLAGS_FILTER_LINEAR:
        // Need to implement a 3D bi-linear filter (2x2x2)
        return E_NOTIMPL;

    case WIC_FLAGS_FILTER_CUBIC:
        // Need to implement a 3D bi-cubic filter (3x3x3)
        return E_NOTIMPL;

    default:
        return HRESULT_FROM_WIN32( ERROR_NOT_SUPPORTED );;
    }
}

HRESULT GenerateMipMaps3D( const Image* srcImages, size_t nimages, const TexMetadata& metadata,
                           DWORD filter, size_t levels, ScratchImage& mipChain )
{
    if ( !srcImages || !nimages || !IsValid(metadata.format)
         || !ispow2(metadata.width) || !ispow2(metadata.height) || !ispow2(metadata.depth) )
        return E_INVALIDARG;

    if ( metadata.dimension != TEX_DIMENSION_TEXTURE3D
         || IsCompressed( metadata.format ) )
        return HRESULT_FROM_WIN32( ERROR_NOT_SUPPORTED );

    if ( !_CalculateMipLevels3D(metadata.width, metadata.height, metadata.depth, levels) )
        return E_INVALIDARG;
    
    std::vector<const Image> baseImages;
    baseImages.reserve( metadata.depth );
    for( size_t slice=0; slice < metadata.depth; ++slice )
    {
        size_t index = metadata.ComputeIndex( 0, 0, slice );
        if ( index >= nimages )
            return E_FAIL;

        const Image& src = srcImages[ index ];
        if ( !src.pixels )
            return E_POINTER;

        if ( src.format != metadata.format || src.width != metadata.width || src.height != metadata.height )
        {
            // All base images must be the same format, width, and height
            return E_FAIL;
        }

        baseImages.push_back( src );
    }

    assert( baseImages.size() == metadata.depth );

    HRESULT hr;

    static_assert( TEX_FILTER_POINT == 0x100000, "TEX_FILTER_ flag values don't match TEX_FILTER_MASK" );
    switch( filter & TEX_FILTER_MASK )
    {
    case 0:
    case TEX_FILTER_FANT:
        hr = _Setup3DMips( &baseImages[0], metadata.depth, levels, mipChain );
        if ( FAILED(hr) )
            return hr;

        // For decimation, Fant is equivalent to a Box filter
        hr = _Generate3DMipsBoxFilter( metadata.depth, levels, mipChain );
        if ( FAILED(hr) )
            mipChain.Release();
        return hr;

    case WIC_FLAGS_FILTER_POINT:
        hr = _Setup3DMips( &baseImages[0], metadata.depth, levels, mipChain );
        if ( FAILED(hr) )
            return hr;

        hr = _Generate3DMipsPointFilter( metadata.depth, levels, mipChain );
        if ( FAILED(hr) )
            mipChain.Release();
        return hr;

    case WIC_FLAGS_FILTER_LINEAR:
        // Need to implement a 3D bi-linear filter (2x2x2)
        return E_NOTIMPL;

    case WIC_FLAGS_FILTER_CUBIC:
        // Need to implement a 3D bi-cubic filter (3x3x3)
        return E_NOTIMPL;

    default:
        return HRESULT_FROM_WIN32( ERROR_NOT_SUPPORTED );;
    }
}

}; // namespace
