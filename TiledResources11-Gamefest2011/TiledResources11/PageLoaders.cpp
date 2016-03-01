//--------------------------------------------------------------------------------------
// PageLoaders.cpp
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "PageLoaders.h"
#include <assert.h>
#include "TiledResourceRuntimeTest.h"

#include "DirectXTex\DirectXTex.h"

//--------------------------------------------------------------------------------------
// Name: ColorTileLoader constructor
//--------------------------------------------------------------------------------------
ColorTileLoader::ColorTileLoader()
{
    m_Grid = TRUE;
}

//--------------------------------------------------------------------------------------
// Name: ColorTileLoader destructor
//--------------------------------------------------------------------------------------
ColorTileLoader::~ColorTileLoader()
{
    CleanupContexts();
}

//--------------------------------------------------------------------------------------
// Name: ColorTileLoader::CreateThreadContext
// Desc: Called at startup time by the title residency manager, this method creates one
//       loader or unloader thread context.  This may be called repeatedly if more than
//       one loader/unloader thread is required.
//--------------------------------------------------------------------------------------
VOID* ColorTileLoader::CreateThreadContext()
{
    LoaderContext* pLC = new LoaderContext();
    pLC->pBuffer = new BYTE[65536];

    return pLC;
}

//--------------------------------------------------------------------------------------
// Name: ColorTileLoader::DestroyThreadContext
// Desc: Called at device reset or app terminate time by the title residency manager to
//       destroy a thread context created at startup time.
//--------------------------------------------------------------------------------------
VOID ColorTileLoader::DestroyThreadContext( VOID* pThreadContext )
{
    LoaderContext* pLC = (LoaderContext*)pThreadContext;
    delete[] pLC->pBuffer;
    delete pLC;
}

//--------------------------------------------------------------------------------------
// Name: ColorTileLoader::LoadAndMapTile
// Desc: Entry point for loader thread operations - this method must be free threaded.
//       Given a tracked tile ID that contains valid virtual and physical addresses,
//       this method creates a tile of colored pixels in the appropriate format, fills
//       the physical tile, and maps the virtual tile to the physical tile.
//--------------------------------------------------------------------------------------
HRESULT ColorTileLoader::LoadAndMapTile( TrackedTileID* pTileID, VOID* pThreadContext )
{
    ASSERT( pTileID->PTileID != D3D11_TILED_INVALID_PHYSICAL_ADDRESS );
    ASSERT( pTileID->VTileID != D3D11_TILED_INVALID_VIRTUAL_ADDRESS );

    LoaderContext* pLC = (LoaderContext*)pThreadContext;
    BYTE* pBuffer = pLC->pBuffer;

    D3DCOLOR MipColor = TiledRuntimeTest::g_MipColors[pTileID->MipLevel];

    D3D11_TILED_SURFACE_DESC SurfDesc;
    pTileID->pResource->GetSubresourceDesc( pTileID->MipLevel, &SurfDesc );

    switch( SurfDesc.Format )
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
        {
            TiledRuntimeTest::TestTileData::FillRect32Bit( pBuffer, SurfDesc.TileTexelWidth, SurfDesc.TileTexelHeight, SurfDesc.TileTexelWidth * sizeof(UINT), MipColor, TRUE );

            if( m_Grid )
            {
                DWORD* pPixels = (DWORD*)pBuffer;
                for( UINT i = 0; i < SurfDesc.TileTexelHeight; ++i )
                {
                    pPixels[i] = 0xFF000000;
                    pPixels[i * SurfDesc.TileTexelWidth] = 0xFF000000;
                }
            }
            *(DWORD*)pBuffer = 0xFFFFFFFF;
            break;
        }
    case DXGI_FORMAT_B5G6R5_UNORM:
        {
            TiledRuntimeTest::TestTileData::FillRect16Bit( pBuffer, SurfDesc.TileTexelWidth, SurfDesc.TileTexelHeight, SurfDesc.TileTexelWidth * sizeof(WORD), MipColor, TRUE );

            if( m_Grid )
            {
                WORD* pPixels = (WORD*)pBuffer;
                for( UINT i = 0; i < SurfDesc.TileTexelWidth; ++i )
                {
                    pPixels[i] = 0;
                }
                for( UINT i = 1; i < SurfDesc.TileTexelHeight; ++i )
                {
                    pPixels[i * SurfDesc.TileTexelWidth] = 0;
                }
            }
            break;
        }
    default:
        ASSERT( FALSE );
        break;
    }

    m_pTilePool->UpdateTileContents( pTileID->PTileID, pBuffer, SurfDesc.Format );

    m_pTilePool->MapVirtualTileToPhysicalTile( pTileID->VTileID, pTileID->PTileID );

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Name: ColorTileLoader::UnmapTile
// Desc: Entry point for unloader thread operations - this method must be free threaded.
//       Unmaps the given virtual tile address from the given physical tile address.
//--------------------------------------------------------------------------------------
HRESULT ColorTileLoader::UnmapTile( TrackedTileID* pTileID, VOID* pThreadContext )
{
    ASSERT( pTileID->PTileID != D3D11_TILED_INVALID_PHYSICAL_ADDRESS );
    ASSERT( pTileID->VTileID != D3D11_TILED_INVALID_VIRTUAL_ADDRESS );

    m_pTilePool->UnmapVirtualAddress( pTileID->VTileID );

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Name: MandelbrotTileLoader constructor
//--------------------------------------------------------------------------------------
MandelbrotTileLoader::MandelbrotTileLoader()
{
    m_DebugColoring = FALSE;
    m_Grid = FALSE;
    m_Julia = FALSE;
    m_JuliaCoordinate = XMVectorSet( -0.726895347709114071439f, 0.188887129043845954792f, 0, 0 );
}

//--------------------------------------------------------------------------------------
// Name: MandelbrotTileLoader destructor
//--------------------------------------------------------------------------------------
MandelbrotTileLoader::~MandelbrotTileLoader()
{
    CleanupContexts();
}

//--------------------------------------------------------------------------------------
// Name: MandelbrotTileLoader::CreateThreadContext
// Desc: Called at startup time by the title residency manager, this method creates one
//       loader or unloader thread context.  This may be called repeatedly if more than
//       one loader/unloader thread is required.
//--------------------------------------------------------------------------------------
VOID* MandelbrotTileLoader::CreateThreadContext()
{
    LoaderContext* pLC = new LoaderContext();
    pLC->pBuffer = new BYTE[65536];

    // Uncompressed buffer is sized for the largest compressed tile size possible.
    // Texels will be rendered to this buffer, then optionally compressed to pBuffer.
    pLC->pUncompressedBuffer = new BYTE[512 * 256 * 4];

    return pLC;
}

//--------------------------------------------------------------------------------------
// Name: MandelbrotTileLoader::DestroyThreadContext
// Desc: Called at device reset or app terminate time by the title residency manager to
//       destroy a thread context created at startup time.
//--------------------------------------------------------------------------------------
VOID MandelbrotTileLoader::DestroyThreadContext( VOID* pThreadContext )
{
    LoaderContext* pLC = (LoaderContext*)pThreadContext;
    delete[] pLC->pBuffer;
    if( pLC->pUncompressedBuffer != NULL )
    {
        delete[] pLC->pUncompressedBuffer;
    }
    delete pLC;
}

//--------------------------------------------------------------------------------------
// Name: MandelbrotTileLoader::LoadAndMapTile
// Desc: Entry point for loader thread operations - this method must be free threaded.
//       Given a tracked tile ID that contains valid virtual and physical addresses,
//       this method determines the virtual tile's rectangle in UV space, generates a
//       subset of the Mandelbrot set for the given rectangle in the resource's format,
//       loads the pixels into the tile specified by the physical address, and then maps
//       the virtual address to the physical address.
//--------------------------------------------------------------------------------------
HRESULT MandelbrotTileLoader::LoadAndMapTile( TrackedTileID* pTileID, VOID* pThreadContext )
{
    PIXBeginNamedEvent( 0, "Mandelbrot Load Tile" );

    ASSERT( pTileID->PTileID != D3D11_TILED_INVALID_PHYSICAL_ADDRESS );
    ASSERT( pTileID->VTileID != D3D11_TILED_INVALID_VIRTUAL_ADDRESS );

    // Select the proper buffer to render Mandelbrot pixels to.
    LoaderContext* pLC = (LoaderContext*)pThreadContext;
    BYTE* pBuffer = pLC->pBuffer;
    if( pLC->pUncompressedBuffer != NULL )
    {
        pBuffer = pLC->pUncompressedBuffer;
    }

    // Get the dimensions of the surface for the given mip level:
    ID3D11TiledTexture2D* pResource = pTileID->pResource;
    D3D11_TILED_SURFACE_DESC MipDesc;
    pResource->GetSubresourceDesc( pTileID->MipLevel, &MipDesc );

    // Get the quilt dimensions from the texture desc:
    D3D11_TILED_TEXTURE2D_DESC TexDesc;
    pResource->GetDesc( &TexDesc );

    FLOAT SampleU = pTileID->U;
    FLOAT SampleV = pTileID->V;
    UINT ArraySlice = pTileID->ArraySlice;

    XMVECTOR vQuiltScale = XMVectorSet( 1, 1, 0, 0 );
    XMVECTOR vQuiltOffset = XMVectorZero();

    // Adjust the quilt scale and offset if the resource is quilted:
    if( TexDesc.MiscFlags & D3D11_RESOURCE_MISC_TEXTUREQUILT )
    {
        UINT QuiltWidth = TexDesc.QuiltWidth;
        UINT QuiltHeight = TexDesc.QuiltHeight;
        vQuiltScale = XMVectorSet( 1.0f / (FLOAT)QuiltWidth, 1.0f / (FLOAT)QuiltHeight, 0, 0 );
        UINT QuiltY = ArraySlice / QuiltWidth;
        UINT QuiltX = ArraySlice % QuiltWidth;
        vQuiltOffset = XMVectorSet( (FLOAT)QuiltX, (FLOAT)QuiltY, 0, 0 ) * vQuiltScale;
        ArraySlice = 0;
    }

    // Compute the tile X and Y indices from the sample UV coordinates:
    UINT TileX = (UINT)( SampleU * ( (FLOAT)MipDesc.TexelWidth / (FLOAT)MipDesc.TileTexelWidth ) );
    TileX = min( TileX, MipDesc.TileWidth - 1 );
    UINT TileY = (UINT)( SampleV * ( (FLOAT)MipDesc.TexelHeight / (FLOAT)MipDesc.TileTexelHeight ) );
    TileY = min( TileY, MipDesc.TileHeight - 1 );

    // Create the Mandelbrot texels for this tile:
    CreateMandelbrot( pBuffer, TileX, TileY, ArraySlice, MipDesc, vQuiltScale, vQuiltOffset );

    // Determine if we need to compress the texels:
    BOOL CompressTexture = FALSE;
    switch( MipDesc.Format )
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
        CompressTexture = TRUE;
        break;
    }

    DirectX::ScratchImage* pCompressedImage = NULL;

    // Compress the texels to a temporary buffer if necessary.  The temporary
    // buffer is provided by the texture compressor module.
    if( CompressTexture )
    {
        BYTE* pCompressedTexels = NULL;
        UINT CompressedTexelsRowStrideBytes = 0;

        DXUT_BeginPerfEvent( 0, L"Texture Compression" );

        DirectX::Image SrcImage;
        ZeroMemory( &SrcImage, sizeof(SrcImage) );
        SrcImage.format = DXGI_FORMAT_R8G8B8A8_UNORM;
        SrcImage.width = MipDesc.TileTexelWidth;
        SrcImage.height = MipDesc.TileTexelHeight;
        SrcImage.rowPitch = MipDesc.TileTexelWidth * 4;
        SrcImage.pixels = pLC->pUncompressedBuffer;

        pCompressedImage = new DirectX::ScratchImage();
        HRESULT hr = DirectX::Compress( SrcImage, MipDesc.Format, DirectX::TEX_COMPRESS_DITHER, 1.0f, *pCompressedImage );
        ASSERT( SUCCEEDED(hr) );

        assert( pCompressedImage->GetImageCount() == 1 );
        const DirectX::Image* pResultImage = pCompressedImage->GetImages();

        CompressedTexelsRowStrideBytes = pResultImage->rowPitch;
        pCompressedTexels = pResultImage->pixels;

        DXUT_EndPerfEvent();

        // Reassign the buffer pointer to the compressed texels in the temp buffer:
        pBuffer = pCompressedTexels;
    }

    // Load the texels into the physical tile:
    m_pTilePool->UpdateTileContents( pTileID->PTileID, pBuffer, MipDesc.Format );

    // Release the texture compressor's temporary buffer if necessary:
    if( pCompressedImage != NULL )
    {
        SAFE_DELETE( pCompressedImage );
    }

    // Map the virtual tile to the physical tile:
    m_pTilePool->MapVirtualTileToPhysicalTile( pTileID->VTileID, pTileID->PTileID );

    PIXEndNamedEvent();

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Name: MandelbrotTileLoader::UnmapTile
// Desc: Entry point for unloader thread operations - this method must be free threaded.
//       Unmaps the given virtual tile address from the given physical tile address.
//--------------------------------------------------------------------------------------
HRESULT MandelbrotTileLoader::UnmapTile( TrackedTileID* pTileID, VOID* pThreadContext )
{
    ASSERT( pTileID->PTileID != D3D11_TILED_INVALID_PHYSICAL_ADDRESS );
    ASSERT( pTileID->VTileID != D3D11_TILED_INVALID_VIRTUAL_ADDRESS );

    m_pTilePool->UnmapVirtualAddress( pTileID->VTileID );

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Name: ImagSq
// Desc: Computes the square of an imaginary value A + Bi stored in the XY components of
//       an XMVECTOR.
//--------------------------------------------------------------------------------------
inline XMVECTOR ImagSq( XMVECTOR Value )
{
    const XMVECTOR vSwizzleA = XMVectorSwizzle( Value, 0, 1, 0, 1 );
    const XMVECTOR vSwizzleB = XMVectorSwizzle( Value, 0, 1, 1, 0 );
    const XMVECTOR vProduct = vSwizzleA * vSwizzleB;
    const XMVECTOR vRealConst = { 1, -1, 0, 0 };
    const XMVECTOR vImagConst = { 0, 0, 1, 1 };
    XMVECTOR vReal = XMVector2Dot( vProduct, vRealConst );
    XMVECTOR vImag = XMVector4Dot( vProduct, vImagConst );
    return XMVectorSelect( vReal, vImag, XMVectorSelectControl( 0, 1, 1, 1 ) );
}

//--------------------------------------------------------------------------------------
// Name: ImagMagnitude
// Desc: Computes the magnitude squared of an imaginary value A + Bi stored in the XY 
//       components of an XMVECTOR.
//--------------------------------------------------------------------------------------
inline FLOAT ImagMagnitudeSq( XMVECTOR Value )
{
    return XMVectorGetX( XMVector2LengthSq( Value ) );
}

//--------------------------------------------------------------------------------------
// Name: VectorMandelIter
// Desc: Performs a Mandelbrot fractal iteration loop given an initial value plus a
//       constant value (Origin) added at each iteration.  The loop terminates when the
//       magnitude exceeds 2.0 (magnitude squared exceeds 4.0).  The return value is the
//       iteration count.
//--------------------------------------------------------------------------------------
inline UINT VectorMandelIter( XMVECTOR Point, const XMVECTOR Origin )
{
    UINT Count = 0;
    while( ImagMagnitudeSq( Point ) <= 4.0f && Count < 512 )
    {
        Point = ImagSq( Point ) + Origin;
        ++Count;
    }
    return Count;
}

//--------------------------------------------------------------------------------------
// Name: CreateColor
// Desc: Creates a final texel color for a Mandelbrot texel.
//--------------------------------------------------------------------------------------
D3DCOLOR CreateColor( const BYTE Pattern, const BYTE Tile, const BYTE MinAlpha, const BOOL DebugColoring )
{
    BYTE Alpha = max( MinAlpha, Pattern );
    if( DebugColoring )
    {
        BYTE TileBrightness = Tile * 32;
        TileBrightness = max( Pattern, TileBrightness );
        return D3DCOLOR_ARGB( Alpha, TileBrightness, Pattern, Pattern );
    }
    else
    {
        BYTE HalfPattern = Pattern >> 1;
        return D3DCOLOR_ARGB( Alpha, HalfPattern, Pattern, Pattern );
    }
}

//--------------------------------------------------------------------------------------
// Name: ModifyColorForSlice
// Desc: Modifies a Mandelbrot texel color for array slices that are non-zero.  This
//       provides a way to tell apart Mandelbrot texels that differ only by array slice
//       index.
//--------------------------------------------------------------------------------------
inline D3DCOLOR ModifyColorForSlice( D3DCOLOR Color, UINT ArraySlice )
{
    BYTE MinRed = (BYTE)( ArraySlice % 8 ) * 16;
    BYTE MinGreen = (BYTE)( ArraySlice / 8 ) * 16;

    BYTE Red = D3DCOLOR_GETRED( Color );
    BYTE Green = D3DCOLOR_GETGREEN( Color );
    BYTE Blue = D3DCOLOR_GETBLUE( Color );
    BYTE Alpha = D3DCOLOR_GETALPHA( Color );

    Red = max( Red, MinRed );
    Green = max( Green, MinGreen );

    return D3DCOLOR_ARGB( Alpha, Red, Green, Blue );
}

//--------------------------------------------------------------------------------------
// Name: MandelbrotTileLoader::CreateMandelbrot
// Desc: Worker method that creates a tile's worth of Mandelbrot texels, given the tile
//       location, quilt transform and array slice.
//--------------------------------------------------------------------------------------
VOID MandelbrotTileLoader::CreateMandelbrot( BYTE* pBuffer, UINT TileX, UINT TileY, UINT ArraySlice, const D3D11_TILED_SURFACE_DESC& MipLevelDesc, const XMVECTOR vQuiltScale, const XMVECTOR vQuiltOffset )
{
    BYTE Tile = ( TileX + TileY ) % 2;

    // Convert the tile X and Y indices into an upper-right UV space coordinate.
    FLOAT TileWidthU = (FLOAT)MipLevelDesc.TileTexelWidth / (FLOAT)MipLevelDesc.TexelWidth;
    FLOAT TileHeightV = (FLOAT)MipLevelDesc.TileTexelHeight / (FLOAT)MipLevelDesc.TexelHeight;

    FLOAT TileU = (FLOAT)TileX * TileWidthU;
    FLOAT TileV = (FLOAT)TileY * TileHeightV;

    // Determine the pixel size in bytes per pixel.
    // The default is 32bpp RGBA; if the resource format differs from that,
    // then the tile loader will convert uncompressed 32bpp RGBA into the
    // final format in the calling method.
    UINT PixelSize = 4;
    switch( MipLevelDesc.Format )
    {
    case DXGI_FORMAT_B5G6R5_UNORM:
        PixelSize = 2;
        break;
    case DXGI_FORMAT_R8_UNORM:
        PixelSize = 1;
        break;
    case DXGI_FORMAT_R16G16B16A16_UNORM:
        PixelSize = 8;
        break;
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
        PixelSize = 16;
        break;
    }

    // Adjust the min alpha value for alpha-only formats (BC4).
    // This ensures that later, during color generation, the Mandelbrot
    // pattern will be replicated into the alpha channel in the uncompressed
    // texels.
    BYTE MinAlpha = 255;
    switch( MipLevelDesc.Format )
    {
    case DXGI_FORMAT_BC4_TYPELESS:
    case DXGI_FORMAT_BC4_UNORM:
    case DXGI_FORMAT_BC4_SNORM:
        MinAlpha = 0;
        break;
    }

    // Select a pleasing real-imaginary plane window for the overall fractal.
    XMVECTOR vNumericRegionScale = { 2.0f, 2.0f, 0, 0 };
    XMVECTOR vNumericRegionOffset = { -1.5f, -1.0f, 0, 0 };
    if( m_Julia )
    {
        vNumericRegionScale = XMVectorSet( 2.0f, 2.0f, 0, 0 );
        vNumericRegionOffset = XMVectorSet( -1.0f, -1.0f, 0, 0 );
    }

    // Loop over the tile size in Y and X dimensions.
    for( UINT Y = 0; Y < MipLevelDesc.TileTexelHeight; ++Y )
    {
        // Early out if the loader threads are being terminated.
        if( !IsRunning() )
        {
            break;
        }

        BYTE* pLine = (BYTE*)( pBuffer + Y * MipLevelDesc.TileTexelWidth * PixelSize );
        for( UINT X = 0; X < MipLevelDesc.TileTexelWidth; ++X )
        {
            // Compute the UV-space coords for the given texel.
            FLOAT XCoord = ( (FLOAT)X / (FLOAT)MipLevelDesc.TileTexelWidth ) * TileWidthU + TileU;
            FLOAT YCoord = ( (FLOAT)Y / (FLOAT)MipLevelDesc.TileTexelHeight ) * TileHeightV + TileV;
            XMVECTOR vPoint = XMVectorSet( XCoord, YCoord, 0, 0 );

            // Transform the coords for quilting.
            vPoint = XMVectorMultiplyAdd( vPoint, vQuiltScale, vQuiltOffset );

            // Transform the coords into real-imaginary space for fractal generation.
            vPoint = XMVectorMultiplyAdd( vPoint, vNumericRegionScale, vNumericRegionOffset );

            // Perform the Mandelbrot iterations.  This is the expensive part.
            UINT Count = VectorMandelIter( vPoint, m_Julia ? m_JuliaCoordinate : vPoint );

            // Compute an 8-bit brightness value from the iteration count.
            BYTE Brightness = (BYTE)( Count % 256 );

            // Create a color from the brightness value.
            D3DCOLOR ColorARGB = CreateColor( Brightness, Tile, MinAlpha, m_DebugColoring );
            if( MinAlpha == 255 && ArraySlice > 0 )
            {
                ColorARGB = ModifyColorForSlice( ColorARGB, ArraySlice );
            }

            // Store the pixel value into the output buffer.
            BYTE* pPixel = pLine + X * PixelSize;
            switch( MipLevelDesc.Format )
            {
            case DXGI_FORMAT_B5G6R5_UNORM:
                *(WORD*)pPixel = TiledRuntimeTest::Color32To565( ColorARGB );
                if( m_Grid && ( X == 0 || Y == 0 ) )
                {
                    *(WORD*)pPixel = 0;
                }
                break;
            case DXGI_FORMAT_R8_UNORM:
                *(BYTE*)pPixel = TiledRuntimeTest::Color32To8( ColorARGB );
                if( m_Grid && ( X == 0 || Y == 0 ) )
                {
                    *(BYTE*)pPixel = 0;
                }
                break;
            case DXGI_FORMAT_R16G16B16A16_UNORM:
                *(UINT64*)pPixel = TiledRuntimeTest::Color32To64RGBA( ColorARGB );
                if( m_Grid && ( X == 0 || Y == 0 ) )
                {
                    *(UINT64*)pPixel = TiledRuntimeTest::Color32To64RGBA( D3DCOLOR_ARGB( 255, 0, 0, 0 ) );
                }
                break;
            case DXGI_FORMAT_R32G32B32A32_FLOAT:
                {
                    *(XMFLOAT4*)pPixel = TiledRuntimeTest::Color32To128ABGR( ColorARGB );
                    if( m_Grid && ( X == 0 || Y == 0 ) )
                    {
                        *(XMFLOAT4*)pPixel = XMFLOAT4( 0, 0, 0, 1.0f );
                    }
                }
                break;
            case DXGI_FORMAT_R10G10B10A2_UNORM:
                *(DWORD*)pPixel = TiledRuntimeTest::Color32To210ABGR( ColorARGB );
                if( m_Grid && ( X == 0 || Y == 0 ) )
                {
                    *(DWORD*)pPixel = TiledRuntimeTest::Color32To210ABGR( D3DCOLOR_ARGB( 255, 0, 0, 0 ) );
                }
                break;
            case DXGI_FORMAT_R8G8B8A8_UNORM:
            default:
                *(DWORD*)pPixel = TiledRuntimeTest::Color32To32RGBA( ColorARGB );
                if( m_Grid && ( X == 0 || Y == 0 ) )
                {
                    *(DWORD*)pPixel = TiledRuntimeTest::Color32To32RGBA( D3DCOLOR_ARGB( 255, 0, 0, 0 ) );
                }
                break;
            }
        }
    }
}

//--------------------------------------------------------------------------------------
// Name: TiledFileLoader constructor
//--------------------------------------------------------------------------------------
TiledFileLoader::TiledFileLoader()
{
    m_hFile = INVALID_HANDLE_VALUE;
    m_ByteSwapped = FALSE;
    ZeroMemory( &m_Header, sizeof(m_Header) );
    m_pSubresources = NULL;
    m_pFlatIndices = NULL;
    m_ppTileIndexes = NULL;
    m_pDefaultTile = NULL;
    m_DefaultPhysicalTile = D3D11_TILED_INVALID_PHYSICAL_ADDRESS;
    InitializeCriticalSection( &m_FileAccessCritSec );
}

//--------------------------------------------------------------------------------------
// Name: TiledFileLoader destructor
//--------------------------------------------------------------------------------------
TiledFileLoader::~TiledFileLoader()
{
    CloseHandle( m_hFile );

    SAFE_DELETE_ARRAY( m_pSubresources );
    SAFE_DELETE_ARRAY( m_pFlatIndices );
    SAFE_DELETE_ARRAY( m_ppTileIndexes );

    CleanupContexts();
    SAFE_DELETE_ARRAY( m_pDefaultTile );
    DeleteCriticalSection( &m_FileAccessCritSec );
}

//--------------------------------------------------------------------------------------
// Name: TiledFileLoader::CreateTiledTexture2D
// Desc: Given the file header that has been loaded, create a matching TiledTexture2D
//       resource.
//--------------------------------------------------------------------------------------
HRESULT TiledFileLoader::CreateTiledTexture2D( ID3D11TiledResourceDevice* pd3dDeviceEx, ID3D11TiledTexture2D** ppTexture )
{
    if( m_Header.BaseWidthTexels == 0 || m_Header.BaseHeightTexels == 0 )
    {
        return E_FAIL;
    }

    // Convert the platform-neutral file format to a DXGI format.
    static const DXGI_FORMAT TiledFormatToD3DFormat[] =
    {
        DXGI_FORMAT_R8_UNORM,           // TILED_FORMAT_8BPP
        DXGI_FORMAT_B5G6R5_UNORM,       // TILED_FORMAT_16BPP_B5G6R5
        DXGI_FORMAT_B5G5R5A1_UNORM,     // TILED_FORMAT_16BPP_B5G5R5A1
        DXGI_FORMAT_UNKNOWN,            // TILED_FORMAT_16BPP_B4G4R4A4
        DXGI_FORMAT_R8G8B8A8_UNORM,     // TILED_FORMAT_32BPP_R8G8B8A8
        DXGI_FORMAT_R10G10B10A2_UNORM,  // TILED_FORMAT_32BPP_R10G10B10A2
        DXGI_FORMAT_R16G16B16A16_UNORM, // TILED_FORMAT_64BPP_R16G16B16A16
        DXGI_FORMAT_R16G16B16A16_FLOAT, // TILED_FORMAT_64BPP_R16G16B16A16F
        DXGI_FORMAT_BC1_UNORM,          // TILED_FORMAT_BC1
        DXGI_FORMAT_BC2_UNORM,          // TILED_FORMAT_BC2
        DXGI_FORMAT_BC3_UNORM,          // TILED_FORMAT_BC3
        DXGI_FORMAT_BC4_UNORM,          // TILED_FORMAT_BC4
        DXGI_FORMAT_BC5_UNORM,          // TILED_FORMAT_BC5
        DXGI_FORMAT_BC6H_UF16,          // TILED_FORMAT_BC6
        DXGI_FORMAT_BC7_UNORM,          // TILED_FORMAT_BC7
        DXGI_FORMAT_R16_UNORM,          // TILED_FORMAT_16BPP_R16
        DXGI_FORMAT_R8G8_UNORM,         // TILED_FORMAT_16BPP_R8G8
    };
    C_ASSERT( ARRAYSIZE(TiledFormatToD3DFormat) == TiledContent::TILED_FORMAT_MAX );

    DXGI_FORMAT Format = TiledFormatToD3DFormat[m_Header.Format];
    if( Format == DXGI_FORMAT_UNKNOWN )
    {
        return E_FAIL;
    }

    // Create the tiled texture2D.
    D3D11_TILED_TEXTURE2D_DESC TexDesc;
    ZeroMemory( &TexDesc, sizeof(TexDesc) );
    TexDesc.Width = m_Header.BaseWidthTexels;
    TexDesc.Height = m_Header.BaseHeightTexels;
    TexDesc.MipLevels = m_Header.MipLevelCount;
    TexDesc.ArraySize = 1;
    TexDesc.Format = Format;
    TexDesc.Usage = D3D11_USAGE_DEFAULT;
    TexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    return pd3dDeviceEx->CreateTexture2D( NULL, &TexDesc, ppTexture );
}

//--------------------------------------------------------------------------------------
// Name: TiledFileLoader::CreateThreadContext
// Desc: Called at startup time by the title residency manager, this method creates one
//       loader or unloader thread context.  This may be called repeatedly if more than
//       one loader/unloader thread is required.
//--------------------------------------------------------------------------------------
VOID* TiledFileLoader::CreateThreadContext()
{
    LoaderContext* pLC = new LoaderContext();
    pLC->pBuffer = new BYTE[65536];

    return pLC;
}

//--------------------------------------------------------------------------------------
// Name: TiledFileLoader::DestroyThreadContext
// Desc: Called at device reset or app terminate time by the title residency manager to
//       destroy a thread context created at startup time.
//--------------------------------------------------------------------------------------
VOID TiledFileLoader::DestroyThreadContext( VOID* pThreadContext )
{
    LoaderContext* pLC = (LoaderContext*)pThreadContext;
    delete[] pLC->pBuffer;
    delete pLC;
}

//--------------------------------------------------------------------------------------
// Name: TiledFileLoader::TileNeedsUniquePhysicalTile
// Desc: This method is called by a loader thread in the tiled residency manager.  Before
//       assigning a new physical tile to each load request, the residency manager needs
//       to know if the tile content for the given virtual address is unique or not.  If
//       the content is not unique, and it has been loaded already, then this method
//       returns FALSE, and the tiled residency manager does not allocate a new tile for
//       the load request.  It is up to the tile loader to track uniqueness and keep
//       track of the "master" copy of non-unique tiles within each resource.
//--------------------------------------------------------------------------------------
BOOL TiledFileLoader::TileNeedsUniquePhysicalTile( TrackedTileID* pTileID )
{
    if( m_DefaultPhysicalTile == D3D11_TILED_INVALID_PHYSICAL_ADDRESS || m_Header.DefaultPage == TiledContent::TILEDFILE_INVALID_LOCATOR )
    {
        return TRUE;
    }

    ASSERT( pTileID->PTileID == D3D11_TILED_INVALID_PHYSICAL_ADDRESS );
    ASSERT( pTileID->VTileID != D3D11_TILED_INVALID_VIRTUAL_ADDRESS );

    // Determine the tile X and Y indices given the sample's UV address and mip level.
    ID3D11TiledTexture2D* pResource = pTileID->pResource;
    D3D11_TILED_SURFACE_DESC MipDesc;
    pResource->GetSubresourceDesc( pTileID->MipLevel, &MipDesc );

    UINT TileX = (UINT)( pTileID->U * (FLOAT)MipDesc.TileWidth );
    TileX = min( TileX, MipDesc.TileWidth - 1 );
    UINT TileY = (UINT)( pTileID->V * (FLOAT)MipDesc.TileHeight );
    TileY = min( TileY, MipDesc.TileHeight - 1 );

    // Search for the tile in the file's tile index.
    TiledContent::TILEDFILE_PAGEDATA_LOCATOR Locator = FindTile( pTileID->MipLevel, TileX, TileY, NULL );

    // The tile locator indicates uniqueness - there can be one "default tile" per file.
    BOOL IsDefaultPage = ( Locator == m_Header.DefaultPage );
    return !IsDefaultPage;
}

//--------------------------------------------------------------------------------------
// Name: TiledFileLoader::LoadAndMapTile
// Desc: Entry point for loader thread operations - this method must be free threaded.
//       Given a tracked tile ID that contains valid virtual and physical addresses,
//       this method loads tile contents from storage that correspond to the virtual
//       tile address, fills the physical tile with those contents, and maps the virtual
//       tile to the physical tile.
//--------------------------------------------------------------------------------------
HRESULT TiledFileLoader::LoadAndMapTile( TrackedTileID* pTileID, VOID* pThreadContext )
{
    ASSERT( pTileID->VTileID != D3D11_TILED_INVALID_VIRTUAL_ADDRESS );

    LoaderContext* pLC = (LoaderContext*)pThreadContext;
    BYTE* pBuffer = pLC->pBuffer;

    // Convert the sample UV address and mip level to a tile X and Y.
    ID3D11TiledTexture2D* pResource = pTileID->pResource;
    D3D11_TILED_SURFACE_DESC MipDesc;
    pResource->GetSubresourceDesc( pTileID->MipLevel, &MipDesc );

    UINT TileX = (UINT)( pTileID->U * (FLOAT)MipDesc.TileWidth );
    TileX = min( TileX, MipDesc.TileWidth - 1 );
    UINT TileY = (UINT)( pTileID->V * (FLOAT)MipDesc.TileHeight );
    TileY = min( TileY, MipDesc.TileHeight - 1 );

    // Find the tile within the tile index.
    UINT BlockOffset = 0;
    TiledContent::TILEDFILE_PAGEDATA_LOCATOR Locator = FindTile( pTileID->MipLevel, TileX, TileY, &BlockOffset );

    // Determine if the tile locator references the "default tile", which is a single physical tile that can be 
    // mapped to multiple virtual locations.
    if( Locator == m_Header.DefaultPage && m_Header.DefaultPage != TiledContent::TILEDFILE_INVALID_LOCATOR )
    {
        if( m_DefaultPhysicalTile == D3D11_TILED_INVALID_PHYSICAL_ADDRESS )
        {
            ASSERT( pTileID->PTileID != D3D11_TILED_INVALID_PHYSICAL_ADDRESS );

            // Fill tile
            m_pTilePool->UpdateTileContents( pTileID->PTileID, m_pDefaultTile, MipDesc.Format );

            // Map tile
            m_pTilePool->MapVirtualTileToPhysicalTile( pTileID->VTileID, pTileID->PTileID );

            // Pin tile so the title residency manager won't get rid of it
            pTileID->PinnedTile = TRUE;

            // Save the physical tile address so we can use it for other locations
            m_DefaultPhysicalTile = pTileID->PTileID;
        }
        else
        {
            ASSERT( pTileID->PTileID == D3D11_TILED_INVALID_PHYSICAL_ADDRESS );

            // Default physical tile already exists; map it to the virtual tile
            m_pTilePool->MapVirtualTileToPhysicalTile( pTileID->VTileID, m_DefaultPhysicalTile );
        }
    }
    else
    {
        ASSERT( pTileID->PTileID != D3D11_TILED_INVALID_PHYSICAL_ADDRESS );

        // unique tile; load the tile, update tile contents, and map the tile to the resource
        HRESULT hr = LoadTile( Locator, BlockOffset, pBuffer );

        //DebugSpew( "SR %d X %d Y %d Locator %d Offset %d VTileID %I64u\n", pTileID->MipLevel, TileX, TileY, Locator.TileOffset, BlockOffset, pTileID->VTileID );

        if( SUCCEEDED(hr) )
        {
            // Fill the physical tile.
            m_pTilePool->UpdateTileContents( pTileID->PTileID, pBuffer, MipDesc.Format );

            // Map the virtual tile to the physical tile.
            m_pTilePool->MapVirtualTileToPhysicalTile( pTileID->VTileID, pTileID->PTileID );
        }
    }

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Name: TiledFileLoader::UnmapTile
// Desc: Entry point for unloader thread operations - this method must be free threaded.
//       Unmaps the given virtual tile address from the given physical tile address.
//--------------------------------------------------------------------------------------
HRESULT TiledFileLoader::UnmapTile( TrackedTileID* pTileID, VOID* pThreadContext )
{
    ASSERT( pTileID->VTileID != D3D11_TILED_INVALID_VIRTUAL_ADDRESS );
    ASSERT( pTileID->PTileID != m_DefaultPhysicalTile );

    m_pTilePool->UnmapVirtualAddress( pTileID->VTileID );

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Name: TiledFileLoader::LoadFile
// Desc: Loads the tiled texture resource from the given filename.
//       Parse the header and load the tile index, keeping the file handle open so that
//       tile contents can be loaded later.
//--------------------------------------------------------------------------------------
HRESULT TiledFileLoader::LoadFile( const WCHAR* strFileName )
{
    // Open the file.
    HANDLE hFile = CreateFile( strFileName, GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL );
    if( hFile == INVALID_HANDLE_VALUE )
    {
        return E_INVALIDARG;
    }

    // Read the header magic.
    DWORD BytesRead = 0;
    ReadFile( hFile, &m_Header.Magic, sizeof(m_Header.Magic), &BytesRead, NULL );

    // Determine the byte ordering of the file.
    switch( m_Header.Magic )
    {
    case TiledContent::TILEDFILE_HEADER_MAGIC:
        m_ByteSwapped = FALSE;
        break;
    case TiledContent::TILEDFILE_HEADER_MAGIC_SWAPPED:
        m_ByteSwapped = TRUE;
        break;
    default:
        CloseHandle( hFile );
        return E_INVALIDARG;
    }

    // Lock access to the file.
    EnterCriticalSection( &m_FileAccessCritSec );

    m_hFile = hFile;

    // Read the remainder of the header and byte swap if necessary.
    ReadFile( hFile, &m_Header.Format, sizeof(TiledContent::TILEDFILE_HEADER) - sizeof(UINT), &BytesRead, NULL );

    if( m_ByteSwapped )
    {
        TiledContent::ByteSwapHeader( &m_Header );
    }

    // Allocate the subresource indexes.
    UINT SubresourceCount = m_Header.ArraySliceCount * m_Header.MipLevelCount;
    m_pSubresources = new TiledContent::TILEDFILE_SUBRESOURCE[SubresourceCount];

    // Read the subresource indexes and byte swap if necessary.
    ReadFile( hFile, m_pSubresources, SubresourceCount * sizeof(TiledContent::TILEDFILE_SUBRESOURCE), &BytesRead, NULL );

    if( m_ByteSwapped )
    {
        for( UINT i = 0; i < SubresourceCount; ++i )
        {
            TiledContent::ByteSwapSubresource( &m_pSubresources[i] );
        }
    }

    // Read the tile indices and byte swap if necessary.
    m_pFlatIndices = new TiledContent::TILEDFILE_PAGEDATA_LOCATOR[m_Header.BlockIndexEntries];
    ReadFile( hFile, m_pFlatIndices, m_Header.BlockIndexEntries * sizeof(TiledContent::TILEDFILE_PAGEDATA_LOCATOR), &BytesRead, NULL );

    if( m_ByteSwapped )
    {
        for( UINT i = 0; i < m_Header.BlockIndexEntries; ++i )
        {
            TiledContent::ByteswapLocator( &m_pFlatIndices[i] );
        }
    }

    // Create a start pointer into the tile indices for each subresource.
    m_ppTileIndexes = new TiledContent::TILEDFILE_PAGEDATA_LOCATOR*[SubresourceCount];
    for( UINT i = 0; i < SubresourceCount; ++i )
    {
        UINT Offset = m_pSubresources[i].BlockIndexLocation;
        ASSERT( Offset < m_Header.BlockIndexEntries );
        if( Offset != (UINT)-1 )
        {
            m_ppTileIndexes[i] = &m_pFlatIndices[Offset];
        }
        else
        {
            m_ppTileIndexes[i] = NULL;
        }
    }

    // Load the default tile contents if they are present in the file.
    if( m_Header.DefaultPage != TiledContent::TILEDFILE_INVALID_LOCATOR )
    {
        m_pDefaultTile = new BYTE[TiledContent::PAGE_SIZE_BYTES];
        LoadTile( m_Header.DefaultPage, 0, m_pDefaultTile );
    }
    else
    {
        m_pDefaultTile = NULL;
    }

    // Unlock access to the tiled resource file.
    LeaveCriticalSection( &m_FileAccessCritSec );

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Name: TiledFileLoader::FindTile
// Desc: Search for the given virtual tile in the tile index, and return its locator.
//--------------------------------------------------------------------------------------
TiledContent::TILEDFILE_PAGEDATA_LOCATOR TiledFileLoader::FindTile( UINT Subresource, UINT TileX, UINT TileY, UINT* pBlockOffset ) const
{
    // Validate the subresource index and tile X and Y indices.
    UINT SubresourceCount = m_Header.ArraySliceCount * m_Header.MipLevelCount;
    ASSERT( Subresource < SubresourceCount );

    const TiledContent::TILEDFILE_SUBRESOURCE& SubR = m_pSubresources[Subresource];
    ASSERT( TileX < SubR.WidthPages && TileY < SubR.HeightPages );

    // Compute the block X and Y indices within the subresource.  A block is a contiguous group of tiles.
    UINT BlockX = TileX / SubR.BlockWidthPages;
    UINT BlockY = TileY / SubR.BlockWidthPages;
    ASSERT( BlockX < SubR.WidthBlocks && BlockY < SubR.HeightBlocks );
    UINT BlockIndex = BlockY * SubR.WidthBlocks + BlockX;

    // Output the offset of the given tile within the block.
    if( pBlockOffset != NULL )
    {
        UINT BlockOffsetX = TileX % SubR.BlockWidthPages;
        UINT BlockOffsetY = TileY % SubR.BlockWidthPages;
        UINT BlockOffset = BlockOffsetY * SubR.BlockWidthPages + BlockOffsetX;
        *pBlockOffset = BlockOffset;
    }

    // Get the tile locator for the block and return it.
    const TiledContent::TILEDFILE_PAGEDATA_LOCATOR* pSubresourceIndex = m_ppTileIndexes[Subresource];
    TiledContent::TILEDFILE_PAGEDATA_LOCATOR Locator = pSubresourceIndex[BlockIndex];

    return Locator;
}

//--------------------------------------------------------------------------------------
// Name: TiledFileLoader::LoadTile
// Desc: Loads one tile's contents from storage into the destination buffer.
//       This method is called by the tile loader - which means it must be free threaded.
//--------------------------------------------------------------------------------------
HRESULT TiledFileLoader::LoadTile( TiledContent::TILEDFILE_PAGEDATA_LOCATOR Locator, UINT TileOffset, VOID* pDestBuffer )
{
    ASSERT( m_hFile != INVALID_HANDLE_VALUE );

    // Compute the file offset from the tile locator.
    UINT64 Offset = (UINT64)( Locator.PageOffset + TileOffset ) * (UINT64)TiledContent::PAGE_SIZE_BYTES;
    LONG OffsetLow = (LONG)( Offset & 0x00000000FFFFFFFF );
    LONG OffsetHigh = (LONG)( Offset >> 32 );

    // Lock access to the tiled texture file.  This is required because we have two separate
    // transactions on the file pointer in this method, and 
    EnterCriticalSection( &m_FileAccessCritSec );

    // Set the file offset on the file pointer.
    DWORD Result = SetFilePointer( m_hFile, OffsetLow, &OffsetHigh, FILE_BEGIN );
    if( Result == INVALID_SET_FILE_POINTER )
    {
        LeaveCriticalSection( &m_FileAccessCritSec );
        return E_FAIL;
    }

    // Read the tile contents into the destination buffer.
    DWORD BytesRead = 0;
    ReadFile( m_hFile, pDestBuffer, TiledContent::PAGE_SIZE_BYTES, &BytesRead, NULL );

    // Unlock access to the tiled texture file.
    LeaveCriticalSection( &m_FileAccessCritSec );

    // Byte swap the tile contents if necessary.
    if( m_ByteSwapped )
    {
        USHORT* pShorts = (USHORT*)pDestBuffer;
        ULONG* pLongs = (ULONG*)pDestBuffer;
        UINT64* pLLongs = (UINT64*)pDestBuffer;

        switch( m_Header.Format )
        {
        case TiledContent::TILED_FORMAT_8BPP:
            break;
        case TiledContent::TILED_FORMAT_16BPP_B5G6R5:
        case TiledContent::TILED_FORMAT_16BPP_B5G5R5A1:
        case TiledContent::TILED_FORMAT_16BPP_B4G4R4A4:
        case TiledContent::TILED_FORMAT_16BPP_R16:
        case TiledContent::TILED_FORMAT_16BPP_R8G8:
            for( UINT i = 0; i < ( TiledContent::PAGE_SIZE_BYTES / sizeof(USHORT) ); ++i )
            {
                pShorts[i] = _byteswap_ushort( pShorts[i] );
            }
            break;
        case TiledContent::TILED_FORMAT_32BPP_R8G8B8A8:
        case TiledContent::TILED_FORMAT_32BPP_R10G10B10A2:
        case TiledContent::TILED_FORMAT_BC2:                      // DXT2/3
        case TiledContent::TILED_FORMAT_BC3:                      // DXT4/5
        case TiledContent::TILED_FORMAT_BC5:                      // DXN
            for( UINT i = 0; i < ( TiledContent::PAGE_SIZE_BYTES / sizeof(ULONG) ); ++i )
            {
                pLongs[i] = _byteswap_ulong( pLongs[i] );
            }
            break;
        case TiledContent::TILED_FORMAT_64BPP_R16G16B16A16:
        case TiledContent::TILED_FORMAT_64BPP_R16G16B16A16F:
        case TiledContent::TILED_FORMAT_BC1:                      // DXT1
        case TiledContent::TILED_FORMAT_BC4:                      // DXT5A
            for( UINT i = 0; i < ( TiledContent::PAGE_SIZE_BYTES / sizeof(UINT64) ); ++i )
            {
                pLLongs[i] = _byteswap_uint64( pLLongs[i] );
            }
            break;
        case TiledContent::TILED_FORMAT_BC6:
        case TiledContent::TILED_FORMAT_BC7:
            break;
        }
    }

    return S_OK;
}
