//--------------------------------------------------------------------------------------
// PageLoaders.h
//
// Implements three different modules that load data into tiled resource tiles, and
// unload tiles as necessary.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#ifdef _XBOX
#include <xtl.h>
#else
#include <windows.h>
#include <d3d11.h>
#endif

#include <xnamath.h>

#include "TitleResidencyManager.h"
#include "TiledFileFormat.h"
#include "Util.h"

#pragma warning (disable: 4324)

//--------------------------------------------------------------------------------------
// Name: ColorTileLoader
// Desc: Implements a page loader that fills pages with a checkerboarded solid color,
//       the color chosen by the mip level.
//--------------------------------------------------------------------------------------
class ColorTileLoader : public ITileLoader
{
protected:
    struct LoaderContext
    {
        BYTE* pBuffer;
    };

public:
    BOOL m_Grid;

    ColorTileLoader();
    virtual ~ColorTileLoader();

    virtual VOID* CreateThreadContext();
    virtual VOID DestroyThreadContext( VOID* pThreadContext );

    virtual HRESULT LoadAndMapTile( TrackedTileID* pTileID, VOID* pThreadContext );
    virtual HRESULT UnmapTile( TrackedTileID* pTileID, VOID* pThreadContext );
};

//--------------------------------------------------------------------------------------
// Name: MandelbrotTileLoader
// Desc: Implements a page loader that fills pages with sections of the Mandelbrot or
//       Julia fractals.
//--------------------------------------------------------------------------------------
class MandelbrotTileLoader : public ITileLoader
{
protected:
    struct LoaderContext
    {
        BYTE* pBuffer;
        BYTE* pUncompressedBuffer;
    };

public:
    XMVECTOR m_JuliaCoordinate;
    BOOL m_Julia;

    BOOL m_DebugColoring;
    BOOL m_Grid;

    MandelbrotTileLoader();
    virtual ~MandelbrotTileLoader();

    virtual VOID* CreateThreadContext();
    virtual VOID DestroyThreadContext( VOID* pThreadContext );

    virtual HRESULT LoadAndMapTile( TrackedTileID* pTileID, VOID* pThreadContext );
    virtual HRESULT UnmapTile( TrackedTileID* pTileID, VOID* pThreadContext );

protected:
    VOID CreateMandelbrot( BYTE* pBuffer, UINT TileX, UINT TileY, UINT ArraySlice, const D3D11_TILED_SURFACE_DESC& MipLevelDesc, const XMVECTOR vQuiltScale, const XMVECTOR vQuiltOffset );
};

//--------------------------------------------------------------------------------------
// Name: TiledFileLoader
// Desc: Implements a page loader that fills pages with data that comes from a file.
//--------------------------------------------------------------------------------------
class TiledFileLoader : public ITileLoader
{
protected:
    struct LoaderContext
    {
        BYTE* pBuffer;
    };

    HANDLE m_hFile;
    CRITICAL_SECTION m_FileAccessCritSec;

    BOOL m_ByteSwapped;
    TiledContent::TILEDFILE_HEADER m_Header;
    TiledContent::TILEDFILE_SUBRESOURCE* m_pSubresources;

    TiledContent::TILEDFILE_PAGEDATA_LOCATOR* m_pFlatIndices;
    TiledContent::TILEDFILE_PAGEDATA_LOCATOR** m_ppTileIndexes;

    BYTE* m_pDefaultTile;
    D3D11_TILED_PHYSICAL_ADDRESS m_DefaultPhysicalTile;

public:
    TiledFileLoader();
    virtual ~TiledFileLoader();

    HRESULT LoadFile( const WCHAR* strFileName );

    HRESULT CreateTiledTexture2D( ID3D11TiledResourceDevice* pd3dDeviceEx, ID3D11TiledTexture2D** ppTexture );

    virtual VOID* CreateThreadContext();
    virtual VOID DestroyThreadContext( VOID* pThreadContext );

    virtual BOOL TileNeedsUniquePhysicalTile( TrackedTileID* pTileID );
    virtual HRESULT LoadAndMapTile( TrackedTileID* pTileID, VOID* pThreadContext );
    virtual HRESULT UnmapTile( TrackedTileID* pTileID, VOID* pThreadContext );

protected:
    TiledContent::TILEDFILE_PAGEDATA_LOCATOR FindTile( UINT Subresource, UINT TileX, UINT TileY, UINT* pBlockOffset ) const;
    HRESULT LoadTile( TiledContent::TILEDFILE_PAGEDATA_LOCATOR Locator, UINT TileOffset, VOID* pDestBuffer );
};
