//--------------------------------------------------------------------------------------
// TiledResourceRuntimeTest.h
//
// A set of utility functions for converting color values, filling texels with solid colors,
// and trace functionality to help with debugging.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#include "TiledResourceCommon.h"

namespace TiledRuntimeTest
{
    using namespace TiledRuntime;

    // Standard colors to identify mip levels:
    static const DWORD g_MipColors[12] = 
    {
        0xFFFF0000,
        0xFFFFFF00,
        0xFF00FF00,
        0xFF00FFFF,
        0xFF0000FF,
        0xFFFF00FF,
        0xFF800000,
        0xFF808000,
        0xFF008000,
        0xFF008080,
        0xFF000080,
        0xFF800080
    };

    //--------------------------------------------------------------------------------------
    // Name: Color32To565
    // Desc: Converts a 32bpp ARGB color to 16bpp 565 BGR.
    //--------------------------------------------------------------------------------------
    inline WORD Color32To565( DWORD Color )
    {
        WORD Red = D3DCOLOR_GETRED( Color );
        WORD Green = D3DCOLOR_GETGREEN( Color );
        WORD Blue = D3DCOLOR_GETBLUE( Color );
        return ( Blue >> 3 ) | ( ( Green >> 2 ) << 5 ) | ( ( Red >> 3 ) << 11 );
    }

    //--------------------------------------------------------------------------------------
    // Name: Color32To4444
    // Desc: Converts a 32bpp ARGB color to 16bpp 4444 BGRA.
    //--------------------------------------------------------------------------------------
    inline WORD Color32To4444( DWORD Color )
    {
        WORD Red = D3DCOLOR_GETRED( Color );
        WORD Green = D3DCOLOR_GETGREEN( Color );
        WORD Blue = D3DCOLOR_GETBLUE( Color );
        WORD Alpha = D3DCOLOR_GETALPHA( Color );
        return ( Blue >> 4 ) | ( ( Green >> 4 ) << 4 ) | ( ( Red >> 4 ) << 8 ) | ( ( Alpha >> 4 ) << 12 );
    }

    //--------------------------------------------------------------------------------------
    // Name: Color32To8
    // Desc: Converts a 32bpp ARGB color to single channel 8bpp.
    //--------------------------------------------------------------------------------------
    inline BYTE Color32To8( DWORD Color )
    {
        BYTE Red = D3DCOLOR_GETRED( Color );
        BYTE Green = D3DCOLOR_GETGREEN( Color );
        BYTE Blue = D3DCOLOR_GETBLUE( Color );
        return max( Red, max( Green, Blue ) );
    }

    //--------------------------------------------------------------------------------------
    // Name: Color32To210ABGR
    // Desc: Converts a 32bpp ARGB color to 32bpp 2:10:10:10 ABGR.
    //--------------------------------------------------------------------------------------
    inline DWORD Color32To210ABGR( DWORD Color )
    {
        WORD Red = D3DCOLOR_GETRED( Color );
        WORD Green = D3DCOLOR_GETGREEN( Color );
        WORD Blue = D3DCOLOR_GETBLUE( Color );
        WORD Alpha = D3DCOLOR_GETALPHA( Color );
        return ( Red << 2 ) | ( ( Green << 2 ) << 10 ) | ( ( Blue << 2 ) << 20 ) | ( ( Alpha >> 6 ) << 30 );
    }

    //--------------------------------------------------------------------------------------
    // Name: Color32To210ARGB
    // Desc: Converts a 32bpp ARGB color to 32bpp 2:10:10:10 ARGB.
    //--------------------------------------------------------------------------------------
    inline DWORD Color32To210ARGB( DWORD Color )
    {
        WORD Red = D3DCOLOR_GETRED( Color );
        WORD Green = D3DCOLOR_GETGREEN( Color );
        WORD Blue = D3DCOLOR_GETBLUE( Color );
        WORD Alpha = D3DCOLOR_GETALPHA( Color );
        return ( Blue << 2 ) | ( ( Green << 2 ) << 10 ) | ( ( Red << 2 ) << 20 ) | ( ( Alpha >> 6 ) << 30 );
    }

    //--------------------------------------------------------------------------------------
    // Name: Color32To32RGBA
    // Desc: Converts a 32bpp ARGB color to 32bpp RGBA.
    //--------------------------------------------------------------------------------------
    inline DWORD Color32To32RGBA( DWORD Color )
    {
        WORD Red = D3DCOLOR_GETRED( Color );
        WORD Green = D3DCOLOR_GETGREEN( Color );
        WORD Blue = D3DCOLOR_GETBLUE( Color );
        WORD Alpha = D3DCOLOR_GETALPHA( Color );
        return ( Blue << 16 ) | ( Green << 8 ) | ( Red << 0 ) | ( Alpha << 24 );
    }

    //--------------------------------------------------------------------------------------
    // Name: Color32To64ABGR
    // Desc: Converts a 32bpp ARGB color to 64bpp unsigned ABGR.
    //--------------------------------------------------------------------------------------
    inline UINT64 Color32To64ABGR( DWORD Color )
    {
        UINT64 Red = D3DCOLOR_GETRED( Color );
        UINT64 Green = D3DCOLOR_GETGREEN( Color );
        UINT64 Blue = D3DCOLOR_GETBLUE( Color );
        UINT64 Alpha = D3DCOLOR_GETALPHA( Color );
        return ( Alpha << 8 ) | ( ( Blue << 8 ) << 16 ) | ( ( Green << 8 ) << 32 ) | ( ( Red << 8 ) << 48 );
    }

    //--------------------------------------------------------------------------------------
    // Name: Color32To64RGBA
    // Desc: Converts a 32bpp ARGB color to 64bpp unsigned RGBA.
    //--------------------------------------------------------------------------------------
    inline UINT64 Color32To64RGBA( DWORD Color )
    {
        UINT64 Red = D3DCOLOR_GETRED( Color );
        UINT64 Green = D3DCOLOR_GETGREEN( Color );
        UINT64 Blue = D3DCOLOR_GETBLUE( Color );
        UINT64 Alpha = D3DCOLOR_GETALPHA( Color );
        return ( ( Alpha << 8 ) << 48 ) | ( ( Blue << 8 ) << 32 ) | ( ( Green << 8 ) << 16 ) | ( ( Red << 8 ) << 0 );
    }

    //--------------------------------------------------------------------------------------
    // Name: Color32To128ABGR
    // Desc: Converts a 32bpp ARGB color to 128bpp float ABGR.
    //--------------------------------------------------------------------------------------
    inline XMFLOAT4 Color32To128ABGR( DWORD Color )
    {
        FLOAT Red = (FLOAT)D3DCOLOR_GETRED( Color ) / 255.0f;
        FLOAT Green = (FLOAT)D3DCOLOR_GETGREEN( Color ) / 255.0f;
        FLOAT Blue = (FLOAT)D3DCOLOR_GETBLUE( Color ) / 255.0f;
        FLOAT Alpha = (FLOAT)D3DCOLOR_GETALPHA( Color ) / 255.0f;
        return XMFLOAT4( Red, Green, Blue, Alpha );
    }

    const CHAR* GetFormatName( DXGI_FORMAT Format );

    //--------------------------------------------------------------------------------------
    // Name: TestTileData
    // Desc: Exposes static methods for filling textures with 32bpp or 16bpp color patterns.
    //--------------------------------------------------------------------------------------
    class TestTileData
    {
    public:
        static VOID FillRect32Bit( BYTE* pDestBits, DWORD Width, DWORD Height, DWORD PitchBytes, DWORD ColorARGB, BOOL Checker );
        static VOID FillRect16Bit( BYTE* pDestBits, DWORD Width, DWORD Height, DWORD PitchBytes, DWORD ColorARGB, BOOL Checker );
    };

    //--------------------------------------------------------------------------------------
    // Name: Trace
    // Desc: Exposes static methods that are called at important points within the tiled
    //       resource runtime.  These traces can be used to help debug problems.
    //--------------------------------------------------------------------------------------
    class Trace
    {
    public:
        static VOID CreatePage( PhysicalPageID PageID, DXGI_FORMAT Format );
        static VOID CreateTexture2D( INT ResourceID, INT Width, INT Height, INT ArraySize, DXGI_FORMAT Format );
        static VOID MovePage( PhysicalPageID PageID, DXGI_FORMAT SrcFormat, DXGI_FORMAT DestFormat );
        static VOID FillPage( PhysicalPageID PageID, DXGI_FORMAT Format );
        static VOID MapPage( VirtualPageID VPageID, PhysicalPageID PageID );
        static VOID QueueMapPageUpdate( VirtualPageID VPageID, PhysicalPageID PageID );
        static VOID RetireMapPageUpdate( VirtualPageID VPageID, PhysicalPageID PageID );
        static VOID UpdatePageBorder( PhysicalPageID CenterPage, PhysicalPageID BorderPage, PageNeighbors BorderLocation );
        static VOID PageCreateFailure( PhysicalPageID PageID, DXGI_FORMAT Format );
        static VOID AddPageToPool( PhysicalPageID PageID, INT PoolIndex, DXGI_FORMAT PoolFormat );
        static VOID RemovePageFromPool( PhysicalPageID PageID, INT PoolIndex, DXGI_FORMAT PoolFormat );
    };
}
