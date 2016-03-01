//--------------------------------------------------------------------------------------
// TiledFileFormat.h
//
// Structs, constants, and enumerations that define a tiled texture file format.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#ifdef _XBOX
#include <xtl.h>
#else
#include <windows.h>
#endif

namespace TiledContent
{
    static const UINT TILEDFILE_HEADER_MAGIC = 'SPF1';
    static const UINT TILEDFILE_HEADER_MAGIC_SWAPPED = '1FPS';

    static const UINT PAGE_SIZE_BYTES = 65536;
    static const UINT MAX_WIDTH_TEXELS = 16384;
    static const UINT MAX_HEIGHT_TEXELS = 16384;

    enum TILEDFILE_FORMAT
    {
        TILED_FORMAT_8BPP = 0,
        TILED_FORMAT_16BPP_B5G6R5,
        TILED_FORMAT_16BPP_B5G5R5A1,
        TILED_FORMAT_16BPP_B4G4R4A4,
        TILED_FORMAT_32BPP_R8G8B8A8,
        TILED_FORMAT_32BPP_R10G10B10A2,
        TILED_FORMAT_64BPP_R16G16B16A16,
        TILED_FORMAT_64BPP_R16G16B16A16F,
        TILED_FORMAT_BC1,                      // DXT1
        TILED_FORMAT_BC2,                      // DXT2/3
        TILED_FORMAT_BC3,                      // DXT4/5
        TILED_FORMAT_BC4,                      // DXT5A
        TILED_FORMAT_BC5,                      // DXN
        TILED_FORMAT_BC6,
        TILED_FORMAT_BC7,
        TILED_FORMAT_16BPP_R16,
        TILED_FORMAT_16BPP_R8G8,
        TILED_FORMAT_MAX,
        TILED_FORMAT_CUSTOM = 0x80000000
    };

#ifdef _XBOX
#pragma bitfield_order(push)
#pragma bitfield_order(lsb_to_msb)
#endif
    struct TILEDFILE_PAGEDATA_LOCATOR
    {
        UINT DataFileIndex : 4;
        UINT UniquePages : 1;
        UINT PageOffset : 27;

        bool operator==( const TILEDFILE_PAGEDATA_LOCATOR& RHS ) const 
        { 
            return ( DataFileIndex == RHS.DataFileIndex && UniquePages == RHS.UniquePages && PageOffset == RHS.PageOffset );
        }
        bool operator!=( const TILEDFILE_PAGEDATA_LOCATOR& RHS ) const 
        { 
            return ( DataFileIndex != RHS.DataFileIndex || UniquePages != RHS.UniquePages || PageOffset != RHS.PageOffset );
        }
    };
#ifdef _XBOX
#pragma bitfield_order(pop)
#endif

    inline VOID ByteswapLocator( __inout TILEDFILE_PAGEDATA_LOCATOR* pLocator )
    {
        ULONG Swap = _byteswap_ulong( *(ULONG*)pLocator );
        *pLocator = *(TILEDFILE_PAGEDATA_LOCATOR*)&Swap;
    }

    struct TILEDFILE_HEADER
    {
        UINT Magic;
        TILEDFILE_FORMAT Format;
        USHORT BaseWidthTexels;
        USHORT BaseHeightTexels;
        USHORT PageWidthTexels;
        USHORT PageHeightTexels;
        USHORT ArraySliceCount;
        USHORT MipLevelCount;
        USHORT DataFileCount;
        UINT BlockIndexEntries;
        TILEDFILE_PAGEDATA_LOCATOR DefaultPage;
    };

    inline VOID ByteSwapHeader( __inout TILEDFILE_HEADER* pHeader )
    {
        pHeader->Magic = _byteswap_ulong( pHeader->Magic );
        pHeader->Format = (TILEDFILE_FORMAT)_byteswap_ulong( pHeader->Format );
        pHeader->BaseWidthTexels = _byteswap_ushort( pHeader->BaseWidthTexels );
        pHeader->BaseHeightTexels = _byteswap_ushort( pHeader->BaseHeightTexels );
        pHeader->PageWidthTexels = _byteswap_ushort( pHeader->PageWidthTexels );
        pHeader->PageHeightTexels = _byteswap_ushort( pHeader->PageHeightTexels );
        pHeader->ArraySliceCount = _byteswap_ushort( pHeader->ArraySliceCount );
        pHeader->MipLevelCount = _byteswap_ushort( pHeader->MipLevelCount );
        pHeader->DataFileCount = _byteswap_ushort( pHeader->DataFileCount );
        pHeader->BlockIndexEntries = _byteswap_ulong( pHeader->BlockIndexEntries );
        ByteswapLocator( &pHeader->DefaultPage );
    }

    struct TILEDFILE_SUBRESOURCE
    {
        USHORT WidthTexels;
        USHORT HeightTexels;
        USHORT WidthPages;
        USHORT HeightPages;
        USHORT WidthBlocks;
        USHORT HeightBlocks;
        USHORT BlockWidthPages;
        USHORT Padding0;
        UINT BlockIndexLocation;
        TILEDFILE_PAGEDATA_LOCATOR PageArrayLocation;
    };

    inline VOID ByteSwapSubresource( __inout TILEDFILE_SUBRESOURCE* pSubresource )
    {
        pSubresource->WidthTexels = _byteswap_ushort( pSubresource->WidthTexels );
        pSubresource->HeightTexels = _byteswap_ushort( pSubresource->HeightTexels );
        pSubresource->WidthPages = _byteswap_ushort( pSubresource->WidthPages );
        pSubresource->HeightPages = _byteswap_ushort( pSubresource->HeightPages );
        pSubresource->WidthBlocks = _byteswap_ushort( pSubresource->WidthBlocks );
        pSubresource->HeightBlocks = _byteswap_ushort( pSubresource->HeightBlocks );
        pSubresource->BlockWidthPages = _byteswap_ushort( pSubresource->BlockWidthPages );
        pSubresource->BlockIndexLocation = _byteswap_ulong( pSubresource->BlockIndexLocation );
        ByteswapLocator( &pSubresource->PageArrayLocation );
    }

    static const TILEDFILE_PAGEDATA_LOCATOR TILEDFILE_INVALID_LOCATOR = { 0, 0, 0 };
}
