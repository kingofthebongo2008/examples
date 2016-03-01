//--------------------------------------------------------------------------------------
// EndianSwap.h
//
// A set of endian swapping helper functions for the tiled texture content library.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#ifdef _XBOX
#include <xtl.h>
#else
#include <windows.h>
#endif

#include "TiledFileFormat.h"

namespace TiledContent
{
    inline USHORT EndianSwap( USHORT Value )
    {
        return _byteswap_ushort( Value );
    }

    inline UINT EndianSwap( UINT Value )
    {
        return _byteswap_ulong( Value );
    }

    inline UINT64 EndianSwap( UINT64 Value )
    {
        return _byteswap_uint64( Value );
    }

    inline TILEDFILE_PAGEDATA_LOCATOR EndianSwap( TILEDFILE_PAGEDATA_LOCATOR Value )
    {
        UINT SwappedValue = _byteswap_ulong( *(UINT*)&Value );
        return *(TILEDFILE_PAGEDATA_LOCATOR*)&SwappedValue;
    }

    inline TILEDFILE_FORMAT EndianSwap( TILEDFILE_FORMAT Format )
    {
        return (TILEDFILE_FORMAT)_byteswap_ulong( (UINT)Format );
    }

    //--------------------------------------------------------------------------------------
    // Name: EndianSwap64bppBufferToFile
    // Desc: Endian swaps the given buffer directly to an output file.  The buffer contains
    //       a packed array of 64-bit quantities.
    //--------------------------------------------------------------------------------------
    inline VOID EndianSwap64bppBufferToFile( HANDLE hFile, const VOID* pSrc, UINT BufferSize )
    {
        const UINT BlockSize = 64 * 1024;
        BYTE* pDest = new BYTE[BlockSize];

        UINT64* pDestItems = (UINT64*)pDest;

        while( BufferSize > 0 )
        {
            const UINT CopySizeBytes = min( BufferSize, BlockSize );
            UINT CopyCount = CopySizeBytes / sizeof(UINT64);

            const UINT64* pSrcItems = (const UINT64*)pSrc;

            for( UINT i = 0; i < CopyCount; ++i )
            {
                pDestItems[i] = EndianSwap( pSrcItems[i] );
            }

            DWORD BytesWritten = 0;
            WriteFile( hFile, pDest, CopySizeBytes, &BytesWritten, NULL );

            BufferSize -= CopySizeBytes;
            pSrc = (const VOID*)( (const BYTE*)pSrc + CopySizeBytes );
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: EndianSwap32bppBufferToFile
    // Desc: Endian swaps the given buffer directly to an output file.  The buffer contains
    //       a packed array of 32-bit quantities.
    //--------------------------------------------------------------------------------------
    inline VOID EndianSwap32bppBufferToFile( HANDLE hFile, const VOID* pSrc, UINT BufferSize )
    {
        const UINT BlockSize = 64 * 1024;
        BYTE* pDest = new BYTE[BlockSize];

        UINT* pDestItems = (UINT*)pDest;

        while( BufferSize > 0 )
        {
            const UINT CopySizeBytes = min( BufferSize, BlockSize );
            UINT CopyCount = CopySizeBytes / sizeof(UINT);

            const UINT* pSrcItems = (const UINT*)pSrc;

            for( UINT i = 0; i < CopyCount; ++i )
            {
                pDestItems[i] = EndianSwap( pSrcItems[i] );
            }

            DWORD BytesWritten = 0;
            WriteFile( hFile, pDest, CopySizeBytes, &BytesWritten, NULL );

            BufferSize -= CopySizeBytes;
            pSrc = (const VOID*)( (const BYTE*)pSrc + CopySizeBytes );
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: EndianSwap16bppBufferToFile
    // Desc: Endian swaps the given buffer directly to an output file.  The buffer contains
    //       a packed array of 16-bit quantities.
    //--------------------------------------------------------------------------------------
    inline VOID EndianSwap16bppBufferToFile( HANDLE hFile, const VOID* pSrc, UINT BufferSize )
    {
        const UINT BlockSize = 64 * 1024;
        BYTE* pDest = new BYTE[BlockSize];

        USHORT* pDestItems = (USHORT*)pDest;

        while( BufferSize > 0 )
        {
            const UINT CopySizeBytes = min( BufferSize, BlockSize );
            UINT CopyCount = CopySizeBytes / sizeof(USHORT);

            const USHORT* pSrcItems = (const USHORT*)pSrc;

            for( UINT i = 0; i < CopyCount; ++i )
            {
                pDestItems[i] = EndianSwap( pSrcItems[i] );
            }

            DWORD BytesWritten = 0;
            WriteFile( hFile, pDest, CopySizeBytes, &BytesWritten, NULL );

            BufferSize -= CopySizeBytes;
            pSrc = (const VOID*)( (const BYTE*)pSrc + CopySizeBytes );
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: EndianSwap32bppBufferInPlace
    // Desc: Endian swaps a buffer of packed 32-bit quantities, in place.
    //--------------------------------------------------------------------------------------
    inline VOID EndianSwap32bppBufferInPlace( VOID* pBuffer, UINT BufferSizeBytes )
    {
        UINT* pItems = (UINT*)pBuffer;
        UINT ItemCount = BufferSizeBytes / sizeof(UINT);

        for( UINT i = 0; i < ItemCount; ++i )
        {
            pItems[i] = EndianSwap( pItems[i] );
        }
    }
}
