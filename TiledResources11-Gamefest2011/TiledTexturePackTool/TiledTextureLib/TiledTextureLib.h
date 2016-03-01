//--------------------------------------------------------------------------------------
// TiledTextureLib.h
//
// Public interface to the tiled texture content library.  A tiled texture file contains
// a single tiled texture.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#ifdef _XBOX
#include <xtl.h>
#else
#include <windows.h>
#endif

#ifdef TILEDTEXTURELIB_STATIC

#define TILEDTEXTURELIB_API

#else

#ifdef TILEDTEXTURELIB_EXPORTS
#define TILEDTEXTURELIB_API __declspec(dllexport) __stdcall
#else
#define TILEDTEXTURELIB_API __declspec(dllimport) __stdcall
#endif

#endif

#include "TiledFileFormat.h"

namespace TiledContent
{

    struct TILEDFILE_LEVEL_DESC
    {
        TILEDFILE_FORMAT Format;

        USHORT WidthTexels;
        USHORT HeightTexels;

        USHORT WidthPages;
        USHORT HeightPages;

        UINT WidthHeightPageBlock;
    };

    typedef UINT TILEDFILE_HANDLE;
    static const TILEDFILE_HANDLE TILEDFILE_INVALID_HANDLE_VALUE = (TILEDFILE_HANDLE)-1;

    TILEDFILE_HANDLE TILEDTEXTURELIB_API CreateTiledTextureFile( TILEDFILE_FORMAT Format, UINT WidthTexels, UINT HeightTexels, UINT ArraySize, UINT Levels );
    VOID TILEDTEXTURELIB_API DeleteTiledTextureFile( TILEDFILE_HANDLE Handle );

    HRESULT TILEDTEXTURELIB_API OpenTiledTextureFile( const CHAR* strFileName, TILEDFILE_HANDLE* pHandle );
    HRESULT TILEDTEXTURELIB_API SaveTiledTextureFile( TILEDFILE_HANDLE Handle, BOOL EndianSwap, const CHAR* strFileName );

    VOID TILEDTEXTURELIB_API GetFormatPageSizeTexels( TILEDFILE_FORMAT Format, UINT* pWidthTexels, UINT* pHeightTexels );

    UINT TILEDTEXTURELIB_API GetLevelCount( TILEDFILE_HANDLE Handle );
    UINT TILEDTEXTURELIB_API GetArraySize( TILEDFILE_HANDLE Handle );
    VOID TILEDTEXTURELIB_API GetLevelDesc( TILEDFILE_HANDLE Handle, UINT Level, TILEDFILE_LEVEL_DESC* pDesc );
    UINT TILEDTEXTURELIB_API ComputeSubresourceIndex( TILEDFILE_HANDLE Handle, UINT ArrayIndex, UINT Level );
    VOID TILEDTEXTURELIB_API ComputeArrayAndLevel( TILEDFILE_HANDLE Handle, UINT Subresource, UINT* pArrayIndex, UINT* pLevel );

    HRESULT TILEDTEXTURELIB_API SetDefaultPageData( TILEDFILE_HANDLE Handle, const VOID* pBuffer );

    HRESULT TILEDTEXTURELIB_API FillRect( TILEDFILE_HANDLE Handle, UINT Subresource, const RECT* pDestRect, const VOID* pFillValue );
    HRESULT TILEDTEXTURELIB_API BlitRect( TILEDFILE_HANDLE Handle, UINT Subresource, const RECT* pDestRect, const VOID* pBuffer, UINT RowPitchBytes );
    HRESULT TILEDTEXTURELIB_API GenerateMipData( TILEDFILE_HANDLE Handle, UINT Subresource );
    HRESULT TILEDTEXTURELIB_API GenerateNormalMap( TILEDFILE_HANDLE DestHandle, UINT DestSubresource, TILEDFILE_HANDLE SourceHandle, UINT SourceSubresource );

    HRESULT TILEDTEXTURELIB_API GetPageData( TILEDFILE_HANDLE Handle, UINT Subresource, UINT PageX, UINT PageY, BOOL CreateMissingPage, VOID** ppBuffer );
    HRESULT TILEDTEXTURELIB_API GetTexel( TILEDFILE_HANDLE Handle, UINT Subresource, UINT X, UINT Y, BOOL CreateMissingPage, VOID** ppTexel );
}

