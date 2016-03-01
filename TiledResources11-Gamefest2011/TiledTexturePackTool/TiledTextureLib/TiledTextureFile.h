//--------------------------------------------------------------------------------------
// TiledTextureFile.h
//
// Internal classes that comprise the functionality of the tiled texture content library,
// including files, subresources, and a data serialization class.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#include "TiledTextureLib.h"

namespace TiledContent
{
    struct TiledTextureFile;

    //--------------------------------------------------------------------------------------

    struct TiledTextureDataFile
    {
        struct DataChunk
        {
            UINT64 FileOffset;
            BYTE* pBuffer;
            UINT64 BufferSizeBytes;
            UINT64 ConsumedBytes;
        };
        std::vector<DataChunk> m_Chunks;
        UINT64 m_CurrentFileOffset;
        UINT m_FileIndex;

        TiledTextureDataFile()
        {
            m_CurrentFileOffset = 0;
            m_FileIndex = 0;
        }
        ~TiledTextureDataFile()
        {
            Cleanup();
        }

        VOID ApplyEmptyOffset( UINT PageCount );
        TILEDFILE_PAGEDATA_LOCATOR AllocatePages( UINT PageCount );

        VOID Serialize( HANDLE hFile, TILEDFILE_FORMAT Format );
        BOOL Deserialize( HANDLE hFile, UINT64 PageCount, TILEDFILE_FORMAT Format );

        VOID* GetLocatorData( TILEDFILE_PAGEDATA_LOCATOR Locator, UINT PageOffset ) const;

    protected:
        UINT64 Allocate( UINT64 SizeBytes );
        VOID Cleanup();
    };

    //--------------------------------------------------------------------------------------

    struct TiledTextureSubresource
    {
        TiledTextureSubresource()
        {
            ZeroMemory( &Header, sizeof(Header) );
            pPageBlockIndex = NULL;
        }
        ~TiledTextureSubresource()
        {
            if( pPageBlockIndex != NULL )
            {
                delete[] pPageBlockIndex;
                pPageBlockIndex = NULL;
            }
        }
        TILEDFILE_SUBRESOURCE Header;
        TILEDFILE_PAGEDATA_LOCATOR* pPageBlockIndex;
        TiledTextureFile* pFile;
        UINT SubresourceIndex;

        VOID* GetPageData( UINT PageX, UINT PageY, BOOL CreatePage );

    protected:
        INT GetBlockIndex( UINT PageX, UINT PageY ) const;
    };

    //--------------------------------------------------------------------------------------

    struct TiledTextureFile
    {
        TILEDFILE_HEADER m_Header;
        TiledTextureSubresource* m_pSubresources;
        TiledTextureDataFile m_RootDataFile;

        TiledTextureFile();
        ~TiledTextureFile();

        HRESULT Initialize();

        HRESULT Serialize( HANDLE hFile, const CHAR* strRootFileName, BOOL EndianSwap );
        HRESULT PrepareForSerialization();

        HRESULT Deserialize( HANDLE hFile, const CHAR* strRootFileName );

        VOID* GetPageData( UINT Subresource, UINT PageX, UINT PageY, BOOL CreatePage );
        VOID GetPageCoordinates( UINT Subresource, UINT TexelX, UINT TexelY, UINT* pPageX, UINT* pPageY );

        VOID* GetLocatorData( TILEDFILE_PAGEDATA_LOCATOR Locator, UINT PageOffset ) const;

        TILEDFILE_PAGEDATA_LOCATOR CreatePageBlock( UINT SubresourceIndex, UINT PageWidth, UINT PageHeight );

        VOID FillRect( UINT SubresourceIndex, const RECT* pDestRect, const VOID* pFillValue );
        VOID BlitRect( UINT SubresourceIndex, const RECT* pDestRect, const VOID* pSourceBuffer, const UINT SourceRowPitchBytes );

        VOID GenerateMipData( UINT ArraySlice, UINT SourceLevel, UINT DestLevel );
    };

    //--------------------------------------------------------------------------------------

    inline UINT NextMultiple( UINT Value, UINT Multiple )
    {
        return ((Value + Multiple - 1) / Multiple) * Multiple;
    }

    //--------------------------------------------------------------------------------------

    inline UINT NextWholeQuotient( UINT Value, UINT Divisor )
    {
        return ( Value + Divisor - 1 ) / Divisor;
    }

    //--------------------------------------------------------------------------------------

    extern const CHAR* g_strErrorLocation;
    inline VOID SetErrorLocation( const CHAR* strLocation )
    {
        g_strErrorLocation = strLocation;
    }

#define SETERRORLOCATION_FUNCTION() SetErrorLocation( __FUNCTION__ )

    //--------------------------------------------------------------------------------------

    inline VOID DisplayUserError( const CHAR* strMessage, ... )
    {
        if( g_strErrorLocation != NULL )
        {
            fprintf_s( stderr, "ERROR in %s: ", g_strErrorLocation );
        }
        else
        {
            fprintf_s( stderr, "ERROR: " );
        }

        va_list pArgList;
        va_start( pArgList, strMessage );
        vfprintf_s( stderr, strMessage, pArgList );
        va_end( pArgList );

        fprintf_s( stderr, "\n" );
    }

    //--------------------------------------------------------------------------------------

    inline SIZE GetPageSize( const TILEDFILE_FORMAT Format )
    {
        SIZE ReturnSize = { 0, 0 };
        switch( Format )
        {
        case TILED_FORMAT_8BPP:
            ReturnSize.cx = 256;
            ReturnSize.cy = 256;
            break;
        case TILED_FORMAT_16BPP_B5G6R5:
        case TILED_FORMAT_16BPP_B5G5R5A1:
        case TILED_FORMAT_16BPP_B4G4R4A4:
        case TILED_FORMAT_16BPP_R16:
        case TILED_FORMAT_16BPP_R8G8:
            ReturnSize.cx = 256;
            ReturnSize.cy = 128;
            break;
        case TILED_FORMAT_32BPP_R8G8B8A8:
        case TILED_FORMAT_32BPP_R10G10B10A2:
            ReturnSize.cx = 128;
            ReturnSize.cy = 128;
            break;
        case TILED_FORMAT_64BPP_R16G16B16A16:
        case TILED_FORMAT_64BPP_R16G16B16A16F:
            ReturnSize.cx = 128;
            ReturnSize.cy = 64;
            break;
        case TILED_FORMAT_BC1:
        case TILED_FORMAT_BC4:
            ReturnSize.cx = 512;
            ReturnSize.cy = 256;
            break;
        case TILED_FORMAT_BC2:
        case TILED_FORMAT_BC3:
        case TILED_FORMAT_BC5:
            ReturnSize.cx = 256;
            ReturnSize.cy = 256;
            break;
        case TILED_FORMAT_BC6:
        case TILED_FORMAT_BC7:
        default:
            DisplayUserError( "Unknown or unsupported format." );
            break;
        }

        return ReturnSize;
    }

    //--------------------------------------------------------------------------------------

    inline UINT GetBytesPerTexel( const TILEDFILE_FORMAT Format )
    {
        switch( Format )
        {
        case TILED_FORMAT_8BPP:
            return 1;
            break;
        case TILED_FORMAT_16BPP_B5G6R5:
        case TILED_FORMAT_16BPP_B5G5R5A1:
        case TILED_FORMAT_16BPP_B4G4R4A4:
        case TILED_FORMAT_16BPP_R16:
        case TILED_FORMAT_16BPP_R8G8:
            return 2;
            break;
        case TILED_FORMAT_32BPP_R8G8B8A8:
        case TILED_FORMAT_32BPP_R10G10B10A2:
            return 4;
            break;
        case TILED_FORMAT_64BPP_R16G16B16A16:
        case TILED_FORMAT_64BPP_R16G16B16A16F:
            return 8;
            break;
        case TILED_FORMAT_BC1:
        case TILED_FORMAT_BC4:
            return 8;
            break;
        case TILED_FORMAT_BC2:
        case TILED_FORMAT_BC3:
        case TILED_FORMAT_BC5:
            return 16;
            break;
        case TILED_FORMAT_BC6:
        case TILED_FORMAT_BC7:
        default:
            DisplayUserError( "Unknown or unsupported format." );
            return 0;
        }
    }

    //--------------------------------------------------------------------------------------

    inline UINT GetTexelBlockSize( const TILEDFILE_FORMAT Format )
    {
        switch( Format )
        {
        case TILED_FORMAT_BC1:
        case TILED_FORMAT_BC4:
            return 4;
        case TILED_FORMAT_BC2:
        case TILED_FORMAT_BC3:
        case TILED_FORMAT_BC5:
            return 4;
        case TILED_FORMAT_BC6:
        case TILED_FORMAT_BC7:
            DisplayUserError( "Unknown or unsupported format." );
            return 4;
        default:
            return 1;
        }
    }

    //--------------------------------------------------------------------------------------
    
    inline BOOL CanMipData( const TILEDFILE_FORMAT Format )
    {
        switch( Format )
        {
        case TILED_FORMAT_32BPP_R8G8B8A8:
        case TILED_FORMAT_16BPP_R16:
        case TILED_FORMAT_16BPP_R8G8:
            return TRUE;
        default:
            return FALSE;
        }
    }

    //--------------------------------------------------------------------------------------

    TILEDFILE_HANDLE CreateHandle( TiledTextureFile* pNewFile );
    TiledTextureFile* GetFile( TILEDFILE_HANDLE Handle );
}

