//--------------------------------------------------------------------------------------
// TiledTextureLib.cpp
//
// Functions that convert the externally visible handle-based tiled texture access to
// internal operation on tiled texture objects.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "stdafx.h"

namespace TiledContent
{
    const CHAR* g_strErrorLocation = NULL;

    //--------------------------------------------------------------------------------------

    TILEDFILE_HANDLE TILEDTEXTURELIB_API CreateTiledTextureFile( TILEDFILE_FORMAT Format, UINT WidthTexels, UINT HeightTexels, UINT ArraySize, UINT Levels )
    {
        SETERRORLOCATION_FUNCTION();

        TiledTextureFile* pNewFile = new TiledTextureFile;

        pNewFile->m_Header.BaseWidthTexels = WidthTexels;
        pNewFile->m_Header.BaseHeightTexels = HeightTexels;
        pNewFile->m_Header.Format = Format;
        pNewFile->m_Header.ArraySliceCount = ArraySize;
        pNewFile->m_Header.MipLevelCount = Levels;

        HRESULT hr = pNewFile->Initialize();
        if( FAILED(hr) )
        {
            delete pNewFile;
            return TILEDFILE_INVALID_HANDLE_VALUE;
        }

        TILEDFILE_HANDLE FileHandle = CreateHandle( pNewFile );
        return FileHandle;
    }

    //--------------------------------------------------------------------------------------

    VOID TILEDTEXTURELIB_API DeleteTiledTextureFile( TILEDFILE_HANDLE Handle )
    {
        SETERRORLOCATION_FUNCTION();

        TiledTextureFile* pFile = GetFile( Handle );
        if( pFile != NULL )
        {
            delete pFile;
        }
    }

    //--------------------------------------------------------------------------------------

    HRESULT TILEDTEXTURELIB_API SaveTiledTextureFile( TILEDFILE_HANDLE Handle, BOOL EndianSwap, const CHAR* strFileName )
    {
        SETERRORLOCATION_FUNCTION();

        TiledTextureFile* pFile = GetFile( Handle );
        if( pFile == NULL )
        {
            DisplayUserError( "Invalid handle." );
            return E_INVALIDARG;
        }

        HANDLE hFile = CreateFileA( strFileName, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL );
        if( hFile == INVALID_HANDLE_VALUE )
        {
            DisplayUserError( "Could not create output file \"%s\".", strFileName );
            return E_INVALIDARG;
        }

        HRESULT hr = pFile->Serialize( hFile, strFileName, EndianSwap );
        if( FAILED(hr) )
        {
            DisplayUserError( "Could not serialize data to file; file may be corrupted." );
        }

        CloseHandle( hFile );
        return hr;
    }

    //--------------------------------------------------------------------------------------
    
    HRESULT TILEDTEXTURELIB_API OpenTiledTextureFile( const CHAR* strFileName, TILEDFILE_HANDLE* pHandle )
    {
        SETERRORLOCATION_FUNCTION();

        if( strFileName == NULL )
        {
            DisplayUserError( "Must provide a filename." );
            return E_INVALIDARG;
        }

        if( pHandle == NULL )
        {
            DisplayUserError( "Must provide a valid pointer to a handle." );
            return E_INVALIDARG;
        }

        HANDLE hFile = CreateFileA( strFileName, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL );
        if( hFile == INVALID_HANDLE_VALUE )
        {
            DisplayUserError( "Could not open file \"%s\".", strFileName );
            return E_INVALIDARG;
        }

        TiledTextureFile* pNewFile = new TiledTextureFile();
        HRESULT hr = pNewFile->Deserialize( hFile, strFileName );

        if( FAILED(hr) )
        {
            DisplayUserError( "Could not successfully load file \"%s\".", strFileName );
            delete pNewFile;
            *pHandle = TILEDFILE_INVALID_HANDLE_VALUE;
            return E_FAIL;
        }

        TILEDFILE_HANDLE NewHandle = CreateHandle( pNewFile );
        *pHandle = NewHandle;

        return S_OK;
    }

    //--------------------------------------------------------------------------------------

    VOID TILEDTEXTURELIB_API GetFormatPageSizeTexels( TILEDFILE_FORMAT Format, UINT* pWidthTexels, UINT* pHeightTexels )
    {
        SETERRORLOCATION_FUNCTION();

        SIZE PageSize = GetPageSize( Format );
        if( pWidthTexels != NULL )
        {
            *pWidthTexels = (UINT)PageSize.cx;
        }
        if( pHeightTexels != NULL )
        {
            *pHeightTexels = (UINT)PageSize.cy;
        }
    }

    //--------------------------------------------------------------------------------------

    UINT TILEDTEXTURELIB_API GetLevelCount( TILEDFILE_HANDLE Handle )
    {
        SETERRORLOCATION_FUNCTION();

        TiledTextureFile* pFile = GetFile( Handle );
        if( pFile == NULL )
        {
            DisplayUserError( "Invalid handle." );
            return 0;
        }

        return pFile->m_Header.MipLevelCount;
    }

    //--------------------------------------------------------------------------------------

    UINT TILEDTEXTURELIB_API GetArraySize( TILEDFILE_HANDLE Handle )
    {
        SETERRORLOCATION_FUNCTION();

        TiledTextureFile* pFile = GetFile( Handle );
        if( pFile == NULL )
        {
            DisplayUserError( "Invalid handle." );
            return 0;
        }

        return pFile->m_Header.ArraySliceCount;
    }

    //--------------------------------------------------------------------------------------

    UINT TILEDTEXTURELIB_API ComputeSubresourceIndex( TILEDFILE_HANDLE Handle, UINT ArrayIndex, UINT Level )
    {
        UINT MipLevelCount = GetLevelCount( Handle );

        return ArrayIndex * MipLevelCount + Level;
    }

    //--------------------------------------------------------------------------------------
    
    VOID TILEDTEXTURELIB_API ComputeArrayAndLevel( TILEDFILE_HANDLE Handle, UINT Subresource, UINT* pArrayIndex, UINT* pLevel )
    {
        UINT MipLevelCount = GetLevelCount( Handle );

        UINT ArrayIndex = Subresource / MipLevelCount;
        UINT Level = Subresource % MipLevelCount;
        
        if( pArrayIndex != NULL )
        {
            *pArrayIndex = ArrayIndex;
        }

        if( pLevel != NULL )
        {
            *pLevel = Level;
        }
    }

    //--------------------------------------------------------------------------------------

    VOID TILEDTEXTURELIB_API GetLevelDesc( TILEDFILE_HANDLE Handle, UINT Level, TILEDFILE_LEVEL_DESC* pDesc )
    {
        SETERRORLOCATION_FUNCTION();

        if( pDesc == NULL )
        {
            DisplayUserError( "NULL pointer passed for pDesc." );
            return;
        }

        TiledTextureFile* pFile = GetFile( Handle );
        if( pFile == NULL )
        {
            DisplayUserError( "Invalid handle." );
            return;
        }

        if( Level >= pFile->m_Header.MipLevelCount )
        {
            DisplayUserError( "Invalid level index; valid range is 0 to %d.", pFile->m_Header.MipLevelCount - 1 );
            return;
        }

        TiledTextureSubresource& SubR = pFile->m_pSubresources[Level];
        pDesc->Format = pFile->m_Header.Format;
        pDesc->WidthPages = SubR.Header.WidthPages;
        pDesc->HeightPages = SubR.Header.HeightPages;
        pDesc->WidthTexels = SubR.Header.WidthTexels;
        pDesc->HeightTexels = SubR.Header.HeightTexels;
        pDesc->WidthHeightPageBlock = SubR.Header.BlockWidthPages;
    }

    //--------------------------------------------------------------------------------------

    HRESULT TILEDTEXTURELIB_API SetDefaultPageData( TILEDFILE_HANDLE Handle, const VOID* pBuffer )
    {
        SETERRORLOCATION_FUNCTION();

        TiledTextureFile* pFile = GetFile( Handle );
        if( pFile == NULL )
        {
            DisplayUserError( "Invalid handle." );
            return E_INVALIDARG;
        }

        TILEDFILE_PAGEDATA_LOCATOR DefaultPage = pFile->m_Header.DefaultPage;
        if( DefaultPage == TILEDFILE_INVALID_LOCATOR )
        {
            DefaultPage = pFile->CreatePageBlock( 0, 1, 1 );
            pFile->m_Header.DefaultPage = DefaultPage;
        }

        VOID* pDestData = pFile->GetLocatorData( DefaultPage, 0 );
        memcpy( pDestData, pBuffer, PAGE_SIZE_BYTES );

        return S_OK;
    }

    //--------------------------------------------------------------------------------------

    HRESULT TILEDTEXTURELIB_API FillRect( TILEDFILE_HANDLE Handle, UINT Subresource, const RECT* pDestRect, const VOID* pFillValue )
    {
        SETERRORLOCATION_FUNCTION();

        TiledTextureFile* pFile = GetFile( Handle );
        if( pFile == NULL )
        {
            DisplayUserError( "Invalid handle." );
            return E_INVALIDARG;
        }

        UINT SubresourceCount = pFile->m_Header.ArraySliceCount * pFile->m_Header.MipLevelCount;
        if( Subresource > SubresourceCount )
        {
            DisplayUserError( "Subresource must be between 0 and %d.", SubresourceCount );
            return E_INVALIDARG;
        }

        RECT WholeRect = { 0, 0, 0, 0 };
        if( pDestRect == NULL )
        {
            UINT Level = Subresource % GetLevelCount( Handle );

            TILEDFILE_LEVEL_DESC LevelDesc;
            GetLevelDesc( Handle, Level, &LevelDesc );
            WholeRect.right = LevelDesc.WidthTexels;
            WholeRect.bottom = LevelDesc.HeightTexels;
            pDestRect = &WholeRect;
        }

        BYTE SixteenZeros[16];
        if( pFillValue == NULL )
        {
            ZeroMemory( SixteenZeros, ARRAYSIZE(SixteenZeros) * sizeof(BYTE) );
            pFillValue = SixteenZeros;
        }

        pFile->FillRect( Subresource, pDestRect, pFillValue );

        return S_OK;
    }

    //--------------------------------------------------------------------------------------
    
    HRESULT TILEDTEXTURELIB_API BlitRect( TILEDFILE_HANDLE Handle, UINT Subresource, const RECT* pDestRect, const VOID* pBuffer, UINT RowPitchBytes )
    {
        SETERRORLOCATION_FUNCTION();

        TiledTextureFile* pFile = GetFile( Handle );
        if( pFile == NULL )
        {
            DisplayUserError( "Invalid handle." );
            return E_INVALIDARG;
        }

        UINT SubresourceCount = pFile->m_Header.ArraySliceCount * pFile->m_Header.MipLevelCount;
        if( Subresource > SubresourceCount )
        {
            DisplayUserError( "Subresource must be between 0 and %d.", SubresourceCount );
            return E_INVALIDARG;
        }

        if( pBuffer == NULL )
        {
            DisplayUserError( "Must provide a source image buffer." );
            return E_INVALIDARG;
        }

        if( RowPitchBytes == 0 )
        {
            DisplayUserError( "Must provide a non-zero row pitch in bytes." );
            return E_INVALIDARG;
        }

        RECT WholeRect = { 0, 0, 0, 0 };
        if( pDestRect == NULL )
        {
            UINT Level = Subresource % GetLevelCount( Handle );

            TILEDFILE_LEVEL_DESC LevelDesc;
            GetLevelDesc( Handle, Level, &LevelDesc );
            WholeRect.right = LevelDesc.WidthTexels;
            WholeRect.bottom = LevelDesc.HeightTexels;
            pDestRect = &WholeRect;
        }

        pFile->BlitRect( Subresource, pDestRect, pBuffer, RowPitchBytes );

        return S_OK;
    }

    //--------------------------------------------------------------------------------------
    
    HRESULT TILEDTEXTURELIB_API GenerateMipData( TILEDFILE_HANDLE Handle, UINT Subresource )
    {
        SETERRORLOCATION_FUNCTION();

        TiledTextureFile* pFile = GetFile( Handle );
        if( pFile == NULL )
        {
            DisplayUserError( "Invalid handle." );
            return E_INVALIDARG;
        }

        UINT SubresourceCount = pFile->m_Header.ArraySliceCount * pFile->m_Header.MipLevelCount;
        if( Subresource > SubresourceCount )
        {
            DisplayUserError( "Subresource must be between 0 and %d.", SubresourceCount );
            return E_INVALIDARG;
        }

        UINT MipLevelIndex = Subresource % pFile->m_Header.MipLevelCount;
        if( MipLevelIndex == 0 )
        {
            DisplayUserError( "Cannot generate mip data on mip level 0, since it is a base level." );
            return E_INVALIDARG;
        }

        BOOL CanMip = CanMipData( pFile->m_Header.Format );
        if( !CanMip )
        {
            DisplayUserError( "Cannot generate mip data for this format." );
            return E_INVALIDARG;
        }

        UINT SourceLevelIndex = MipLevelIndex - 1;
        UINT ArraySlice = Subresource / pFile->m_Header.MipLevelCount;

        pFile->GenerateMipData( ArraySlice, SourceLevelIndex, MipLevelIndex );

        return S_OK;
    }

    //--------------------------------------------------------------------------------------

    HRESULT TILEDTEXTURELIB_API GetPageData( TILEDFILE_HANDLE Handle, UINT Subresource, UINT PageX, UINT PageY, BOOL CreateMissingPage, VOID** ppBuffer )
    {
        SETERRORLOCATION_FUNCTION();

        TiledTextureFile* pFile = GetFile( Handle );
        if( pFile == NULL )
        {
            DisplayUserError( "Invalid handle." );
            return E_INVALIDARG;
        }

        UINT SubresourceCount = pFile->m_Header.ArraySliceCount * pFile->m_Header.MipLevelCount;
        if( Subresource > SubresourceCount )
        {
            DisplayUserError( "Subresource must be between 0 and %d.", SubresourceCount );
            return E_INVALIDARG;
        }

        if( ppBuffer == NULL )
        {
            DisplayUserError( "Must provide a valid pointer to ppBuffer." );
            return E_INVALIDARG;
        }

        VOID* pPage = pFile->GetPageData( Subresource, PageX, PageY, CreateMissingPage );
        *ppBuffer = pPage;

        return S_OK;
    }

    //--------------------------------------------------------------------------------------

    HRESULT TILEDTEXTURELIB_API GetTexel( TILEDFILE_HANDLE Handle, UINT Subresource, UINT X, UINT Y, BOOL CreateMissingPage, VOID** ppBuffer )
    {
        SETERRORLOCATION_FUNCTION();

        TiledTextureFile* pFile = GetFile( Handle );
        if( pFile == NULL )
        {
            DisplayUserError( "Invalid handle." );
            return E_INVALIDARG;
        }

        UINT SubresourceCount = pFile->m_Header.ArraySliceCount * pFile->m_Header.MipLevelCount;
        if( Subresource > SubresourceCount )
        {
            DisplayUserError( "Subresource must be between 0 and %d.", SubresourceCount );
            return E_INVALIDARG;
        }

        if( ppBuffer == NULL )
        {
            DisplayUserError( "Must provide a valid pointer to ppBuffer." );
            return E_INVALIDARG;
        }

        SIZE PageSize = GetPageSize( pFile->m_Header.Format );
        UINT PageX = X / PageSize.cx;
        UINT PageY = Y / PageSize.cy;

        UINT TexelSizeBytes = GetBytesPerTexel( pFile->m_Header.Format );
        UINT StrideBytes = TexelSizeBytes * PageSize.cx;

        VOID* pPage = pFile->GetPageData( Subresource, PageX, PageY, CreateMissingPage );
        if( pPage == NULL )
        {
            *ppBuffer = NULL;
            return S_OK;
        }

        UINT PageOffsetX = X % PageSize.cx;
        UINT PageOffsetY = Y % PageSize.cy;

        BYTE* pPageData = (BYTE*)pPage;
        pPageData += StrideBytes * PageOffsetY + PageOffsetX * TexelSizeBytes;
        *ppBuffer = (VOID*)pPageData;

        return S_OK;
    }

    //--------------------------------------------------------------------------------------

    inline BOOL FetchThreeHeightValues( TILEDFILE_HANDLE Handle, UINT Subresource, UINT X, UINT Y, VOID** ppCenterValue, VOID** ppRightValue, VOID** ppBottomValue )
    {
        VOID* pCenterValue = NULL;
        GetTexel( Handle, Subresource, X, Y, FALSE, &pCenterValue );
        if( pCenterValue == NULL )
        {
            return FALSE;
        }

        VOID* pRightValue = NULL;
        VOID* pBottomValue = NULL;
        GetTexel( Handle, Subresource, X + 1, Y, FALSE, &pRightValue );
        GetTexel( Handle, Subresource, X, Y + 1, FALSE, &pBottomValue );
        if( pRightValue == NULL )
        {
            pRightValue = pCenterValue;
        }
        if( pBottomValue == NULL )
        {
            pBottomValue = pCenterValue;
        }

        *ppCenterValue = pCenterValue;
        *ppRightValue = pRightValue;
        *ppBottomValue = pBottomValue;

        return TRUE;
    }

    //--------------------------------------------------------------------------------------
    
    inline VOID GenerateNormalValues( BYTE* pNormalX, BYTE* pNormalY, UINT CenterValue, UINT RightValue, UINT BottomValue, UINT MaxValue, FLOAT HeightMultiplier )
    {
        INT XDifference = (INT)CenterValue - (INT)RightValue;
        INT YDifference = (INT)CenterValue - (INT)BottomValue;

        FLOAT Distance = (FLOAT)( MaxValue / 256 );
        if( MaxValue <= 255 )
        {
            Distance = 16.0f;
        }
        FLOAT XAngle = atan2f( (FLOAT)XDifference * HeightMultiplier, Distance );
        FLOAT YAngle = atan2f( (FLOAT)YDifference * HeightMultiplier, Distance );

        XAngle += (FLOAT)M_PI;
        YAngle += (FLOAT)M_PI;

        *pNormalX = (BYTE)( ( XAngle * 255.0f ) / ( 2.0f * M_PI ) );
        *pNormalY = (BYTE)( ( YAngle * 255.0f ) / ( 2.0f * M_PI ) );
    }

    //--------------------------------------------------------------------------------------
    
    HRESULT TILEDTEXTURELIB_API GenerateNormalMap( TILEDFILE_HANDLE DestHandle, UINT DestSubresource, TILEDFILE_HANDLE SourceHandle, UINT SourceSubresource )
    {
        SETERRORLOCATION_FUNCTION();

        if( SourceHandle == TILEDFILE_INVALID_HANDLE_VALUE || DestHandle == TILEDFILE_INVALID_HANDLE_VALUE )
        {
            DisplayUserError( "Invalid handle." );
            return E_INVALIDARG;
        }

        UINT SourceLevel, DestLevel;
        ComputeArrayAndLevel( SourceHandle, SourceSubresource, NULL, &SourceLevel );
        ComputeArrayAndLevel( DestHandle, DestSubresource, NULL, &DestLevel );

        TILEDFILE_LEVEL_DESC SourceLevelDesc, DestLevelDesc;
        GetLevelDesc( SourceHandle, SourceLevel, &SourceLevelDesc );
        GetLevelDesc( DestHandle, DestLevel, &DestLevelDesc );

        if( SourceLevelDesc.WidthTexels != DestLevelDesc.WidthTexels || SourceLevelDesc.HeightTexels != DestLevelDesc.HeightTexels )
        {
            DisplayUserError( "Source subresource (%d x %d) must be the same size as destination subresource (%d x %d).", SourceLevelDesc.WidthTexels, SourceLevelDesc.HeightTexels, DestLevelDesc.WidthTexels, DestLevelDesc.HeightTexels );
            return E_INVALIDARG;
        }

        BOOL ValidSourceFormat = FALSE;
        switch( SourceLevelDesc.Format )
        {
        case TILED_FORMAT_16BPP_R16:
        case TILED_FORMAT_8BPP:
            ValidSourceFormat = TRUE;
            break;
        }

        if( !ValidSourceFormat )
        {
            DisplayUserError( "The source texture is not in a supported format." );
            return E_INVALIDARG;
        }

        BOOL ValidDestFormat = FALSE;
        switch( DestLevelDesc.Format )
        {
        case TILED_FORMAT_16BPP_R8G8:
        case TILED_FORMAT_32BPP_R8G8B8A8:
            ValidDestFormat = TRUE;
        }

        if( !ValidDestFormat )
        {
            DisplayUserError( "The destination texture is not in a supported format." );
            return E_INVALIDARG;
        }

        UINT MaxValue = 65536;
        switch( SourceLevelDesc.Format )
        {
        case TILED_FORMAT_8BPP:
            MaxValue = 255;
            break;
        }

        const FLOAT HeightMultiplier = (FLOAT)max( SourceLevelDesc.WidthTexels, SourceLevelDesc.HeightTexels ) / 1024.0f;

        for( UINT y = 0; y < DestLevelDesc.HeightTexels; ++y )
        {
            for( UINT x = 0; x < DestLevelDesc.WidthTexels; ++x )
            {
                VOID* pCenterTexel = NULL;
                VOID* pRightTexel = NULL;
                VOID* pBottomTexel = NULL;

                BOOL ValidTexel = FetchThreeHeightValues( SourceHandle, SourceSubresource, x, y, &pCenterTexel, &pRightTexel, &pBottomTexel );
                if( !ValidTexel )
                {
                    continue;
                }

                UINT CenterValue, RightValue, BottomValue;
                switch( SourceLevelDesc.Format )
                {
                case TILED_FORMAT_16BPP_R16:
                    CenterValue = *(USHORT*)pCenterTexel;
                    RightValue = *(USHORT*)pRightTexel;
                    BottomValue = *(USHORT*)pBottomTexel;
                    break;
                case TILED_FORMAT_8BPP:
                    CenterValue = *(BYTE*)pCenterTexel;
                    RightValue = *(BYTE*)pRightTexel;
                    BottomValue = *(BYTE*)pBottomTexel;
                    break;
                }

                BYTE NormalX;
                BYTE NormalY;

                GenerateNormalValues( &NormalX, &NormalY, CenterValue, RightValue, BottomValue, MaxValue, HeightMultiplier );

                BYTE* pDestTexel = NULL;
                GetTexel( DestHandle, DestSubresource, x, y, TRUE, (VOID**)&pDestTexel );
                pDestTexel[0] = NormalX;
                pDestTexel[1] = NormalY;
            }
        }

        return S_OK;
    }
}
