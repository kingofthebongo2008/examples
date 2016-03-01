//--------------------------------------------------------------------------------------
// TiledTextureFile.cpp
//
// Implementation of the tiled texture manipulation functions in the tiled texture content
// library, including file I/O, memory allocation, and texel manipulation.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "stdafx.h"

namespace TiledContent
{
    const UINT HandleOffset = 0x80000000;

    std::vector<TiledTextureFile*> g_TiledTextureFiles;

    const UINT DEFAULT_CHUNK_PAGE_COUNT = 32;
    const UINT64 DEFAULT_CHUNK_SIZE = ( PAGE_SIZE_BYTES * DEFAULT_CHUNK_PAGE_COUNT );

    //--------------------------------------------------------------------------------------

    UINT64 TiledTextureDataFile::Allocate( UINT64 SizeBytes )
    {
        assert( SizeBytes > 0 );
        UINT64 CurrentOffset = m_CurrentFileOffset;

        if( m_Chunks.size() == 0 )
        {
            UINT64 ChunkSize = max( SizeBytes, DEFAULT_CHUNK_SIZE );
            DataChunk NewChunk;
            NewChunk.FileOffset = m_CurrentFileOffset;
            NewChunk.BufferSizeBytes = ChunkSize;
            NewChunk.ConsumedBytes = SizeBytes;
            NewChunk.pBuffer = new BYTE[(UINT)NewChunk.BufferSizeBytes];
            m_Chunks.push_back( NewChunk );

            m_CurrentFileOffset += SizeBytes;

            return CurrentOffset;
        }

        while( SizeBytes > 0 )
        {
            DataChunk& LastChunk = m_Chunks[ m_Chunks.size() - 1 ];
            UINT64 ChunkSizeRemaining = ( LastChunk.BufferSizeBytes - LastChunk.ConsumedBytes );
            UINT64 ConsumeSize = min( SizeBytes, ChunkSizeRemaining );
            SizeBytes -= ConsumeSize;
            m_CurrentFileOffset += ConsumeSize;
            LastChunk.ConsumedBytes += ConsumeSize;

            if( SizeBytes > 0 )
            {
                UINT64 NewChunkSize = max( SizeBytes, DEFAULT_CHUNK_SIZE );
                DataChunk NewChunk;
                NewChunk.FileOffset = m_CurrentFileOffset;
                NewChunk.BufferSizeBytes = NewChunkSize;
                NewChunk.ConsumedBytes = SizeBytes;
                NewChunk.pBuffer = new BYTE[(UINT)NewChunk.BufferSizeBytes];
                m_Chunks.push_back( NewChunk );

                m_CurrentFileOffset += SizeBytes;
                SizeBytes = 0;
            }
        }

        return CurrentOffset;
    }

    //--------------------------------------------------------------------------------------

    VOID TiledTextureDataFile::Cleanup()
    {
        UINT ChunkCount = (UINT)m_Chunks.size();
        for( UINT i = 0; i < ChunkCount; ++i )
        {
            delete[] m_Chunks[i].pBuffer;
            m_Chunks[i].pBuffer = NULL;
        }
    }

    //--------------------------------------------------------------------------------------

    VOID TiledTextureDataFile::ApplyEmptyOffset( UINT PageCount )
    {
        assert( m_Chunks.size() == 0 );
        m_CurrentFileOffset += (UINT64)( PageCount * PAGE_SIZE_BYTES );
    }

    //--------------------------------------------------------------------------------------

    TILEDFILE_PAGEDATA_LOCATOR TiledTextureDataFile::AllocatePages( UINT PageCount )
    {
        UINT64 Offset = Allocate( PageCount * PAGE_SIZE_BYTES );
        TILEDFILE_PAGEDATA_LOCATOR Locator;
        Locator.DataFileIndex = m_FileIndex;
        Locator.UniquePages = 1;
        Locator.PageOffset = (UINT)( Offset / PAGE_SIZE_BYTES );
        return Locator;
    }

    //--------------------------------------------------------------------------------------

    VOID TiledTextureDataFile::Serialize( HANDLE hFile, TILEDFILE_FORMAT Format )
    {
        UINT ChunkCount = (UINT)m_Chunks.size();
        for( UINT i = 0; i < ChunkCount; ++i )
        {
            DataChunk& Chunk = m_Chunks[i];
            DWORD BytesWritten = 0;
            assert( Chunk.ConsumedBytes <= 0xFFFFFFFF );

            switch( Format )
            {
            case TILED_FORMAT_32BPP_R8G8B8A8:
            case TILED_FORMAT_32BPP_R10G10B10A2:
                EndianSwap32bppBufferToFile( hFile, Chunk.pBuffer, (UINT)Chunk.ConsumedBytes );
                break;
            case TILED_FORMAT_16BPP_R16:
            case TILED_FORMAT_16BPP_B5G6R5:
            case TILED_FORMAT_16BPP_B5G5R5A1:
            case TILED_FORMAT_16BPP_B4G4R4A4:
            case TILED_FORMAT_16BPP_R8G8:
                EndianSwap16bppBufferToFile( hFile, Chunk.pBuffer, (UINT)Chunk.ConsumedBytes );
                break;
            case TILED_FORMAT_64BPP_R16G16B16A16:
            case TILED_FORMAT_64BPP_R16G16B16A16F:
                EndianSwap64bppBufferToFile( hFile, Chunk.pBuffer, (UINT)Chunk.ConsumedBytes );
                break;
            default:
                WriteFile( hFile, Chunk.pBuffer, (DWORD)Chunk.ConsumedBytes, &BytesWritten, NULL );
                break;
            }
        }
    }

    //--------------------------------------------------------------------------------------
    
    BOOL TiledTextureDataFile::Deserialize( HANDLE hFile, UINT64 PageCount, TILEDFILE_FORMAT Format )
    {
        while( PageCount > 0 )
        {
            DataChunk NewChunk;
            NewChunk.BufferSizeBytes = DEFAULT_CHUNK_SIZE;
            NewChunk.pBuffer = new BYTE[(size_t)NewChunk.BufferSizeBytes];

            if( NewChunk.pBuffer == NULL )
            {
                return FALSE;
            }

            UINT64 CurrentChunkPages = min( PageCount, DEFAULT_CHUNK_PAGE_COUNT );
            PageCount -= CurrentChunkPages;
            NewChunk.ConsumedBytes = CurrentChunkPages * PAGE_SIZE_BYTES;
            NewChunk.FileOffset = m_CurrentFileOffset;
            m_CurrentFileOffset += NewChunk.ConsumedBytes;

            m_Chunks.push_back( NewChunk );

            DWORD BytesRead = 0;
            BOOL Success = ReadFile( hFile, NewChunk.pBuffer, (DWORD)NewChunk.ConsumedBytes, &BytesRead, NULL );
            if( !Success )
            {
                return FALSE;
            }

            switch( Format )
            {
            case TILED_FORMAT_32BPP_R8G8B8A8:
            case TILED_FORMAT_32BPP_R10G10B10A2:
                EndianSwap32bppBufferInPlace( NewChunk.pBuffer, (UINT)NewChunk.ConsumedBytes );
                break;
            }
        }
        return TRUE;
    }

    //--------------------------------------------------------------------------------------

    VOID* TiledTextureDataFile::GetLocatorData( TILEDFILE_PAGEDATA_LOCATOR Locator, UINT PageOffset ) const
    {
        assert( Locator.DataFileIndex == m_FileIndex );

        UINT PageLocation = Locator.PageOffset + PageOffset;

        UINT ChunkCount = (UINT)m_Chunks.size();
        for( UINT i = 0; i < ChunkCount; ++i )
        {
            const DataChunk& DC = m_Chunks[i];
            UINT64 PageStart = DC.FileOffset / PAGE_SIZE_BYTES;
            UINT64 PageEnd = PageStart + ( DC.ConsumedBytes / PAGE_SIZE_BYTES );
            if( PageLocation >= PageStart && PageLocation < PageEnd )
            {
                UINT64 ChunkOffset = ( PageLocation - PageStart ) * PAGE_SIZE_BYTES;
                VOID* pData = (VOID*)( DC.pBuffer + ChunkOffset );
                return pData;
            }
        }
        return NULL;
    }

    //--------------------------------------------------------------------------------------

    VOID* TiledTextureSubresource::GetPageData( UINT PageX, UINT PageY, BOOL CreatePage )
    {
        INT BlockIndex = GetBlockIndex( PageX, PageY );
        if( BlockIndex == -1 )
        {
            return NULL;
        }

        TILEDFILE_PAGEDATA_LOCATOR Locator = pPageBlockIndex[BlockIndex];

        if( Locator == TILEDFILE_INVALID_LOCATOR )
        {
            if( CreatePage )
            {
                // create new block
                Locator = pFile->CreatePageBlock( SubresourceIndex, Header.BlockWidthPages, Header.BlockWidthPages );

                // store block in block index
                pPageBlockIndex[BlockIndex] = Locator;
            }
        }

        if( Locator != TILEDFILE_INVALID_LOCATOR )
        {
            assert( Locator != pFile->m_Header.DefaultPage );

            // compute offset within block
            UINT BlockX = PageX % Header.BlockWidthPages;
            UINT BlockY = PageY % Header.BlockWidthPages;
            UINT BlockOffset = BlockY * Header.BlockWidthPages + BlockX;

            // return data address within block
            return pFile->GetLocatorData( Locator, BlockOffset );
        }
        else
        {
            return NULL;
        }
    }

    //--------------------------------------------------------------------------------------

    INT TiledTextureSubresource::GetBlockIndex( UINT PageX, UINT PageY ) const
    {
        INT BlockX = PageX / Header.BlockWidthPages;
        INT BlockY = PageY / Header.BlockWidthPages;

        if( BlockX >= Header.WidthBlocks || BlockY >= Header.HeightBlocks )
        {
            return -1;
        }

        return BlockY * Header.WidthBlocks + BlockX;
    }

    //--------------------------------------------------------------------------------------

    TiledTextureFile::TiledTextureFile()
    {
        m_pSubresources = NULL;
        ZeroMemory( &m_Header, sizeof(m_Header) );
        m_Header.Magic = TILEDFILE_HEADER_MAGIC;
    }

    //--------------------------------------------------------------------------------------

    TiledTextureFile::~TiledTextureFile()
    {
        if( m_pSubresources != NULL )
        {
            delete[] m_pSubresources;
        }
    }

    //--------------------------------------------------------------------------------------

    TILEDFILE_HANDLE CreateHandle( TiledTextureFile* pNewFile )
    {
        TILEDFILE_HANDLE Handle = (TILEDFILE_HANDLE)( g_TiledTextureFiles.size() + HandleOffset );
        g_TiledTextureFiles.push_back( pNewFile );
        return Handle;
    }

    //--------------------------------------------------------------------------------------

    TiledTextureFile* GetFile( TILEDFILE_HANDLE Handle )
    {
        UINT Index = (UINT)Handle - HandleOffset;
        if( Index >= g_TiledTextureFiles.size() )
        {
            return NULL;
        }
        return g_TiledTextureFiles[Index];
    }

    //--------------------------------------------------------------------------------------

    HRESULT TiledTextureFile::Initialize()
    {
        TILEDFILE_HEADER& Header = m_Header;

        if( Header.BaseWidthTexels == 0 || Header.BaseWidthTexels > MAX_WIDTH_TEXELS )
        {
            DisplayUserError( "Texture base width in texels must be between 0 and %d.", MAX_WIDTH_TEXELS );
            return E_INVALIDARG;
        }
        if( Header.BaseHeightTexels == 0 || Header.BaseHeightTexels > MAX_HEIGHT_TEXELS )
        {
            DisplayUserError( "Texture base height in texels must be between 0 and %d.", MAX_HEIGHT_TEXELS );
            return E_INVALIDARG;
        }
        if( Header.ArraySliceCount == 0 )
        {
            DisplayUserError( "Texture must have at least 1 array level." );
            return E_INVALIDARG;
        }

        SIZE PageSize = GetPageSize( Header.Format );
        if( PageSize.cx == 0 || PageSize.cy == 0 )
        {
            DisplayUserError( "Could not create texture with desired format." );
            return E_INVALIDARG;
        }

        Header.PageWidthTexels = (UINT)PageSize.cx;
        Header.PageHeightTexels = (UINT)PageSize.cy;

        UINT MipLevels = Header.MipLevelCount;
        UINT MaxMipLevelCount = 1;
        if( MipLevels == 0 )
        {
            UINT Width = Header.BaseWidthTexels;
            UINT Height = Header.BaseHeightTexels;
            UINT WidthPages = NextWholeQuotient( Width, Header.PageWidthTexels );
            UINT HeightPages = NextWholeQuotient( Height, Header.PageHeightTexels );
            while( WidthPages > 1 || HeightPages > 1 )
            {
                ++MaxMipLevelCount;
                Width = max( 1, Width >> 1 );
                Height = max( 1, Height >> 1 );
                WidthPages = NextWholeQuotient( Width, Header.PageWidthTexels );
                HeightPages = NextWholeQuotient( Height, Header.PageHeightTexels );
            }
            MipLevels = MaxMipLevelCount;
        }

        if( MipLevels > MaxMipLevelCount )
        {
            DisplayUserError( "Can only create %d mip levels for texture size %d x %d.", MaxMipLevelCount, Header.BaseWidthTexels, Header.BaseHeightTexels );
            return E_INVALIDARG;
        }

        Header.MipLevelCount = MipLevels;

        Header.DefaultPage = TILEDFILE_INVALID_LOCATOR;

        const UINT SubresourceCount = Header.ArraySliceCount * Header.MipLevelCount;
        m_pSubresources = new TiledTextureSubresource[SubresourceCount];

        for( UINT ArrayIndex = 0; ArrayIndex < Header.ArraySliceCount; ++ArrayIndex )
        {
            UINT WidthTexels = Header.BaseWidthTexels;
            UINT HeightTexels = Header.BaseHeightTexels;

            for( UINT MipIndex = 0; MipIndex < Header.MipLevelCount; ++MipIndex )
            {
                UINT SubresourceIndex = ArrayIndex * Header.MipLevelCount + MipIndex;
                TiledTextureSubresource& SubR = m_pSubresources[SubresourceIndex];

                SubR.pFile = this;

                SubR.Header.WidthTexels = WidthTexels;
                SubR.Header.HeightTexels = HeightTexels;

                UINT WidthPages = NextWholeQuotient( WidthTexels, Header.PageWidthTexels );
                UINT HeightPages = NextWholeQuotient( HeightTexels, Header.PageHeightTexels );

                SubR.Header.WidthPages = WidthPages;
                SubR.Header.HeightPages = HeightPages;

                UINT BlockSize = 2;
                if( WidthPages < BlockSize || HeightPages < BlockSize )
                {
                    BlockSize = 1;
                }

                SubR.Header.BlockWidthPages = BlockSize;

                UINT WidthBlocks = NextWholeQuotient( WidthPages, BlockSize );
                UINT HeightBlocks = NextWholeQuotient( HeightPages, BlockSize );

                SubR.Header.WidthBlocks = WidthBlocks;
                SubR.Header.HeightBlocks = HeightBlocks;

                UINT BlockCount = WidthBlocks * HeightBlocks;
                SubR.pPageBlockIndex = new TILEDFILE_PAGEDATA_LOCATOR[BlockCount];
                ZeroMemory( SubR.pPageBlockIndex, sizeof(TILEDFILE_PAGEDATA_LOCATOR) * BlockCount );

                WidthTexels = max( 1, WidthTexels >> 1 );
                HeightTexels = max( 1, HeightTexels >> 1 );
            }
        }

        PrepareForSerialization();
        UINT PageDataOffsetBytes = sizeof(TILEDFILE_HEADER) + sizeof(TILEDFILE_SUBRESOURCE) * SubresourceCount;
        PageDataOffsetBytes += Header.BlockIndexEntries * sizeof(TILEDFILE_PAGEDATA_LOCATOR);

        UINT HeaderAndIndexPageCount = NextWholeQuotient( PageDataOffsetBytes, PAGE_SIZE_BYTES );

        m_RootDataFile.ApplyEmptyOffset( HeaderAndIndexPageCount );

        return S_OK;
    }

    //--------------------------------------------------------------------------------------

    inline DWORD WritePageBlockIndex( HANDLE hFile, const TILEDFILE_PAGEDATA_LOCATOR* pPageBlockIndex, UINT IndexSize, const TILEDFILE_PAGEDATA_LOCATOR DefaultPage, BOOL Swap )
    {
        DWORD BytesWritten = 0;
        if( DefaultPage == TILEDFILE_INVALID_LOCATOR && !Swap )
        {
            WriteFile( hFile, pPageBlockIndex, IndexSize * sizeof(TILEDFILE_PAGEDATA_LOCATOR), &BytesWritten, NULL );
            return BytesWritten;
        }

        for( UINT i = 0; i < IndexSize; ++i )
        {
            DWORD BlockWritten = 0;
            TILEDFILE_PAGEDATA_LOCATOR Locator = pPageBlockIndex[i];

            // on file write, replace null locator with the default page
            if( Locator == TILEDFILE_INVALID_LOCATOR )
            {
                Locator = DefaultPage;
            }

            if( Swap )
            {
                Locator = EndianSwap( Locator );
            }
            WriteFile( hFile, &Locator, sizeof(TILEDFILE_PAGEDATA_LOCATOR), &BlockWritten, NULL );
            BytesWritten += BlockWritten;
        }

        return BytesWritten;
    }

    //--------------------------------------------------------------------------------------
    
    inline VOID SwapHeader( TILEDFILE_HEADER* pSwappedHeader, const TILEDFILE_HEADER* pHeader )
    {
        memcpy( pSwappedHeader, pHeader, sizeof(TILEDFILE_HEADER) );
        ByteSwapHeader( pSwappedHeader );
    }

    //--------------------------------------------------------------------------------------
    
    inline VOID SwapSubresourceHeader( TILEDFILE_SUBRESOURCE* pSwappedHeader, const TILEDFILE_SUBRESOURCE* pHeader )
    {
        memcpy( pSwappedHeader, pHeader, sizeof(TILEDFILE_SUBRESOURCE) );
        ByteSwapSubresource( pSwappedHeader );
    }

    //--------------------------------------------------------------------------------------

    HRESULT TiledTextureFile::Serialize( HANDLE hFile, const CHAR* strRootFileName, BOOL EndianSwap )
    {
        HRESULT hr = PrepareForSerialization();
        if( FAILED(hr) )
        {
            return hr;
        }

        DWORD TotalBytesWritten = 0;
        DWORD BytesWritten;

        // write header
        if( EndianSwap )
        {
            TILEDFILE_HEADER SwappedHeader;
            SwapHeader( &SwappedHeader, &m_Header );
            WriteFile( hFile, &SwappedHeader, sizeof(SwappedHeader), &BytesWritten, NULL );
        }
        else
        {
            WriteFile( hFile, &m_Header, sizeof(m_Header), &BytesWritten, NULL );
        }
        TotalBytesWritten += BytesWritten;

        // write subresource headers
        BytesWritten = 0;
        UINT SubresourceCount = m_Header.ArraySliceCount * m_Header.MipLevelCount;
        if( EndianSwap )
        {
            for( UINT i = 0; i < SubresourceCount; ++i )
            {
                TILEDFILE_SUBRESOURCE SwappedHeader;
                SwapSubresourceHeader( &SwappedHeader, &m_pSubresources[i].Header );
                DWORD HeaderBytesWritten = 0;
                WriteFile( hFile, &SwappedHeader, sizeof(TILEDFILE_SUBRESOURCE), &HeaderBytesWritten, NULL );
                BytesWritten += HeaderBytesWritten;
            }
        }
        else
        {
            for( UINT i = 0; i < SubresourceCount; ++i )
            {
                DWORD HeaderBytesWritten = 0;
                WriteFile( hFile, &m_pSubresources[i].Header, sizeof(TILEDFILE_SUBRESOURCE), &HeaderBytesWritten, NULL );
                BytesWritten += HeaderBytesWritten;
            }
        }
        TotalBytesWritten += BytesWritten;

        // write block indexes
        for( UINT i = 0; i < SubresourceCount; ++i )
        {
            TiledTextureSubresource& SubR = m_pSubresources[i];

            if( SubR.Header.BlockIndexLocation != (UINT)-1 )
            {
                UINT CurrentIndexCount = SubR.Header.WidthBlocks * SubR.Header.HeightBlocks;
                BytesWritten = WritePageBlockIndex( hFile, SubR.pPageBlockIndex, CurrentIndexCount, m_Header.DefaultPage, EndianSwap );
                TotalBytesWritten += BytesWritten;
            }
        }

        // pad file to next 64KB boundary
        UINT FirstPageOffset = NextMultiple( TotalBytesWritten, PAGE_SIZE_BYTES );
        UINT Padding = FirstPageOffset - TotalBytesWritten;
        if( Padding > 0 )
        {
            BYTE* pPadding = new BYTE[Padding];
            ZeroMemory( pPadding, Padding );
            WriteFile( hFile, pPadding, Padding, &BytesWritten, NULL );
            delete[] pPadding;
        }

        // write out all of the page data
        m_RootDataFile.Serialize( hFile, EndianSwap ? m_Header.Format : TILED_FORMAT_8BPP );

        return S_OK;
    }

    //--------------------------------------------------------------------------------------

    HRESULT TiledTextureFile::PrepareForSerialization()
    {
        UINT SubresourceCount = m_Header.ArraySliceCount * m_Header.MipLevelCount;

        UINT IndexOffsetBytes = sizeof(TILEDFILE_HEADER) + sizeof(TILEDFILE_SUBRESOURCE) * SubresourceCount;

        UINT IndexEntryCount = 0;

        for( UINT i = 0; i < SubresourceCount; ++i )
        {
            TiledTextureSubresource& SubR = m_pSubresources[i];

            if( SubR.Header.BlockWidthPages > 0 )
            {
                SubR.Header.BlockIndexLocation = IndexEntryCount;
                SubR.Header.PageArrayLocation = TILEDFILE_INVALID_LOCATOR;

                UINT CurrentIndexCount = SubR.Header.WidthBlocks * SubR.Header.HeightBlocks;
                IndexEntryCount += CurrentIndexCount;
                IndexOffsetBytes += CurrentIndexCount * sizeof(TILEDFILE_PAGEDATA_LOCATOR);
            }
            else
            {
                SubR.Header.BlockIndexLocation = (UINT)-1;
            }
        }

        m_Header.BlockIndexEntries = IndexEntryCount;

        return S_OK;
    }

    //--------------------------------------------------------------------------------------
    
    HRESULT TiledTextureFile::Deserialize( HANDLE hFile, const CHAR* strRootFileName )
    {
        DWORD BytesRead = 0;

        // load the header
        TILEDFILE_HEADER ReadHeader;
        BOOL ReadSuccessful = ReadFile( hFile, &ReadHeader, sizeof(ReadHeader), &BytesRead, NULL );
        if( !ReadSuccessful )
        {
            return E_FAIL;
        }

        BOOL NeedsEndianSwap = FALSE;

        // endian swap the header based on the magic value
        if( ReadHeader.Magic == TILEDFILE_HEADER_MAGIC )
        {
            NeedsEndianSwap = FALSE;
            m_Header = ReadHeader;
        }
        else if( ReadHeader.Magic == TILEDFILE_HEADER_MAGIC_SWAPPED )
        {
            NeedsEndianSwap = TRUE;
            SwapHeader( &m_Header, &ReadHeader );
        }
        else
        {
            return E_FAIL;
        }

        // create and load subresources
        const UINT SubresourceCount = m_Header.ArraySliceCount * m_Header.MipLevelCount;
        m_pSubresources = new TiledTextureSubresource[SubresourceCount];
        for( UINT i = 0; i < SubresourceCount; ++i )
        {
            TILEDFILE_SUBRESOURCE ReadSubHeader;
            ReadSuccessful = ReadFile( hFile, &ReadSubHeader, sizeof(ReadSubHeader), &BytesRead, NULL );
            if( !ReadSuccessful )
            {
                return E_FAIL;
            }

            if( NeedsEndianSwap )
            {
                SwapSubresourceHeader( &m_pSubresources[i].Header, &ReadSubHeader );
            }
            else
            {
                m_pSubresources[i].Header = ReadSubHeader;
            }

            m_pSubresources[i].pFile = this;
            m_pSubresources[i].SubresourceIndex = i;
        }

        // create and load page block indexes
        for( UINT i = 0; i < SubresourceCount; ++i )
        {
            TiledTextureSubresource& SubR = m_pSubresources[i];
            UINT BlockCount = SubR.Header.WidthBlocks * SubR.Header.HeightBlocks;
            SubR.pPageBlockIndex = new TILEDFILE_PAGEDATA_LOCATOR[BlockCount];
            ZeroMemory( SubR.pPageBlockIndex, sizeof(TILEDFILE_PAGEDATA_LOCATOR) * BlockCount );

            ReadSuccessful = ReadFile( hFile, SubR.pPageBlockIndex, BlockCount * sizeof(TILEDFILE_PAGEDATA_LOCATOR), &BytesRead, FALSE );
            if( !ReadSuccessful )
            {
                return E_FAIL;
            }

            // endian swap locators
            if( NeedsEndianSwap )
            {
                for( UINT j = 0; j < BlockCount; ++j )
                {
                    SubR.pPageBlockIndex[j] = EndianSwap( SubR.pPageBlockIndex[j] );
                }
            }

            // if the locator is the default page, replace it with the null page so the sparse algorithm will work properly
            for( UINT j = 0; j < BlockCount; ++j )
            {
                if( SubR.pPageBlockIndex[j] == m_Header.DefaultPage )
                {
                    SubR.pPageBlockIndex[j] = TILEDFILE_INVALID_LOCATOR;
                }
            }
        }

        LARGE_INTEGER FileSize;
        GetFileSizeEx( hFile, &FileSize );

        LARGE_INTEGER CurrentLocation;
        LARGE_INTEGER MoveLocation;
        MoveLocation.QuadPart = 0;
        SetFilePointerEx( hFile, MoveLocation, &CurrentLocation, FILE_CURRENT );
        assert( CurrentLocation.HighPart == 0 );

        MoveLocation.LowPart = NextMultiple( CurrentLocation.LowPart, PAGE_SIZE_BYTES );
        SetFilePointerEx( hFile, MoveLocation, &CurrentLocation, FILE_BEGIN );

        m_RootDataFile.ApplyEmptyOffset( MoveLocation.LowPart / PAGE_SIZE_BYTES );

        UINT64 PageDataSizeBytes = FileSize.QuadPart - CurrentLocation.QuadPart;
        assert( PageDataSizeBytes % PAGE_SIZE_BYTES == 0 );
        UINT64 PageCount = PageDataSizeBytes / PAGE_SIZE_BYTES;

        TILEDFILE_FORMAT Format = m_Header.Format;
        if( !NeedsEndianSwap )
        {
            Format = TILED_FORMAT_8BPP;
        }

        ReadSuccessful = m_RootDataFile.Deserialize( hFile, PageCount, Format );
        if( !ReadSuccessful )
        {
            return E_FAIL;
        }

        return S_OK;
    }

    //--------------------------------------------------------------------------------------

    VOID* TiledTextureFile::GetLocatorData( TILEDFILE_PAGEDATA_LOCATOR Locator, UINT PageOffset ) const
    {
        if( Locator == TILEDFILE_INVALID_LOCATOR )
        {
            return NULL;
        }

        // TODO: support multiple data files
        assert( Locator.DataFileIndex == 0 );

        VOID* pData = m_RootDataFile.GetLocatorData( Locator, PageOffset );
        return pData;
    }

    //--------------------------------------------------------------------------------------

    TILEDFILE_PAGEDATA_LOCATOR TiledTextureFile::CreatePageBlock( UINT SubresourceIndex, UINT PageWidth, UINT PageHeight )
    {
        UINT PageCount = PageWidth * PageHeight;
        TILEDFILE_PAGEDATA_LOCATOR Locator = m_RootDataFile.AllocatePages( PageCount );
        Locator.UniquePages = ( PageCount > 1 );

        if( m_Header.DefaultPage != TILEDFILE_INVALID_LOCATOR )
        {
            const VOID* pDefaultData = GetLocatorData( m_Header.DefaultPage, 0 );
            for( UINT i = 0; i < PageCount; ++i )
            {
                VOID* pDestData = GetLocatorData( Locator, i );
                memcpy( pDestData, pDefaultData, PAGE_SIZE_BYTES );
            }
        }

        return Locator;
    }

    //--------------------------------------------------------------------------------------

    inline VOID FillPartialRect( VOID* pDestBuffer, const RECT PageRect, RECT DestRect, const VOID* pFillValue, const UINT BytesPerTexel )
    {
        // clip dest rect to page rect
        DestRect.left = max( DestRect.left, PageRect.left );
        DestRect.top = max( DestRect.top, PageRect.top );
        DestRect.right = min( DestRect.right, PageRect.right );
        DestRect.bottom = min( DestRect.bottom, PageRect.bottom );

        UINT StartRow = DestRect.top - PageRect.top;
        UINT EndRow = DestRect.bottom - PageRect.top;
        UINT RowCount = EndRow - StartRow;

        UINT StartColumn = DestRect.left - PageRect.left;
        UINT EndColumn = DestRect.right - PageRect.left;
        UINT ColumnCount = EndColumn - StartColumn;

        UINT RowStrideBytes = ( PageRect.right - PageRect.left ) * BytesPerTexel;

        BYTE* pFillDest = (BYTE*)pDestBuffer;

        // offset start position by the start row
        pFillDest += RowStrideBytes * StartRow;

        // offset start position by the start column
        pFillDest += StartColumn * BytesPerTexel;

        // perform the fill
        for( UINT Row = 0; Row < RowCount; ++Row )
        {
            BYTE* pCurrentElement = pFillDest;
            for( UINT Column = 0; Column < ColumnCount; ++Column )
            {
                memcpy( pCurrentElement, pFillValue, BytesPerTexel );
                pCurrentElement += BytesPerTexel;
            }

            pFillDest += RowStrideBytes;
        }
    }

    //--------------------------------------------------------------------------------------

    VOID* TiledTextureFile::GetPageData( UINT Subresource, UINT PageX, UINT PageY, BOOL CreatePage )
    {
        return m_pSubresources[Subresource].GetPageData( PageX, PageY, CreatePage );
    }

    //--------------------------------------------------------------------------------------

    VOID TiledTextureFile::FillRect( UINT SubresourceIndex, const RECT* pDestRect, const VOID* pFillValue )
    {
        UINT BytesPerTexel = GetBytesPerTexel( m_Header.Format );
        SIZE PageSizeTexels = GetPageSize( m_Header.Format );
        UINT TexelBlockSize = GetTexelBlockSize( m_Header.Format );

        UINT SubresourceWidth = m_pSubresources[SubresourceIndex].Header.WidthTexels;
        UINT SubresourceHeight = m_pSubresources[SubresourceIndex].Header.HeightTexels;
        RECT ClippedRect = *pDestRect;
        ClippedRect.left = max( 0, ClippedRect.left );
        ClippedRect.right = min( (INT)SubresourceWidth, ClippedRect.right );
        ClippedRect.top = max( 0, ClippedRect.top );
        ClippedRect.bottom = min( (INT)SubresourceHeight, ClippedRect.bottom );
        pDestRect = &ClippedRect;

        RECT DestPageRect;
        DestPageRect.left = pDestRect->left / PageSizeTexels.cx;
        DestPageRect.top = pDestRect->top / PageSizeTexels.cy;
        DestPageRect.right = NextWholeQuotient( pDestRect->right, PageSizeTexels.cx );
        DestPageRect.bottom = NextWholeQuotient( pDestRect->bottom, PageSizeTexels.cy );

        PageSizeTexels.cx /= TexelBlockSize;
        PageSizeTexels.cy /= TexelBlockSize;

        RECT DestRect = *pDestRect;
        DestRect.left /= TexelBlockSize;
        DestRect.top /= TexelBlockSize;
        DestRect.right /= TexelBlockSize;
        DestRect.bottom /= TexelBlockSize;

        TiledTextureSubresource& SubR = m_pSubresources[SubresourceIndex];

        for( INT PageY = DestPageRect.top; PageY <= DestPageRect.bottom; ++PageY )
        {
            for( INT PageX = DestPageRect.left; PageX <= DestPageRect.right; ++PageX )
            {
                RECT PageRect = { PageX * PageSizeTexels.cx, PageY * PageSizeTexels.cy, ( PageX + 1 ) * PageSizeTexels.cx, ( PageY + 1 ) * PageSizeTexels.cy };

                if( DestRect.right <= PageRect.left || DestRect.bottom <= PageRect.top )
                {
                    continue;
                }

                VOID* pData = SubR.GetPageData( PageX, PageY, TRUE );

                FillPartialRect( pData, PageRect, DestRect, pFillValue, BytesPerTexel );
            }
        }
    }

    //--------------------------------------------------------------------------------------
    
    inline VOID BlitPartialRect( VOID* pDestBuffer, const RECT PageRect, RECT DestRect, const VOID* pSourceBuffer, const UINT SourceRowPitchBytes, const UINT BytesPerTexel )
    {
        UINT SourceOriginY = DestRect.top;
        UINT SourceOriginX = DestRect.left;

        // clip dest rect to page rect
        DestRect.left = max( DestRect.left, PageRect.left );
        DestRect.top = max( DestRect.top, PageRect.top );
        DestRect.right = min( DestRect.right, PageRect.right );
        DestRect.bottom = min( DestRect.bottom, PageRect.bottom );

        UINT StartRow = DestRect.top - PageRect.top;
        UINT EndRow = DestRect.bottom - PageRect.top;
        UINT RowCount = EndRow - StartRow;

        UINT SourceX = DestRect.left - SourceOriginX;
        UINT SourceY = DestRect.top - SourceOriginY;

        UINT StartColumn = DestRect.left - PageRect.left;
        UINT EndColumn = DestRect.right - PageRect.left;
        UINT ColumnCount = EndColumn - StartColumn;

        UINT RowStrideBytes = ( PageRect.right - PageRect.left ) * BytesPerTexel;

        BYTE* pFillDest = (BYTE*)pDestBuffer;

        // offset start position by the start row
        pFillDest += RowStrideBytes * StartRow;

        // offset start position by the start column
        pFillDest += StartColumn * BytesPerTexel;

        BYTE* pSourceBits = (BYTE*)pSourceBuffer;
        pSourceBits += SourceY * SourceRowPitchBytes;
        pSourceBits += SourceX * BytesPerTexel;

        UINT CopyRowBytes = ColumnCount * BytesPerTexel;

        // perform the fill
        for( UINT Row = 0; Row < RowCount; ++Row )
        {
            BYTE* pCurrentElement = pFillDest;

            memcpy( pCurrentElement, pSourceBits, CopyRowBytes );

            pFillDest += RowStrideBytes;
            pSourceBits += SourceRowPitchBytes;
        }
    }

    //--------------------------------------------------------------------------------------
    
    VOID TiledTextureFile::BlitRect( UINT SubresourceIndex, const RECT* pDestRect, const VOID* pSourceBuffer, const UINT SourceRowPitchBytes )
    {
        UINT BytesPerTexel = GetBytesPerTexel( m_Header.Format );
        SIZE PageSizeTexels = GetPageSize( m_Header.Format );
        UINT TexelBlockSize = GetTexelBlockSize( m_Header.Format );

        UINT SubresourceWidth = m_pSubresources[SubresourceIndex].Header.WidthTexels;
        UINT SubresourceHeight = m_pSubresources[SubresourceIndex].Header.HeightTexels;
        RECT ClippedRect = *pDestRect;
        ClippedRect.left = max( 0, ClippedRect.left );
        ClippedRect.right = min( (INT)SubresourceWidth, ClippedRect.right );
        ClippedRect.top = max( 0, ClippedRect.top );
        ClippedRect.bottom = min( (INT)SubresourceHeight, ClippedRect.bottom );
        pDestRect = &ClippedRect;

        RECT DestPageRect;
        DestPageRect.left = pDestRect->left / PageSizeTexels.cx;
        DestPageRect.top = pDestRect->top / PageSizeTexels.cy;
        DestPageRect.right = NextWholeQuotient( pDestRect->right, PageSizeTexels.cx );
        DestPageRect.bottom = NextWholeQuotient( pDestRect->bottom, PageSizeTexels.cy );

        PageSizeTexels.cx /= TexelBlockSize;
        PageSizeTexels.cy /= TexelBlockSize;

        RECT DestRect = *pDestRect;
        DestRect.left /= TexelBlockSize;
        DestRect.top /= TexelBlockSize;
        DestRect.right /= TexelBlockSize;
        DestRect.bottom /= TexelBlockSize;

        UINT DestRowPitchBytes = ( DestRect.right - DestRect.left ) * BytesPerTexel;
        if( DestRowPitchBytes > SourceRowPitchBytes )
        {
            DisplayUserError( "The destination rectangle must be the same width as or smaller than the source image." );
            return;
        }

        TiledTextureSubresource& SubR = m_pSubresources[SubresourceIndex];

        for( INT PageY = DestPageRect.top; PageY <= DestPageRect.bottom; ++PageY )
        {
            for( INT PageX = DestPageRect.left; PageX <= DestPageRect.right; ++PageX )
            {
                RECT PageRect = { PageX * PageSizeTexels.cx, PageY * PageSizeTexels.cy, ( PageX + 1 ) * PageSizeTexels.cx, ( PageY + 1 ) * PageSizeTexels.cy };

                if( DestRect.right <= PageRect.left || DestRect.bottom <= PageRect.top )
                {
                    continue;
                }

                VOID* pData = SubR.GetPageData( PageX, PageY, TRUE );

                BlitPartialRect( pData, PageRect, DestRect, pSourceBuffer, SourceRowPitchBytes, BytesPerTexel );
            }
        }
    }

    //--------------------------------------------------------------------------------------
    
    inline VOID BlendFourTexels( const TILEDFILE_FORMAT Format, BYTE* pDestTexel, const BYTE* pSrcA, const BYTE* pSrcB, const BYTE* pSrcC, const BYTE* pSrcD )
    {
        switch( Format )
        {
        case TILED_FORMAT_32BPP_R8G8B8A8:
            for( UINT i = 0; i < 4; ++i )
            {
                UINT Sum = *pSrcA++ + *pSrcB++ + *pSrcC++ + *pSrcD++;
                *pDestTexel++ = (BYTE)( Sum >> 2 );
            }
            break;
        case TILED_FORMAT_16BPP_R8G8:
            for( UINT i = 0; i < 2; ++i )
            {
                UINT Sum = *pSrcA++ + *pSrcB++ + *pSrcC++ + *pSrcD++;
                *pDestTexel++ = (BYTE)( Sum >> 2 );
            }
            break;
        case TILED_FORMAT_16BPP_R16:
            UINT Sum = *(USHORT*)pSrcA + *(USHORT*)pSrcB + *(USHORT*)pSrcC + *(USHORT*)pSrcD;
            Sum >>= 2;
            *(USHORT*)pDestTexel = (USHORT)Sum;
            break;
        }
    }

    //--------------------------------------------------------------------------------------
    
    VOID TiledTextureFile::GenerateMipData( UINT ArraySlice, UINT SourceLevel, UINT DestLevel )
    {
        assert( ArraySlice < m_Header.ArraySliceCount );
        assert( SourceLevel < m_Header.MipLevelCount );
        assert( DestLevel < m_Header.MipLevelCount );
        assert( SourceLevel == ( DestLevel - 1 ) );

        UINT DestSubresourceIndex = ArraySlice * m_Header.MipLevelCount + DestLevel;
        TiledTextureSubresource& DestSubR = m_pSubresources[DestSubresourceIndex];

        UINT SrcSubresourceIndex = ArraySlice * m_Header.MipLevelCount + SourceLevel;
        TiledTextureSubresource& SrcSubR = m_pSubresources[SrcSubresourceIndex];

        const UINT TexelSizeBytes = GetBytesPerTexel( m_Header.Format );
        const UINT RowPitchBytes = TexelSizeBytes * m_Header.PageWidthTexels;

        for( UINT PageY = 0; PageY < SrcSubR.Header.HeightPages; ++PageY )
        {
            for( UINT PageX = 0; PageX < SrcSubR.Header.WidthPages; ++PageX )
            {
                const BYTE* pSrcPageData = (const BYTE*)GetPageData( SrcSubresourceIndex, PageX, PageY, FALSE );
                if( pSrcPageData == NULL )
                {
                    continue;
                }

                UINT DestQuadrantX = PageX % 2;
                UINT DestQuadrantY = PageY % 2;

                UINT DestPageX = PageX >> 1;
                UINT DestPageY = PageY >> 1;

                BYTE* pDestPageData = (BYTE*)GetPageData( DestSubresourceIndex, DestPageX, DestPageY, TRUE );
                assert( pDestPageData != NULL );

                for( UINT TexelY = 0; TexelY < m_Header.PageHeightTexels; TexelY += 2 )
                {
                    const BYTE* pRowAB = pSrcPageData + TexelY * RowPitchBytes;
                    const BYTE* pRowCD = pRowAB + RowPitchBytes;

                    UINT DestTexelY = ( TexelY + DestQuadrantY * m_Header.PageHeightTexels ) >> 1;

                    for( UINT TexelX = 0; TexelX < m_Header.PageWidthTexels; TexelX += 2 )
                    {
                        const BYTE* pSrcTexelA = pRowAB + TexelSizeBytes * TexelX;
                        const BYTE* pSrcTexelB = pSrcTexelA + TexelSizeBytes;
                        const BYTE* pSrcTexelC = pRowCD + TexelSizeBytes * TexelX;
                        const BYTE* pSrcTexelD = pSrcTexelC + TexelSizeBytes;

                        UINT DestTexelX = ( TexelX + DestQuadrantX * m_Header.PageWidthTexels ) >> 1;

                        BYTE* pDestTexel = pDestPageData + ( DestTexelY * RowPitchBytes ) + DestTexelX * TexelSizeBytes;

                        BlendFourTexels( m_Header.Format, pDestTexel, pSrcTexelA, pSrcTexelB, pSrcTexelC, pSrcTexelD );
                    }
                }
            }
        }
    }
}
