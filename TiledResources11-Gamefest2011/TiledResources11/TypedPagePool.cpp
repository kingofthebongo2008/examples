//--------------------------------------------------------------------------------------
// TypedPagePool.cpp
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "TypedPagePool.h"
#include "PhysicalPageManager.h"

#include "TiledResourceRuntimeTest.h"
using namespace TiledRuntimeTest;

namespace TiledRuntime
{
    ID3D11SamplerState* TypedPagePool::s_pPagePoolSamplerStatePoint = NULL;
    ID3D11SamplerState* TypedPagePool::s_pPagePoolSamplerStateBilinear = NULL;

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool constructor
    //--------------------------------------------------------------------------------------
    TypedPagePool::TypedPagePool( ID3D11Device* pd3dDevice, PhysicalPageManager* pPageManager, DXGI_FORMAT TextureFormat, UINT MaxPageCount )
    {
        ASSERT( TextureFormat != DXGI_FORMAT_UNKNOWN );
        m_TextureFormat = TextureFormat;
        m_PageCapacity = 0;
        m_pPageManager = pPageManager;

        // Create the page pool array texture:
        CreateArrayTexture( pd3dDevice, MaxPageCount );

        // Create the physical page directories:
        CreateAtlasDirectory();

        // Create the shader constants for accessing the page pool array texture from shaders:
        CreateShaderConstants();

        // Create the sampler states:
        if( s_pPagePoolSamplerStatePoint == NULL )
        {
            D3D11_SAMPLER_DESC SamplerDesc;
            ZeroMemory( &SamplerDesc, sizeof(SamplerDesc) );
            SamplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
            SamplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
            SamplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
            SamplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
            SamplerDesc.MipLODBias = 0;
            SamplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
            SamplerDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;

            pd3dDevice->CreateSamplerState( &SamplerDesc, &s_pPagePoolSamplerStatePoint );

            SamplerDesc.Filter = D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
            pd3dDevice->CreateSamplerState( &SamplerDesc, &s_pPagePoolSamplerStateBilinear );
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool destructor
    // Desc: Releases D3D11 objects associated with the typed page pool, and deallocates the
    //       physical page directories:
    //--------------------------------------------------------------------------------------
    TypedPagePool::~TypedPagePool()
    {
        SAFE_RELEASE( s_pPagePoolSamplerStateBilinear );
        SAFE_RELEASE( s_pPagePoolSamplerStatePoint );

        SAFE_RELEASE( m_pPagePoolArrayTexture );

        m_PageLocationMap.clear();
        delete[] m_pAtlasDirectory;
        m_pAtlasDirectory = NULL;
        m_pFreeEntryList = NULL;
    }

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool::CreateArrayTexture
    // Desc: Creates a single array texture that will hold the physical pages, as well as
    //       room for border texels surrounding each physical page.
    //--------------------------------------------------------------------------------------
    VOID TypedPagePool::CreateArrayTexture( ID3D11Device* pd3dDevice, UINT MaxPageCount )
    {
        // Determine how many pages can be stored by the array texture:
        m_PageCapacity = XGNextMultiple( MaxPageCount, ATLAS_PAGES_PER_SLICE );
        ASSERT( m_PageCapacity > 0 );

        // Compute the array slice count:
        const DWORD SliceCount = m_PageCapacity / ATLAS_PAGES_PER_SLICE;
        ASSERT( SliceCount <= MAX_ARRAY_SLICES );
        ASSERT( SliceCount > 0 );
        m_ArraySliceCount = SliceCount;

        // Compute the page size and border sizes for the texture format:
        const SIZE PageSizeTexels = GetPageSizeTexels( m_TextureFormat );
        const UINT BorderTexelCount = GetPageBorderTexelCount( m_TextureFormat );

        // Compute the dimensions of the array texture:
        const UINT AtlasWidth = ATLAS_COLUMNS * ( PageSizeTexels.cx + BorderTexelCount * 2 );
        const UINT AtlasHeight = ATLAS_ROWS * ( PageSizeTexels.cy + BorderTexelCount * 2 );
        m_AtlasPageSizeTexels.cx = AtlasWidth;
        m_AtlasPageSizeTexels.cy = AtlasHeight;

        // Create the array texture:
        const DXGI_FORMAT ArrayTextureFormat = m_TextureFormat;
        CreateZeroedArrayTexture( pd3dDevice, AtlasWidth, AtlasHeight, SliceCount, 1, ArrayTextureFormat, FALSE, &m_pPagePoolArrayTexture );
        ASSERT( m_pPagePoolArrayTexture != NULL );
    }

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool::CreateAtlasDirectory
    // Desc: Creates a static directory of physical pages, one for each slot in the page
    //       pool.  The entries are in a flat array, but they are linked to each other in a
    //       linked list, so that they can be relinked in separate active/free lists.
    //--------------------------------------------------------------------------------------
    VOID TypedPagePool::CreateAtlasDirectory()
    {
        const UINT DirectorySize = m_PageCapacity;

        // Create the flat directory:
        ASSERT( DirectorySize > 0 );
        m_pAtlasDirectory = new AtlasEntry[DirectorySize];
        ZeroMemory( m_pAtlasDirectory, DirectorySize * sizeof(AtlasEntry) );

        for( DWORD i = 0; i < DirectorySize; ++i )
        {
            // Link each entry to the next entry:
            if( i < ( DirectorySize - 1 ) )
            {
                m_pAtlasDirectory[i].pNextFree = &m_pAtlasDirectory[i+1];
            }

            // Fill in the slot address of each entry:
            UINT SliceIndex = i / ATLAS_PAGES_PER_SLICE;
            UINT AtlasIndex = i % ATLAS_PAGES_PER_SLICE;
            UINT RowIndex = AtlasIndex / ATLAS_COLUMNS;
            UINT ColumnIndex = AtlasIndex % ATLAS_COLUMNS;

            m_pAtlasDirectory[i].Slice = SliceIndex;
            m_pAtlasDirectory[i].X = ColumnIndex;
            m_pAtlasDirectory[i].Y = RowIndex;
        }

        // The free list initially points to the entire directory:
        m_pFreeEntryList = &m_pAtlasDirectory[0];
    }

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool::CreateShaderConstants
    // Desc: Populates a constant buffer with information describing the size of the page
    //       pool array texture, the atlasing dimensions, and the border size relative to the
    //       page contents.
    //--------------------------------------------------------------------------------------
    VOID TypedPagePool::CreateShaderConstants()
    {
        const SIZE PageSizeTexels = GetPageSizeTexels( m_TextureFormat );
        const UINT BorderTexelCount = GetPageBorderTexelCount( m_TextureFormat );

        FLOAT TotalWidth = (FLOAT)( PageSizeTexels.cx + BorderTexelCount * 2 );
        FLOAT TotalHeight = (FLOAT)( PageSizeTexels.cy + BorderTexelCount * 2 );
        m_CBData.PageBorderUVTransform.x = (FLOAT)PageSizeTexels.cx / TotalWidth;
        m_CBData.PageBorderUVTransform.y = (FLOAT)PageSizeTexels.cy / TotalHeight;
        m_CBData.PageBorderUVTransform.z = (FLOAT)BorderTexelCount / TotalWidth;
        m_CBData.PageBorderUVTransform.w = (FLOAT)BorderTexelCount / TotalHeight;

        m_CBData.ArrayTexConstant.x = (FLOAT)( max( 1, m_ArraySliceCount ) );
        m_CBData.ArrayTexConstant.y = 0;
        m_CBData.ArrayTexConstant.z = 1.0f / (FLOAT)ATLAS_COLUMNS;
        m_CBData.ArrayTexConstant.w = 1.0f / (FLOAT)ATLAS_ROWS;
    }

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool::CreateArrayTextureView
    // Desc: Creates a shader resource view for the page pool array texture in the given
    //       format.
    //--------------------------------------------------------------------------------------
    ID3D11ShaderResourceView* TypedPagePool::CreateArrayTextureView( ID3D11Device* pd3dDevice, DXGI_FORMAT Format )
    {
        ASSERT( Format == m_TextureFormat );

        D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc;
        ZeroMemory( &SRVDesc, sizeof(SRVDesc) );
        SRVDesc.Format = Format;
        SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
        SRVDesc.Texture2DArray.MipLevels = (UINT)-1;
        SRVDesc.Texture2DArray.ArraySize = (UINT)-1;
        SRVDesc.Texture2DArray.MostDetailedMip = 0;
        SRVDesc.Texture2DArray.FirstArraySlice = 0;

        ID3D11ShaderResourceView* pSRV = NULL;
        pd3dDevice->CreateShaderResourceView( m_pPagePoolArrayTexture, &SRVDesc, &pSRV );
        return pSRV;
    }

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool::FindPage
    // Desc: Locates a page in the page pool by physical address.  This has O(log n) run time.
    //--------------------------------------------------------------------------------------
    INT TypedPagePool::FindPage( PhysicalPageID PageID ) const
    {
        PhysicalPageLocationMap::const_iterator iter = m_PageLocationMap.find( PageID );
        if( iter != m_PageLocationMap.end() )
        {
            return iter->second;
        }
        return -1;
    }

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool::AddPage
    // Desc: Finds or adds the given physical page to the typed page pool.
    //--------------------------------------------------------------------------------------
    INT TypedPagePool::AddPage( PhysicalPageID PageID )
    {
        // Check if the page already exists:
        INT Index = FindPage( PageID );
        if( Index != -1 )
        {
            return Index;
        }

        // Check if we have any free slots available:
        if( m_pFreeEntryList == NULL )
        {
            return INVALID_PAGE_POOL_INDEX;
        }

        // Grab the top entry from the free list:
        AtlasEntry* pFreeEntry = m_pFreeEntryList;
        m_pFreeEntryList = m_pFreeEntryList->pNextFree;

        // Fill in the entry:
        pFreeEntry->pNextFree = NULL;
        pFreeEntry->PageID = PageID;

        Index = GetPageIndex( pFreeEntry );

        Trace::AddPageToPool( PageID, Index, GetFormat() );

        // Add entry to the hash map using the physical address:
        m_PageLocationMap[PageID] = Index;

        return Index;
    }

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool::RemovePage
    // Desc: Removes a page from the page pool, either by index or physical page ID. By index,
    //       this method is O(1), by physical address it is O(log N).
    //--------------------------------------------------------------------------------------
    BOOL TypedPagePool::RemovePage( PhysicalPageID PageID, INT PageIndex )
    {
        // Find the page entry:
        AtlasEntry* pPageEntry = NULL;
        if( PageIndex == -1 )
        {
            PageIndex = FindPage( PageID );
            if( PageIndex != -1 )
            {
                pPageEntry = GetAtlasEntry( PageIndex );
            }
        }
        else
        {
            pPageEntry = GetAtlasEntry( PageIndex );
            ASSERT( PageID == pPageEntry->PageID );
        }

        if( pPageEntry == NULL )
        {
            return FALSE;
        }

        Trace::RemovePageFromPool( PageID, PageIndex, GetFormat() );

        // Clear the page entry:
        pPageEntry->PageID = INVALID_PHYSICAL_PAGE_ID;

        // Add the page entry to the free list:
        pPageEntry->pNextFree = m_pFreeEntryList;
        m_pFreeEntryList = pPageEntry;

        // Remove the page entry from the map:
        m_PageLocationMap.erase( PageID );

        return TRUE;
    }

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool::GetPageCount
    // Desc: Returns the number of occupied pages in this page pool.
    //--------------------------------------------------------------------------------------
    UINT TypedPagePool::GetPageCount() const
    {
        return (UINT)m_PageLocationMap.size();
    }

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool::IsPagePresent
    // Desc: Returns TRUE if the given physical page is a member of this page pool, otherwise
    //       returns FALSE.
    //--------------------------------------------------------------------------------------
    BOOL TypedPagePool::IsPagePresent( PhysicalPageID PageID ) const
    {
        PhysicalPageLocationMap::const_iterator iter = m_PageLocationMap.find( PageID );
        return ( iter != m_PageLocationMap.end() );
    }

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool::GetPageIndex
    // Desc: Returns the linear index of the given page entry relative to the beginning of
    //       the directory.  It can use simple pointer subtraction since all of the entries
    //       are in the same allocation.
    //--------------------------------------------------------------------------------------
    INT TypedPagePool::GetPageIndex( AtlasEntry* pEntry ) const
    {
        ASSERT( pEntry != NULL );
        INT Index = (INT)( pEntry - m_pAtlasDirectory );
        ASSERT( Index >= 0 && Index < (INT)m_PageCapacity );
        return Index;
    }

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool::GetAtlasEntry
    // Desc: Returns the atlas entry at the given index.
    //--------------------------------------------------------------------------------------
    AtlasEntry* TypedPagePool::GetAtlasEntry( INT Index )
    {
        ASSERT( Index >= 0 && Index < (INT)m_PageCapacity );
        return &m_pAtlasDirectory[Index];
    }

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool::GetAtlasEntry
    // Desc: Returns the atlas entry at the given index.
    //--------------------------------------------------------------------------------------
    const AtlasEntry* TypedPagePool::GetAtlasEntry( INT Index ) const
    {
        ASSERT( Index >= 0 && Index < (INT)m_PageCapacity );
        return &m_pAtlasDirectory[Index];
    }

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool::GetPageRect
    // Desc: For the given atlas entry, this method computes the atlas rectangle in texels 
    //       for that physical page, not including border texels.
    //--------------------------------------------------------------------------------------
    RECT TypedPagePool::GetPageRect( const AtlasEntry* pEntry ) const
    {
        ASSERT( pEntry != NULL );
        RECT PageRect = { 0 };

        const UINT BorderTexels = GetPageBorderTexelCount( m_TextureFormat );
        const SIZE PageSizeTexels = GetPageSizeTexels( m_TextureFormat );

        PageRect.left = ( BorderTexels * 2 + PageSizeTexels.cx ) * pEntry->X + BorderTexels;
        PageRect.top = ( BorderTexels * 2 + PageSizeTexels.cy ) * pEntry->Y + BorderTexels;
        PageRect.right = PageRect.left + PageSizeTexels.cx;
        PageRect.bottom = PageRect.top + PageSizeTexels.cy;

        return PageRect;
    }

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool::GetPageByAtlasLocation
    // Desc: Returns the atlas entry at a given atlas location (X, Y, and slice).  The atlas
    //       index is computed from the coordinates and then that index is used to look up
    //       the entry within the directory.
    //--------------------------------------------------------------------------------------
    PhysicalPageID TypedPagePool::GetPageByAtlasLocation( UINT Slice, UINT X, UINT Y ) const
    {
        UINT EncodedIndex = ( Slice * ATLAS_PAGES_PER_SLICE ) + ( Y * ATLAS_COLUMNS ) + X;
        ASSERT( EncodedIndex < m_PageCapacity );

        const AtlasEntry& Entry = m_pAtlasDirectory[EncodedIndex];
        ASSERT( Entry.X == X && Entry.Y == Y && Entry.Slice == Slice );

        return Entry.PageID;
    }

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool::VSSetSRVSamplerStateAndConstants
    // Desc: Sets the page pool array texture and the samplers into the vertex shader pipeline
    //       inputs, given the slot index.
    //--------------------------------------------------------------------------------------
    VOID TypedPagePool::VSSetSRVSamplerStateAndConstants( ID3D11DeviceContext* pd3dDeviceContext, ID3D11ShaderResourceView* pSRV, UINT SlotIndex )
    {
        ID3D11SamplerState* SamplerStates[] = { s_pPagePoolSamplerStatePoint, s_pPagePoolSamplerStateBilinear };
        pd3dDeviceContext->VSSetSamplers( 14, 2, SamplerStates );

        pd3dDeviceContext->VSSetShaderResources( GetVSBaseSlotIndex() + SlotIndex, 1, &pSRV );
    }

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool::PSSetSRVSamplerStateAndConstants
    // Desc: Sets the page pool array texture and the samplers into the pixel shader pipeline
    //       inputs, given the slot index.
    //--------------------------------------------------------------------------------------
    VOID TypedPagePool::PSSetSRVSamplerStateAndConstants( ID3D11DeviceContext* pd3dDeviceContext, ID3D11ShaderResourceView* pSRV, UINT SlotIndex )
    {
        ID3D11SamplerState* SamplerStates[] = { s_pPagePoolSamplerStatePoint, s_pPagePoolSamplerStateBilinear };
        pd3dDeviceContext->PSSetSamplers( 14, 2, SamplerStates );

        pd3dDeviceContext->PSSetShaderResources( GetPSBaseSlotIndex() + SlotIndex, 1, &pSRV );
    }

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool::GetMemoryUsage
    // Desc: Fills in a memory usage struct with details about the capacity of this page pool
    //       and the amount of contents within.
    //--------------------------------------------------------------------------------------
    VOID TypedPagePool::GetMemoryUsage( D3D11_TILED_MEMORY_USAGE* pMemoryUsage ) const
    {
        ASSERT( pMemoryUsage != NULL );

        pMemoryUsage->FormatPoolsActive++;

        pMemoryUsage->TileCapacity += m_PageCapacity;
        pMemoryUsage->TilesAllocated += m_PageLocationMap.size();

        pMemoryUsage->TileTextureMemoryBytesAllocated += GetTextureSizeBytes( m_pPagePoolArrayTexture );

        UINT AtlasDirectorySizeBytes = m_PageCapacity * sizeof(AtlasEntry);
        pMemoryUsage->OverheadMemoryBytesAllocated += AtlasDirectorySizeBytes;

        pMemoryUsage->OverheadMemoryBytesAllocated += m_PageLocationMap.size() * ( sizeof(INT) + sizeof(PhysicalPageID) );
    }
}
