//--------------------------------------------------------------------------------------
// PhysicalPageManager.cpp
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "PhysicalPageManager.h"
#include "TypedPagePool.h"
#include "TiledResourceBase.h"
#include "PageRenderer.h"

#include "TiledResourceRuntimeTest.h"
using namespace TiledRuntimeTest;

namespace TiledRuntime
{
    const UINT STARTING_PAGE_ID = 100;
    const UINT STARTING_RESOURCE_ID = 1;

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager constructor
    //--------------------------------------------------------------------------------------
    PhysicalPageManager::PhysicalPageManager( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dDeviceContext, const PhysicalPageManagerDesc* pDesc )
    {
        InitializeCriticalSection( &m_CriticalSection );

        m_pd3dDevice = pd3dDevice;
        m_pd3dDevice->AddRef();
        m_pd3dDeviceContext = pd3dDeviceContext;
        m_pd3dDeviceContext->AddRef();

        // Copy initialization params to a local copy:
        ZeroMemory( &m_Desc, sizeof(m_Desc) );
        if( pDesc != NULL )
        {
            m_Desc = *pDesc;
        }

        // Fill in any parameters that are zero:
        SetDefaultParameters();

        m_NextPageID = STARTING_PAGE_ID;
        m_NextResourceID = STARTING_RESOURCE_ID;
        for( UINT i = 0; i < ARRAYSIZE(m_pTypedPagePools); ++i )
        {
            m_pTypedPagePools[i] = NULL;
        }

        // Create the page renderer:
        m_pPageRenderer = new PageRenderer( pd3dDevice, pd3dDeviceContext );
    }

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager destructor
    // Desc: Clean up D3D11 objects, delete the page renderer, and delete all of the typed
    //       page pools.
    //--------------------------------------------------------------------------------------
    PhysicalPageManager::~PhysicalPageManager()
    {
        SAFE_RELEASE( m_pd3dDevice );
        SAFE_RELEASE( m_pd3dDeviceContext );

        delete m_pPageRenderer;
        m_pPageRenderer = NULL;

        for( UINT i = 0; i < ARRAYSIZE(m_pTypedPagePools); ++i )
        {
            if( m_pTypedPagePools[i] != NULL )
            {
                delete m_pTypedPagePools[i];
                m_pTypedPagePools[i] = NULL;
            }
        }
        DeleteCriticalSection( &m_CriticalSection );
    }

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager::SetDefaultParameters
    // Desc: Fills in a default value for any zero-value initialization parameters.
    //--------------------------------------------------------------------------------------
    VOID PhysicalPageManager::SetDefaultParameters()
    {
        if( m_Desc.MaxPhysicalPages == 0 )
        {
            m_Desc.MaxPhysicalPages = 1024;
        }

        if( m_Desc.EmulationParams.DefaultResourceFormat == 0 )
        {
            m_Desc.EmulationParams.DefaultResourceFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager::GetPhysicalPageCount
    // Desc: Returns the current number of allocated physical pages.
    //--------------------------------------------------------------------------------------
    UINT PhysicalPageManager::GetPhysicalPageCount() const
    {
        return (UINT)m_MasterIndex.size();
    }

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager::AllocatePage
    // Desc: Creates a new physical page.
    //--------------------------------------------------------------------------------------
    HRESULT PhysicalPageManager::AllocatePage( PhysicalPageID* pNewPageID, DXGI_FORMAT DefaultFormat )
    {
        if( PARAMETER_CHECK )
        {
            if( pNewPageID == NULL )
            {
                return E_INVALIDARG;
            }
        }

        // Lock the data structures, since we will be modifying them:
        EnterLock();

        // Check if we have hit the page count limit:
        const UINT PageCount = GetPhysicalPageCount();
        if( PageCount >= m_Desc.MaxPhysicalPages )
        {
            LeaveLock();
            return E_OUTOFMEMORY;
        }

        // Create a new physical page ID:
        const PhysicalPageID NewPageID = m_NextPageID++;

        // The physical page must be hinted to a specific format in the software emulation:
        if( DefaultFormat == DXGI_FORMAT_UNKNOWN )
        {
            DefaultFormat = m_Desc.EmulationParams.DefaultResourceFormat;
        }

        if( DefaultFormat == DXGI_FORMAT_UNKNOWN )
        {
            LeaveLock();
            return E_INVALIDARG;
        }

        // Get the typed page pool for this format:
        TypedPagePool* pTPP = CreateTypedPagePool( DefaultFormat );

        // Add a new page to the typed page pool:
        INT PoolIndex = pTPP->AddPage( NewPageID );

        // Make sure the typed page pool could create the page:
        if( PoolIndex < 0 )
        {
            LeaveLock();
            Trace::PageCreateFailure( NewPageID, pTPP->GetFormat() );
            return E_OUTOFMEMORY;
        }

        // Create a new physical page entry to add to the master index:
        PhysicalPageEntry NewPageEntry;
        ZeroMemory( &NewPageEntry, sizeof(NewPageEntry) );

        // Record the physical page ID (the physical address):
        NewPageEntry.m_PageID = NewPageID;

        // Record the format (which typed page pool contains the page):
        NewPageEntry.m_Location.m_Format = DefaultFormat;

        // Record the location within the typed page pool:
        NewPageEntry.m_Location.m_PoolIndex = PoolIndex;

        // Add the physical page entry to the index:
        m_MasterIndex[NewPageID] = NewPageEntry;

        // Unlock the data structures:
        LeaveLock();

        // Return the new physical page ID:
        *pNewPageID = NewPageID;

        Trace::CreatePage( NewPageID, DefaultFormat );

        return S_OK;
    }

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager::FreePage
    // Desc: Frees a physical page that has been allocated.
    //       This method has not been implemented yet.
    //--------------------------------------------------------------------------------------
    HRESULT PhysicalPageManager::FreePage( PhysicalPageID PageID )
    {
        NOTIMPL;
        return E_NOTIMPL;
    }

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager::RegisterResource
    // Desc: Assigns a new resource ID to the given resource.  The resource ID is used to
    //       generate virtual addresses that correspond to that resource.
    //--------------------------------------------------------------------------------------
    HRESULT PhysicalPageManager::RegisterResource( TiledResourceBase* pResource )
    {
        // Ensure that the resource does not already have a resource ID:
        if( pResource->m_ResourceID != 0 || pResource->m_pPageManager != NULL )
        {
            return E_FAIL;
        }

        // Lock data structures for access:
        EnterLock();

        // Create a new resource ID:
        UINT NewResourceID = m_NextResourceID++;

        // Fill in pointers and the resource ID on the resource:
        pResource->m_pPageManager = this;
        pResource->m_pTypedPagePool = CreateTypedPagePool( pResource->GetResourceFormat() );
        pResource->m_ResourceID = NewResourceID;

        // Add the resource to our mapping:
        m_Resources[NewResourceID] = pResource;

        // Unlock data structures:
        LeaveLock();

        return S_OK;
    }

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager::UnregisterResource
    // Desc: Unregisters a resource with the physical page manager.  This method will unmap
    //       all virtual pages associated with the resource, and then remove the resource
    //       from tracking.
    //       Not implemented yet.
    //--------------------------------------------------------------------------------------
    HRESULT PhysicalPageManager::UnregisterResource( TiledResourceBase* pResource )
    {
        NOTIMPL;
        return E_NOTIMPL;
    }

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager::CreateTypedPagePool
    // Desc: Finds or creates a typed page pool for the given format.
    //--------------------------------------------------------------------------------------
    TypedPagePool* PhysicalPageManager::CreateTypedPagePool( DXGI_FORMAT TextureFormat )
    {
        ASSERT( TextureFormat != DXGI_FORMAT_UNKNOWN );

        // Check if the typed page pool already exists:
        TypedPagePool* pTPP = m_pTypedPagePools[TextureFormat];

        if( pTPP != NULL )
        {
            return pTPP;
        }

        EnterLock();

        // Create a new typed page pool:
        pTPP = new TypedPagePool( m_pd3dDevice, this, TextureFormat, m_Desc.MaxPhysicalPages );

        // Add the typed page pool to the index:
        m_pTypedPagePools[TextureFormat] = pTPP;

        LeaveLock();

        return pTPP;
    }

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager::UpdateSinglePageContents
    // Desc: Copies the given 64KB buffer into the given physical page.  The data format of
    //       the buffer is given, since the software emulation needs to know the type of the
    //       data in order to copy the data to the proper region within the proper typed 
    //       page pool.
    //--------------------------------------------------------------------------------------
    HRESULT PhysicalPageManager::UpdateSinglePageContents( PhysicalPageID PageID, const VOID* pPageBuffer, DXGI_FORMAT BufferDataFormat )
    {
        if( PARAMETER_CHECK )
        {
            if( PageID == INVALID_PHYSICAL_PAGE_ID )
            {
                RIP;
                return E_INVALIDARG;
            }
        }

        // Ensure that the physical page is valid and get a pointer to the entry:
        PhysicalPageEntry* pEntry = FindPhysicalPage( PageID );
        if( pEntry == NULL )
        {
            return E_INVALIDARG;
        }

        // If the buffer format is unknown, assume that it is the same as the page's format:
        DXGI_FORMAT DataFormat = BufferDataFormat;
        if( DataFormat == DXGI_FORMAT_UNKNOWN )
        {
            DataFormat = pEntry->m_Location.m_Format;
        }

        // Get the typed page pool for the page's format:
        TypedPagePool* pCurrentPagePool = CreateTypedPagePool( pEntry->m_Location.m_Format );
        ASSERT( pCurrentPagePool != NULL );

        // Determine if the page is already in the format of the buffer.
        // If it is not, we have to move the page from the current page pool to the desired page pool:
        if( pCurrentPagePool->GetFormat() != DataFormat )
        {
            // The destination page pool is the typed page pool of the buffer format:
            TypedPagePool* pDestPagePool = CreateTypedPagePool( BufferDataFormat );

            Trace::MovePage( PageID, pCurrentPagePool->GetFormat(), pDestPagePool->GetFormat() );

            // Protect the data structures, since we are about to do an atomic operation:
            EnterLock();

            // Add page to the destination typed page pool:
            INT DestIndex = pDestPagePool->AddPage( PageID );
            if( DestIndex == -1 )
            {
                Trace::PageCreateFailure( PageID, pDestPagePool->GetFormat() );
                
                LeaveLock();

                // Couldn't add page to the new location!
                return E_OUTOFMEMORY;
            }

            // Remove the page from its current location:
            BOOL RemoveResult = pCurrentPagePool->RemovePage( PageID, pEntry->m_Location.m_PoolIndex );

            // Removing a page should always succeed, since we validated that the page exists earlier:
            ASSERT( RemoveResult == TRUE );

            LeaveLock();

            // Change location in physical page entry:
            pEntry->m_Location.m_Format = pDestPagePool->GetFormat();
            pEntry->m_Location.m_PoolIndex = DestIndex;

            // Reassign the page pool pointer to the destination:
            pCurrentPagePool = pDestPagePool;
        }

        ASSERT( pCurrentPagePool != NULL );
        ASSERT( pCurrentPagePool->GetFormat() == DataFormat );

        // Queue a page update operation on the page renderer, which will perform the copy from the buffer into
        // the typed page pool:
        m_pPageRenderer->QueuePageUpdate( pCurrentPagePool, pEntry->m_Location.m_PoolIndex, pPageBuffer );

        return S_OK;
    }

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager::MapVirtualPageToPhysicalPage
    // Desc: Maps the given virtual address to the given physical address.  If the physical
    //       address is invalid, then this method removes any existing virtual to physical
    //       mapping for the given virtual page.
    //--------------------------------------------------------------------------------------
    HRESULT PhysicalPageManager::MapVirtualPageToPhysicalPage( VirtualPageID VPageID, PhysicalPageID PageID )
    {
        if( VPageID == INVALID_VIRTUAL_PAGE_ID || !VPageID.Valid )
        {
            return E_INVALIDARG;
        }

        // Find the resource corresponding to the virtual page ID:
        TiledResourceBase* pResource = GetResource( VPageID );
        if( pResource == NULL )
        {
            // Invalid resource ID in the virtual address:
            return E_INVALIDARG;
        }

        // Determine format of the resource:
        const DXGI_FORMAT DataFormat = pResource->GetResourceFormat();

        EnterLock();

        // Find the typed page pool corresponding to the resource:
        TypedPagePool* pResourcePool = CreateTypedPagePool( DataFormat );

        INT PagePoolIndex = -1;

        if( PageID != INVALID_PHYSICAL_PAGE_ID )
        {
            // Ensure that the physical page exists:
            PhysicalPageEntry* pPhysicalEntry = FindPhysicalPage( PageID );
            if( pPhysicalEntry == NULL )
            {
                // Physical page not found:
                LeaveLock();
                return E_INVALIDARG;
            }

            // Ensure that the physical page exists within the desired typed page pool.
            // If the physical page isn't already there, we need to remove it from its current location
            // and move it to a new location.
            // Note that in the current implementation, we do not attempt to copy the physical page
            // contents from one format to another.
            if( pPhysicalEntry->m_Location.m_Format != DataFormat )
            {
                Trace::MovePage( PageID, pPhysicalEntry->m_Location.m_Format, DataFormat );

                // Add physical page to the new location:
                INT DestIndex = pResourcePool->AddPage( PageID );
                if( DestIndex == -1 )
                {
                    // Couldn't add page to the new location:
                    LeaveLock();

                    Trace::PageCreateFailure( PageID, pResourcePool->GetFormat() );

                    return E_OUTOFMEMORY;
                }

                // Remove the page from its current location:
                DXGI_FORMAT SrcFormat = pPhysicalEntry->m_Location.m_Format;
                TypedPagePool* pSrcPool = CreateTypedPagePool( SrcFormat );
                BOOL RemoveResult = pSrcPool->RemovePage( PageID, pPhysicalEntry->m_Location.m_PoolIndex );
                ASSERT( RemoveResult == TRUE );

                // Change location in the physical page entry:
                pPhysicalEntry->m_Location.m_Format = DataFormat;
                pPhysicalEntry->m_Location.m_PoolIndex = DestIndex;

                // Record the location of the physical page in the destination page pool:
                PagePoolIndex = DestIndex;
            }
            else
            {
                // Record the location of the physical page in the page pool:
                PagePoolIndex = pPhysicalEntry->m_Location.m_PoolIndex;
            }

            ASSERT( PagePoolIndex != -1 );
        }

        // Find an existing virtual-physical mapping for the virtual address, if it exists:
        VirtualToPhysicalIndex::iterator iter = m_VirtualToPhysicalIndex.find( VPageID );
        if( iter != m_VirtualToPhysicalIndex.end() )
        {
            // The virtual page is already mapped to a different physical page:
            PhysicalPageID ExistingPhysicalPageID = iter->second;
            PhysicalPageEntry* pPhysicalEntry = FindPhysicalPage( ExistingPhysicalPageID );
            ASSERT( pPhysicalEntry != NULL );
            if( pPhysicalEntry != NULL )
            {
                // Decrement refcount of existing physical page:
                pPhysicalEntry->m_RefCount--;
                ASSERT( pPhysicalEntry->m_RefCount >= 0 );
            }
        }

        BOOL MultipleMappingsToPhysicalPage = FALSE;

        // Update virtual-physical mapping:
        if( PageID == INVALID_PHYSICAL_PAGE_ID )
        {
            // New mapping is to a NULL physical address; remove entry from index
            if( iter != m_VirtualToPhysicalIndex.end() )
            {
                m_VirtualToPhysicalIndex.erase( iter );
            }
        } 
        else
        {
            // Increase refcount on new physical page:
            PhysicalPageEntry* pCurrentEntry = FindPhysicalPage( PageID );
            ASSERT( pCurrentEntry != NULL );
            pCurrentEntry->m_RefCount++;

            if( pCurrentEntry->m_RefCount > 1 )
            {
                MultipleMappingsToPhysicalPage = TRUE;
            }

            // Map virtual page to physical page in the master copy of the virtual to physical mappings:
            m_VirtualToPhysicalIndex[VPageID] = PageID;
        }

        // Extract the virtual page's neighborhood from the resource:
        PageNeighborhood Neighborhood;
        HRESULT hr = pResource->GetNeighborhood( m_pd3dDeviceContext, VPageID, &Neighborhood );
        if( FAILED(hr) )
        {
            // This should never fail:
            RIP;
            LeaveLock();
            return E_FAIL;
        }
        Neighborhood.m_CenterPage = PageID;

        // Queue border texel updates for the physical pages affected by this new virtual to physical mapping:

        // Update borders of center page:
        if( Neighborhood.m_CenterPage != INVALID_PHYSICAL_PAGE_ID )
        {
            for( INT i = 0; i < PN_COUNT; ++i )
            {
                PhysicalPageID BorderPageID = Neighborhood.m_Neighbors[i];

                // If the center page is a multiple mapped physical page, make sure its borders are itself:
                if( MultipleMappingsToPhysicalPage )
                {
                    BorderPageID = Neighborhood.m_CenterPage;
                }

                // Determine if we need to make the center page border itself, to reduce visual corruption:
                PageNeighbors BorderRelationship = (PageNeighbors)i;
                BOOL InvertSourceRelationship = FALSE;
                if( FETCH_ONLY_VALID_TEXELS_OPTION && BorderPageID == INVALID_PHYSICAL_PAGE_ID )
                {
                    BorderPageID = Neighborhood.m_CenterPage;
                    InvertSourceRelationship = TRUE;
                }

                // Queue the border update on the center page, copying edge pixels from the neighbor page to the border of the center page:
                m_pPageRenderer->QueueBorderUpdate( pResourcePool, Neighborhood.m_CenterPage, BorderPageID, BorderRelationship, InvertSourceRelationship );
            }
        }

        // Update borders of neighbor pages:
        for( INT i = 0; i < PN_COUNT; ++i )
        {
            PhysicalPageID BorderPageID = Neighborhood.m_Neighbors[i];

            if( BorderPageID != INVALID_PHYSICAL_PAGE_ID && BorderPageID != Neighborhood.m_CenterPage )
            {
                PhysicalPageEntry* pBorderEntry = FindPhysicalPage( BorderPageID );

                // If the border page is a multiple mapped physical page, make sure we don't update its borders
                if( pBorderEntry->m_RefCount < 2 )
                {
                    // Queue the border update on the neighbor page, copying edge pixels from the center page to the border of the neighbor page:
                    m_pPageRenderer->QueueBorderUpdate( pResourcePool, BorderPageID, Neighborhood.m_CenterPage, GetOppositeNeighbor( (PageNeighbors)i ) );
                }
            }
        }

        // Send the virtual to physical mapping to the resource, which will update its index map texture:
        pResource->SetIndexMapEntry( VPageID, PageID, PagePoolIndex );

        LeaveLock();

        Trace::MapPage( VPageID, PageID );

        return S_OK;
    }

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager::UnmapPhysicalPage
    // Desc: Given a physical page address, this method unmaps all virtual pages that are
    //       mapped to that physical page.
    //--------------------------------------------------------------------------------------
    HRESULT PhysicalPageManager::UnmapPhysicalPage( PhysicalPageID PageID )
    {
        if( PageID == INVALID_PHYSICAL_PAGE_ID )
        {
            return E_INVALIDARG;
        }

        // A vector to store all of the virtual page addresses:
        std::vector<VirtualPageID> MappedVirtualPages;

        EnterLock();

        // Iterate over each virtual-to-physical mapping:
        VirtualToPhysicalIndex::iterator iter = m_VirtualToPhysicalIndex.begin();
        VirtualToPhysicalIndex::iterator end = m_VirtualToPhysicalIndex.end();
        while( iter != end )
        {
            // If the mapping is to the physical page, add the virtual page to the vector:
            if( iter->second == PageID )
            {
                MappedVirtualPages.push_back( iter->first );
            }
            ++iter;
        }

        // Unmap all of the virtual pages that we found:
        for( UINT i = 0; i < (UINT)MappedVirtualPages.size(); ++i )
        {
            UnmapVirtualPage( MappedVirtualPages[i] );
        }

        LeaveLock();

        return S_OK;
    }

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager::ExecutePageDataOperations
    // Desc: Executes per-frame rendering operations that reflect tiled resource manipulations.
    //       It flushes all pending operations in the page renderer, and updates the GPU
    //       index map for each resource (as necessary).
    //--------------------------------------------------------------------------------------
    VOID PhysicalPageManager::ExecutePageDataOperations()
    {
        // Flush pending updates in the page renderer:
        m_pPageRenderer->FlushPendingUpdates();

        // Update the GPU index map texture in each tiled resource:
        ResourceIndex::iterator iter = m_Resources.begin();
        ResourceIndex::iterator end = m_Resources.end();
        while( iter != end )
        {
            TiledResourceBase* pResource = iter->second;
            pResource->GPUTextureUpdate( m_pd3dDeviceContext );
            ++iter;
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager::FindPhysicalPage
    // Desc: For a given physical address, find the physical page entry in the physical page
    //       index.
    //--------------------------------------------------------------------------------------
    PhysicalPageEntry* PhysicalPageManager::FindPhysicalPage( PhysicalPageID PageID )
    {
        PhysicalPageEntry* pEntry = NULL;
        
        // Even though this method is read only, we must make sure that we protect the
        // data structures so that the physical page index isn't modified while we are
        // reading from it:
        EnterLock();

        // Search for the physical page address in the hash map:
        PhysicalPageMasterIndex::iterator iter = m_MasterIndex.find( PageID );
        if( iter == m_MasterIndex.end() )
        {
            // Page not found:
            LeaveLock();
            return NULL;
        }

        // Return a pointer to the page entry:
        pEntry = &(iter->second);

        LeaveLock();

        return pEntry;
    }

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager::GetResource
    // Desc: For the given virtual address, decode the resource ID and then return the
    //       resource that corresponds with that resource ID.
    //--------------------------------------------------------------------------------------
    TiledResourceBase* PhysicalPageManager::GetResource( VirtualPageID VPageID ) const
    {
        ResourceIndex::const_iterator iter = m_Resources.find( VPageID.ResourceID );
        if( iter == m_Resources.end() )
        {
            return NULL;
        }
        return iter->second;
    }

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager::EnterLock
    // Desc: Locks the critical section that protects the physical page manager data structures.
    //--------------------------------------------------------------------------------------
    VOID PhysicalPageManager::EnterLock()
    {
        EnterCriticalSection( &m_CriticalSection );
    }

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager::LeaveLock
    // Desc: Unlocks the critical section that protects the physical page manager data structures.
    //--------------------------------------------------------------------------------------
    VOID PhysicalPageManager::LeaveLock()
    {
        LeaveCriticalSection( &m_CriticalSection );
    }

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager::GetMemoryUsage
    // Desc: Reports overhead memory used by the physical page manager, and accumulates the
    //       memory usage for typed page pools and resources.
    //--------------------------------------------------------------------------------------
    VOID PhysicalPageManager::GetMemoryUsage( D3D11_TILED_MEMORY_USAGE* pMemoryUsage ) const
    {
        if( pMemoryUsage == NULL )
        {
            return;
        }

        // Zero out the memory usage struct:
        ZeroMemory( pMemoryUsage, sizeof(D3D11_TILED_MEMORY_USAGE) );

        // Accumulate the memory usage of each typed page pool:
        for( UINT i = 0; i < ARRAYSIZE(m_pTypedPagePools); ++i )
        {
            if( m_pTypedPagePools[i] != NULL )
            {
                m_pTypedPagePools[i]->GetMemoryUsage( pMemoryUsage );
            }
        }

        // Accumulate the memory usage of each tiled resource:
        ResourceIndex::const_iterator iter = m_Resources.begin();
        ResourceIndex::const_iterator end = m_Resources.end();
        while( iter != end )
        {
            TiledResourceBase* pResource = iter->second;
            pResource->GetMemoryUsage( pMemoryUsage );
            ++iter;
        }

        // Add overhead bytes for the indices in the physical page manager (representative, not entirely accurate):
        pMemoryUsage->OverheadMemoryBytesAllocated += m_MasterIndex.size() * ( sizeof(PhysicalPageID) + sizeof(PhysicalPageEntry) );
        pMemoryUsage->OverheadMemoryBytesAllocated += m_VirtualToPhysicalIndex.size() * ( sizeof(VirtualPageID) + sizeof(PhysicalPageID) );
    }
}
