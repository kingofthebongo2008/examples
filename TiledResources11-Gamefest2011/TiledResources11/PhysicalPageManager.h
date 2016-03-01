//--------------------------------------------------------------------------------------
// PhysicalPageManager.h
//
// The physical page manager is the central module in the tiled resources software
// emulation.  It provides a layer of abstraction between operations on physical
// pages, virtual pages, the page pool, and tiled resources; and the underlying real
// concepts, such as typed page pools and index map textures.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#include "TiledResourceCommon.h"
#include "d3d11tiled.h"

namespace TiledRuntime
{
    // In the current version, you cannot map a physical page to more than one virtual location.
    static const UINT MAX_PAGE_LOCATIONS = 1;

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageLocation
    // Desc: A location that specifies an index within a typed page pool.
    //--------------------------------------------------------------------------------------
    struct PhysicalPageLocation
    {
        DXGI_FORMAT m_Format;
        UINT m_PoolIndex;
    };

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageEntry
    // Desc: A struct that represents a single physical page within the physical page manager.
    //--------------------------------------------------------------------------------------
    struct PhysicalPageEntry
    {
        PhysicalPageID m_PageID;
        PhysicalPageLocation m_Location;
        INT m_RefCount;
    };

    typedef stdext::hash_map<PhysicalPageID, PhysicalPageEntry> PhysicalPageMasterIndex;
    typedef stdext::hash_map<VirtualPageID, PhysicalPageID> VirtualToPhysicalIndex;

    typedef stdext::hash_map<UINT, TiledResourceBase*> ResourceIndex;

    //--------------------------------------------------------------------------------------
    // Name: EmulationParameters
    // Desc: A struct that defines various attributes that are used to initialize the 
    //       physical page manager's software emulation of tiled resources.
    //--------------------------------------------------------------------------------------
    struct EmulationParameters
    {
        DXGI_FORMAT DefaultResourceFormat;
    };

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManagerDesc
    // Desc: Initialization parameters for the physical page manager.
    //--------------------------------------------------------------------------------------
    struct PhysicalPageManagerDesc
    {
        // The maximum number of physical pages that can be tracked by the physical page manager:
        UINT MaxPhysicalPages;

        // Software emulation parameters:
        EmulationParameters EmulationParams;
    };

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageManager
    // Desc: The physical page manager represents the interface to a physical page pool as 
    //       well as the virtual-to-physical page mapping tables.  Through the physical page
    //       manager, you can allocate and free physical pages, fill physical pages, and map
    //       or unmap virtual pages to physical pages.
    //--------------------------------------------------------------------------------------
    class PhysicalPageManager
    {
    protected:
        // The real D3D device and immediate context:
        ID3D11Device* m_pd3dDevice;
        ID3D11DeviceContext* m_pd3dDeviceContext;

        // Initialization parameters for the physical page manager:
        PhysicalPageManagerDesc m_Desc;

        // The map of each physical page and its attributes:
        PhysicalPageMasterIndex m_MasterIndex;

        // The ID of the next physical page that will be allocated:
        PhysicalPageID m_NextPageID;

        // The authoritative mapping of virtual pages to physical pages.
        // This data is also replicated in a different form in the tiled resources' index maps:
        VirtualToPhysicalIndex m_VirtualToPhysicalIndex;

        // One typed page pool for each DXGI format:
        TypedPagePool* m_pTypedPagePools[DXGI_FORMAT_MAX];

        // A map of tiled resources, each with their own resource ID:
        ResourceIndex m_Resources;

        // The resource ID that will be assigned to the next created resource:
        UINT m_NextResourceID;

        // The page renderer, which implements most of the actual memory mapping operations 
        // to operations on traditional textures:
        PageRenderer* m_pPageRenderer;

        // This critical section protects access to all of the data structures in this class:
        CRITICAL_SECTION m_CriticalSection;

    public:
		PhysicalPageManager( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dDeviceContext, const PhysicalPageManagerDesc* pDesc );
        ~PhysicalPageManager();

        const PhysicalPageManagerDesc& GetDesc() const { return m_Desc; }

        HRESULT AllocatePage( PhysicalPageID* pNewPageID, DXGI_FORMAT DefaultFormat = DXGI_FORMAT_UNKNOWN );
        HRESULT FreePage( PhysicalPageID PageID );

        HRESULT UpdateSinglePageContents( PhysicalPageID PageID, const VOID* pPageBuffer, DXGI_FORMAT BufferDataFormat = DXGI_FORMAT_UNKNOWN );

        HRESULT RegisterResource( TiledResourceBase* pResource );
        HRESULT UnregisterResource( TiledResourceBase* pResource );

        HRESULT MapVirtualPageToPhysicalPage( VirtualPageID VPageID, PhysicalPageID PageID );
        HRESULT UnmapVirtualPage( VirtualPageID VPageID ) { return MapVirtualPageToPhysicalPage( VPageID, INVALID_PHYSICAL_PAGE_ID ); }
        HRESULT UnmapPhysicalPage( PhysicalPageID PageID );

        VOID ExecutePageDataOperations();

        UINT GetPhysicalPageCount() const;

        PageRenderer* GetPageRenderer() const { return m_pPageRenderer; }

        VOID GetMemoryUsage( D3D11_TILED_MEMORY_USAGE* pMemoryUsage ) const;

    protected:
        VOID SetDefaultParameters();

        TypedPagePool* CreateTypedPagePool( DXGI_FORMAT TextureFormat );

        PhysicalPageEntry* FindPhysicalPage( PhysicalPageID PageID );

        TiledResourceBase* GetResource( VirtualPageID VPageID ) const;

        VOID EnterLock();
        VOID LeaveLock();
    };
}

