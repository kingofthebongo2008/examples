//--------------------------------------------------------------------------------------
// TypedPagePool.h
//
// Part of the tiled resources software emulation, the typed page pool represents a pool 
// of physical pages in a particular format.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#include "TiledResourceCommon.h"
#include "d3d11tiled.h"

namespace TiledRuntime
{
    //--------------------------------------------------------------------------------------
    // Name: AtlasEntry
    // Desc: Represents a physical page and its location within the page pool array texture.
    //--------------------------------------------------------------------------------------
    struct AtlasEntry
    {
        struct
        {
            DWORD X: 8;
            DWORD Y: 8;
            DWORD Slice: 16;
        };
        PhysicalPageID PageID;
        AtlasEntry* pNextFree;
    };

    typedef stdext::hash_map<PhysicalPageID, INT> PhysicalPageLocationMap;

    //--------------------------------------------------------------------------------------
    // Name: TypedPagePool
    // Desc: Class that represents a collection of physical pages sharing a common texture
    //       format.
    //--------------------------------------------------------------------------------------
    class TypedPagePool
    {
    public:
        struct CB_PagePool
        {
            XMFLOAT4 PageBorderUVTransform;
            XMFLOAT4 ArrayTexConstant;
        };

    protected:
        PhysicalPageManager* m_pPageManager;

        // The format of the typed page pool and its contents:
        DXGI_FORMAT m_TextureFormat;

        // The number of array slices in the page pool array texture:
        UINT m_ArraySliceCount;

        // The total number of physical pages that this pool can contain:
        UINT m_PageCapacity;

        // The directory of physical pages, indexable by location within
        // the array texture:
        AtlasEntry* m_pAtlasDirectory;

        // The free entry list, so that we can locate an empty slot in the
        // page pool with O(1) efficiency:
        AtlasEntry* m_pFreeEntryList;

        // The directory of physical pages, indexable by physical address:
        PhysicalPageLocationMap m_PageLocationMap;

        // The array texture that stores physical pages as a stack of atlased pages:
        ID3D11Texture2D* m_pPagePoolArrayTexture;

        // The atlas texture dimensions in texels
        SIZE m_AtlasPageSizeTexels;

        // The constant buffer for shader access to the page pool array texture:
        CB_PagePool m_CBData;

        // Sampler states for shader access to the page pool array texture:
        static ID3D11SamplerState* s_pPagePoolSamplerStatePoint;
        static ID3D11SamplerState* s_pPagePoolSamplerStateBilinear;

    public:
        TypedPagePool( ID3D11Device* pd3dDevice, PhysicalPageManager* pPageManager, DXGI_FORMAT TextureFormat, UINT MaxPageCount );
        ~TypedPagePool();

        static UINT GetVSBaseSlotIndex() { return 11; }
        VOID VSSetSRVSamplerStateAndConstants( ID3D11DeviceContext* pd3dDeviceContext, ID3D11ShaderResourceView* pPoolSRV, UINT SlotIndex );

        static UINT GetPSBaseSlotIndex() { return 11; }
        VOID PSSetSRVSamplerStateAndConstants( ID3D11DeviceContext* pd3dDeviceContext, ID3D11ShaderResourceView* pPoolSRV, UINT SlotIndex );

        const CB_PagePool& GetShaderConstants() const { return m_CBData; }

        DXGI_FORMAT GetFormat() const { return m_TextureFormat; }

        INT FindPage( PhysicalPageID PageID ) const;
        INT AddPage( PhysicalPageID PageID );
        BOOL RemovePage( PhysicalPageID PageID, INT PageIndex = -1 );
        UINT GetPageCount() const;
        BOOL IsPagePresent( PhysicalPageID PageID ) const;
        PhysicalPageID GetPageByAtlasLocation( UINT Slice, UINT X, UINT Y ) const;

        AtlasEntry* GetAtlasEntry( INT Index );
        const AtlasEntry* GetAtlasEntry( INT Index ) const;

        RECT GetPageRect( const AtlasEntry* pEntry ) const;

        ID3D11Texture2D* GetArrayTexture() const { return m_pPagePoolArrayTexture; }
        ID3D11ShaderResourceView* CreateArrayTextureView( ID3D11Device* pd3dDevice, DXGI_FORMAT ResourceFormat );

        SIZE GetAtlasPageSizeTexels() const { return m_AtlasPageSizeTexels; }

        VOID GetMemoryUsage( D3D11_TILED_MEMORY_USAGE* pMemoryUsage ) const;

    protected:
        VOID CreateArrayTexture( ID3D11Device* pd3dDevice, UINT MaxPageCount );
        VOID CreateAtlasDirectory();
        VOID CreateShaderConstants();

        INT GetPageIndex( AtlasEntry* pEntry ) const;
    };
}

