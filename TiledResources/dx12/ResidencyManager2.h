//// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
//// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
//// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
//// PARTICULAR PURPOSE.
////
//// Copyright (c) Microsoft Corporation. All rights reserved

#pragma once

#include "DeviceResources2.h"
#include "TileLoader.h"

#include "GpuTexture2D.h"

namespace TiledResources
{
    // Data and metadata for a tiled resource layer.
    struct ManagedTiledResource
    {
        GpuTexture2D*                           texture;
        D3D12_RESOURCE_DESC                     textureDesc;
        UINT                                    totalTiles;
        D3D12_PACKED_MIP_INFO                   packedMipDesc;
        D3D12_TILE_SHAPE                        tileShape;
        std::vector<D3D12_SUBRESOURCE_TILING>   subresourceTilings;
        std::unique_ptr<TileLoader>             loader;
        std::vector<byte>                       residency[6];

        GpuTexture2D                            residencyTexture;
    };

    // Unique identifier for a tile.
    struct TileKey
    {
        D3D12_TILED_RESOURCE_COORDINATE coordinate;
        GpuTexture2D*                   resource;
    };

    // Define the < relational operator for use as a key in std::map.
    static bool operator<(const TiledResources::TileKey & a, const TiledResources::TileKey & b)
    {
        if (a.resource < b.resource) return true;
        if (a.resource > b.resource) return false;
        if (a.coordinate.Subresource < b.coordinate.Subresource) return true;
        if (a.coordinate.Subresource > b.coordinate.Subresource) return false;
        if (a.coordinate.Z < b.coordinate.Z) return true;
        if (a.coordinate.Z > b.coordinate.Z) return false;
        if (a.coordinate.Y < b.coordinate.Y) return true;
        if (a.coordinate.Y > b.coordinate.Y) return false;
        return a.coordinate.X < b.coordinate.X;
    }

    enum class TileState
    {
        Seen,
        Loading,
        Loaded,
        Mapped
    };

    struct TrackedTile
    {
        ManagedTiledResource* managedResource;
        D3D12_TILED_RESOURCE_COORDINATE coordinate;
        short mipLevel;
        short face;
        UINT physicalTileOffset;
        unsigned int lastSeen;
        std::vector<byte> tileData;
        TileState state;
    };

    class ResidencyManager
    {
    public:
        ResidencyManager(const std::shared_ptr<DeviceResources>& deviceResources);
        void CreateDeviceDependentResources();
        concurrency::task<void> CreateDeviceDependentResourcesAsync();
        void ReleaseDeviceDependentResources();

        //ID3D11ShaderResourceView* ManageTexture(ID3D11Texture2D* texture, const std::wstring& filename);
        concurrency::task<void> InitializeManagedResourcesAsync();
        //void EnqueueSamples(const std::vector<DecodedSample>& samples);
        void ProcessQueues();

        void RenderVisualization();

        void SetDebugMode(bool value);

        void Reset();

        void SetBorderMode(bool value);

    private:
        // Cached pointer to device resources.
        std::shared_ptr<DeviceResources> m_deviceResources;
        GpuTexture2D                     m_tempTexture; //null mapped tile;

    };

    static bool LoadPredicate(const std::shared_ptr<TiledResources::TrackedTile>& a, const std::shared_ptr<TiledResources::TrackedTile>& b)
    {
        // Prefer more recently seen tiles.
        if (a->lastSeen > b->lastSeen) return true;
        if (a->lastSeen < b->lastSeen) return false;

        // Break ties by loading less detailed tiles first.
        return a->mipLevel > b->mipLevel;
    }
    
    static bool MapPredicate(const std::shared_ptr<TiledResources::TrackedTile>& a, const std::shared_ptr<TiledResources::TrackedTile>& b)
    {
        // Only loaded tiles can be mapped, so put those first.
        if (a->state == TileState::Loaded && b->state == TileState::Loading) return true;
        if (a->state == TileState::Loading && b->state == TileState::Loaded) return false;

        // Then prefer more recently seen tiles.
        if (a->lastSeen > b->lastSeen) return true;
        if (a->lastSeen < b->lastSeen) return false;

        // Break ties by mapping less detailed tiles first.
        return a->mipLevel > b->mipLevel;
    }

    static bool EvictPredicate(const std::shared_ptr<TiledResources::TrackedTile>& a, const std::shared_ptr<TiledResources::TrackedTile>& b)
    {
        // Evict older tiles first.
        if (a->lastSeen < b->lastSeen) return true;
        if (a->lastSeen > b->lastSeen) return false;

        // To break ties, evict more detailed tiles first.
        return a->mipLevel < b->mipLevel;
    }
}
