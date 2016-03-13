#pragma once

#include "GpuResource.h"
#include "GpuDescriptorHeap.h"

#include "GpuTexture2D.h"
#include "GpuTiledCubeTexture.h"

#include "GpuUploadBuffer.h"
#include "GpuReadbackBuffer.h"

#include "GpuColorBuffer.h"
#include "GpuDepthBuffer.h"
#include "GpuBackBuffer.h"

namespace TiledResources
{
    namespace details
    {
        inline SIZE_T Align(SIZE_T size, SIZE_T alignment)
        {
            return ( size + alignment - 1 ) & ( alignment - 1 );
        }

        /*
        inline UINT64 Align(UINT64 size, UINT64 alignment)
        {
            return (size + alignment - 1) & (alignment - 1);
        }
        */

        inline SIZE_T MB(SIZE_T size)
        {
            return size * 1024 * 1024;
        }

        class PlacementHeapAllocator
        {
            public:

            PlacementHeapAllocator();
            PlacementHeapAllocator(ID3D12Device* device, Microsoft::WRL::ComPtr<ID3D12Heap> heap, SIZE_T size);

            void CreatePlacedResource(const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES initialState, const D3D12_CLEAR_VALUE *optimizedClearValue, REFIID riid, void **resource);
            void CreatePlacedResource(const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES initialState, REFIID riid, void **resource);

            private:
            ID3D12Device*					   m_device;
            Microsoft::WRL::ComPtr<ID3D12Heap> m_heap;
            UINT64							   m_heapOffset;
            SIZE_T							   m_size;
        };
    }

    class GpuResourceCreateContext
    {

        public:

        GpuResourceCreateContext(ID3D12Device* device);
    
        //Read only assets
        GpuTexture2D            CreateTexture2D();

        //Transfer accross the pci bus
        GpuUploadBuffer         CreateUploadBuffer(SIZE_T size);
        GpuReadBackBuffer       CreateReadBackBuffer(SIZE_T size);

        //Render Targets
        GpuColorBuffer          CreateColorBuffer(UINT width, UINT height, DXGI_FORMAT format);

        //Depth Buffer
        GpuDepthBuffer          CreateDepthBuffer(UINT width, UINT height, DXGI_FORMAT format);

        //Tiled Resources
        GpuTiledCubeTexture     CreateTiledCubeTexture();

        GpuBackBuffer           CreateBackBuffer(ID3D12Resource* r);

        void Sync();

        private:

        Microsoft::WRL::ComPtr<ID3D12Device>    m_device;
        GpuResourceDescriptorHeap               m_texturesDescriptorHeap;

        details::PlacementHeapAllocator         m_uploadAllocator[3];
        details::PlacementHeapAllocator         m_readBackAllocator[3];

        details::PlacementHeapAllocator         m_renderTargetAllocator;
        details::PlacementHeapAllocator         m_tiledResourcesAllocator;

        UINT                                    m_frameIndex;
        details::PlacementHeapAllocator*        GetUploadAllocator()
        {
            return &m_uploadAllocator[m_frameIndex];
        }

        details::PlacementHeapAllocator*   GetReadBackAllocator()
        {
            return &m_readBackAllocator[m_frameIndex];
        }
    };
}
