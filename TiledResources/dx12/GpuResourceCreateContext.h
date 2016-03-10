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

        inline UINT64 Align(UINT64 size, UINT64 alignment)
        {
            return (size + alignment - 1) & (alignment - 1);
        }

        inline SIZE_T MB(SIZE_T size)
        {
            return size * 1024 * 1024;
        }

        class FramePlacementHeapAllocator
        {
            public:

			FramePlacementHeapAllocator();
			FramePlacementHeapAllocator(ID3D12Device* device, Microsoft::WRL::ComPtr<ID3D12Heap> heap, SIZE_T size); 

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

		GpuTexture2D CreateTexture2D();
		GpuUploadBuffer CreateUploadBuffer(SIZE_T size);
		GpuReadBackBuffer CreateReadBackBuffer(SIZE_T size);
		GpuTiledCubeTexture CreateTiledCubeTexture();

		void Sync();

        private:

        Microsoft::WRL::ComPtr<ID3D12Device>    m_device;
        GpuResourceDescriptorHeap               m_texturesDescriptorHeap;

        Microsoft::WRL::ComPtr<ID3D12Heap>      m_renderTargets;
        Microsoft::WRL::ComPtr<ID3D12Heap>      m_tiledResources;
        
        details::FramePlacementHeapAllocator    m_uploadAllocator[3];
        details::FramePlacementHeapAllocator    m_readbackAllocator[3];
        

        UINT                                    m_frameIndex;
    };
}
