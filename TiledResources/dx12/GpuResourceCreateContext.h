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

        inline SIZE_T MB(SIZE_T size)
        {
            return size * 1024 * 1024;
        }

        inline Microsoft::WRL::ComPtr<ID3D12Heap> CreateUploadHeap( ID3D12Device* device, SIZE_T size )
        {
            D3D12_HEAP_DESC d                 = {};
            d.Properties.Type                 = D3D12_HEAP_TYPE_UPLOAD;
            d.Properties.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE;
            d.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
            d.Properties.CreationNodeMask     = 1;
            d.Properties.VisibleNodeMask      = 1;

            d.Alignment                       = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
            d.SizeInBytes                     = Align(size, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);

            Microsoft::WRL::ComPtr<ID3D12Heap> result;

            DX::ThrowIfFailed(device->CreateHeap(&d, IID_PPV_ARGS(&result)));
            return result;
        }

        inline Microsoft::WRL::ComPtr<ID3D12Heap> CreateReadBackHeap(ID3D12Device* device, SIZE_T size)
        {
            D3D12_HEAP_DESC d = {};
            d.Properties.Type = D3D12_HEAP_TYPE_READBACK;
            d.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
            d.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
            d.Properties.CreationNodeMask = 1;
            d.Properties.VisibleNodeMask = 1;

            d.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
            d.SizeInBytes = Align(size, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);

            Microsoft::WRL::ComPtr<ID3D12Heap> result;

            DX::ThrowIfFailed(device->CreateHeap(&d, IID_PPV_ARGS(&result)));
            return result;
        }
    }

    class GpuResourceCreateContext
    {
        public:

        GpuResourceCreateContext( ID3D12Device* device) :
        m_device(device)
        , m_texturesDescriptorHeap( device, 256 )
        , m_frameIndex(0)
        {
            m_uploadHeap[0] = details::CreateUploadHeap( device, details::MB(24) );
            m_uploadHeap[1] = details::CreateUploadHeap(device, details::MB(24));
            m_uploadHeap[2] = details::CreateUploadHeap(device, details::MB(24));
        }


        GpuTexture2D CreateTexture2D()
        {
            return GpuTexture2D(nullptr, m_texturesDescriptorHeap.Allocate(), m_texturesDescriptorHeap.Allocate() ) ;
        }

        GpuUploadBuffer CreateUploadBuffer(SIZE_T size)
        {
            return GpuUploadBuffer(nullptr);
        }

        GpuReadBackBuffer CreateReadBackBuffer(SIZE_T size)
        {
            return GpuReadBackBuffer(nullptr);
        }

        GpuTiledCubeTexture CreateTiledCubeTexture()
        {
            return GpuTiledCubeTexture(nullptr, m_texturesDescriptorHeap.Allocate(), m_texturesDescriptorHeap.Allocate());
        }

        void Sync()
        {
            m_frameIndex++;
            m_frameIndex %= 3;
        }

        private:

        Microsoft::WRL::ComPtr<ID3D12Device>    m_device;
        GpuResourceDescriptorHeap               m_texturesDescriptorHeap;

        Microsoft::WRL::ComPtr<ID3D12Heap>      m_renderTargets;
        Microsoft::WRL::ComPtr<ID3D12Heap>      m_tiledResources;
        
        Microsoft::WRL::ComPtr<ID3D12Heap>      m_uploadHeap[3];
        Microsoft::WRL::ComPtr<ID3D12Heap>      m_readBackHeap[3];
        UINT                                    m_frameIndex;
    };
}
