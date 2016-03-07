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
        inline Microsoft::WRL::ComPtr<ID3D12Heap> CreateUploadHeap( ID3D12Device* device, SIZE_T size )
        {

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
