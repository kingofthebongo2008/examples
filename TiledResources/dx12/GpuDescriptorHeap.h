#pragma once

#include <d3d12.h>

#include "DirectXHelper.h"

namespace TiledResources
{
    class DescriptorHandle
    {
        public: 
        DescriptorHandle(D3D12_CPU_DESCRIPTOR_HANDLE h0, D3D12_GPU_DESCRIPTOR_HANDLE h1) :
            m_h0(h0)
            , m_h1(h1)
        {

        }

        operator D3D12_GPU_DESCRIPTOR_HANDLE() const
        {
            return m_h1;
        }

        operator D3D12_CPU_DESCRIPTOR_HANDLE() const
        {
            m_h0;
        }

    private:
        D3D12_CPU_DESCRIPTOR_HANDLE m_h0;
        D3D12_GPU_DESCRIPTOR_HANDLE m_h1;
    };


    template < D3D12_DESCRIPTOR_HEAP_TYPE heap, D3D12_DESCRIPTOR_HEAP_FLAGS flags >
    class GpuDescriptorHeap
    {
        public: 

        GpuDescriptorHeap( ID3D12Device* device, UINT descriptorCount, UINT NodeMask = 0 )
        {
            D3D12_DESCRIPTOR_HEAP_DESC desc = {};

            desc.Type           = static_cast<D3D12_DESCRIPTOR_HEAP_TYPE> (heap);
            desc.Flags          = static_cast<D3D12_DESCRIPTOR_HEAP_FLAGS> (flags);
            desc.NumDescriptors = descriptorCount;
            desc.NodeMask       = NodeMask;

            DX::ThrowIfFailed(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_resource)));

            m_cpuBegin        = m_resource->GetCPUDescriptorHandleForHeapStart();
            m_gpuBegin        = m_resource->GetGPUDescriptorHandleForHeapStart();
            m_offset          = 0;
            m_incrementSize   = device->GetDescriptorHandleIncrementSize(desc.Type);
        }

        DescriptorHandle Allocate(uint32_t count)
        {
            assert(!Full(count));

            D3D12_CPU_DESCRIPTOR_HANDLE cpu; 
            D3D12_GPU_DESCRIPTOR_HANDLE gpu;

            cpu.ptr = static_cast<SIZE_T>(m_cpuBegin.ptr + m_offset);
            gpu.ptr = static_cast<SIZE_T>(m_gpuBegin.ptr + m_offset);

            m_offset += count * m_incrementSize;
            return DescriptorHandle(cpu, gpu);
        }

        DescriptorHandle Allocate()
        {
            return Allocate(1);
        }

        bool Full(uint32_t count) const
        {
            auto desc = m_resource->GetDesc();
            return (m_offset + count >= desc.NumDescriptors);
        }


        private:

        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_resource;
        D3D12_CPU_DESCRIPTOR_HANDLE                  m_cpuBegin;
        D3D12_GPU_DESCRIPTOR_HANDLE                  m_gpuBegin;
        UINT64                                       m_offset;
        UINT64                                       m_incrementSize;
    };

    using GpuResourceDescriptorHeap = GpuDescriptorHeap< D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE >;
}
