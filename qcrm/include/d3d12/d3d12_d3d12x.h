#pragma once

#include <cstdint>
#include <d3d12.h>

#include <os/windows/com_error.h>
#include <os/windows/dxgi_pointers.h>

#include <d3d12/d3d12_exception.h>
#include <d3d12/d3d12_pointers.h>

namespace d3d12x
{
    class gpu_descriptor_heap
    {
    public:

        gpu_descriptor_heap
            (
                const D3D12_GPU_DESCRIPTOR_HANDLE begin,
                const uint32_t                    handle_increment_size
                ) :
            m_begin(begin)
            , m_handle_increment_size(handle_increment_size)
        {

        }

        D3D12_CPU_DESCRIPTOR_HANDLE operator() (uint32_t index) const
        {
            D3D12_CPU_DESCRIPTOR_HANDLE r = { m_begin.ptr + index *  m_handle_increment_size };
            return r;
        }

    private:

        const D3D12_GPU_DESCRIPTOR_HANDLE m_begin;
        const uint32_t                    m_handle_increment_size;
    };

    class cpu_descriptor_heap
    {
    public:

        cpu_descriptor_heap
            (
                const D3D12_CPU_DESCRIPTOR_HANDLE begin,
                const uint32_t                    handle_increment_size
                ) :
            m_begin(begin)
            , m_handle_increment_size(handle_increment_size)
        {

        }

        D3D12_CPU_DESCRIPTOR_HANDLE operator() (uint32_t index) const
        {
            D3D12_CPU_DESCRIPTOR_HANDLE r = { m_begin.ptr + index * m_handle_increment_size };
            return r;
        }

    private:

        const D3D12_CPU_DESCRIPTOR_HANDLE m_begin;
        const uint32_t                    m_handle_increment_size;
    };

    template < uint32_t descriptor_heap_type, uint32_t descriptor_heap_flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE >
    class descriptor_heap
    {
    public:
        descriptor_heap(ID3D12Device* device, const uint32_t descriptor_count, const uint32_t node_mask = 0)
        {
            using namespace d3d12;
            using namespace os::windows;

            D3D12_DESCRIPTOR_HEAP_DESC desc = {};

            desc.Type = static_cast<D3D12_DESCRIPTOR_HEAP_TYPE> (descriptor_heap_type);
            desc.Flags = static_cast<D3D12_DESCRIPTOR_HEAP_FLAGS> (descriptor_heap_flags);
            desc.NumDescriptors = descriptor_count;
            desc.NodeMask = node_mask;
            os::windows::throw_if_failed<create_descriptor_heap_exception >(device->CreateDescriptorHeap(&desc, __uuidof(ID3D12DescriptorHeap), reinterpret_cast<void**> (&m_heap)));

            m_begin_cpu = m_heap->GetCPUDescriptorHandleForHeapStart();
            m_begin_gpu = m_heap->GetGPUDescriptorHandleForHeapStart();
            m_handle_increment_size = device->GetDescriptorHandleIncrementSize(desc.Type);
        }

        cpu_descriptor_heap create_cpu_heap() const
        {
            return cpu_descriptor_heap(m_begin_cpu, m_handle_increment_size);
        }

        gpu_descriptor_heap create_gpu_heap() const
        {
            return gpu_descriptor_heap(m_begin_gpu, m_handle_increment_size);
        }

    private:

        D3D12_CPU_DESCRIPTOR_HANDLE m_begin_cpu;
        D3D12_GPU_DESCRIPTOR_HANDLE m_begin_gpu;
        uint32_t                    m_handle_increment_size;
        d3d12::descriptor_heap      m_heap;
    };

    inline d3d12::device create_device(_In_opt_ IUnknown* pAdapter, D3D_FEATURE_LEVEL MinimumFeatureLevel)
    {
        using namespace os::windows;
        using namespace d3d12;

        device  device;

        throw_if_failed<create_device_exception>(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void**)&device));
        return std::move(device);
    }

    inline dxgi::factory create_factory()
    {
        using namespace os::windows;
        using namespace d3d12;

        dxgi::factory factory;
        throw_if_failed<create_dxgi_factory_exception>(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory));

        return std::move(factory);
    }

    inline dxgi::factory1 create_factory1()
    {
        using namespace os::windows;
        using namespace d3d12;

        dxgi::factory1 factory;
        throw_if_failed<create_dxgi_factory_exception>(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory));

        return std::move(factory);
    }

    inline dxgi::factory1 create_factory3()
    {
        using namespace os::windows;
        using namespace d3d12;

        dxgi::factory3 factory;
        throw_if_failed<create_dxgi_factory_exception>(CreateDXGIFactory1(__uuidof(IDXGIFactory3), (void**)&factory));

        return std::move(factory);
    }

    inline dxgi::factory4 create_factory4()
    {
        using namespace os::windows;
        using namespace d3d12;

        dxgi::factory4 factory;
        throw_if_failed<create_dxgi_factory_exception>(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&factory));

        return std::move(factory);
    }

    inline d3d12::command_allocator create_command_allocator(ID3D12Device* d, D3D12_COMMAND_LIST_TYPE type)
    {
        using namespace os::windows;
        using namespace d3d12;

        d3d12::command_allocator r;

        throw_if_failed<create_command_allocator_exception>(d->CreateCommandAllocator(type, __uuidof(ID3D12CommandAllocator), (void**)&r));
        return std::move(r);
    }

    inline d3d12::command_queue create_command_queue(ID3D12Device* d, const D3D12_COMMAND_QUEUE_DESC*  desc)
    {
        using namespace os::windows;
        using namespace d3d12;

        d3d12::command_queue r;

        throw_if_failed<create_command_queue_exception>(d->CreateCommandQueue(desc, __uuidof(ID3D12CommandQueue), (void**)&r));
        return std::move(r);
    }

    inline d3d12::heap create_heap(ID3D12Device* d, const D3D12_HEAP_DESC*  desc)
    {
        using namespace os::windows;
        using namespace d3d12;

        d3d12::heap r;

        throw_if_failed<create_heap_exception>(d->CreateHeap(desc, __uuidof(ID3D12Heap), (void**)&r));
        return std::move(r);
    }

    inline d3d12::command_list create_command_list(ID3D12Device* d, _In_  UINT nodeMask, _In_  D3D12_COMMAND_LIST_TYPE type, _In_  ID3D12CommandAllocator *command_allocator, _In_opt_  ID3D12PipelineState *initial_state)
    {
        using namespace os::windows;
        using namespace d3d12;

        d3d12::command_list r;

        throw_if_failed<create_heap_exception>(d->CreateCommandList(nodeMask, type, command_allocator, initial_state, __uuidof(ID3D12CommandList), (void**)&r));
        return std::move(r);
    }

    inline d3d12::graphics_command_list create_graphics_command_list(ID3D12Device* d, _In_  UINT nodeMask, _In_  D3D12_COMMAND_LIST_TYPE type, _In_  ID3D12CommandAllocator *command_allocator, _In_opt_  ID3D12PipelineState *initial_state)
    {
        using namespace os::windows;
        using namespace d3d12;

        d3d12::graphics_command_list r;

        throw_if_failed<create_heap_exception>(d->CreateCommandList(nodeMask, type, command_allocator, initial_state, __uuidof(ID3D12GraphicsCommandList), (void**)&r));
        return std::move(r);
    }

    

    inline d3d12::command_signature create_command_signature(ID3D12Device* d, _In_  const D3D12_COMMAND_SIGNATURE_DESC* desc, _In_opt_  ID3D12RootSignature *root_signature)
    {
        using namespace os::windows;
        using namespace d3d12;

        d3d12::command_signature r;
        throw_if_failed<create_heap_exception>(d->CreateCommandSignature(desc, root_signature, __uuidof(ID3D12CommandSignature), (void**)&r));
        return std::move(r);
    }

    inline d3d12::resource create_committed_resource(ID3D12Device* d, _In_  const D3D12_HEAP_PROPERTIES *heap_properties, D3D12_HEAP_FLAGS heap_flags, _In_  const D3D12_RESOURCE_DESC *resource_desc, D3D12_RESOURCE_STATES initial_resource_state, _In_opt_  const D3D12_CLEAR_VALUE * optimized_clear_value)
    {
        using namespace os::windows;
        using namespace d3d12;

        d3d12::resource r;
        throw_if_failed<create_heap_exception>(d->CreateCommittedResource(heap_properties, heap_flags, resource_desc, initial_resource_state, optimized_clear_value, __uuidof(ID3D12Resource), (void**)&r));
        return std::move(r);
    }

    inline d3d12::pipeline_state create_compute_pipeline_state(ID3D12Device* d, _In_  const D3D12_COMPUTE_PIPELINE_STATE_DESC * desc)
    {
        using namespace os::windows;
        using namespace d3d12;

        d3d12::pipeline_state r;
        throw_if_failed<create_heap_exception>(d->CreateComputePipelineState(desc, __uuidof(ID3D12PipelineState), (void**)&r));
        return std::move(r);
    }

    inline d3d12::descriptor_heap create_descriptor_heap(ID3D12Device* d, _In_  const D3D12_DESCRIPTOR_HEAP_DESC * desc)
    {
        using namespace os::windows;
        using namespace d3d12;

        d3d12::descriptor_heap r;
        throw_if_failed<create_heap_exception>(d->CreateDescriptorHeap(desc, __uuidof(ID3D12DescriptorHeap), (void**)&r));
        return std::move(r);
    }

    inline d3d12::fence create_fence(ID3D12Device* d, _In_  uint64_t initial_value, D3D12_FENCE_FLAGS flags)
    {
        using namespace os::windows;
        using namespace d3d12;

        d3d12::fence r;
        throw_if_failed<create_heap_exception>(d->CreateFence(initial_value, flags, __uuidof(ID3D12Fence), (void**)&r));
        return std::move(r);
    }

    inline d3d12::fence create_fence(ID3D12Device* d, _In_  uint64_t initial_value)
    {
        return create_fence(d, initial_value, D3D12_FENCE_FLAG_NONE);
    }

    inline d3d12::fence create_fence(ID3D12Device* d)
    {
        return create_fence(d, 0, D3D12_FENCE_FLAG_NONE);
    }

    inline d3d12::pipeline_state create_graphics_pipeline_state(ID3D12Device* d, _In_  const D3D12_GRAPHICS_PIPELINE_STATE_DESC * desc)
    {
        using namespace os::windows;
        using namespace d3d12;

        d3d12::pipeline_state r;
        throw_if_failed<create_heap_exception>(d->CreateGraphicsPipelineState(desc, __uuidof(ID3D12PipelineState), (void**)&r));
        return std::move(r);
    }

    inline d3d12::resource create_placed_resource(ID3D12Device* d, _In_  ID3D12Heap * heap, UINT64 heap_offset, _In_  const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES initial_state, _In_opt_  const D3D12_CLEAR_VALUE * optimized_clear_value)
    {
        using namespace os::windows;
        using namespace d3d12;

        d3d12::resource r;
        throw_if_failed<create_heap_exception>(d->CreatePlacedResource(heap, heap_offset, desc, initial_state, optimized_clear_value, __uuidof(ID3D12Resource), (void**)&r));
        return std::move(r);
    }

    inline d3d12::query_heap create_query_heap(ID3D12Device* d, _In_  const D3D12_QUERY_HEAP_DESC * desc)
    {
        using namespace os::windows;
        using namespace d3d12;

        d3d12::query_heap r;
        throw_if_failed<create_heap_exception>(d->CreateQueryHeap(desc, __uuidof(ID3D12QueryHeap), (void**)&r));
        return std::move(r);
    }

    inline d3d12::resource create_reserved_resource(ID3D12Device* d, _In_  const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES initial_state, _In_opt_  const D3D12_CLEAR_VALUE * optimized_clear_value)
    {
        using namespace os::windows;
        using namespace d3d12;

        d3d12::resource r;
        throw_if_failed<create_heap_exception>(d->CreateReservedResource(desc, initial_state, optimized_clear_value, __uuidof(ID3D12Resource), (void**)&r));
        return std::move(r);
    }

    inline d3d12::root_signature create_root_signature(ID3D12Device* d, _In_  uint32_t node_mask, _In_reads_(blob_length_in_bytes)  const void *blob_with_root_signature, _In_  size_t blob_length_in_bytes)
    {
        using namespace os::windows;
        using namespace d3d12;

        d3d12::root_signature r;
        throw_if_failed<create_heap_exception>(d->CreateRootSignature(node_mask, blob_with_root_signature, blob_length_in_bytes, __uuidof(ID3D12RootSignature), (void**)&r));
        return std::move(r);
    }

    inline HANDLE create_shared_handle(ID3D12Device* d, _In_  ID3D12DeviceChild *object, _In_opt_  const SECURITY_ATTRIBUTES *attributes, DWORD access, _In_opt_  LPCWSTR name, _Out_  HANDLE * handle)
    {
        using namespace os::windows;
        using namespace d3d12;

        HANDLE r;
        throw_if_failed<create_heap_exception>(d->CreateSharedHandle(object, attributes, access, name, &r));
        return std::move(r);
    }

    inline descriptor_heap < D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE > create_cbv_srv_uav_descriptor_heap(ID3D12Device* d, uint32_t descriptor_count)
    {
        descriptor_heap < D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE > heap1(d, descriptor_count);
        return std::move(heap1);
    }

    inline descriptor_heap < D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE > create_sampler_descriptor_heap(ID3D12Device* d, uint32_t descriptor_count)
    {
        descriptor_heap < D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE > heap1(d, descriptor_count);
        return std::move(heap1);
    }

    inline descriptor_heap < D3D12_DESCRIPTOR_HEAP_TYPE_RTV > create_rtv_descriptor_heap(ID3D12Device* d, uint32_t descriptor_count)
    {
        descriptor_heap < D3D12_DESCRIPTOR_HEAP_TYPE_RTV> heap1(d, descriptor_count);
        return std::move(heap1);
    }

    inline descriptor_heap < D3D12_DESCRIPTOR_HEAP_TYPE_DSV > create_dsv_descriptor_heap(ID3D12Device* d, uint32_t descriptor_count)
    {
        descriptor_heap < D3D12_DESCRIPTOR_HEAP_TYPE_DSV> heap1(d, descriptor_count);
        return std::move(heap1);
    }
}
