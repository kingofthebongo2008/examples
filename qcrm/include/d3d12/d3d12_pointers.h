#pragma once

#include <os/windows/com_pointers.h>

#include <D3D12.h>

namespace d3d12
{
    typedef os::windows::com_ptr<ID3D12Device>              device;
    typedef os::windows::com_ptr<ID3D12DescriptorHeap>      descriptor_heap;
    typedef os::windows::com_ptr<ID3D12CommandAllocator>    command_allocator;
    typedef os::windows::com_ptr<ID3D12CommandQueue>        command_queue;
    typedef os::windows::com_ptr<ID3D12Heap>                heap;
    typedef os::windows::com_ptr<ID3D12CommandList>         command_list;
    typedef os::windows::com_ptr<ID3D12CommandSignature>    command_signature;
    typedef os::windows::com_ptr<ID3D12RootSignature>       root_signature;
    typedef os::windows::com_ptr<ID3D12Resource>            resource;
    typedef os::windows::com_ptr<ID3D12PipelineState>       pipeline_state;
    typedef os::windows::com_ptr<ID3D12Fence>               fence;
    typedef os::windows::com_ptr<ID3D12QueryHeap>           query_heap;
    

}

