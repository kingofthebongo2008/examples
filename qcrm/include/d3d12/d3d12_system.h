#pragma once

#include <d3d12/d3d12_d3d12x.h>
#include <os/windows/dxgi_pointers.h>

namespace d3d12
{
    struct system_context
    {
        dxgi::adapter1          m_adapter;
        dxgi::iswapchain3       m_swap_chain;
        dxgi::factory4          m_factory;
        device	                m_device;
        command_queue           m_direct_command_queue;
        HWND                    m_hwnd;
    };

    system_context create_system_context(HWND hwnd);

    namespace
    {
        inline static DXGI_SWAP_CHAIN_DESC create_default_swap_chain_desc(HWND hwnd)
        {
            DXGI_MODE_DESC mode = {};
            DXGI_SWAP_CHAIN_DESC desc  = {};

            mode.RefreshRate.Numerator = 60;
            mode.RefreshRate.Denominator = 1;

            mode.Format = DXGI_FORMAT_R8G8B8A8_UNORM;//DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;//DXGI_FORMAT_R8G8B8A8_UNORM; //DXGI_FORMAT_R8G8B8A8_UNORM_SRGB; //DXGI_FORMAT_R10G10B10A2_UNORM;//DXGI_FORMAT_B8G8R8A8_UNORM;//DXGI_FORMAT_R10G10B10A2_UNORM;//DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM;//DXGI_FORMAT_R10G10B10A2_UNORM;//DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM; //DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;

            desc.BufferDesc = mode;
            desc.Windowed = hwnd !=0;
            desc.OutputWindow = hwnd;
            desc.BufferCount = 3;
            desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            desc.SampleDesc.Count = 1;
            desc.SampleDesc.Quality = 0;
            desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
            desc.Flags = 0;
            return desc;
        }
    }

    inline system_context create_system_context(HWND hwnd)
    {
        using namespace os::windows;

        dxgi::adapter1              adapter;
        dxgi::iswapchain            swap_chain;

        auto device                 = d3d12x::create_device(nullptr, D3D_FEATURE_LEVEL_11_0);
        auto factory                = d3d12x::create_factory4();


        D3D12_COMMAND_QUEUE_DESC queue_desc = {};
        queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

        auto command_queue = d3d12x::create_command_queue(device, &queue_desc);

        throw_if_failed( factory->EnumAdapters1(0, &adapter));
        throw_if_failed( factory->CreateSwapChain(command_queue, &create_default_swap_chain_desc(hwnd), &swap_chain));

        dxgi::iswapchain3            swap_chain3;

        throw_if_failed(swap_chain->QueryInterface(__uuidof(IDXGISwapChain3), (void**)&swap_chain3));

        system_context result = { adapter, swap_chain3, factory, device, command_queue, hwnd };

        return std::move(result);
    }
}


