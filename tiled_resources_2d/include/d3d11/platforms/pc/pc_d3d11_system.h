#pragma once

#include <d3d11/d3d11_helpers.h>
#include <os/windows/dxgi_pointers.h>

namespace d3d11
{
    struct system_context
    {
        dxgi::adapter1          m_adapter;
        dxgi::swapchain3        m_swap_chain;
        dxgi::factory4          m_factory;
        device2	                m_device;
        device_context2         m_immediate_context;
        HWND                    m_hwnd;
    };

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

    inline system_context create_system_context(HWND hwnd, uint32_t adapter_index)
    {
        using namespace os::windows;

        dxgi::adapter1              adapter;
        dxgi::swapchain            swap_chain;

        
        auto factory                = d3d11::helpers::create_factory4();

        throw_if_failed(factory->EnumAdapters1(adapter_index, &adapter));

        auto t = d3d11::helpers::create_device(adapter);
        auto&& device = std::get<0>(t);
        auto&& ctx    = std::get<1>(t);

        auto desc = create_default_swap_chain_desc(hwnd);

        throw_if_failed( factory->CreateSwapChain(device, &desc, &swap_chain));

        dxgi::swapchain3            swap_chain3;

        throw_if_failed(swap_chain->QueryInterface(__uuidof(IDXGISwapChain3), (void**)&swap_chain3));

        device2 d2;
        throw_if_failed(device->QueryInterface(__uuidof(ID3D11Device2), (void**)&d2));

        device_context2 ctx2;
        throw_if_failed(ctx->QueryInterface(__uuidof(ID3D11DeviceContext2), (void**)&ctx2));

        system_context result = { adapter, swap_chain3, factory, d2, ctx2, hwnd };

        return result;
    }
}


