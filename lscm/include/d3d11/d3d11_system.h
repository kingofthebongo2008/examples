#ifndef __d3d11_SYSTEM_H__
#define __d3d11_SYSTEM_H__

#include <d3d11/d3d11_pointers.h>
#include <os/windows/dxgi_pointers.h>

namespace d3d11
{
    struct system_context
    {
        dxgi::iadapter_ptr          m_adapter;
        dxgi::iswapchain_ptr        m_swap_chain;
        idevice_ptr	                m_device;
        idevicecontext_ptr          m_immediate_context;
        HWND                        m_hwnd;
    };

    system_context create_system_context(HWND hwnd);
    system_context create_system_context(HWND hwnd, system_context context);
}

namespace d3d11
{
    namespace
    {
        inline static DXGI_SWAP_CHAIN_DESC create_default_swap_chain_desc(HWND hwnd)
        {
            DXGI_MODE_DESC mode = {};
            DXGI_SWAP_CHAIN_DESC desc  = {};

            mode.RefreshRate.Numerator = 60;
            mode.RefreshRate.Denominator = 1;

            mode.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB; //DXGI_FORMAT_R8G8B8A8_UNORM_SRGB; //DXGI_FORMAT_R10G10B10A2_UNORM;//DXGI_FORMAT_B8G8R8A8_UNORM;//DXGI_FORMAT_R10G10B10A2_UNORM;//DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM;//DXGI_FORMAT_R10G10B10A2_UNORM;//DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM; //DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;

            desc.BufferDesc = mode;
            desc.Windowed = (hwnd !=0);
            desc.OutputWindow = hwnd;
            desc.BufferCount = 2;
            desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            desc.SampleDesc.Count = 1;
            desc.SampleDesc.Quality = 0;
            //desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
            desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
            desc.Flags = 0;
            return desc;
        }
    }

    inline system_context create_system_context(HWND hwnd)
    {
        auto flags                  = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

        auto level                  = D3D_FEATURE_LEVEL_11_0;
        auto desc                   = create_default_swap_chain_desc(hwnd);

        dxgi::iadapter_ptr          adapter;
        dxgi::iswapchain_ptr        swap_chain;

        idevice_ptr                 device;
        idevicecontext_ptr          context;

        dxgi::ifactory2_ptr factory;

        using namespace os::windows;

        throw_if_failed<create_dxgi_factory_exception>(CreateDXGIFactory1(__uuidof(IDXGIFactory2), (void**)&factory));

        auto i = 0;
        std::vector <dxgi::iadapter_ptr> adapters;
        adapters.reserve(6);

        while (factory->EnumAdapters(i, &adapter) != DXGI_ERROR_NOT_FOUND)
        {
            adapters.push_back(adapter);
            ++i;
        }

        using namespace os::windows;
        auto hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, 0, flags, &level, 1, D3D11_SDK_VERSION, &desc, &swap_chain, &device, 0, &context ) ;

        throw_if_failed<create_device_exception> (hr);

        unsigned int result1 = 0;
        device->CheckFormatSupport( DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM, &result1);

        system_context result = { adapters[0], swap_chain, device, context, hwnd };
        return result;
    }

    inline system_context create_system_context(HWND hwnd, system_context context)
    {
        dxgi::ifactory2_ptr factory;

        using namespace os::windows;
        throw_if_failed<create_dxgi_factory_exception>( CreateDXGIFactory1(__uuidof(IDXGIFactory2), (void**) &factory  ) );

        DXGI_SWAP_CHAIN_DESC        desc = create_default_swap_chain_desc(hwnd);
      
        dxgi::iswapchain_ptr        swap_chain;
        using namespace os::windows;
        throw_if_failed<create_swap_chain_exception>( factory->CreateSwapChain(context.m_device.get(), &desc,  &swap_chain ) ); 
        
        system_context result = {context.m_adapter, swap_chain, context.m_device,  context.m_immediate_context, hwnd};

        return result;
    }
}

#endif

