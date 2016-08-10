#pragma once

#include <cstdint>
#include <d3d11_2.h>

#include <os/windows/com_error.h>
#include <os/windows/dxgi_pointers.h>

#include <d3d11/d3d11_exception.h>
#include <d3d11/d3d11_pointers.h>

namespace d3d11
{
    namespace helpers
    {
        inline dxgi::factory create_factory()
        {
            using namespace os::windows;
            using namespace d3d11;

            dxgi::factory factory;
            throw_if_failed(CreateDXGIFactory1(__uuidof(IDXGIFactory), (void**)&factory));

            return factory;
        }

        inline dxgi::factory1 create_factory1()
        {
            using namespace os::windows;
            using namespace d3d11;

            dxgi::factory1 factory;
            throw_if_failed(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory));

            return factory;
        }

        inline dxgi::factory1 create_factory3()
        {
            using namespace os::windows;
            using namespace d3d11;

            dxgi::factory3 factory;
            throw_if_failed(CreateDXGIFactory1(__uuidof(IDXGIFactory3), (void**)&factory));

            return factory;
        }

        inline dxgi::factory4 create_factory4()
        {
            using namespace os::windows;
            using namespace d3d11;

            dxgi::factory4 factory;
            throw_if_failed(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&factory));

            return factory;
        }
    }
}
