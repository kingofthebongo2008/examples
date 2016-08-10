#ifndef __DXGI_POINTERS_H__
#define __DXGI_POINTERS_H__

#include <os/windows/com_ptr.h>

#include <dxgi.h>
#include <dxgi1_2.h>
#include <dxgi1_3.h>
#include <dxgi1_4.h>

namespace dxgi
{
    using factory = os::windows::com_ptr<IDXGIFactory>;
    using factory1 = os::windows::com_ptr<IDXGIFactory1>;
    using factory2 = os::windows::com_ptr<IDXGIFactory2>;
    using factory3 = os::windows::com_ptr<IDXGIFactory3>;
    using factory4 = os::windows::com_ptr<IDXGIFactory4>;

    using adapter = os::windows::com_ptr<IDXGIAdapter>;
    using adapter1 = os::windows::com_ptr<IDXGIAdapter1>;
    using swapchain = os::windows::com_ptr<IDXGISwapChain>;
    using swapchain2 = os::windows::com_ptr<IDXGISwapChain2>;
    using swapchain3 = os::windows::com_ptr<IDXGISwapChain3>;
    using surface = os::windows::com_ptr<IDXGISurface>;
}

#endif

