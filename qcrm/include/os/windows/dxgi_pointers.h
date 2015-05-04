#ifndef __DXGI_POINTERS_H__
#define __DXGI_POINTERS_H__

#include <os/windows/com_ptr.h>

#include <dxgi.h>
#include <dxgi1_2.h>
#include <dxgi1_3.h>
#include <dxgi1_4.h>

namespace dxgi
{
    typedef os::windows::com_ptr<IDXGIFactory>                factory;
    typedef os::windows::com_ptr<IDXGIFactory1>               factory1;
    typedef os::windows::com_ptr<IDXGIFactory2>               factory2;
    typedef os::windows::com_ptr<IDXGIFactory3>               factory3;
    typedef os::windows::com_ptr<IDXGIFactory4>               factory4;

    typedef os::windows::com_ptr<IDXGIAdapter>                adapter;
    typedef os::windows::com_ptr<IDXGIAdapter1>               adapter1;
    typedef os::windows::com_ptr<IDXGISwapChain>              iswapchain;
    typedef os::windows::com_ptr<IDXGISwapChain2>             iswapchain2;
    typedef os::windows::com_ptr<IDXGISwapChain3>             iswapchain3;

    typedef os::windows::com_ptr<IDXGISurface>                isurface;

}

#endif

