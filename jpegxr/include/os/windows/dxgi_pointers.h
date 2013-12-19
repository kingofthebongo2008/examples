#ifndef __DXGI_POINTERS_H__
#define __DXGI_POINTERS_H__

#include <os/windows/com_ptr.h>

#include <DXGI.h>
#include <DXGI1_2.h>

namespace dxgi
{
    typedef os::windows::com_ptr<IDXGIFactory>                ifactory_ptr;
    typedef os::windows::com_ptr<IDXGIFactory1>               ifactory1_ptr;
    typedef os::windows::com_ptr<IDXGIFactory2>               ifactory2_ptr;

    typedef os::windows::com_ptr<IDXGIAdapter>                iadapter_ptr;
    typedef os::windows::com_ptr<IDXGISwapChain>              iswapchain_ptr;

    typedef os::windows::com_ptr<IDXGISurface>                isurface_ptr;

}

#endif

