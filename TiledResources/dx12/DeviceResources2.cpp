//// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
//// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
//// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
//// PARTICULAR PURPOSE.
////
//// Copyright (c) Microsoft Corporation. All rights reserved

#include "pch.h"
#include "DeviceResources2.h"

#include "DirectXHelper.h"                   // For ThrowIfFailed
#include <windows.ui.xaml.media.dxinterop.h> // For SwapChainBackgroundPanel native methods

using namespace TiledResources;

//using namespace D2D1;
using namespace DirectX;
using namespace Microsoft::WRL;

using namespace std;

// Constructor for DeviceResources.
DeviceResources::DeviceResources() :
    m_screenViewport(),
    m_d3dFeatureLevel(D3D_FEATURE_LEVEL_9_1),
    m_d3dRenderTargetSize(),
    m_windowBounds(),
    m_tiledResourcesTier(D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED)
{
    CreateDeviceIndependentResources();
    CreateDeviceResources();
}

// Configures resources that don't depend on the Direct3D device.
void DeviceResources::CreateDeviceIndependentResources()
{
   
}

// Configures the Direct3D device, and stores handles to it and the device context.
void DeviceResources::CreateDeviceResources()
{

    HR hr = S_OK;

#if defined(_DEBUG) // this must be first, before everything else, otherwise d3d12 has problems
    DX::ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&m_debug)));
    m_debug->EnableDebugLayer();
#endif

#if defined(_DEBUG)
    hr = CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG,IID_PPV_ARGS(&m_dxgiFactory));
#else
    hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&m_dxgiFactory));
#endif

    wcout << L"Finding an adapter that supports tiled resources..." << endl;
    UINT i = 0;
    while (true)
    {
        //if (i == 0) { i++; continue; } // reenable this (assuming the hw adapter is adapter 0) to make it run on warp
        //if (i == 1) { i++; continue; } // reenable this (assuming the hw adapter is adapter 0) to make it run on warp
        ComPtr<IDXGIAdapter1> adapter;
        HRESULT ehr = m_dxgiFactory->EnumAdapters1(i, &adapter);
        if (ehr == DXGI_ERROR_NOT_FOUND)
        {
            wcout << L"couldn't find an adapter supporting tiled resources, and WARP didn't work for some reason" << endl;
            hr = E_UNEXPECTED;
        }
        else
        {
            hr = ehr;
            DXGI_ADAPTER_DESC1 desc = { 0 };
            hr = adapter->GetDesc1(&desc);
            wcout << i << L": " << desc.Description << L" - ";

            ComPtr<ID3D12Device> dev;
            HRESULT chr = D3D12CreateDevice( adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev));

            if (FAILED(chr))
            {
                wcout << L"couldn't create feature level 11 device " << endl;
            }
            else
            {
                D3D12_FEATURE feature = D3D12_FEATURE_D3D12_OPTIONS;
                D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
                hr = dev->CheckFeatureSupport(feature, &options, sizeof(options));

                std::wstring messages[] =
                {
                    L"no support for tiled resources ",
                    L"supports tier 1 ",
                    L"supports tier 2 ",
                    L"supports tier 3 ",
                    L"Unknown tier"
                };

                UINT index = options.TiledResourcesTier <= D3D12_TILED_RESOURCES_TIER_3 ? options.TiledResourcesTier : D3D12_TILED_RESOURCES_TIER_3 + 1;

                wcout << messages[index] << endl;

                if ( options.TiledResourcesTier > D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED )
                {
                    m_tiledResourcesTier = D3D12_TILED_RESOURCES_TIER_1; // for now, force tier-1 behavior - still some issues in the sample code
                    wcout << L"using this adapter" << endl;
                    m_d3dDevice = dev;
                    break;
                }
            }
        }
        i++;
    }

    //Create the direct command queue to the gpu
    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    DX::ThrowIfFailed(m_d3dDevice->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&m_directQueue)));

}

// These resources need to be recreated every time the window size is changed.
void DeviceResources::CreateWindowSizeDependentResources()
{
    // Store the window bounds so the next time we get a SizeChanged event we can
    // avoid rebuilding everything if the size is identical.
    GetWindowRect(m_window, &m_windowBounds);

    // Calculate the necessary swap chain and render target size in pixels.
    float windowWidth = (float)(m_windowBounds.right - m_windowBounds.left);
    float windowHeight = (float)(m_windowBounds.bottom - m_windowBounds.top);

    // The width and height of the swap chain must be based on the window's
    // natively-oriented width and height. If the window is not in the native
    // orientation, the dimensions must be reversed.

    m_d3dRenderTargetSize.cx = (LONG)windowWidth;
    m_d3dRenderTargetSize.cy = (LONG)windowHeight;

    if (m_swapChain)
    {
        // If the swap chain already exists, resize it.
        HRESULT hr = m_swapChain->ResizeBuffers(
            2, // Double-buffered swap chain.
            static_cast<UINT>(m_d3dRenderTargetSize.cx),
            static_cast<UINT>(m_d3dRenderTargetSize.cy),
            DXGI_FORMAT_B8G8R8A8_UNORM,
            0
            );

        if (hr == DXGI_ERROR_DEVICE_REMOVED)
        {
            // If the device was removed for any reason, a new device and swap chain will need to be created.
            //HandleDeviceLost();

            // Everything is set up now. Do not continue execution of this method. 
            return;
        }
        else
        {
            DX::ThrowIfFailed(hr);
        }
    }
    else
    {
        // Otherwise, create a new one using the same adapter as the existing Direct3D device.
        DXGI_MODE_DESC mode = {};
        DXGI_SWAP_CHAIN_DESC desc = {};

        mode.RefreshRate.Numerator = 60;
        mode.RefreshRate.Denominator = 1;

        mode.Format = DXGI_FORMAT_B8G8R8A8_UNORM;

        desc.BufferDesc = mode;
        desc.Windowed = m_window != 0;
        desc.OutputWindow = m_window;
        desc.BufferCount = 2;
        desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        desc.Flags = 0;

        DX::ThrowIfFailed(
            m_dxgiFactory->CreateSwapChain(
                m_directQueue.Get(),
                &desc,
                &m_swapChain
                )
            );
    }

    // Create a render target view of the swap chain back buffer.
    ComPtr<ID3D12Resource> backBuffer;
    DX::ThrowIfFailed( m_swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer)) );

    m_resourceCreateContext = std::make_unique< GpuResourceCreateContext >(m_d3dDevice.Get());

    m_d3dRenderTargetView   =  m_resourceCreateContext->CreateBackBuffer(backBuffer.Get());
    m_d3dDepthStencilView   =  m_resourceCreateContext->CreateDepthBuffer(m_d3dRenderTargetSize.cx, m_d3dRenderTargetSize.cy, DXGI_FORMAT_D24_UNORM_S8_UINT);

    // Set the 3D rendering viewport to target the entire window.
    m_screenViewport.TopLeftX = 0.0f;
    m_screenViewport.TopLeftY = 0.0f;
    m_screenViewport.Width    = (float)m_d3dRenderTargetSize.cx;
    m_screenViewport.Height   = (float)m_d3dRenderTargetSize.cy;
    m_screenViewport.MinDepth = 0.0f;
    m_screenViewport.MaxDepth = 1.0f;
   
}

// This method is called when the CoreWindow is created (or re-created)
void DeviceResources::SetWindow(HWND window)
{
    m_window = window;

    // SetDpi() will call CreateWindowSizeDependentResources()
    // if those resources have not been created yet.
    UpdateForWindowSizeChange();
}

// This method is called in the event handler for the SizeChanged event.
void DeviceResources::UpdateForWindowSizeChange()
{
    /*
    ID3D11RenderTargetView* nullViews[] = {nullptr};
    m_d3dContext->OMSetRenderTargets(ARRAYSIZE(nullViews), nullViews, nullptr);
    m_d3dContext->Flush();
    */
    CreateWindowSizeDependentResources();
}


// Present the contents of the swap chain to the screen.
void DeviceResources::Present()
{
    // The first argument instructs DXGI to block until VSync, putting the application
    // to sleep until the next VSync. This ensures we don't waste any cycles rendering
    // frames that will never be displayed to the screen.
    HRESULT hr = m_swapChain->Present(1, 0);

    // Discard the contents of the render target.
    // This is a valid operation only when the existing contents will be entirely
    // overwritten. If dirty or scroll rects are used, this call should be removed.
    //m_d3dContext->DiscardView(m_d3dRenderTargetView.Get());

    // Discard the contents of the depth stencil.
    //m_d3dContext->DiscardView(m_d3dDepthStencilView.Get());

    // If the device was removed either by a disconnect or a driver upgrade, we 
    // must recreate all device resources.
    if (hr == DXGI_ERROR_DEVICE_REMOVED)
    {
        DX::ThrowIfFailed(hr);
    }
    else
    {
        DX::ThrowIfFailed(hr);
    }
}
