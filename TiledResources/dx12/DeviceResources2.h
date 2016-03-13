//// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
//// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
//// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
//// PARTICULAR PURPOSE.
////
//// Copyright (c) Microsoft Corporation. All rights reserved

#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include "GpuBackBuffer.h"
#include "GpuDepthBuffer.h"
#include "GpuCommandContext.h"
#include "GpuResourceCreateContext.h"


namespace TiledResources
{
    // Controls all the DirectX device resources.
    class DeviceResources
    {
    public:
        DeviceResources();
        void SetWindow(HWND window);
        void UpdateForWindowSizeChange();
        void Present();

        // Device Accessors.
        RECT                        GetWindowBounds() const                 { return m_windowBounds; }
        // D3D Accessors.
        ID3D12Device*               GetD3DDevice() const                    { return m_d3dDevice.Get(); }
        const GpuCommandContext*    GetD3DDeviceContext() const             { return &m_d3dContext; }
        IDXGISwapChain*             GetSwapChain() const                    { return m_swapChain.Get(); }
        D3D_FEATURE_LEVEL           GetDeviceFeatureLevel() const           { return m_d3dFeatureLevel; }
        const GpuBackBuffer*        GetBackBufferRenderTargetView() const   { return &m_d3dRenderTargetView; }
        const GpuDepthBuffer*       GetDepthStencilView() const             { return &m_d3dDepthStencilView; }
        
        D3D12_VIEWPORT              GetScreenViewport() const               { return m_screenViewport; }

        // Sample-specific Accessors.
        D3D12_TILED_RESOURCES_TIER GetTiledResourcesTier() const            { return m_tiledResourcesTier; }

    private:
        void CreateDeviceIndependentResources();
        void CreateDeviceResources();
        void CreateWindowSizeDependentResources();

        DXGI_MODE_ROTATION ComputeDisplayRotation();

        std::unique_ptr<GpuResourceCreateContext>                           m_resourceCreateContext;
        // Direct3D objects.
        Microsoft::WRL::ComPtr<ID3D12Device>                                m_d3dDevice;
        Microsoft::WRL::ComPtr <ID3D12CommandQueue>                         m_directQueue;
        GpuCommandContext                                                   m_d3dContext;
        Microsoft::WRL::ComPtr<IDXGISwapChain>                              m_swapChain;
        Microsoft::WRL::ComPtr<IDXGIFactory4>                               m_dxgiFactory;
        #if defined(_DEBUG)
        Microsoft::WRL::ComPtr<ID3D12Debug>                                 m_debug;
        #endif

        // Direct3D rendering objects. Required for 3D.
        GpuBackBuffer                                                       m_d3dRenderTargetView;
        GpuDepthBuffer                                                      m_d3dDepthStencilView;
        D3D12_VIEWPORT                                                      m_screenViewport;

        // Cached reference to the Window.
        HWND m_window;

        // Cached device properties.
        D3D_FEATURE_LEVEL                                                   m_d3dFeatureLevel;
        SIZE                                                                m_d3dRenderTargetSize;
        RECT                                                                m_windowBounds;

        // Tiled Resources Tier.
        D3D12_TILED_RESOURCES_TIER                                          m_tiledResourcesTier;
    };
}
