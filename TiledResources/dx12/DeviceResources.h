//// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
//// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
//// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
//// PARTICULAR PURPOSE.
////
//// Copyright (c) Microsoft Corporation. All rights reserved
/*
#pragma once

#include "GpuBackBuffer.h"
#include "GpuDepthBuffer.h"

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
        RECT                    GetWindowBounds() const                 { return m_windowBounds; }

        // D3D Accessors.
        ID3D12Device*               GetD3DDevice() const                { return m_d3dDevice.Get(); }
//        ID3D11DeviceContext2*       GetD3DDeviceContext() const             { return m_d3dContext.Get(); }
 //       IDXGISwapChain1*            GetSwapChain() const                    { return m_swapChain.Get(); }
        D3D_FEATURE_LEVEL           GetDeviceFeatureLevel() const           { return m_d3dFeatureLevel; }
        GpuBackBuffer*              GetBackBufferRenderTargetView() const   { return m_d3dRenderTargetView.Get(); }
        GpuDepthBuffer*             GetDepthStencilView() const             { return m_d3dDepthStencilView.Get(); }
        
        D3D12_VIEWPORT              GetScreenViewport() const               { return m_screenViewport; }

        // Sample-specific Accessors.
        D3D12_TILED_RESOURCES_TIER GetTiledResourcesTier() const        { return m_tiledResourcesTier; }

    private:
        void CreateDeviceIndependentResources();
        void CreateDeviceResources();
        void CreateWindowSizeDependentResources();

        DXGI_MODE_ROTATION ComputeDisplayRotation();

        // Direct3D objects.
        Microsoft::WRL::ComPtr<ID3D12Device>        m_d3dDevice;
        Microsoft::WRL::ComPtr<ID3D11DeviceContext2> m_d3dContext;
        Microsoft::WRL::ComPtr<IDXGISwapChain1>      m_swapChain;
        Microsoft::WRL::ComPtr<IDXGIFactory2>        m_dxgiFactory;

        // Direct3D rendering objects. Required for 3D.
        Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_d3dRenderTargetView;
        Microsoft::WRL::ComPtr<ID3D11DepthStencilView> m_d3dDepthStencilView;
        D3D11_VIEWPORT                                 m_screenViewport;

        // Cached reference to the Window.
        HWND m_window;

        // Cached device properties.
        D3D_FEATURE_LEVEL m_d3dFeatureLevel;
        SIZE              m_d3dRenderTargetSize;
        RECT              m_windowBounds;

        // Tiled Resources Tier.
        D3D12_TILED_RESOURCES_TIER m_tiledResourcesTier;
    };
}
*/