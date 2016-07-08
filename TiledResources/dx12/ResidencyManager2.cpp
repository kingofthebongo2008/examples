//// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
//// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
//// THE IMPLIED ARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
//// PARTICULAR PURPOSE.
////
//// Copyright (c) Microsoft Corporation. All rights reserved

#include "pch.h"
#include "DirectXHelper.h"
#include "ResidencyManager2.h"

using namespace TiledResources;

using namespace concurrency;
using namespace DirectX;
using namespace Windows::Foundation;

unsigned g_frame = 0;

namespace TiledResources
{
    namespace details
    {
        static inline GpuTexture2D CreateTempTexture(GpuResourceCreateContext* context)
        {
            return context->CreateTexture2D(256, 256, DXGI_FORMAT_R8_UNORM);

        }
    }

    ResidencyManager::ResidencyManager(const std::shared_ptr<DeviceResources>& deviceResources) :
        m_deviceResources(deviceResources)
    {

    }

    void ResidencyManager::CreateDeviceDependentResources()
    {
        m_tempTexture = details::CreateTempTexture(m_deviceResources->GetResourceCreateContext());
    }

    task<void> ResidencyManager::CreateDeviceDependentResourcesAsync()
    {
        return concurrency::create_task([] {});
    }

    void ResidencyManager::ReleaseDeviceDependentResources()
    {

    }

    void ResidencyManager::RenderVisualization()
    {

    }

    //void ResidencyManager::EnqueueSamples(const std::vector<DecodedSample>& samples)
    //{
    //}

    void ResidencyManager::ProcessQueues()
    {

    }

    void ResidencyManager::SetDebugMode(bool value)
    {
    }

    void ResidencyManager::Reset()
    {

    }

    std::wstring g_diffuseFilename;

    task<void> ResidencyManager::InitializeManagedResourcesAsync()
    {
        return concurrency::create_task([]{});
    }

    void ResidencyManager::SetBorderMode(bool value)
    {
    }
}
