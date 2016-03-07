#pragma once

#include "GpuResource.h"
#include "GpuDescriptorHeap.h"

namespace TiledResources
{
    class GpuReadBackBuffer : public GpuResource
    {
        private:

        using Base = GpuResource;

        public:

        GpuReadBackBuffer( ID3D12Resource* resource ) : Base(resource)
        {

        }

        private:

            void* map()
            {
                D3D12_RESOURCE_DESC d;
                d = m_resource->GetDesc();

                D3D12_RANGE range = {};

                range.Begin = 0;
                range.End   = static_cast<SIZE_T>(d.Width);

                void* r;
                DX::ThrowIfFailed(m_resource->Map(0, &range, &r));
                return r;
            }

            void unmap()
            {
                m_resource->Unmap(0, nullptr);
            }
    };
}
