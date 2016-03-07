#pragma once

#include "GpuResource.h"
#include "GpuDescriptorHeap.h"

namespace TiledResources
{
    class GpuTexture2D : public GpuResource
    {
        private:

        using Base = GpuResource;

        public:

        GpuTexture2D( ID3D12Resource* resource, DescriptorHandle uav, DescriptorHandle srv ) : Base(resource)
        , m_uav(uav)
        , m_srv(srv)
        {

        }

        auto UAV() const
        {
            return m_uav;
        }

        auto SRV() const
        {
            return m_srv;
        }

        private:

        DescriptorHandle    m_uav;
        DescriptorHandle    m_srv;
    };
}
