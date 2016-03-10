#pragma once

#include "GpuResource.h"
#include "GpuPixelBuffer.h"

namespace TiledResources
{
    class GpuColorBuffer : public GpuPixelBuffer
    {
        private:
        
        using Base = GpuPixelBuffer;

        public:
        GpuColorBuffer(ID3D12Resource* resource, DescriptorHandle rtv, DescriptorHandle srv, DescriptorHandle uav ) : Base(resource)
            , m_RTV(rtv)
            , m_SRV(srv)
            , m_UAV(uav)
        {

        }

        DescriptorHandle   RTV() const
        {
            return m_RTV;
        }

        DescriptorHandle   SRV() const
        {
            return m_SRV;
        }

        DescriptorHandle   UAV() const
        {
            return m_UAV;
        }

        private:

        DescriptorHandle   m_RTV;
        DescriptorHandle   m_SRV;
        DescriptorHandle   m_UAV;

    };
}
