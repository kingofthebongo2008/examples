#pragma once

#include "GpuResource.h"
#include "GpuPixelBuffer.h"

namespace TiledResources
{
    class GpuDepthBuffer : public GpuPixelBuffer
    {
    private:

        using Base = GpuPixelBuffer;

    public:
        GpuDepthBuffer(ID3D12Resource* resource, DescriptorHandle srv, DescriptorHandle dsv) : Base(resource)
            , m_SRV(srv)
            , m_DSV(uav)
        {

        }

        DescriptorHandle   DSV() const
        {
            return m_DSV;
        }

        DescriptorHandle   SRV() const
        {
            return m_SRV;
        }

    private:

        DescriptorHandle   m_SRV;
        DescriptorHandle   m_DSV;

    };
}
