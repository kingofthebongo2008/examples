#pragma once

#include "GpuResource.h"
#include "GpuPixelBuffer.h"
#include "GpuDescriptorHeap.h"

namespace TiledResources
{
    class GpuDepthBuffer : public GpuPixelBuffer
    {
        private:

        using Base = GpuPixelBuffer;

        public:

        GpuDepthBuffer(ID3D12Resource* resource, DescriptorHandle srv[2], DescriptorHandle dsv[4] ) : Base(resource)
        {
            m_SRV[0] = srv[0];
            m_SRV[1] = srv[1];

            m_DSV[0] = dsv[0];
            m_DSV[1] = dsv[1];
            m_DSV[2] = dsv[2];
            m_DSV[3] = dsv[3];
        }

        DescriptorHandle DSV() const
        {
            return m_DSV[0];
        }

        DescriptorHandle DSVReadDepth() const
        {
            return m_DSV[1];
        }

        DescriptorHandle DSVReadStencil() const
        {
            return m_DSV[2];
        }

        DescriptorHandle DSVReadDepthStencil() const
        {
            return m_DSV[3];
        }

        DescriptorHandle SRVDepth() const
        {
            return m_SRV[0];
        }

        DescriptorHandle SRVStencil() const
        {
            return m_SRV[1];
        }

    private:

        DescriptorHandle   m_SRV[4];
        DescriptorHandle   m_DSV[2];
    };
}
