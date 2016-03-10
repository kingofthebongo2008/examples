#pragma once

#include "GpuResource.h"

namespace TiledResources
{
    class GpuPixelBuffer : public GpuResource
    {
        private:

        using Base = GpuResource;

        public:

        GpuPixelBuffer( ID3D12Resource* resource ) : Base(resource)
        {

        }

        UINT Width() const
        {
            auto d = GetResource()->GetDesc();
            return static_cast<uint32_t>(d.Width);
        }

        UINT Height() const
        {
            auto d = GetResource()->GetDesc();
            return static_cast<uint32_t>(d.Height);
        }

        UINT ArraySize() const
        {
            auto d = GetResource()->GetDesc();
            return d.DepthOrArraySize;
        }

        UINT Depth() const
        {
            auto d = GetResource()->GetDesc();
            return static_cast<uint32_t>(d.DepthOrArraySize);
        }

        DXGI_FORMAT Format() const
        {
            auto d = GetResource()->GetDesc();
            return (d.Format);
        }
    };
}
