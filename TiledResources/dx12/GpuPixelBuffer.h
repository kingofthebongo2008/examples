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
    };
}
