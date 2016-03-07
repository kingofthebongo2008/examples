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

        GpuDepthBuffer( ID3D12Resource* resource ) : Base(resource)
        {

        }
    };
}
