#pragma once

#include "GpuResource.h"
#include "GpuPixelBuffer.h"

namespace TiledResources
{
    class GpuBackBuffer : public GpuPixelBuffer
    {

    private:

        using Base = GpuPixelBuffer;

    public:

        GpuBackBuffer(ID3D12Resource* resource) : Base(resource)
        {

        }
    };
}
