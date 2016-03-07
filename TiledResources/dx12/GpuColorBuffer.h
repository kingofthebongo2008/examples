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

        GpuColorBuffer(ID3D12Resource* resource) : Base(resource)
        {

        }
    };
}
