#pragma once

#include <d3d11_2.h>
#include <cstdint>

namespace d3d11
{
    struct vertex_buffer_view
    {
        ID3D11Buffer* buffer;
        uint32_t      stride;
        uint32_t      offset;
    };

    struct index_buffer_view
    {
        ID3D11Buffer* buffer;
        DXGI_FORMAT   format;
        uint32_t      offset;
    };
}
