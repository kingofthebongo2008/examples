#pragma once

namespace app
{
    struct tiled_texture
    {
        uint32_t            m_width_texels = 0;
        uint32_t            m_height_texels = 0;

        uint32_t            m_width_tiles = 0;
        uint32_t            m_height_tiles = 0;

        d3d11::buffer       m_tile_pool;
        d3d11::texture2d    m_resource;
    };
}
