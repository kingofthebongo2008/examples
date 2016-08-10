#pragma once

#include <cstdint>
#include <strstream>
#include <iomanip>
#include <string>
#include <sstream>


namespace app
{
    using tile = uint16_t;
    using texel = uint16_t;

    inline texel to_texel(tile t)
    {
        return t * 128U;
    }

    inline tile to_tile(texel t)
    {
        return t / 128U;
    }

    struct map
    {
        tile m_width = 128;
        tile m_height = 128;
    };

    template <uint32_t n> inline size_t mip_levels()
    {
        return math::log2_c<n>::value + 1;
    }

    inline std::string make_tile_file_name(tile row, tile col)
    {
        std::stringstream str;
        str << "tile_" << std::setfill('0') << std::setw(4) << row << "_" << std::setfill('0') << std::setw(4) << col << ".tga";

        return str.str();
    }

    inline std::string make_tile_file_name(uint32_t mip, tile row, tile col)
    {
        std::stringstream str;
        str << "tile_" << std::setfill('0') << std::setw(4) << mip << "_" << std::setfill('0') << std::setw(4) << row << "_" << std::setfill('0') << std::setw(4) << col << ".tga";
        return str.str();
    }
}
