#pragma once

namespace app
{
    struct tile_coordinates
    {
        uint8_t m_tile_x;
        uint8_t m_tile_y;
        uint8_t m_mip;
        uint8_t m_padding;

        operator uint32_t() const
        {
            return (m_tile_x << 24) | (m_tile_y << 16) | (m_mip << 8);
        }
    };
}

inline bool operator==(app::tile_coordinates a, app::tile_coordinates b)
{
    return static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
}

namespace std
{
    template<>
    struct hash<app::tile_coordinates>
        : public _Bitwise_hash<app::tile_coordinates>
    {	// hash functor for bool
    };
}




