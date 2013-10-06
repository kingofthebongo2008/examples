#ifndef __GX_GEOMETRY_HELPERS_H__
#define __GX_GEOMETRY_HELPERS_H__

#include <cstdint>
#include <vector>

#include <math/math_half.h>


namespace gx
{
    std::vector<math::half>  create_positions_x_y_z( const float * positions_x_y_z, uint32_t count_triplet, float w = 1.0f )
    {
        auto size = count_triplet / 3;
        auto padded_size = 24 * ((size + 23) / 24);

        std::vector<math::half> positions_h(4 * padded_size);

        math::convert_3_x_f32_f16_stream(positions_x_y_z, 3 * padded_size, w, &positions_h[0]);

        return std::move(positions_h);
    }

    std::vector<math::half>  create_positions_x_y_z_w(const float * positions_x_y_z_w, uint32_t count)
    {
        auto size = count;

        std::vector<math::half> positions_h( 4 * size );

        math::convert_f32_f16_stream(positions_x_y_z_w, size, &positions_h[0]);

        return std::move(positions_h);
    }
}

#endif