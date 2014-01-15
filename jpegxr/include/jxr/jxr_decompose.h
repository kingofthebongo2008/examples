#ifndef __jxr_decompose_h__
#define __jxr_decompose_h__

#include <cstdint>
#include <jxr/jxr_filter.h>

namespace jpegxr
{
    namespace decompose
    {
        //gather dc coefficients after the pct4x4 transform
        __global__ void split_lp_hp( const transforms::pixel* in, transforms::pixel* out, const uint32_t in_pitch, const uint32_t width, const uint32_t height, const uint32_t out_pitch)
        {
            auto x = blockIdx.x * blockDim.x + threadIdx.x;
            auto y = blockIdx.y * blockDim.y + threadIdx.y;

            auto row = 4 * y;
            auto col = 4 * x;
            auto pitch = in_pitch;

            //out of image bounds
            if ( col >= width || row >= height )
            {
                return;
            }

            auto element_in_index  = row * pitch + col;

            auto dc = in[ element_in_index ];

            auto element_out_index  = (y + 0) * out_pitch + x + 0;
            out[ element_out_index ] = dc;
        }

        __global__ void combine_lp_hp( const transforms::pixel* in, transforms::pixel* out, const uint32_t in_pitch, const uint32_t width, const uint32_t height, const uint32_t out_pitch)
        {
            auto x = blockIdx.x * blockDim.x + threadIdx.x;
            auto y = blockIdx.y * blockDim.y + threadIdx.y;

            auto row = 4 * y;
            auto col = 4 * x;
            auto pitch = in_pitch;

            //out of image bounds
            if ( col >= width || row >= height )
            {
                return;
            }

            auto element_in_index  = x * pitch + y;

            auto dc = in[ element_in_index ];

            auto element_out_index  = row * out_pitch + col;
            out[ element_out_index ] = dc;
        }
    }
}

#endif

