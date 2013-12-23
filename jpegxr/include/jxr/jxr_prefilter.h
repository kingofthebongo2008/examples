#ifndef __jxr_prefilter_h__
#define __jxr_prefilter_h__

#include <jxr/jxr_filter.h>

namespace jpegxr
{
    namespace cuda
    {
        __global__ void prefilter_4x4( const uint32_t* in, uint32_t* out, const uint32_t image_pitch, const uint32_t width, const uint32_t height  )
        {
            auto x = blockIdx.x * blockDim.x + threadIdx.x;
            auto y = blockIdx.y * blockDim.y + threadIdx.y;

            auto row = 4 * y + 2;
            auto col = 4 * x + 2;
            auto pitch = image_pitch;

            //out of image bounds
            if ( col >= width - 2  || row >= height - 2 )
            {
                return;
            }
        
            auto element_index0  = (row + 0) * pitch + col + 0;
            auto element_index1  = (row + 0) * pitch + col + 1;
            auto element_index2  = (row + 0) * pitch + col + 2;
            auto element_index3  = (row + 0) * pitch + col + 3;

            auto element_index4  = (row + 1) * pitch + col + 0;
            auto element_index5  = (row + 1) * pitch + col + 1;
            auto element_index6  = (row + 1) * pitch + col + 2;
            auto element_index7  = (row + 1) * pitch + col + 3;

            auto element_index8  = (row + 2) * pitch + col + 0;
            auto element_index9  = (row + 2) * pitch + col + 1;
            auto element_index10 = (row + 2) * pitch + col + 2;
            auto element_index11 = (row + 2) * pitch + col + 3;

            auto element_index12 = (row + 3) * pitch + col + 0;
            auto element_index13 = (row + 3) * pitch + col + 1;
            auto element_index14 = (row + 3) * pitch + col + 2;
            auto element_index15 = (row + 3) * pitch + col + 3;

            out [ element_index0 ] = 0xfffffff0;
            out [ element_index1 ] = 0xfffffff1;
            out [ element_index2 ] = 0xfffffff2;
            out [ element_index3 ] = 0xfffffff3;

            out [ element_index4 ] = 0xfffffff4;
            out [ element_index5 ] = 0xfffffff5;
            out [ element_index6 ] = 0xfffffff6;
            out [ element_index7 ] = 0xfffffff7;

            out [ element_index8 ]  = 0xfffffff8;
            out [ element_index9 ]  = 0xfffffff9;
            out [ element_index10 ] = 0xfffffffa;
            out [ element_index11 ] = 0xfffffffb;

            out [ element_index12 ] = 0xfffffffc;
            out [ element_index13 ] = 0xfffffffd;
            out [ element_index14 ] = 0xfffffffe;
            out [ element_index15 ] = 0xffffffff;
        }


        __global__ void prefilter4_vertical( const uint32_t* in, uint32_t* out, const uint32_t pitch, const uint32_t width, const uint32_t height)
        {
            auto x = blockIdx.x * blockDim.x + threadIdx.x;
            auto y = blockIdx.y * blockDim.y + threadIdx.y;

            auto row = 4 * y + 2;
            auto col = x ;

            //out of bounds checks
            if ( x > 1 && x < width - 2 )
            {
                return;
            }

            //out of bounds checks
            if ( row >= height  - 2 )
            {
                return;
            }
        
            auto element_index0  = (row + 0) * pitch + col + 0;
            auto element_index1  = (row + 1) * pitch + col + 0;

            auto element_index2  = (row + 2) * pitch + col + 0;
            auto element_index3  = (row + 3) * pitch + col + 0;

            out [ element_index0 ] = (col << 16) | row;//0x3;
            out [ element_index1 ] = (col << 16) | row;//0x3;
            out [ element_index2 ] = (col << 16) | row;//0x3;
            out [ element_index3 ] = (col << 16) | row;//0x3;
        }

        __global__ void prefilter4_horizontal( const uint32_t* in, uint32_t* out, const uint32_t pitch, const uint32_t width, const uint32_t height)
        {
            auto x = blockIdx.x * blockDim.x + threadIdx.x;
            auto y = blockIdx.y * blockDim.y + threadIdx.y;

            auto row = y;
            auto col = 4 * x + 2;

            //out of bounds checks
            if ( y > 1 && y < height - 2 )
            {
                return;
            }

            //out of bounds checks
            if ( col >= width  - 2 )
            {
                return;
            }
        
            auto element_index0  = (row + 0) * pitch + col + 0;
            auto element_index1  = (row + 0) * pitch + col + 1;

            auto element_index2  = (row + 0) * pitch + col + 2;
            auto element_index3  = (row + 0) * pitch + col + 3;

            out [ element_index0 ] = (row << 16) | 0x2;
            out [ element_index1 ] = (row << 16) | 0x2;
            out [ element_index2 ] = (row << 16) | 0x2;
            out [ element_index3 ] = (row << 16) | 0x2;
        }

        __global__ void prefilter2x2_edge( const uint32_t* in, uint32_t* out, const uint32_t pitch, const uint32_t width, const uint32_t height)
        {
            auto x = blockIdx.x * blockDim.x + threadIdx.x;
            auto y = blockIdx.y * blockDim.y + threadIdx.y;

            auto row = y ;
            auto col = x ;

            //out of bounds checks
            if ( 
                !
                (
                    (x == 0 && y == 0) || 
                    (x == 0 && y == height - 2 ) ||
                    (x == width - 2 && y == 0) ||
                    (x == width - 2 && y == height - 2 )
                )
                )
            {
                return;
            }

            auto element_index0  = (row + 0) * pitch + col + 0;
            auto element_index1  = (row + 0) * pitch + col + 1;

            auto element_index2  = (row + 1) * pitch + col + 0;
            auto element_index3  = (row + 1) * pitch + col + 1;

            out [ element_index0 ] = 0x1;
            out [ element_index1 ] = 0x1;
            out [ element_index2 ] = 0x1;
            out [ element_index3 ] = 0x1;
        }
    }

    inline void prefilter4x4 (  std::shared_ptr< ::cuda::memory_buffer > in, uint32_t width, uint32_t height, uint32_t pitch ) 
    {
        filter_image( *in, *in, pitch, width, height, [=] 
        ( dim3 blocks, dim3 threads_per_block, const uint32_t* in, uint32_t* out, const uint32_t image_pitch, const uint32_t width, const uint32_t height )
        {
            jpegxr::cuda::prefilter_4x4<<<blocks, threads_per_block>>>( in, out, image_pitch, width, height );
        });
    }

    inline void prefilter4_vertical(  std::shared_ptr< ::cuda::memory_buffer > in, uint32_t width, uint32_t height, uint32_t pitch ) 
    {
        filter_image( *in, *in, pitch, width, height, [=] 
        ( dim3 blocks, dim3 threads_per_block, const uint32_t* in, uint32_t* out, const uint32_t image_pitch, const uint32_t width, const uint32_t height )
        {
            jpegxr::cuda::prefilter4_vertical<<<blocks, threads_per_block>>>( in, out, image_pitch, width, height );
        });
        }

    inline void prefilter4_horizontal(  std::shared_ptr< ::cuda::memory_buffer > in, uint32_t width, uint32_t height, uint32_t pitch ) 
    {
        filter_image( *in, *in, pitch, width, height, [=] 
        ( dim3 blocks, dim3 threads_per_block, const uint32_t* in, uint32_t* out, const uint32_t image_pitch, const uint32_t width, const uint32_t height )
        {
            jpegxr::cuda::prefilter4_horizontal<<<blocks, threads_per_block>>>( in, out, image_pitch, width, height );
        });
    }

    inline void prefilter2x2_edge (  std::shared_ptr< ::cuda::memory_buffer > in, uint32_t width, uint32_t height, uint32_t pitch ) 
    {
        filter_image( *in, *in, pitch, width, height, [=] 
        ( dim3 blocks, dim3 threads_per_block, const uint32_t* in, uint32_t* out, const uint32_t image_pitch, const uint32_t width, const uint32_t height )
        {
            jpegxr::cuda::prefilter2x2_edge<<<blocks, threads_per_block>>>( in, out, image_pitch, width, height );
        });
    }
}

#endif

