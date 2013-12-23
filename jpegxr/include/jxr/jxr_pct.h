#ifndef __jxr_pct_h__
#define __jxr_pct_h__

#include <jxr/jxr_filter.h>
#include <jxr/jxr_analysis.h>
#include <jxr/jxr_synthesis.h>

namespace jpegxr
{
    namespace cuda
    {
        __global__ void pct_4x4( const transforms::pixel* in, transforms::pixel* out, const uint32_t image_pitch, const uint32_t width, const uint32_t height  )
        {
            auto x = blockIdx.x * blockDim.x + threadIdx.x;
            auto y = blockIdx.y * blockDim.y + threadIdx.y;

            auto row = 4 * y;
            auto col = 4 * x;
            auto pitch = image_pitch;

            //out of image bounds
            if ( col >= width || row >= height )
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

            jpegxr::transforms::pixel pixels[16] =
            {
                in [ element_index0 ],
                in [ element_index1 ],
                in [ element_index2 ],
                in [ element_index3 ],

                in [ element_index4 ],
                in [ element_index5 ],
                in [ element_index6 ],
                in [ element_index7 ],

                in [ element_index8 ],
                in [ element_index9 ],
                in [ element_index10 ],
                in [ element_index11 ],

                in [ element_index12 ],
                in [ element_index13 ],
                in [ element_index14 ],
                in [ element_index15 ],
            };

            transforms::analysis::pct4x4
                (
                    pixels
                );


            out [ element_index0 ] = pixels[0];
            out [ element_index1 ] = pixels[1];
            out [ element_index2 ] = pixels[2];
            out [ element_index3 ] = pixels[3];

            out [ element_index4 ] = pixels[4];
            out [ element_index5 ] = pixels[5];
            out [ element_index6 ] = pixels[6];
            out [ element_index7 ] = pixels[7];

            out [ element_index8 ]  = pixels[8];
            out [ element_index9 ]  = pixels[9];
            out [ element_index10 ] = pixels[10];
            out [ element_index11 ] = pixels[11];

            out [ element_index12 ] = pixels[12];
            out [ element_index13 ] = pixels[13];
            out [ element_index14 ] = pixels[14];
            out [ element_index15 ] = pixels[15];
        }

        __global__ void ipct_4x4( const transforms::pixel* in, transforms::pixel* out, const uint32_t image_pitch, const uint32_t width, const uint32_t height  )
        {
            auto x = blockIdx.x * blockDim.x + threadIdx.x;
            auto y = blockIdx.y * blockDim.y + threadIdx.y;

            auto row = 4 * y;
            auto col = 4 * x;
            auto pitch = image_pitch;

            //out of image bounds
            if ( col >= width || row >= height )
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

            jpegxr::transforms::pixel pixels[16] =
            {
                in [ element_index0 ],
                in [ element_index1 ],
                in [ element_index2 ],
                in [ element_index3 ],

                in [ element_index4 ],
                in [ element_index5 ],
                in [ element_index6 ],
                in [ element_index7 ],

                in [ element_index8 ],
                in [ element_index9 ],
                in [ element_index10 ],
                in [ element_index11 ],

                in [ element_index12 ],
                in [ element_index13 ],
                in [ element_index14 ],
                in [ element_index15 ],
            };

            transforms::synthesis::pct4x4
                (
                   pixels
                );


            out [ element_index0 ] = pixels[0];
            out [ element_index1 ] = pixels[1];
            out [ element_index2 ] = pixels[2];
            out [ element_index3 ] = pixels[3];

            out [ element_index4 ] = pixels[4];
            out [ element_index5 ] = pixels[5];
            out [ element_index6 ] = pixels[6];
            out [ element_index7 ] = pixels[7];

            out [ element_index8 ]  = pixels[8];
            out [ element_index9 ]  = pixels[9];
            out [ element_index10 ] = pixels[10];
            out [ element_index11 ] = pixels[11];

            out [ element_index12 ] = pixels[12];
            out [ element_index13 ] = pixels[13];
            out [ element_index14 ] = pixels[14];
            out [ element_index15 ] = pixels[15];
        }
    }

    inline void pct4x4 (  std::shared_ptr< ::cuda::memory_buffer > in, uint32_t width, uint32_t height, uint32_t pitch ) 
    {
        filter_image( *in, *in, pitch, width, height, [=] 
        ( dim3 blocks, dim3 threads_per_block, const transforms::pixel* in, transforms::pixel* out, const uint32_t image_pitch, const uint32_t width, const uint32_t height )
        {
            jpegxr::cuda::pct_4x4<<<blocks, threads_per_block>>>( in, out, image_pitch, width, height );
        });
    }

    inline void ipct4x4 (  std::shared_ptr< ::cuda::memory_buffer > in, uint32_t width, uint32_t height, uint32_t pitch ) 
    {
        filter_image( *in, *in, pitch, width, height, [=] 
        ( dim3 blocks, dim3 threads_per_block, const transforms::pixel* in, transforms::pixel* out, const uint32_t image_pitch, const uint32_t width, const uint32_t height )
        {
            jpegxr::cuda::ipct_4x4<<<blocks, threads_per_block>>>( in, out, image_pitch, width, height );
        });
    }
}

#endif

