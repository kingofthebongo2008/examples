#ifndef __jxr_prefilter_h__
#define __jxr_prefilter_h__

#include <jxr/jxr_filter.h>
#include <jxr/jxr_analysis.h>

namespace jpegxr
{
    namespace cuda
    {
        __global__ void prefilter_4x4( const transforms::pixel* in, transforms::pixel* out, const uint32_t image_pitch, const uint32_t width, const uint32_t height  )
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

            auto a = in [ element_index0 ];
            auto b = in [ element_index1 ];
            auto c = in [ element_index2 ];
            auto d = in [ element_index3 ];

            auto e = in [ element_index4 ];
            auto f = in [ element_index5 ];
            auto g = in [ element_index6 ];
            auto h = in [ element_index7 ];

            auto i = in [ element_index8 ];
            auto j = in [ element_index9 ];
            auto k = in [ element_index10 ];
            auto l = in [ element_index11 ];

            auto m = in [ element_index12 ];
            auto n = in [ element_index13 ];
            auto o = in [ element_index14 ];
            auto p = in [ element_index15 ];

            transforms::analysis::prefilter4x4
                (
                    &a, &b, &c, &d, 
                    &e, &f, &g, &h, 
                    &i, &j, &k, &l,
                    &m, &n, &o, &p
                );


            out [ element_index0 ] = a;
            out [ element_index1 ] = b;
            out [ element_index2 ] = c;
            out [ element_index3 ] = d;

            out [ element_index4 ] = e;
            out [ element_index5 ] = f;
            out [ element_index6 ] = g;
            out [ element_index7 ] = h;

            out [ element_index8 ]  = i;
            out [ element_index9 ]  = j;
            out [ element_index10 ] = k;
            out [ element_index11 ] = l;

            out [ element_index12 ] = m;
            out [ element_index13 ] = n;
            out [ element_index14 ] = o;
            out [ element_index15 ] = p;
        }


        __global__ void prefilter4_vertical( const transforms::pixel* in, transforms::pixel* out, const uint32_t pitch, const uint32_t width, const uint32_t height)
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

            auto a = in [ element_index0 ];
            auto b = in [ element_index1 ];

            auto c = in [ element_index2 ];
            auto d = in [ element_index3 ];

            transforms::analysis::prefilter4 ( &a, &b, &c, &d );

            out [ element_index0 ] = a;
            out [ element_index1 ] = b;
            out [ element_index2 ] = c;
            out [ element_index3 ] = d;
        }

        __global__ void prefilter4_horizontal( const transforms::pixel* in, transforms::pixel* out, const uint32_t pitch, const uint32_t width, const uint32_t height)
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

            auto a = in [ element_index0 ];
            auto b = in [ element_index1 ];

            auto c = in [ element_index2 ];
            auto d = in [ element_index3 ];

            transforms::analysis::prefilter4 ( &a, &b, &c, &d );

            out [ element_index0 ] = a;
            out [ element_index1 ] = b;
            out [ element_index2 ] = c;
            out [ element_index3 ] = d;
        }

        __global__ void prefilter2x2_edge( const transforms::pixel* in, transforms::pixel* out, const uint32_t pitch, const uint32_t width, const uint32_t height)
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

            auto a = in [ element_index0 ];
            auto b = in [ element_index1 ];

            auto c = in [ element_index2 ];
            auto d = in [ element_index3 ];

            transforms::analysis::prefilter2x2( &a, &b, &c, &d );

            out [ element_index0 ] = a;
            out [ element_index1 ] = b;
            out [ element_index2 ] = c;
            out [ element_index3 ] = d;
        }
    }

    inline void prefilter4x4 (  std::shared_ptr< ::cuda::memory_buffer > in, uint32_t width, uint32_t height, uint32_t pitch ) 
    {
        filter_image( *in, *in, pitch, width, height, [=] 
        ( dim3 blocks, dim3 threads_per_block, const transforms::pixel* in, transforms::pixel* out, const uint32_t image_pitch, const uint32_t width, const uint32_t height )
        {
            jpegxr::cuda::prefilter_4x4<<<blocks, threads_per_block>>>( in, out, image_pitch, width, height );
        });
    }

    inline void prefilter4_vertical(  std::shared_ptr< ::cuda::memory_buffer > in, uint32_t width, uint32_t height, uint32_t pitch ) 
    {
        filter_image( *in, *in, pitch, width, height, [=] 
        ( dim3 blocks, dim3 threads_per_block, const transforms::pixel* in, transforms::pixel* out, const uint32_t image_pitch, const uint32_t width, const uint32_t height )
        {
            jpegxr::cuda::prefilter4_vertical<<<blocks, threads_per_block>>>( in, out, image_pitch, width, height );
        });
        }

    inline void prefilter4_horizontal(  std::shared_ptr< ::cuda::memory_buffer > in, uint32_t width, uint32_t height, uint32_t pitch ) 
    {
        filter_image( *in, *in, pitch, width, height, [=] 
        ( dim3 blocks, dim3 threads_per_block, const transforms::pixel* in, transforms::pixel* out, const uint32_t image_pitch, const uint32_t width, const uint32_t height )
        {
            jpegxr::cuda::prefilter4_horizontal<<<blocks, threads_per_block>>>( in, out, image_pitch, width, height );
        });
    }

    inline void prefilter2x2_edge (  std::shared_ptr< ::cuda::memory_buffer > in, uint32_t width, uint32_t height, uint32_t pitch ) 
    {
        filter_image( *in, *in, pitch, width, height, [=] 
        ( dim3 blocks, dim3 threads_per_block, const transforms::pixel* in, transforms::pixel* out, const uint32_t image_pitch, const uint32_t width, const uint32_t height )
        {
            jpegxr::cuda::prefilter2x2_edge<<<blocks, threads_per_block>>>( in, out, image_pitch, width, height );
        });
    }
}

#endif

