#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

#include <jxr/cuda_helper.h>
#include <jxr/jxr_transforms.h>
#include <jxr/jxr_analysis.h>

#include <jxr/jxr_filter.h>
#include <jxr/jxr_prefilter.h>
#include <jxr/jxr_overlapfilter.h>
#include <jxr/jxr_pct.h>
#include <jxr/jxr_decompose.h>

#include <os/windows/com_initializer.h>

#include "img_images.h"
#include "img_loader.h"

namespace example
{
    class cuda_initializer
    {
        public:
        cuda_initializer()
        {
            // Choose which GPU to run on, change this on a multi-GPU system.
            cuda::throw_if_failed<cuda::exception> (  cudaSetDevice(0) );
        }

        ~cuda_initializer()
        {
            // cudaDeviceReset must be called before exiting in order for profiling and
            // tracing tools such as Nsight and Visual Profiler to show complete traces.
            cuda::throw_if_failed<cuda::exception> ( cudaDeviceReset() );
        }
    };
}

namespace example
{
    __global__ void addKernel( int32_t * c, const int32_t * a )
    {
        int i = threadIdx.x;

        jpegxr::transforms::pixel v[16] =
        { 
            a[0], a[1], a[2], a[3],
            a[4], a[5], a[6], a[7],
            a[8], a[9], a[10], a[11],
            a[12], a[13], a[14], a[15]
        };

        jpegxr::transforms::analysis::pct4x4
            (
               v
            );

        c[i]   = v[0];
        c[i+1] = v[1];
        c[i+2] = v[2];
        c[i+3] = v[3];

        c[i+4] = v[4];
        c[i+5] = v[5];
        c[i+6] = v[6];
        c[i+7] = v[7];

        c[i+8] = v[8];
        c[i+9] = v[9];
        c[i+10] = v[10];
        c[i+11] = v[11];

        c[i+12] = v[12];
        c[i+13] = v[13];
        c[i+14] = v[14];
        c[i+15] = v[15];
    }

    struct rgb 
    {
        uint8_t color[3];
    };

    __global__ void decompose_ycocg_kernel( const rgb* in, uint32_t* y_color, uint32_t* co_color, uint32_t* cg_color, const uint32_t read_pitch, const uint32_t write_pitch )
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        auto row = y;
        auto col = x;
        
        auto element = reinterpret_cast<const rgb*> (  (uint8_t*) in + ( row * read_pitch )  + sizeof(rgb) * col ); 

        jpegxr::transforms::pixel r_y  = element->color[0];
        jpegxr::transforms::pixel g_co = element->color[1];
        jpegxr::transforms::pixel b_cg = element->color[2];

        jpegxr::transforms::rgb_2_ycocg(&r_y, &g_co, &b_cg );

        y_color [ row * write_pitch + col ] = r_y;
        co_color[ row * write_pitch + col ] = g_co;
        cg_color[ row * write_pitch + col ] = b_cg;
    }

    
    ycocg_image decompose_ycocg ( const image& image ) 
    {
        auto w         = image.get_width();
        auto h         = image.get_height();
        auto size      = w * h * sizeof(int32_t) ;
        
        auto y_buffer  = std::make_shared < cuda::memory_buffer > ( cuda::allocate<void*> ( size ) );
        auto co_buffer = std::make_shared < cuda::memory_buffer > ( cuda::allocate<void*> ( size ) );
        auto cg_buffer = std::make_shared < cuda::memory_buffer > ( cuda::allocate<void*> ( size ) );

        auto blocks = 1;
        auto threads_per_block = dim3( w, h );

        decompose_ycocg_kernel<<<blocks, threads_per_block>>>( image, *y_buffer, *co_buffer, *cg_buffer, image.get_pitch(), w );

        cuda::throw_if_failed<cuda::exception> ( cudaGetLastError() );
        cuda::throw_if_failed<cuda::exception> ( cudaDeviceSynchronize() );

        //debug purposes
        auto y  = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );
        auto co = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );
        auto cg = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );

        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( y.get(),  y_buffer->get(),  size   , cudaMemcpyDeviceToHost) );
        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( co.get(), co_buffer->get(), size   , cudaMemcpyDeviceToHost) );
        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( cg.get(), cg_buffer->get(), size   , cudaMemcpyDeviceToHost) );

        // element access into this image looks like this
        auto row = 15;
        auto col = 15;
        auto res1 = reinterpret_cast<int32_t*> ( y.get()  );
        auto res2 = reinterpret_cast<int32_t*> ( co.get() );
        auto res3 = reinterpret_cast<int32_t*> ( cg.get() );
        auto el1 = res1[ row * w + col ];

        return ycocg_image ( y_buffer, co_buffer, cg_buffer, w, h, w );
    }
}

int32_t main()
{
    try
    {
        auto com_initializer  =  os::windows::com_initializer();
        auto cuda_initializer = example::cuda_initializer();
        auto image  =  example::create_image ( L"test_32x32.png" );

        const int32_t arraySize = 16;
        const jpegxr::transforms::pixel a[arraySize] = 
        { 
            0, 0, 0, 0,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1
        };

        jpegxr::transforms::pixel c[arraySize] = { 0 };

        #define _CC(r, g, b) (b -= r, r += ((b + 1) >> 1) - g, g += ((r + 0) >> 1))

        for (int32_t i = 0; i < 255; ++i)
        {
            for (int32_t j = 0; j <255; ++j)
            {
                for (int32_t k = 0; k < 255; ++k )
                {
                    auto r = i;
                    auto g = j;
                    auto b = k;

                    /*
                    auto r = 144;
                    auto g = 126;
                    auto b = 47;

                    auto r1 = 144;
                    auto g1 = 126;
                    auto b1 = 47;

                    _CC(r1, g1, b1);

                    //auto  pU[iPos] = -r, pV[iPos] = b, pY[iPos] = g - iOffset;
                    auto  y = g1 - 128;
                    auto  u = -r1;
                    auto  v = b1;
                    
                    */
                    using namespace jpegxr::transforms;

                    rgb_2_yuv< scale::no_scale, bias::bd8 >(&r, &g, &b);
                    yuv_2_rgb< scale::no_scale, bias::bd8 >(&r, &g, &b);

                    if  ( !( r == i && g == j && b == k ) )
                    {
                        throw std::exception("error");
                    }


                }
            }
        }

        auto ycocg = decompose_ycocg(*image);

        auto w      = ycocg.get_width();
        auto h      = ycocg.get_height();
        auto pitch  = ycocg.get_width();
        auto size   = w * h * sizeof(jpegxr::transforms::pixel) ;

        auto copy_of_y_1  = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );
        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( copy_of_y_1.get(),  *ycocg.get_y(), size   , cudaMemcpyDeviceToHost) );

        jpegxr::prefilter2x2_edge( ycocg.get_y(), w, h, pitch );
        jpegxr::prefilter4x4( ycocg.get_y(), w, h, pitch );
        jpegxr::prefilter4_horizontal( ycocg.get_y(), w, h, pitch );
        jpegxr::prefilter4_vertical( ycocg.get_y(), w, h, pitch );
        jpegxr::pct4x4( ycocg.get_y(), w, h, pitch );

        jpegxr::ipct4x4( ycocg.get_y(), w, h, pitch );
        jpegxr::overlapfilter2x2_edge( ycocg.get_y(), w, h, pitch );
        jpegxr::overlapfilter4x4( ycocg.get_y(), w, h, pitch );
        jpegxr::overlapfilter4_horizontal( ycocg.get_y(), w, h, pitch );
        jpegxr::overlapfilter4_vertical( ycocg.get_y(), w, h, pitch );

        auto copy_of_y_2  = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );
        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( copy_of_y_2.get(),  *ycocg.get_y(), size   , cudaMemcpyDeviceToHost) );

        auto result = std::memcmp ( copy_of_y_1.get(), copy_of_y_2.get(), size );

        if (result == 0 )
        {
            std::cout <<"Prefect reconstruction." << std::endl;
        }
        else
        {
            std::cerr <<"Error in reconstruction." << std::endl;
        }
    }

    catch (const cuda::exception& e)
    {
        std::cerr<<e.what()<<std::endl;
        return 1;
    }

    return 0;
}
