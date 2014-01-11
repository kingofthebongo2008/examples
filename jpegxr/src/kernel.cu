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

        using namespace jpegxr::transforms;

        rgb_2_ycocg(&r_y, &g_co, &b_cg );

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


    __global__ void decompose_yuv_kernel( const rgb* in, uint32_t* y_color, uint32_t* u_color, uint32_t* v_color, const uint32_t read_pitch, const uint32_t write_pitch )
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        auto row = y;
        auto col = x;
        
        auto element = reinterpret_cast<const rgb*> (  (uint8_t*) in + ( row * read_pitch )  + sizeof(rgb) * col ); 

        jpegxr::transforms::pixel r_y  = element->color[0];
        jpegxr::transforms::pixel g_u  = element->color[1];
        jpegxr::transforms::pixel b_v  = element->color[2];

        using namespace jpegxr::transforms;

        rgb_2_yuv(&r_y, &g_u, &b_v );

        y_color[ row * write_pitch + col ] = r_y;
        u_color[ row * write_pitch + col ] = g_u;
        v_color[ row * write_pitch + col ] = b_v;
    }

    
    ycbcr_image decompose_yuv ( const image& image ) 
    {
        auto w         = image.get_width();
        auto h         = image.get_height();
        auto size      = w * h * sizeof(int32_t) ;
        
        auto y_buffer  = std::make_shared < cuda::memory_buffer > ( cuda::allocate<void*> ( size ) );
        auto u_buffer = std::make_shared < cuda::memory_buffer > ( cuda::allocate<void*> ( size ) );
        auto v_buffer = std::make_shared < cuda::memory_buffer > ( cuda::allocate<void*> ( size ) );

        auto blocks = 1;
        auto threads_per_block = dim3( w, h );

        decompose_ycocg_kernel<<<blocks, threads_per_block>>>( image, *y_buffer, *u_buffer, *v_buffer, image.get_pitch(), w );

        cuda::throw_if_failed<cuda::exception> ( cudaGetLastError() );
        cuda::throw_if_failed<cuda::exception> ( cudaDeviceSynchronize() );

        //debug purposes
        auto y  = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );
        auto co = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );
        auto cg = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );

        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( y.get(),  y_buffer->get(),  size   , cudaMemcpyDeviceToHost) );
        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( co.get(), u_buffer->get(), size   , cudaMemcpyDeviceToHost) );
        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( cg.get(), v_buffer->get(), size   , cudaMemcpyDeviceToHost) );

        // element access into this image looks like this
        auto row = 15;
        auto col = 15;
        auto res1 = reinterpret_cast<int32_t*> ( y.get()  );
        auto res2 = reinterpret_cast<int32_t*> ( co.get() );
        auto res3 = reinterpret_cast<int32_t*> ( cg.get() );
        auto el1 = res1[ row * w + col ];

        return ycbcr_image ( y_buffer, u_buffer, v_buffer, w, h, w );
    }
}

static void block_shuffle444(int*data)
{
    int32_t tmp[256];

    int32_t idx;
    for (idx = 0 ; idx < 256 ; idx += 4) {
        int blk = idx/16;
        int mbx = blk%4;
        int mby = blk/4;
        int pix = idx%16;
        int py = pix/4;

        int ptr = 16*4*mby + 4*mbx + 16*py;
        tmp[idx+0] = data[ptr+0];
        tmp[idx+1] = data[ptr+1];
        tmp[idx+2] = data[ptr+2];
        tmp[idx+3] = data[ptr+3];
    }

    for (idx = 0 ; idx < 256 ; idx += 1)
        data[idx] = tmp[idx];
}

static void unblock_shuffle444(int*data)
{
    int tmp[256];

    int idx;
    for (idx = 0 ; idx < 256 ; idx += 4) {
        int blk = idx/16;
        int mbx = blk%4;
        int mby = blk/4;
        int pix = idx%16;
        int py = pix/4;

        int ptr = 16*4*mby + 4*mbx + 16*py;
        tmp[ptr+0] = data[idx+0];
        tmp[ptr+1] = data[idx+1];
        tmp[ptr+2] = data[idx+2];
        tmp[ptr+3] = data[idx+3];
    }

    for (idx = 0 ; idx < 256 ; idx += 1)
        data[idx] = tmp[idx];
}

int32_t main()
{
    try
    {
        auto com_initializer  =  os::windows::com_initializer();
        auto cuda_initializer = example::cuda_initializer();
        auto image  =  example::create_image ( L"test_32x32.png" );

        int32_t data[256];

        for (int32_t i = 0; i < 256; ++i )
        {
            data[i] = i;
        }

        block_shuffle444(data);
        unblock_shuffle444(data);

        for (int32_t i = 0; i <255; ++i)
        {
            for (int32_t j = 0; j < 255; ++j)
            {
                for (int32_t k = 0; k < 255; ++k )
                {
                    auto r0 = i;
                    auto g0 = j;
                    auto b0 = k;

                    auto r1 = i;
                    auto g1 = j;
                    auto b1 = k;

                    using namespace jpegxr::transforms;

                    rgb_2_yuv(&r0, &g0, &b0 );
                    rgb_2_ycocg(&r1, &g1, &b1 );

                    if (! (  r0 == r1 && g0==g1 && b0==b1) )
                    {
                        //__debugbreak();
                    }
                }
            }
        }


        auto yuv = decompose_ycocg(*image);

        auto w      = yuv.get_width();
        auto h      = yuv.get_height();
        auto pitch  = yuv.get_width();
        auto size   = w * h * sizeof(jpegxr::transforms::pixel) ;

        auto copy_of_y_1  = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );
        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( copy_of_y_1.get(),  *yuv.get_y(), size   , cudaMemcpyDeviceToHost) );

        jpegxr::prefilter2x2_edge( yuv.get_y(), w, h, pitch );
        jpegxr::prefilter4x4( yuv.get_y(), w, h, pitch );
        jpegxr::prefilter4_horizontal( yuv.get_y(), w, h, pitch );
        jpegxr::prefilter4_vertical( yuv.get_y(), w, h, pitch );
        jpegxr::pct4x4( yuv.get_y(), w, h, pitch );

        jpegxr::ipct4x4( yuv.get_y(), w, h, pitch );
        jpegxr::overlapfilter2x2_edge( yuv.get_y(), w, h, pitch );
        jpegxr::overlapfilter4x4( yuv.get_y(), w, h, pitch );
        jpegxr::overlapfilter4_horizontal( yuv.get_y(), w, h, pitch );
        jpegxr::overlapfilter4_vertical( yuv.get_y(), w, h, pitch );

        auto copy_of_y_2  = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );
        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( copy_of_y_2.get(),  *yuv.get_y(), size   , cudaMemcpyDeviceToHost) );

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
