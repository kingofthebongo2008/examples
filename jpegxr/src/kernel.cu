#include "precompiled.h"

#include "img_utils.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>

#include <util/util_memory.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions_decls.h>
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

    __global__ void scale_decompose_ycocg_kernel( const rgb* in, jpegxr::transforms::pixel* y_color, jpegxr::transforms::pixel* co_color, jpegxr::transforms::pixel* cg_color, const uint32_t read_pitch, const uint32_t write_pitch )
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

        scale_bias_bd8_analysis< no_scale, bd8 >(&r_y, &g_co, &b_cg);
        rgb_2_ycocg(&r_y, &g_co, &b_cg );

        y_color [ row * write_pitch + col ] = r_y;
        co_color[ row * write_pitch + col ] = g_co;
        cg_color[ row * write_pitch + col ] = b_cg;
    }

    std::shared_ptr< ycocg_image > make_ycocg ( std::shared_ptr<image> image ) 
    {
        auto w         = image->get_width();
        auto h         = image->get_height();
        auto size      = w * h * sizeof(int32_t) ;
        
        auto y_buffer  = cuda::make_memory_buffer ( size );
        auto co_buffer = cuda::make_memory_buffer ( size );
        auto cg_buffer = cuda::make_memory_buffer ( size );

        auto blocks = 1;
        auto threads_per_block = dim3( w, h );

        scale_decompose_ycocg_kernel<<<blocks, threads_per_block>>>( reinterpret_cast<rgb*> ( image->get() ), *y_buffer, *co_buffer, *cg_buffer, image->get_pitch(), w );

        cuda::throw_if_failed<cuda::exception> ( cudaGetLastError() );
        cuda::throw_if_failed<cuda::exception> ( cudaDeviceSynchronize() );


        /*
        //debug purposes
        auto y  = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );
        auto co = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );
        auto cg = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );

        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( y.get(),  y_buffer->get(),  size   , cudaMemcpyDeviceToHost) );
        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( co.get(), co_buffer->get(), size   , cudaMemcpyDeviceToHost) );
        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( cg.get(), cg_buffer->get(), size   , cudaMemcpyDeviceToHost) );

        // element access into this image looks like this
        auto res1 = reinterpret_cast<int32_t*> ( y.get()  );
        auto res2 = reinterpret_cast<int32_t*> ( co.get() );
        auto res3 = reinterpret_cast<int32_t*> ( cg.get() );
        */

        return std::make_shared<ycocg_image> ( make_image_2d ( y_buffer, w, h, w ), make_image_2d (co_buffer, w, h, w) , make_image_2d( cg_buffer, w, h, w ) ) ;
    }

    __global__ void scale_compose_ycocg_kernel( const jpegxr::transforms::pixel* y_color, const jpegxr::transforms::pixel* u_color, const jpegxr::transforms::pixel* v_color, rgb* out, const uint32_t read_pitch, const uint32_t write_pitch )
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        auto row = y;
        auto col = x;
        
        auto element = reinterpret_cast<rgb*> (  (uint8_t*) out + ( row * write_pitch )  + sizeof(rgb) * col ); 

        jpegxr::transforms::pixel r_y = y_color[ row * read_pitch + col ];
        jpegxr::transforms::pixel g_u = u_color[ row * read_pitch + col ];
        jpegxr::transforms::pixel b_v = v_color[ row * read_pitch + col ];

        using namespace jpegxr::transforms;

        ycocg_2_rgb(&r_y, &g_u, &b_v );
        scale_bias_bd8_synthesis< no_scale, bd8 >(&r_y, &g_u, &b_v);

        r_y = r_y < 0 ? 0 : r_y;
        g_u = g_u < 0 ? 0 : g_u;
        b_v = b_v < 0 ? 0 : b_v;

        r_y = r_y > 255 ? 255 : r_y;
        g_u = g_u > 255 ? 255 : g_u;
        b_v = b_v > 255 ? 255 : b_v;

        element->color[0] = r_y;
        element->color[1] = g_u;
        element->color[2] = b_v;
    }

    std::shared_ptr< image > make_rgb( std::shared_ptr<ycocg_image> img )
    {
        auto w              = get_y( *img )->get_width();
        auto h              = get_y( *img)->get_height();

        auto rgb_row_pitch  = (w * 24 + 7) / 8; 
        auto rgb_image_size = rgb_row_pitch * h;

       
        auto rgb_buffer     = cuda::make_memory_buffer (  rgb_image_size) ;

        auto blocks = 1;
        auto threads_per_block = dim3( w, h );

        scale_compose_ycocg_kernel<<<blocks, threads_per_block>>>( get_pixels( get_y( img ) ), get_pixels( get_co(img) ), get_pixels( get_cg(img) ), *rgb_buffer, w, rgb_row_pitch );

        cuda::throw_if_failed<cuda::exception> ( cudaGetLastError() );
        cuda::throw_if_failed<cuda::exception> ( cudaDeviceSynchronize() );

        return std::shared_ptr<image> ( new image (image::format_24bpp_rgb, rgb_row_pitch, w, h, std::move(rgb_buffer) ) );
    }

    std::shared_ptr<image_2d> make_low_pass( std::shared_ptr<image_2d> img )
    {
        auto w              = img -> get_width() / 4;
        auto h              = img -> get_height() / 4;
        auto pitch          = w;
        auto image_size     = pitch * h * sizeof(jpegxr::transforms::pixel);

        auto buffer         = cuda::make_memory_buffer ( image_size ) ;

        auto kernel_params      = cuda::make_threads_blocks_16( w, h );

        jpegxr::decompose::split_lp_hp <<< std::get<0>( kernel_params), std::get<1>(kernel_params) >>> ( get_pixels( img ), *buffer,  img->get_pitch(), w, h, pitch ); 

        cuda::throw_if_failed<cuda::exception> ( cudaGetLastError() );
        cuda::throw_if_failed<cuda::exception> ( cudaDeviceSynchronize() );

        return make_image_2d( buffer, w, h, pitch) ;
    }

    __global__ void scale_decompose_yuv_kernel( const rgb* in, uint32_t* y_color, uint32_t* u_color, uint32_t* v_color, const uint32_t read_pitch, const uint32_t write_pitch )
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

        scale_bias_bd8_analysis< no_scale, bd8 >(&r_y, &g_u, &b_v);
        rgb_2_yuv(&r_y, &g_u, &b_v );

        y_color[ row * write_pitch + col ] = r_y;
        u_color[ row * write_pitch + col ] = g_u;
        v_color[ row * write_pitch + col ] = b_v;
    }

    std::shared_ptr<ycbcr_image> make_yuv ( std::shared_ptr<image> image ) 
    {
        auto w         = image->get_width();
        auto h         = image->get_height();
        auto size      = w * h * sizeof(int32_t) ;
        
        auto y_buffer  = cuda::make_memory_buffer( size );
        auto u_buffer  = cuda::make_memory_buffer( size );
        auto v_buffer  = cuda::make_memory_buffer( size );

        auto blocks = 1;
        auto threads_per_block = dim3( w, h );

        scale_decompose_yuv_kernel<<<blocks, threads_per_block>>>( reinterpret_cast<rgb*> ( image->get() ), *y_buffer, *u_buffer, *v_buffer, image->get_pitch(), w );

        cuda::throw_if_failed<cuda::exception> ( cudaGetLastError() );
        cuda::throw_if_failed<cuda::exception> ( cudaDeviceSynchronize() );

        //debug purposes
        /*
        auto y  = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );
        auto co = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );
        auto cg = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );

        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( y.get(),  y_buffer->get(),  size  , cudaMemcpyDeviceToHost) );
        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( co.get(), u_buffer->get(), size   , cudaMemcpyDeviceToHost) );
        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( cg.get(), v_buffer->get(), size   , cudaMemcpyDeviceToHost) );

        // element access into this image looks like this
        auto res1 = reinterpret_cast<int32_t*> ( y.get()  );
        auto res2 = reinterpret_cast<int32_t*> ( co.get() );
        auto res3 = reinterpret_cast<int32_t*> ( cg.get() );
        */
        return std::make_shared<ycbcr_image> ( make_image_2d (y_buffer, w, h, w),  make_image_2d (u_buffer, w, h, w),  make_image_2d (v_buffer, w, h, w)  );
    }

    __global__ void scale_compose_yuv_kernel( const uint32_t* y_color, const uint32_t* u_color, const uint32_t* v_color, rgb* out, const uint32_t read_pitch, const uint32_t write_pitch )
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        auto row = y;
        auto col = x;
        
        auto element = reinterpret_cast<rgb*> (  (uint8_t*) out + ( row * write_pitch )  + sizeof(rgb) * col ); 

        jpegxr::transforms::pixel r_y = y_color[ row * read_pitch + col ];
        jpegxr::transforms::pixel g_u = u_color[ row * read_pitch + col ];
        jpegxr::transforms::pixel b_v = v_color[ row * read_pitch + col ];

        using namespace jpegxr::transforms;

        yuv_2_rgb(&r_y, &g_u, &b_v );
        scale_bias_bd8_synthesis< no_scale, bd8 >(&r_y, &g_u, &b_v);

        r_y = r_y < 0 ? 0 : r_y;
        g_u = g_u < 0 ? 0 : g_u;
        b_v = b_v < 0 ? 0 : b_v;

        r_y = r_y > 255 ? 255 : r_y;
        g_u = g_u > 255 ? 255 : g_u;
        b_v = b_v > 255 ? 255 : b_v;

        element->color[0] = r_y;
        element->color[1] = g_u;
        element->color[2] = b_v;
    }

    std::shared_ptr< image > make_rgb( std::shared_ptr<ycbcr_image> img)
    {
        auto w              = get_y(*img)->get_width();
        auto h              = get_y(*img)->get_height();

        auto rgb_row_pitch  = (w * 24 + 7) / 8; 
        auto rgb_image_size = rgb_row_pitch * h;

        auto rgb_buffer     = cuda::make_memory_buffer (  rgb_image_size );

        auto blocks = 1;
        auto threads_per_block = dim3( w, h );

        scale_compose_yuv_kernel<<<blocks, threads_per_block>>>( reinterpret_cast<uint32_t*> ( get_data ( *get_y(img) ) ), reinterpret_cast<uint32_t*> ( get_data (*get_cb( img ) ) ), reinterpret_cast<uint32_t*> ( get_data ( *get_cr(img) ) ) , *rgb_buffer, w, rgb_row_pitch );

        cuda::throw_if_failed<cuda::exception> ( cudaGetLastError() );
        cuda::throw_if_failed<cuda::exception> ( cudaDeviceSynchronize() );

        return std::make_shared<image> ( image::format_24bpp_rgb, rgb_row_pitch, w, h, std::move(rgb_buffer) ) ;
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

namespace example
{

}

int32_t main()
{
    try
    {
        auto com_initializer  =  os::windows::com_initializer();
        auto cuda_initializer = example::cuda_initializer();
        auto image  =  example::create_image ( L"test_32x32.png" );

        auto image1  =  example::make_test_image_linear_16x16();

        
        print_image(image1);

        return 0;
                
        auto yuv  = make_ycocg(image);

        auto y      = get_y(yuv);
        auto w      = y->get_width();
        auto h      = y->get_height();
        auto pitch  = y->get_width();

        std::cout<<"YCoCg...."<<std::endl;
        print_image ( y  );
        //
        jpegxr::prefilter2x2_edge( *y , w, h, pitch );
        jpegxr::prefilter4x4( *y, w, h, pitch );
        jpegxr::prefilter4_horizontal( *y , w, h, pitch );
        jpegxr::prefilter4_vertical( *y, w, h, pitch );
        jpegxr::pct4x4( *y, w, h, pitch );

        std::cout<<"Stage 1..."<<std::endl;
        print_image ( y  );

        auto lp = make_low_pass( y ) ;

        std::cout<<"Low pass..."<<std::endl;
        print_image ( lp  );

        jpegxr::pct4x4( *lp, lp->get_width(), lp->get_height(), lp->get_pitch() );
        std::cout<<"Second Stage..."<<std::endl;

        print_image ( lp  );

        //auto lp = make_low_pass(yuv);

        jpegxr::ipct4x4( *y, w, h, pitch );
        jpegxr::overlapfilter4_vertical( *y, w, h, pitch );
        jpegxr::overlapfilter4_horizontal( *y, w, h, pitch );
        jpegxr::overlapfilter4x4( *y, w, h, pitch );
        jpegxr::overlapfilter2x2_edge( *y, w, h, pitch );

        auto image_out = make_rgb(yuv);

        //print_image ( example::make_test_image_2x2( 16, 16, 0, 1, 2, 3) );

        if ( cuda::is_equal( image->get_buffer(), image_out->get_buffer() ) )
        {
            std::cout <<"Prefect reconstruction." << std::endl;
        }
        else
        {
            std::cout <<"Error in reconstruction." << std::endl;
        }
    }

    catch (const cuda::exception& e)
    {
        std::cerr<<e.what()<<std::endl;
        return 1;
    }

    return 0;
}
