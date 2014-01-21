#ifndef __img_utils_h__
#define __img_utils_h__

#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>



#include <jxr/cuda_helper.h>

#include <jxr/jxr_transforms.h>

#include "img_images.h"

namespace example
{
    inline const jpegxr::transforms::pixel* get_pixels( const image_2d& image )
    {
        return reinterpret_cast<const jpegxr::transforms::pixel*> ( get_data(image) );
    }

    inline jpegxr::transforms::pixel* get_pixels( image_2d& image )
    {
        return reinterpret_cast<jpegxr::transforms::pixel*> ( get_data(image) );
    }

    inline jpegxr::transforms::pixel* get_pixels( const std::shared_ptr<image_2d> image )
    {
        return reinterpret_cast<jpegxr::transforms::pixel*> ( get_data(image) );
    }

    __global__ void make_test_image_kernel( jpegxr::transforms::pixel* pixels, const uint32_t pixel_value, const uint32_t width, const uint32_t height, const uint32_t write_pitch )
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        auto row = y;
        auto col = x;

        if (  row > ( height - 1) )
        {
            return;
        }

        if ( col > ( width - 1 ) ) 
        {
            return;
        }

        pixels [ row * write_pitch + col ] = pixel_value;
    }

    __global__ void make_test_image_kernel_2x2( jpegxr::transforms::pixel* pixels, const uint32_t lt, const uint32_t rt, const uint32_t lb, const uint32_t rb, const uint32_t width, const uint32_t height, const uint32_t write_pitch )
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        auto row = 2 * y;
        auto col = 2 * x;

        if (  row > ( height - 1) )
        {
            return;
        }

        if ( col > ( width - 1 ) ) 
        {
            return;
        }

        pixels [ row * write_pitch + col ] = lt;
        pixels [ row * write_pitch + col + 1] = rt;

        pixels [ (row + 1) * write_pitch + col ] = lb;
        pixels [ (row + 1) * write_pitch + col + 1 ] = rb;
    }

    __global__ void make_test_image_kernel_4x4( jpegxr::transforms::pixel* pixels, const jpegxr::transforms::pixel* pixel_values, const uint32_t width, const uint32_t height, const uint32_t write_pitch )
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        auto row = 4 * y;
        auto col = 4 * x;

        if (  row > ( height - 1) )
        {
            return;
        }

        if ( col > ( width - 1 ) ) 
        {
            return;
        }

        pixels [ row * write_pitch + col + 0] = pixel_values[0];
        pixels [ row * write_pitch + col + 1] = pixel_values[1];
        pixels [ row * write_pitch + col + 2] = pixel_values[2];
        pixels [ row * write_pitch + col + 3] = pixel_values[3];

        pixels [ (row + 1) * write_pitch + col + 0] = pixel_values[4];
        pixels [ (row + 1) * write_pitch + col + 1] = pixel_values[5];
        pixels [ (row + 1) * write_pitch + col + 2] = pixel_values[6];
        pixels [ (row + 1) * write_pitch + col + 3] = pixel_values[7];

        pixels [ (row + 2) * write_pitch + col + 0] = pixel_values[8];
        pixels [ (row + 2) * write_pitch + col + 1] = pixel_values[9];
        pixels [ (row + 2) * write_pitch + col + 2] = pixel_values[10];
        pixels [ (row + 2) * write_pitch + col + 3] = pixel_values[11];

        pixels [ (row + 3) * write_pitch + col + 0] = pixel_values[12];
        pixels [ (row + 3) * write_pitch + col + 1] = pixel_values[13];
        pixels [ (row + 3) * write_pitch + col + 2] = pixel_values[14];
        pixels [ (row + 3) * write_pitch + col + 3] = pixel_values[15];

    }

    inline std::pair< dim3, dim3> make_threads_blocks_16 ( uint32_t pixel_width, uint32_t pixel_height )
    {
        auto w = pixel_width;
        auto h = pixel_height;

        return std::make_pair( dim3 ( w, h,  1 ), dim3 ( ( w + 15 )  / 16 , ( h + 15 ) / 16, 1 ) );
    }

    inline std::shared_ptr< image_2d > make_test_image( uint32_t width, uint32_t height, jpegxr::transforms::pixel pixel_value)
    {
        auto image_size = width * height * sizeof(jpegxr::transforms::pixel);

        auto w                  = width;
        auto h                  = height;
        auto pitch              = w;

        auto kernel_params      = make_threads_blocks_16( w, h );
        

        auto buffer             = cuda::make_memory_buffer (  image_size );

        make_test_image_kernel<<< std::get<0>( kernel_params), std::get<1>(kernel_params) >>> ( *buffer, pixel_value, w, h, pitch );

        return make_image_2d( buffer, width, height, width );
    }

    inline std::shared_ptr< image_2d > make_test_image_2x2( uint32_t width, uint32_t height, jpegxr::transforms::pixel lt, jpegxr::transforms::pixel rt, jpegxr::transforms::pixel lb, jpegxr::transforms::pixel rb)
    {
        auto image_size = width * height * sizeof(jpegxr::transforms::pixel);

        auto w                  = width;
        auto h                  = height;
        auto pitch              = w;

        auto kernel_params      = make_threads_blocks_16( w, h );
        

        auto buffer             = cuda::make_memory_buffer (  image_size );

        make_test_image_kernel_2x2<<< std::get<0>( kernel_params), std::get<1>(kernel_params) >>> ( *buffer, lt, rt, lb, rb, w, h, pitch );

        ::cuda::throw_if_failed<::cuda::exception> ( cudaGetLastError() );
        ::cuda::throw_if_failed<::cuda::exception> ( cudaDeviceSynchronize() );

        return make_image_2d( buffer, width, height, width );
    }


    inline std::shared_ptr< image_2d > make_test_image_4x4( uint32_t width, uint32_t height, const jpegxr::transforms::pixel pixels[16])
    {
        auto image_size = width * height * sizeof(jpegxr::transforms::pixel);

        auto w                  = width;
        auto h                  = height;
        auto pitch              = w;

        auto kernel_params      = make_threads_blocks_16( w, h );
        

        auto buffer_out         = cuda::make_memory_buffer (  image_size );
        auto buffer_in          = cuda::make_memory_buffer_host ( sizeof(jpegxr::transforms::pixel) * 16 , pixels );

        make_test_image_kernel_4x4<<< std::get<0>( kernel_params), std::get<1>(kernel_params) >>> ( *buffer_out, *buffer_in, w, h, pitch );
        
        ::cuda::throw_if_failed<::cuda::exception> ( cudaGetLastError() );
        ::cuda::throw_if_failed<::cuda::exception> ( cudaDeviceSynchronize() );

        return make_image_2d( buffer_out, width, height, width );
    }

    inline std::shared_ptr< image_2d > make_test_image_linear_4x4( uint32_t width, uint32_t height )
    {
        const jpegxr::transforms::pixel pixels[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        return make_test_image_4x4(width, height, pixels);
    }

    inline std::shared_ptr< image_2d > make_zero_image( uint32_t width, uint32_t height, jpegxr::transforms::pixel pixel_value)
    {
        return make_test_image( width, height, 0 );
    }

    inline void print_image( std::shared_ptr<image_2d> image )
    {
        auto size               = image->get_width() * image->get_height() * sizeof(jpegxr::transforms::pixel) ;
        auto y                  = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );

        ::cuda::throw_if_failed<::cuda::exception> ( cudaMemcpy( y.get(), get_pixels(image), size , cudaMemcpyDeviceToHost) );

        auto ptr = reinterpret_cast<jpegxr::transforms::pixel*> (&y[0]);

        for( uint32_t i = 0; i < image->get_height(); ++i )
        {
            for (uint32_t j = 0; j < image->get_width(); ++j)
            {
                std::cout << *( ptr++ ) <<"\t";

                if ( j == image->get_width() - 1 )
                {
                    std::cout<<std::endl;
                }
            }
        }
    }

}

#endif
