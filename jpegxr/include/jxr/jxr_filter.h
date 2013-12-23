#ifndef __jxr_filter_h__
#define __jxr_filter_h__

#include <cstdint>
#include <memory>

#include <jxr/cuda_helper.h>

namespace jpegxr
{
    template <typename functor > void filter_image( const uint32_t* in, uint32_t* out, const uint32_t image_pitch, const uint32_t image_width, const uint32_t image_height, functor f )
    {
        auto w                  = image_width;
        auto h                  = image_height;
        auto pitch              = image_pitch;
        auto size               = w * h * sizeof(int32_t) ;

        auto blocks             = dim3 ( ( w + 15 )  / 16 , ( h + 15 ) / 16, 1 );
        auto threads_per_block  = dim3 ( 16,  16,  1 );

        ::cuda::throw_if_failed<::cuda::exception> ( cudaMemset( out, 0, size ) );

        f( blocks, threads_per_block, in, out, pitch, w, h ); 

        ::cuda::throw_if_failed<::cuda::exception> ( cudaGetLastError() );
        ::cuda::throw_if_failed<::cuda::exception> ( cudaDeviceSynchronize() );

        auto y  = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );

        ::cuda::throw_if_failed<::cuda::exception> ( cudaMemcpy( y.get(), in, size , cudaMemcpyDeviceToHost) );

        auto res = reinterpret_cast< uint32_t* > ( y.get() );
        int i  = 0; i++;
    }
}

#endif

