#ifndef __jxr_filter_h__
#define __jxr_filter_h__

#include <cstdint>
#include <memory>

#include <jxr/cuda_helper.h>
#include <jxr/jxr_transforms.h>

namespace jpegxr
{
    template <typename functor > inline void filter_image( const transforms::pixel* in, transforms::pixel* out, const uint32_t image_pitch, const uint32_t image_width, const uint32_t image_height, functor f )
    {
        auto w                  = image_width;
        auto h                  = image_height;
        auto pitch              = image_pitch;

        auto blocks             = dim3 ( ( w + 15 )  / 16 , ( h + 15 ) / 16, 1 );
        auto threads_per_block  = dim3 ( 16,  16,  1 );

        //debug purposes
        //::cuda::throw_if_failed<::cuda::exception> ( cudaMemset( out, 0, size ) );

        f( blocks, threads_per_block, in, out, pitch, w, h ); 

        ::cuda::throw_if_failed<::cuda::exception> ( cudaGetLastError() );
        ::cuda::throw_if_failed<::cuda::exception> ( cudaDeviceSynchronize() );

        //debug purposes
        /*
        auto size               = w * h * sizeof(jpegxr::transforms::pixel) ;
        auto y                  = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );

        ::cuda::throw_if_failed<::cuda::exception> ( cudaMemcpy( y.get(), in, size , cudaMemcpyDeviceToHost) );
        auto res = reinterpret_cast< jpegxr::transforms::pixel* > ( y.get() );
        */
    }
}

#endif

