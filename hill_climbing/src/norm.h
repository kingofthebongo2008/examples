#ifndef __norm_h__
#define __norm_h__

#include <thrust/transform_reduce.h>

namespace cuda_rt
{
    namespace details
    {
        struct norm_l2_op : public thrust::unary_function<float, float>
        {
            __host__ __device__  float operator()(float f) const
            {
                return f * f;
            }
        };

        typedef thrust::tuple<float, float> f2;

        struct mse_op : public thrust::unary_function< float, f2 >
        {
            __host__ __device__  float operator()( const f2& f) const
            {
                float difference = thrust::get<0>(f) - thrust::get<1>(f) ;
                return ( difference ) * ( difference ) ;
            }
        };
    }    

    template <typename iterator> float norm_l2( iterator begin, iterator end ) 
    {
        float result = sqrtf( thrust::transform_reduce(begin, end, details::norm_l2_op(), 0.0f, thrust::plus<float>() ) );
        return result;
    }

    //mean square error
    template <typename iterator> float mse( iterator begin1, iterator end1, iterator begin2, iterator end2 )
    {
        auto z0 = thrust::make_zip_iterator ( thrust::make_tuple( begin1, begin2 ) );
        auto z1 = thrust::make_zip_iterator ( thrust::make_tuple( end1, end2 ) );

        float result = ( thrust::transform_reduce(z0, z1, details::mse_op(), 0.0f, thrust::plus<float>() ) ) / ( end1 - begin1 );

        return result;
    }

    //peak signal to noise ratio
    inline float psnr( float mse, float max )
    {
        return 10 * log10f( (max * max) / mse );
    }
}




#endif