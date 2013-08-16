// hill_climbing.cpp : Defines the entry point for the console application.
//

#include "precompiled.h"
#include <cstdint>

#include <vector>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/functional.h>

#include <thrust/iterator/zip_iterator.h>

#include <vector_functions.h>

#include "morton_order.h"

#include "strided_range_iterator.h"

struct grayscale_image
{
    uint32_t    m_width;
    uint32_t    m_height;
    uint32_t    m_pitch;

    std::vector< uint8_t > m_image;
};

typedef thrust::tuple< float,float, float, float > f4;

static inline __host__ __device__ float dot3( float3 a, float3 b )
{
    return a.x * b.x + a.y + b.y + a.z + b.z;
}

//haar transform on 4x4
struct haar_2d_transform : public thrust::unary_function< f4, f4 >
{
    __host__ __device__ f4 operator() ( const f4& o ) const
    {
        float a = thrust::get<0> ( o );
        float b = thrust::get<1> ( o );
        float c = thrust::get<2> ( o );
        float d = thrust::get<3> ( o );

        float ll = 0.5f *  ( a + b + c + d );
        float hl = 0.5f *  ( a - b + c - d );
        float lh = 0.5f *  ( a + b - c - d );
        float hh = 0.5f *  ( a - b - c - d );

        return thrust::make_tuple ( ll, hl, lh, hh );
    }
};

//partially inverted haar transform on 4x4
struct haar_pi_2d_transform : public thrust::unary_function< f4, f4 >
{

    haar_pi_2d_transform( float w ) : m_w ( w )
    {

    }

    __host__ __device__ f4 operator() ( const f4& o ) const
    {
        float a = thrust::get<0> ( o );
        float b = thrust::get<1> ( o );
        float c = thrust::get<2> ( o );
        float d = thrust::get<3> ( o );

        //haar 2d transform
        float ll = 0.5f *  ( a + b + c + d );
        float hl = 0.5f *  ( a - b + c - d );
        float lh = 0.5f *  ( a + b - c - d );
        float hh = 0.5f *  ( a - b - c - d );

        float3 r0 = make_float3( 1.0f, 1.0f, m_w );
        float3 v = make_float3( hl, lh, hh );

        //partially inverted haar transform
        float hlp = dot3 ( make_float3(  1.0f, 1.0f,  m_w ), v );
        float lhp = dot3 ( make_float3( -1.0f, 1.0f, -m_w ), v );
        float hhp = dot3 ( make_float3( 1.0f, -1.0f, -m_w ), v );
            
        return thrust::make_tuple ( ll, hlp, lhp, hhp );
    }

    float   m_w;
};

typedef thrust::tuple<float, float, float, float > Float4;

struct arbitrary_functor : public thrust::unary_function< Float4, Float4 >
{
    __host__ __device__
    Float4 operator()(const Float4& t ) const
    {
        float a = thrust::get<0>(t);
        float b = thrust::get<1>(t);
        float c = thrust::get<2>(t);
        float d = thrust::get<3>(t);

        return thrust::make_tuple ( a, b, c, d );
    }
};

int main(int argc, _TCHAR* argv[])
{
    // generate 20 random numbers on the host
    thrust::host_vector<float> h_vec(16);
    thrust::host_vector<float> h_vec_out(16);

    thrust::host_vector<float> h_vec_transformed(16);
    
    h_vec[0] = 0.0f;
    h_vec[1] = 1.0f;
    h_vec[2] = 2.0f;
    h_vec[3] = 3.0f;

    h_vec[4] = 4.0f;
    h_vec[5] = 5.0f;
    h_vec[6] = 6.0f;
    h_vec[7] = 7.0f;

    h_vec[8] = 8.0f;
    h_vec[9] = 9.0f;
    h_vec[10] = 10.0f;
    h_vec[11] = 11.0f;

    h_vec[12] = 12.0f;
    h_vec[13] = 13.0f;
    h_vec[14] = 14.0f;
    h_vec[15] = 15.0f;

    // interface to CUDA code
    convert_to_morton_order_2d( h_vec, 4, 4, h_vec_out);

    auto b0  = h_vec.begin();
    auto b1  = h_vec.begin() + 1;
    auto b2  = h_vec.begin() + 2;
    auto b3  = h_vec.begin() + 3;

    auto e0  = h_vec.end()-3;
    auto e1  = h_vec.end()-2;
    auto e2  = h_vec.end()-1;
    auto e3  = h_vec.end();

    auto itb0  = make_strided_range ( b0, e0, 4 );
    auto itb1  = make_strided_range ( b1, e1, 4 );
    auto itb2  = make_strided_range ( b2, e2, 4 );
    auto itb3  = make_strided_range ( b3, e3, 4 );

    //------

    auto bb0  = h_vec_transformed.begin();
    auto bb1  = h_vec_transformed.begin() + 1;
    auto bb2  = h_vec_transformed.begin() + 2;
    auto bb3  = h_vec_transformed.begin() + 3;

    auto ee0  = h_vec_transformed.end()-3;
    auto ee1  = h_vec_transformed.end()-2;
    auto ee2  = h_vec_transformed.end()-1;
    auto ee3  = h_vec_transformed.end();

    auto itbb0  = make_strided_range ( bb0, ee0, 4 );
    auto itbb1  = make_strided_range ( bb1, ee1, 4 );
    auto itbb2  = make_strided_range ( bb2, ee2, 4 );
    auto itbb3  = make_strided_range ( bb3, ee3, 4 );

    auto z0 = thrust::make_zip_iterator ( thrust::make_tuple( itb0.begin(), itb1.begin(), itb2.begin(), itb3.begin() ) );
    auto z1 = thrust::make_zip_iterator ( thrust::make_tuple( itb0.end(),   itb1.end(), itb2.end(), itb3.end()  ) );
    auto z2 = thrust::make_zip_iterator ( thrust::make_tuple( itbb0.begin(), itbb1.begin(), itbb2.begin(), itbb3.begin() ) );

    thrust::transform ( z0, z1, z2, arbitrary_functor() );

    // print sorted array
    thrust::copy(h_vec_transformed.begin(), h_vec_transformed.end(), std::ostream_iterator<int>(std::cout, "\n"));

    return 0;
}

