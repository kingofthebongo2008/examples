﻿#include "precompiled.h"

#include <cstdint>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include <xmmintrin.h>


namespace svd
{
    //given a and b, returns c and s such that, givens rotation
    // [ c  s ] T [ a ]     =  [ r ]
    // [ -s c ]   [ b ]        [ 0 ]

    inline std::tuple<float, float> givens(float a, float b )
    {
        if ( b == 0.0f )
        {
            return std::make_tuple( 1.0f, 0.0f );
        }
        else
        {
            if ( abs(b) > abs(a) )
            {
                auto tau = -a / b;
                auto s = 1.0f / sqrtf( 1 + tau * tau );
                auto c = s * tau;

                return std::make_tuple( c, s );
            }
            else
            {
                auto tau = -b / a;
                auto c = 1 / sqrtf( 1 + tau * tau );
                auto s = c * tau;

                return std::make_tuple( c, s );
            }

        }
    }

    //non trigonometric approximation of the givens angle
    //given coefficients of a symmetric matrix, returns c and s, such that they diagonalize this matrix
    std::tuple<float, float> approximate_givens(float a11, float a12, float a22)
    {
        const auto sqrtf_5 = sqrtf(0.5f); // sin(pi/4), cos (pi/4)
        auto b =  a12 * a12 < ( a11 - a22 ) * ( a11 - a22 );
        auto w =  1.0f / sqrtf( a12 * a12  + ( a11 - a22 ) * ( a11 - a22 ) ) ;
        auto s = b ? w * a12 : sqrtf_5;
        auto c = b ? w * ( a11 - a22 ) : sqrtf_5;
        return std::make_tuple( c, s );
    }

    const float four_gamma_squared      =   static_cast<float> ( sqrt(8.)+3. );
    const float sine_pi_over_eight      =   static_cast<float> ( .5*sqrt( 2. - sqrt(2.) ) );
    const float cosine_pi_over_eight    =   static_cast<float> ( .5*sqrt( 2. + sqrt(2.) ) );
    const float tiny_number             =   static_cast<float> ( 1.e-20 );
    const float small_number            =   static_cast<float> ( 1.e-12 );


    namespace math
    {
        template <typename t> inline t add( t a, t b );
        template <typename t> inline t sub( t a, t b );
        template <typename t> inline t mul( t a, t b );
        template <typename t> inline t madd( t a, t b, t c );
        template <typename t> inline t max( t a, t b );

        template <typename t> inline t rsqrt( t a );

        template <typename t> inline t cmp_ge( t a, t b );
        template <typename t> inline t cmp_le( t a, t b );
        template <typename t> inline t blend(  t a, t b, t mask );

        template <typename t> inline t zero();
        template <typename t> inline t one();

        template <typename t> inline t splat( float f );

        template <typename t> inline t and ( t a, t mask );

        template <typename t> inline t operator+( t a, t b )
        {
            return add ( a, b );
        }

        template <typename t> inline t operator-( t a, t b )
        {
            return sub ( a, b );
        }

        template <typename t> inline t operator*( t a, t b )
        {
            return mul ( a, b );
        }
    }
}


namespace svd
{
    typedef union
    {
        float    f;
        uint32_t u;
    } cpu_scalar;

    cpu_scalar inline make_cpu_scalar(float f )
    {
        cpu_scalar r;
        r.f = f;
        return r;
    }

    cpu_scalar inline make_cpu_scalar_mask( uint32_t mask )
    {
        cpu_scalar r;
        r.u = mask !=0 ? 0xffffffff : 0 ;
        return r;
    }

    namespace math
    {
        template <> inline cpu_scalar splat( float f )
        {
            cpu_scalar r;
            r.f = f;
            return r;
        }

        template <> inline cpu_scalar zero( )
        {
            cpu_scalar r;
            r.f = 0.0f;
            return r;
        }

        template <> inline cpu_scalar one( )
        {
            cpu_scalar r;
            r.f = 1.0f;
            return r;
        }

        template <> inline cpu_scalar add( cpu_scalar a, cpu_scalar b )
        {
            cpu_scalar r;
            r.f = a.f + b.f;
            return r;
        }

        template <> inline cpu_scalar sub( cpu_scalar a, cpu_scalar b )
        {
            cpu_scalar r;
            r.f = a.f - b.f;
            return r;
        }

        template <> inline cpu_scalar mul( cpu_scalar a, cpu_scalar b )
        {
            cpu_scalar r;
            r.f = a.f * b.f;
            return r;
        }

        template <> inline cpu_scalar madd( cpu_scalar a, cpu_scalar b, cpu_scalar c )
        {
            cpu_scalar r;
            r.f = a.f * b.f + c.f;
            return r;
        }

        template <> inline cpu_scalar max( cpu_scalar a, cpu_scalar b )
        {
            cpu_scalar r;
            r.f = a.f > b.f ? a.f : b.f;
            return r;
        }

        template <> inline cpu_scalar rsqrt( cpu_scalar a )
        {
            //16 byte alignment
            cpu_scalar buf[4];
            buf[0].f=a.f;
            __m128 v=_mm_loadu_ps(&buf[0].f);
            v=_mm_rsqrt_ss(v);
            _mm_storeu_ps(&buf[0].f,v);
            return buf[0];
        }

        template <> inline cpu_scalar cmp_ge( cpu_scalar a, cpu_scalar b )
        {
            cpu_scalar r;
            r.u = a.f < b.f ? 0 : 0xffffffff;
            return r;
        }

        template <> inline cpu_scalar cmp_le( cpu_scalar a, cpu_scalar b )
        {
            cpu_scalar r;
            r.u = a.f > b.f ? 0 : 0xffffffff;
            return r;
        }

        // r = (mask == 0) ? a : b;
        template <> inline cpu_scalar blend( cpu_scalar a, cpu_scalar b, cpu_scalar mask )
        {
            cpu_scalar r;

            r.u = a.u & ~mask.u;
            r.u = r.u | ( mask.u & b.u );
            return r;
        }

        template <> inline cpu_scalar and( cpu_scalar a, cpu_scalar mask )
        {
            cpu_scalar r;
            r.u = a.u & mask.u;
            return r;
        }
    }
}

namespace svd
{
    typedef __m128 sse_vector;

    namespace math
    {
        template <> inline sse_vector splat( float f )
        {
            return _mm_set_ps(f, f, f, f );
        }

        template <> inline sse_vector zero( )
        {
            return _mm_setzero_ps();
        }

        template <> inline sse_vector one( )
        {
            return _mm_set_ps(1.0f, 1.0f, 1.0f, 1.0f);
        }

        template <> inline sse_vector add( sse_vector a, sse_vector b )
        {
            return _mm_add_ps(a, b);
        }

        template <> inline sse_vector sub( sse_vector a, sse_vector b )
        {
            return _mm_sub_ps(a, b);
        }

        template <> inline sse_vector mul( sse_vector a, sse_vector b )
        {
            return _mm_mul_ps(a, b);
        }

        template <> inline sse_vector madd( sse_vector a, sse_vector b, sse_vector c )
        {
            return add( mul(a, b ), c );
        }

        template <> inline sse_vector max( sse_vector a, sse_vector b )
        {
            return _mm_max_ps(a, b);
        }

        template <> inline sse_vector rsqrt( sse_vector a )
        {
            return _mm_rsqrt_ss(a);
        }

        template <> inline sse_vector cmp_ge( sse_vector a, sse_vector b )
        {
            return _mm_cmpge_ps(a, b);
        }

        template <> inline sse_vector cmp_le( sse_vector a, sse_vector b )
        {
            return _mm_cmple_ps(a, b);
        }

        // r = (mask == 0) ? a : b;
        template <> inline sse_vector blend( sse_vector a, sse_vector b, sse_vector mask )
        {
            sse_vector v1 = _mm_andnot_ps(mask, a);
            sse_vector v2 = _mm_and_ps(b, mask);
            return _mm_or_ps(v1, v2);
        }

        template <> inline sse_vector and( sse_vector a, sse_vector mask )
        {
            return _mm_and_ps( a, mask );
        }
    }
}

inline std::ostream& operator<<(std::ostream& s, svd::cpu_scalar scalar)
{
    s << scalar.f;
    return s;
}

namespace svd
{
    template <typename t> inline std::tuple< t, t > approximate_givens_quaternion( t a11, t a12, t a22 )
    {
        using namespace math;
        auto half = splat<t> ( 0.5f );
        auto sh = a12 * half;
        auto id  = cmp_ge( sh * sh,  splat<t> ( tiny_number ) );
        
        //if sh squared is tiny, make sh = 0 and ch = 1. this comes from the several jacobi iterations
        sh = and( id, sh );
        auto ch = blend ( one<t>(), a11 - a22, id );

        auto sh_2 = sh * sh;
        auto ch_2 = ch * ch;

        auto x = ch_2 + sh_2;
        auto w = rsqrt ( x );

        //one iteration of newton rhapson.
        //todo remove the loading of constant with alu 3 * x = x + x + x
        //optional
        w = half * w *  ( splat<t>(3.0f) - x * w * w  ) ;

        sh = w * sh;
        ch = w * ch;

        auto b = cmp_le( ch_2, sh_2 * splat<t>( four_gamma_squared ) ) ;

        sh = blend ( sh, splat<t>( sine_pi_over_eight ), b );
        ch = blend ( ch, splat<t>( cosine_pi_over_eight ), b );

        return std::make_tuple<t,t>( ch, sh );
    }

    template <typename t> struct matrix3x3
    {
        //row major
        t a11; t a12; t a13;
        t a21; t a22; t a23;
        t a31; t a32; t a33;

    };

    template <typename t> struct symmetric_matrix3x3
    {
        //row major
        t a11; //t a12; t a13;
        t a21; t a22; //t a23;
        t a31; t a32; t a33;
    };


    //(1,2), (1,3), (2,3)
    //jacobi conjugation of a symmetric matrix
    template < typename t, int p, int q > inline void jacobi_conjugation
                                                        (   
                                                            t&  a11, //t&  a12, t&  a13,
                                                            t&  a21, t&  a22, //t&  a23,
                                                            t&  a31, t&  a32, t&  a33 
                                                        )
    {
        using namespace math;

        if ( p == 1 && q == 2 )
        {
            auto r  = approximate_givens_quaternion<t> ( a11, a21, a22 );
            auto ch = std::get<0>( r );
            auto sh = std::get<1>( r );

            auto ch_plus_sh_2  = sh * sh + ch * ch ;
            auto ch_minus_sh_2 = ch * ch - sh * sh;
            auto ch_sh_2       = ch * sh + ch * sh;

            //Q matrix in the jaocobi method, formed from quaternion
            auto r11 = ch_minus_sh_2;
            auto r12 = ch_sh_2;             //zero<t>() - ch_sh_2; 
            auto r21 = zero<t>() - ch_sh_2; //ch_sh_2 ;
            auto r22 = ch_minus_sh_2;

            auto c = r11;
            auto s = r21;

            auto t1 = a31;
            auto t2 = a32;

            a31 = ( c * t1 - s * t2 ) * ch_plus_sh_2;
            a32 = ( s * t1 + c * t2 ) * ch_plus_sh_2;;
            a33 = a33 * ch_plus_sh_2 * ch_plus_sh_2;

            auto t3 = a11;
            auto t4 = a21;
            auto t5 = a22;

            a11 = c * c * t3 - ( s * c + s * c) * t4 + s * s * t5;
            a22 = s * s * t3 + ( s * c + s * c) * t4 + c * c * t5;
            a21 = s * c * ( t3 - t5 ) + ( c * c - s * s ) * t4; 

            std::cout<<a21<<std::endl;
        }
        else if ( p == 2 && q == 3 )
        {
            auto r  = approximate_givens_quaternion<t> ( a22, a32, a33 );
            auto ch = std::get<0>( r );
            auto sh = std::get<1>( r );

            auto ch_plus_sh_2  = sh * sh + ch * ch ;
            auto ch_minus_sh_2 = ch * ch - sh * sh;
            auto ch_sh_2       = ch * sh + ch * sh;

            //Q matrix in the jaocobi method, formed from quaternion
            auto r11 = ch_minus_sh_2;
            auto r12 = zero<t>() - ch_sh_2; 
            auto r21 = ch_sh_2 ;
            auto r22 = ch_minus_sh_2;

            auto c = r11;
            auto s = r21;

            auto t1 = a21;
            auto t2 = a31;

            a21 = ( c * t1 - s * t2 ) * ch_plus_sh_2;
            a31 = ( s * t1 + c * t2 ) * ch_plus_sh_2;;
            a11 = a11 * ch_plus_sh_2 * ch_plus_sh_2;

            auto t3 = a22;
            auto t4 = a32;
            auto t5 = a33;

            a22 = c * c * t3 - ( s * c + s * c) * t4 + s * s * t5;
            a33 = s * s * t3 + ( s * c + s * c) * t4 + c * c * t5;
            a32 = s * c * ( t3 - t5 ) + ( c * c - s * s ) * t4; 
        }
        else if ( p == 1 && q == 3 )
        {
            auto r  = approximate_givens_quaternion<t> ( a11, a31, a33 );
            auto ch = std::get<0>( r );
            auto sh = std::get<1>( r );

            auto ch_plus_sh_2  = sh * sh + ch * ch ;
            auto ch_minus_sh_2 = ch * ch - sh * sh;
            auto ch_sh_2       = ch * sh + ch * sh;

            //Q matrix in the jaocobi method, formed from quaternion
            auto r11 = ch_minus_sh_2;
            auto r12 = zero<t>() - ch_sh_2; 
            auto r21 = ch_sh_2 ;
            auto r22 = ch_minus_sh_2;

            auto c = r11;
            auto s = r21;

            auto t1 = a32;
            auto t2 = a21;

            a21 = ( c * t1 - s * t2 ) * ch_plus_sh_2;
            a32 = ( s * t1 + c * t2 ) * ch_plus_sh_2;
            a22 = a22 * ch_plus_sh_2 * ch_plus_sh_2;

            auto t3 = a33;
            auto t4 = a31;
            auto t5 = a11;

            a11 = c * c * t3 - ( s * c + s * c) * t4 + s * s * t5;
            a33 = s * s * t3 + ( s * c + s * c) * t4 + c * c * t5;
            a31 = s * c * ( t3 - t5 ) + ( c * c - s * s ) * t4; 
        }
    }

    //(1,2), (1,3), (2,3)
    //jacobi conjugation of a symmetric matrix
    template < typename t, int p, int q > inline void jacobi_conjugation ( symmetric_matrix3x3<t>& m )
    {
        return jacobi_conjugation<t, p, q> ( m.a11, m.a21, m.a22, m.a31, m.a32, m.a33 ) ;
    }

    template < typename t> inline symmetric_matrix3x3<t> create_symmetric_matrix ( const matrix3x3<t>& in  )
    {
        using namespace svd::math;

        auto a11 = in.a11 * in.a11 + in.a21 * in.a21 + in.a31 * in.a31;
        auto a12 = in.a11 * in.a12 + in.a21 * in.a22 + in.a31 * in.a32;
        auto a13 = in.a11 * in.a13 + in.a21 * in.a23 + in.a31 * in.a33;

        auto a21 = a12;
        auto a22 = in.a12 * in.a12 + in.a22 * in.a22 + in.a32 * in.a32;
        auto a23 = in.a12 * in.a13 + in.a22 * in.a23 + in.a32 * in.a33;

        auto a31 = a13;
        auto a32 = a23;
        auto a33 = in.a13 * in.a13 + in.a23 * in.a23 + in.a33 * in.a33;


        symmetric_matrix3x3<t> r = { a11, a21, a22, a31, a32, a33 };
        return r;
    }

    template < typename t> inline matrix3x3<t> create_matrix 
        ( 
            t a11, t a12, t a13,
            t a21, t a22, t a23,
            t a31, t a32, t a33
        )
    {
        using namespace svd::math;
        matrix3x3<t> r =  { a11, a12, a13, a21, a22, a23, a31, a32, a33 };
        return r;
    }
}


std::int32_t main(int argc, _TCHAR* argv[])
{
    auto m11 = svd::math::splat<svd::cpu_scalar>( 2.0f );
    auto m12 = svd::math::splat<svd::cpu_scalar>( -0.2f );
    auto m13 = svd::math::splat<svd::cpu_scalar>( 1.0f );

    auto m21 = svd::math::splat<svd::cpu_scalar>( -0.2f);
    auto m22 = svd::math::splat<svd::cpu_scalar>( 1.0f);
    auto m23 = svd::math::splat<svd::cpu_scalar>( 0.0f);

    auto m31 = svd::math::splat<svd::cpu_scalar>( 1.0f);
    auto m32 = svd::math::splat<svd::cpu_scalar>( 0.0f);
    auto m33 = svd::math::splat<svd::cpu_scalar>( 1.0f);

    auto m = svd::create_symmetric_matrix( svd::create_matrix ( m11, m12, m13, m21, m22, m23, m31, m32, m33 ) );

    svd::jacobi_conjugation< svd::cpu_scalar, 1, 2 > ( m );

    using namespace svd::math;

    auto a11 = m11 * m11 + m21 * m21 + m31 * m31;
    auto a12 = m11 * m12 + m21 * m22 + m31 * m32;
    auto a13 = m11 * m13 + m21 * m23 + m31 * m33;

    auto a21 = a12;
    auto a22 = m12 * m12 + m22 * m22 + m32 * m32;
    auto a23 = m12 * m13 + m22 * m23 + m32 * m33;

    auto a31 = a13;
    auto a32 = a23;
    auto a33 = m13 * m13 + m23 * m23 + m33 * m33;


    svd::jacobi_conjugation< svd::cpu_scalar, 1, 2 > ( a11, a21, a22, a31, a32, a33 );
    std::cout<<"a11:" << a11 << " a21:" << a21 << " a22:" << a22  << " a31:" << a31 << " a32:" << a32 << " a33:" << a33 << std::endl;
    std::cout<<"sum:" << a21 * a21 + a31 * a31 + a32 * a32 << std::endl;
   
    std::cout<<std::endl;


    svd::jacobi_conjugation< svd::cpu_scalar, 1, 2 > ( a11, a21, a22, a31, a32, a33 );
    std::cout<<"a11:" << a11 << " a21:" << a21 << " a22:" << a22  << " a31:" << a31 << " a32:" << a32 << " a33:" << a33 << std::endl;
    std::cout<<"sum:" << a21 * a21 + a31 * a31 + a32 * a32 << std::endl;
   
    std::cout<<std::endl;




    

    return 0;
}

