#include "precompiled.h"

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
    //given coefficients of a symetric matrix, returns c and s, such that they diagonalize this matrix
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
        auto ch = blend ( one<t>(), a11 - a12, id );

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


    //(2,1), (3,1), (3,2)
    //jacobi conjugation of symetric matrix
    template < typename t, int p, int q > inline void jacobi_conjugation
                                                        (   
                                                            t&  a11, //t&  a12, t&  a13,
                                                            t&  a21, t&  a22, //t&  a23,
                                                            t&  a31, t&  a32, t&  a33 
                                                        )
    {
        using namespace math;

        t* a[3][3] = { &a11, &a21, &a31, &a21, &a22, &a32, &a31, &a32, &a33 };
        auto r = approximate_givens_quaternion<t> ( *a[p][p], *a[p][q], *a[q][q] );

        auto ch = std::get<0>( r );
        auto sh = std::get<1>( r );

        auto ch_plus_sh_2 = sh * sh + ch * ch ;
        auto ch_minus_sh_2 = ch * ch - sh * sh;
        auto ch_sh_2 = ch * sh + ch * sh;

        //Q matrix in the jaocobi method, formed from quaternion
        auto r11 = ch_minus_sh_2;
        auto r12 = zero<t>() - ch_sh_2; 
        auto r21 = ch_sh_2 ;
        auto r22 = ch_minus_sh_2;

        auto c = r11;
        auto s = r21;

        if ( p == 0 && q == 1 )
        {
            
        }
    }
}


std::int32_t main(int argc, _TCHAR* argv[])
{
    auto a11 = svd::math::splat<svd::sse_vector>( 1.0f );
    auto a12 = svd::math::splat<svd::sse_vector>( 0.0f );
    auto a13 = svd::math::splat<svd::sse_vector>( 1.0f );

    auto a21 = svd::math::splat<svd::sse_vector>( 0.0f);
    auto a22 = svd::math::splat<svd::sse_vector>( 1.0f);
    auto a23 = svd::math::splat<svd::sse_vector>( 0.0f);

    auto a31 = svd::math::splat<svd::sse_vector>( 0.0f);
    auto a32 = svd::math::splat<svd::sse_vector>( 0.0f);
    auto a33 = svd::math::splat<svd::sse_vector>( 1.0f);


    using namespace svd::math;
    auto ra = svd::approximate_givens_quaternion( a11, a12, a22 );

    
    svd::matrix3x3<svd::sse_vector> m =
    { 
        a11, a12, a13,
        a21, a22, a23,
        a31, a32, a33
    };

    /*
    auto sum0 = svd::jacobi_conjugation< svd::sse_vector, 0, 1 > ( m.a11, m.a12, m.a13, m.a21, m.a22, m.a23, m.a31, m.a32, m.a33  );
    auto sum1 = svd::jacobi_conjugation< svd::sse_vector, 0, 2 > ( m.a11, m.a12, m.a13, m.a21, m.a22, m.a23, m.a31, m.a32, m.a33  );
    auto sum2 = svd::jacobi_conjugation< svd::sse_vector, 1, 2 > ( m.a11, m.a12, m.a13, m.a21, m.a22, m.a23, m.a31, m.a32, m.a33  );
    

    auto sum0 = svd::jacobi_conjugation< svd::sse_vector, 0, 1 > ( a11, a12, a13, a21, a22, a23, a31, a32, a33  );
    auto sum1 = svd::jacobi_conjugation< svd::sse_vector, 0, 2 > ( a11, a12, a13, a21, a22, a23, a31, a32, a33  );
    auto sum2 = svd::jacobi_conjugation< svd::sse_vector, 1, 2 > ( a11, a12, a13, a21, a22, a23, a31, a32, a33  );
    */

    svd::jacobi_conjugation< svd::sse_vector, 0, 1 > ( a11, a21, a22, a31, a32, a33 );
    svd::jacobi_conjugation< svd::sse_vector, 1, 2 > ( a11, a21, a22, a31, a32, a33 );

    float buf0[4];
    float buf1[4];
    float buf2[4];

    _mm_storeu_ps(&buf0[0],a11);
    _mm_storeu_ps(&buf1[0],a21);
    _mm_storeu_ps(&buf2[0],a22);

    _mm_storeu_ps(&buf0[0],a31);
    _mm_storeu_ps(&buf1[0],a32);
    _mm_storeu_ps(&buf2[0],a33);

    

    return 0;
}

