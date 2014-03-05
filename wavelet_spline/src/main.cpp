#include "precompiled.h"

#include <cstdint>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>


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
    const float tiny_number             =   1.e-20;
    const float small_number            =   1.e-12;


    namespace math
    {
        template <typename t> inline t add( t a, t b );
        template <typename t> inline t sub( t a, t b );
        template <typename t> inline t mul( t a, t b );
        template <typename t> inline t max( t a, t b );

        template <typename t> inline t rsqrt( t a );

        template <typename t> inline t cmp_ge( t a, t b );
        template <typename t> inline t cmp_le( t a, t b );
        template <typename t> inline t blend(  t a, t b, t mask );

        template <typename t> inline t zero();
        template <typename t> inline t one();

        template <typename t> inline t splat( float f );

        template <typename t> inline t and ( t a, t mask );
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

    template <typename t> std::tuple< t, t > approximate_givens_quaternion( t a11, t a12, t a22 )
    {
        using namespace math;
        auto half = splat<t> ( 0.5f );
        auto sh = mul( a12,  half );
        auto id  = cmp_ge( mul( sh, sh ), splat<t> ( tiny_number ) );
        
        //if sh squared is tiny, make sh = 0 and ch = 1. this comes from the several jacobi iterations
        sh = and( id, sh );
        auto ch = blend ( one<t>(), sub( a11, a22 ), id );

        auto sh_2 = mul ( sh, sh );
        auto ch_2 = mul ( ch, ch );

        auto x = add( sh_2, ch_2 );
        auto w = rsqrt ( x );

        //one iteration of newton rhapson.
        //todo remove the loading of constant with alu 3 * x = x + x + x
        w = mul ( half, mul (w, sub ( splat<t>(3.0f), mul (x, mul(w, w)) )) ) ;

        sh = mul ( w, sh );
        ch = mul ( w, ch );

        auto b = cmp_le( ch_2, mul( sh_2, splat<t>( four_gamma_squared ) ) );

        sh = blend ( sh, splat<t>( sine_pi_over_eight ), b );
        ch = blend ( ch, splat<t>( cosine_pi_over_eight ), b );

        return std::make_tuple<t,t>( ch, sh );
    }

    template <typename t, int p, int q > inline void jacobi_conjugation(   t& a11, t& a12, t& a13,
                                                            t& a21, t& a22, t& a23,
                                                            t& a31, t& a32, t& a33 )
    {
        

    }
}


std::int32_t main(int argc, _TCHAR* argv[])
{
    auto a11 = svd::math::splat<svd::cpu_scalar>( 1.0f);
    auto a12 = svd::math::splat<svd::cpu_scalar>( 0.0f);
    auto a22 = svd::math::splat<svd::cpu_scalar>( 0.0f);

    auto givens_rotation = svd::approximate_givens_quaternion( a11, a12, a22 );

    return 0;
}

