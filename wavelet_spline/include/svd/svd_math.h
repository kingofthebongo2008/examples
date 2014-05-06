#ifndef __svd_svd_math_h__
#define __svd_svd_math_h__

#include <xmmintrin.h>
#include <immintrin.h>

#include <cstdint>

namespace svd
{
    const float four_gamma_squared      =   static_cast<float> ( sqrt(8.)+3. );
    const float sine_pi_over_eight      =   static_cast<float> ( .5*sqrt( 2. - sqrt(2.) ) );
    const float cosine_pi_over_eight    =   static_cast<float> ( .5*sqrt( 2. + sqrt(2.) ) );
    const float tiny_number             =   static_cast<float> ( 1.e-20 );
    const float small_number            =   static_cast<float> ( 1.e-12 );


    namespace math
    {
        template <typename t> inline t add( t a, t b );
        template <typename t> inline t sub( t a, t b );
        template <typename t> inline t div( t a, t b );
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
        template <typename t> inline t xor ( t a, t b );

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

        template <typename t> inline t operator/( t a, t b )
        {
            return div ( a, b );
        }

        template <typename t> inline t operator<( t a, t b );


        template <typename t> inline t dot3( t a1, t a2, t a3, t b1, t b2, t b3)
        {
            return a1 * b1 + a2 * b2 + a3 * b3;
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

        template <> inline cpu_scalar div( cpu_scalar a, cpu_scalar b )
        {
            cpu_scalar r;
            r.f = a.f / b.f;
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

        template <> inline cpu_scalar operator<( cpu_scalar a, cpu_scalar b )
        {
            cpu_scalar r;
            r.u = a.f < b.f ? 0xffffffff : 0;
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

        template <> inline cpu_scalar xor( cpu_scalar a, cpu_scalar b )
        {
            cpu_scalar r;
            r.u = a.u ^ b.u;
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
            return _mm_rsqrt_ps(a);
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

        template <> inline sse_vector xor( sse_vector a, sse_vector b )
        {
            return _mm_xor_ps( a, b );
        }

        template <> inline sse_vector operator<( sse_vector a, sse_vector b )
        {
            return _mm_cmplt_ps( a, b);
        }
    }
}

namespace svd
{
    typedef __m256 avx_vector;

    namespace math
    {
        template <> inline avx_vector splat( float f )
        {
            return _mm256_set_ps(f, f, f, f, f, f, f, f );
        }

        template <> inline avx_vector zero( )
        {
            return _mm256_setzero_ps();
        }

        template <> inline avx_vector one( )
        {
            return _mm256_set_ps(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
        }

        template <> inline avx_vector add( avx_vector a, avx_vector b )
        {
            return _mm256_add_ps(a, b);
        }

        template <> inline avx_vector sub( avx_vector a, avx_vector b )
        {
            return _mm256_sub_ps(a, b);
        }

        template <> inline avx_vector mul( avx_vector a, avx_vector b )
        {
            return _mm256_mul_ps(a, b);
        }

        template <> inline avx_vector madd( avx_vector a, avx_vector b, avx_vector c )
        {
            return add( mul(a, b ), c );
        }

        template <> inline avx_vector max( avx_vector a, avx_vector b )
        {
            return _mm256_max_ps(a, b);
        }

        template <> inline avx_vector rsqrt( avx_vector a )
        {
            return _mm256_rsqrt_ps(a);
        }

        template <> inline avx_vector cmp_ge( avx_vector a, avx_vector b )
        {
            return _mm256_cmp_ps(a, b, _CMP_GE_OS);
        }

        template <> inline avx_vector cmp_le( avx_vector a, avx_vector b )
        {
            return _mm256_cmp_ps(a, b, _CMP_LE_OS);
        }

        // r = (mask == 0) ? a : b;
        template <> inline avx_vector blend( avx_vector a, avx_vector b, avx_vector mask )
        {
            return _mm256_blendv_ps( a, b, mask) ;
        }

        template <> inline avx_vector and( avx_vector a, avx_vector mask )
        {
            return _mm256_and_ps( a, mask );
        }

        template <> inline avx_vector xor( avx_vector a, avx_vector b )
        {
            return _mm256_xor_ps( a, b );
        }

        template <> inline avx_vector operator<( avx_vector a, avx_vector b )
        {
            return _mm256_cmp_ps(a, b, _CMP_LT_OS);
        }
    }
}



#endif
