#include "precompiled.h"

#include <cstdint>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include <xmmintrin.h>


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

inline std::ostream& operator<<(std::ostream& s, svd::cpu_scalar scalar)
{
    s << scalar.f;
    return s;
}

inline std::ostream& operator<<(std::ostream& s, svd::sse_vector scalar)
{
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
        w = w + ( w * half ) - ( ( w * half )  *  w * w * x  );

        sh = w * sh;
        ch = w * ch;

        auto b = cmp_le( ch_2, sh_2 * splat<t>( four_gamma_squared ) ) ;

        sh = blend ( sh, splat<t>( sine_pi_over_eight ), b );
        ch = blend ( ch, splat<t>( cosine_pi_over_eight ), b );

        return std::make_tuple<t,t>( std::move(ch), std::move(sh) );
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

    template <typename t> struct quaternion
    {
        t x;
        t y;
        t z;
        t w;
    };

    template <typename t> struct vector3
    {
        t x;
        t y;
        t z;
    };


    //(1,2), (1,3), (2,3)
    //jacobi conjugation of a symmetric matrix
    template < typename t, int p, int q > inline void jacobi_conjugation
                                                        (   
                                                            t&  a11, //t&  a12, t&  a13,
                                                            t&  a21, t&  a22, //t&  a23,
                                                            t&  a31, t&  a32, t&  a33,
                                                            t&  qx,  t&  qy,  t&  qz, t& qw
                                                        )
    {
        using namespace math;

        if ( p == 1 && q == 2 )
        {
            auto r  = approximate_givens_quaternion<t> ( a11, a21, a22 );
            auto ch = std::get<0>( r );
            auto sh = std::get<1>( r );

            auto ch_minus_sh_2 = ch * ch - sh * sh;
            auto ch_sh_2       = ch * sh + ch * sh;

            //Q matrix in the jaocobi method, formed from quaternion
            auto r11 = ch_minus_sh_2;
            auto r12 = ch_sh_2;             
            auto r21 = zero<t>() - ch_sh_2; 
            auto r22 = ch_minus_sh_2;

            auto c = r11;
            auto s = r12;

            auto t1 = a31;
            auto t2 = a32;

            a31 = c * t1 + s * t2;
            a32 = c * t2 - s * t1;
            a33 = a33;

            auto t3 = a11;
            auto t4 = a21;
            auto t5 = a22;

            a11 = s * s * t5 + c * c * t3 + ( s * c + s * c) * t4;
            a22 = c * c * t5 + s * s * t3 - ( s * c + s * c) * t4;
            a21 = s * c * ( t5 - t3 ) + ( c * c - s * s ) * t4; 


            //now create the apply the total quaternion transformation7
            auto q0 = qw;
            auto q1 = qx;
            auto q2 = qy;
            auto q3 = qz;

            auto r0 = ch;
            auto r1 = 0;
            auto r2 = 0;
            auto r3 = sh;

            qw = r0 * q0 - r3 * q3;
            qx = r0 * q1 + r3 * q2;
            qy = r0 * q2 - r3 * q1;
            qz = r0 * q3 + r3 * q0;

        }
        else if ( p == 2 && q == 3 )
        {
            auto r  = approximate_givens_quaternion<t> ( a22, a32, a33 );
            auto ch = std::get<0>( r );
            auto sh = std::get<1>( r );

            auto ch_minus_sh_2 = ch * ch - sh * sh;
            auto ch_sh_2       = ch * sh + ch * sh;

            //Q matrix in the jaocobi method, formed from quaternion
            auto r11 = ch_minus_sh_2;
            auto r12 = ch_sh_2;             
            auto r21 = zero<t>() - ch_sh_2; 
            auto r22 = ch_minus_sh_2;

            auto c = r11;
            auto s = r12;

            auto t1 = a21;
            auto t2 = a31;

            a21 = c * t1 + s * t2;
            a31 = c * t2 - s * t1;
            a11 = a11;

            auto t3 = a22;
            auto t4 = a32;
            auto t5 = a33;

            a22 = s * s * t5 + c * c * t3 + ( s * c + s * c) * t4;
            a33 = c * c * t5 + s * s * t3 - ( s * c + s * c) * t4;
            a32 = s * c * ( t5 - t3 ) + ( c * c - s * s ) * t4; 

            //now create the apply the total quaternion transformation7
            auto q0 = qw;
            auto q1 = qx;
            auto q2 = qy;
            auto q3 = qz;

            auto r0 = ch;
            auto r1 = sh;
            auto r2 = 0;
            auto r3 = 0;

            qw = r0 * q0 - r1 * q1;
            qx = r0 * q1 + r1 * q0;
            qy = r0 * q2 + r1 * q3;
            qz = r0 * q3 - r1 * q2;
        }
        else if ( p == 1 && q == 3 )
        {
            auto r  = approximate_givens_quaternion<t> ( a33, a31, a11 );
            auto ch = std::get<0>( r );
            auto sh = std::get<1>( r );

            auto ch_minus_sh_2 = ch * ch - sh * sh;
            auto ch_sh_2       = ch * sh + ch * sh;

            //Q matrix in the jaocobi method, formed from quaternion
            auto r11 = ch_minus_sh_2;
            auto r12 = ch_sh_2;             
            auto r21 = zero<t>() - ch_sh_2; 
            auto r22 = ch_minus_sh_2;

            auto c = r11;
            auto s = r12;

            auto t1 = a32;
            auto t2 = a21;

            a32 = c * t1 + s * t2;
            a21 = c * t2 - s * t1;
            a22 = a22;

            auto t3 = a33;
            auto t4 = a31;
            auto t5 = a11;

            a33 = s * s * t5 + c * c * t3 + ( s * c + s * c) * t4;
            a11 = c * c * t5 + s * s * t3 - ( s * c + s * c) * t4;
            a31 = s * c * ( t5 - t3 ) + ( c * c - s * s ) * t4; 

            //now create the apply the total quaternion transformation7
            auto q0 = qw;
            auto q1 = qx;
            auto q2 = qy;
            auto q3 = qz;

            auto r0 = ch;
            auto r1 = 0;
            auto r2 = sh;
            auto r3 = 0;

            qw = r0 * q0 - r2 * q2;
            qx = r0 * q1 - r2 * q3;
            qy = r0 * q2 + r2 * q0;
            qz = r0 * q3 + r2 * q1;
        }
    }

    //(1,2), (1,3), (2,3)
    //jacobi conjugation of a symmetric matrix
    template < typename t, int p, int q > inline void jacobi_conjugation ( symmetric_matrix3x3<t>& m, quaternion<t>& quaternion )
    {
        jacobi_conjugation< t, p, q > ( m.a11, m.a21, m.a22, m.a31, m.a32, m.a33, quaternion.x, quaternion.y, quaternion.z, quaternion.w) ;
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

    template <typename t> inline quaternion<t> create_quaternion( t qx, t qy, t qz, t qw)
    {
        using namespace svd::math;

        quaternion<t> r = { qx, qy, qz, qw };
        return r;
    }

    template <typename t> inline quaternion<t> normalize( const quaternion<t>& q )
    {
        using namespace math;
        using namespace svd::math;
        
        auto half = svd::math::splat<t> ( 0.5f );
        
        auto x = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
        auto w = rsqrt( x );

        //one iteration of newton rhapson.
        w = w + ( w * half ) - ( ( w * half )  *  w * w * x  );

        return create_quaternion( q.x * w, q.y * w, q.z * w, q.w * w);
    }

    template <typename t> inline void normalize( quaternion<t>& q )
    {
        using namespace math;
        using namespace svd::math;
        
        auto half = svd::math::splat<t> ( 0.5f );
        
        auto x = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
        auto w = rsqrt( x );

        //one iteration of newton rhapson.
        w = w + ( w * half ) - ( ( w * half )  *  w * w * x  );

        q.x = q.x * w;
        q.y = q.y * w;
        q.z = q.z * w;
        q.w = q.w * w;
    }

    template <typename t> inline void conditional_swap( t c, t& x, t& y )
    {
        using namespace math;
        auto t = xor( x, y );
        auto m = and( c, t );
        x = xor ( x, m );
        y = xor ( y, m );
    }

    //returns -1.0f or 1.0f depending on c
    //used for conditional_negative_swap
    template <typename t> inline t negative_conditional_swap_multiplier( t c )
    {
        using namespace math;
        auto two = splat<t>(-2.0f);
        auto m = and( c, two );
        return one<t>() + m;
    }

    template < typename t, uint32_t axis > inline void conditional_swap( quaternion<t>& v, t c )
    {
        using namespace svd::math;

        if (axis == 3 )
        {
            // If columns 1-2 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)
            // do v*vr, where vr= (1, 0, 0, -c) -> this represents column swap as a quaternion, see the paper for more details

            auto w = v.w;
            auto x = v.x;
            auto y = v.y;
            auto z = v.z;

            v.w = w + c * z;
            v.x = x - c * y;
            v.y = y + c * x;
            v.z = z - c * w;
        }
        else if ( axis == 2 )
        {
            // If columns 1-3 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)
            // do v*vr, where vr= (1, 0, -c, 0) -> this represents column swap as a quaternion, see the paper for more details

            auto w = v.w;
            auto x = v.x;
            auto y = v.y;
            auto z = v.z;

            v.w = w + c * y;
            v.x = x + c * z;
            v.y = y - c * w;
            v.z = z - c * x;

        }
        else if ( axis == 1 )
        {
            // If columns 2-3 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)
            // do v*vr, where vr= (1, -c, 0, 0) -> this represents column swap as a quaternion, see the paper for more details

            auto w = v.w;
            auto x = v.x;
            auto y = v.y;
            auto z = v.z;

            v.w = w + c * x;
            v.x = x - c * w;
            v.y = y - c * z;
            v.z = z + c * y;
        }
    }

    template <typename t> inline std::tuple< t, t > givens_quaternion( t a1, t a2 )
    {
        using namespace math;

        auto half = splat<t> ( 0.5f );

        auto id = cmp_ge( a2 * a2,  splat<t> ( small_number ) );
        auto sh = and( id, a2 );

        auto ch = max ( a1, zero<t>() - a1 );
        auto c = cmp_le( a1, zero<t>() );
        ch = max ( ch, splat<t>(small_number ) );

        // compute sqrt(ch * ch + sh * sh )
        auto x = ch * ch + sh * sh;
        auto w = rsqrt( x );
        //one iteration of newton rhapson.
        w = w + ( w * half ) - ( ( w * half )  *  w * w * x  );
        w = x * w; 

        auto rho = w;
        ch = ch + rho;

        conditional_swap(c, ch, sh );

        x = ch * ch + sh * sh;
        w = rsqrt( x );
        //one iteration of newton rhapson.
        w = w + ( w * half ) - ( ( w * half )  *  w * w * x  );

        ch = w * ch;
        sh = w * sh;

        return std::make_pair ( ch, sh );
    }

    //(1,2), (1,3), (2,3)
    //jacobi conjugation of a symmetric matrix
    template < typename t, int p, int q > inline void givens_conjugation
                                                        (   
                                                            t&  a11, t&  a12, t&  a13,
                                                            t&  a21, t&  a22, t&  a23,
                                                            t&  a31, t&  a32, t&  a33,
                                                            t&  qx,  t&  qy,  t&  qz, t& qw
                                                        )
    {
        using namespace math;

        if ( p == 1 && q == 2 )
        {
            auto r  = givens_quaternion<t> ( a11, a21 );
            auto ch = std::get<0>( r );
            auto sh = std::get<1>( r );

            auto ch_minus_sh_2 = ch * ch - sh * sh;
            auto ch_sh_2       = ch * sh + ch * sh;

            //Q matrix in the jaocobi method, formed from quaternion
            auto r11 = ch_minus_sh_2;
            auto r12 = ch_sh_2;             
            auto r21 = zero<t>() - ch_sh_2; 
            auto r22 = ch_minus_sh_2;

            auto c = r11;
            auto s = r12;

            auto t11 = a11;
            auto t12 = a12;
            auto t13 = a13;

            auto t21 = a21;
            auto t22 = a22;
            auto t23 = a23;

            a11 = c * t11 + s * t21;
            a21 = c * t21 - s * t11;   

            a12 = c * t12 + s * t22;
            a22 = c * t22 - s * t12;
            
            a13 = c * t13 + s * t23;
            a23 = c * t23 - s * t13;

            //now create the apply the total quaternion transformation7
            auto w = qw;
            auto x = qx;
            auto y = qy;
            auto z = qz;

            //use the fact, that x,y,z = 0
            qw = ch * w;
            qx = x;
            qy = x;
            qz = sh * w;

        }
        else if ( p == 2 && q == 3 )
        {
            auto r  = givens_quaternion<t> ( a22, a32 );
            auto ch = std::get<0>( r );
            auto sh = std::get<1>( r );

            auto ch_minus_sh_2 = ch * ch - sh * sh;
            auto ch_sh_2       = ch * sh + ch * sh;

            //Q matrix in the jaocobi method, formed from quaternion
            auto r11 = ch_minus_sh_2;
            auto r12 = ch_sh_2;             
            auto r21 = zero<t>() - ch_sh_2; 
            auto r22 = ch_minus_sh_2;

            auto c = r11;
            auto s = r12;

            auto t11 = a21;
            auto t12 = a22;
            auto t13 = a23;

            auto t21 = a31;
            auto t22 = a32;
            auto t23 = a33;

            a21 = c * t11 + s * t21;
            a31 = c * t21 - s * t11;   

            a22 = c * t12 + s * t22;
            a32 = c * t22 - s * t12;
            
            a23 = c * t13 + s * t23;
            a33 = c * t23 - s * t13;

            //now create the apply the total quaternion transformation7
            auto w = qw;
            auto x = qx;
            auto y = qy;
            auto z = qz;

            //Quaternion[ ch, sh, 0, 0] -> q * r
            qw = ch * w - sh * x;
            qx = ch * x + sh * w;
            qy = ch * y + sh * z;
            qz = ch * z - sh * y;

        }
        else if ( p == 1 && q == 3 )
        {
            auto r  = givens_quaternion<t> ( a11, a31 );
            auto ch = std::get<0>( r );
            auto sh = std::get<1>( r );

            auto ch_minus_sh_2 = ch * ch - sh * sh;
            auto ch_sh_2       = ch * sh + ch * sh;

            //Q matrix in the jaocobi method, formed from quaternion
            auto r11 = ch_minus_sh_2;
            auto r12 = ch_sh_2;             
            auto r21 = zero<t>() - ch_sh_2; 
            auto r22 = ch_minus_sh_2;

            auto c = r11;
            auto s = r12;

            auto t11 = a11;
            auto t12 = a12;
            auto t13 = a13;

            auto t21 = a31;
            auto t22 = a32;
            auto t23 = a33;

            a11 = c * t11 + s * t21;
            a31 = c * t21 - s * t11;   

            a12 = c * t12 + s * t22;
            a32 = c * t22 - s * t12;
            
            a13 = c * t13 + s * t23;
            a33 = c * t23 - s * t13;

            //now create the apply the total quaternion transformation7
            auto w = qw;
            auto x = qx;
            auto y = qy;
            auto z = qz;

            //use the fact, that x = 0 and y = 0 from the previous iteration
            qw = ch * w;
            qx = sh * z;
            qy = zero<t>()-sh * w;
            qz = ch * z;
        }
    }

    //obtain A = USV' 
    template < typename t > inline void compute( const matrix3x3<t>& in, quaternion<t>& u, vector3<t>& s, quaternion<t>& v )
    {
        using namespace svd::math;

        // initial value of v as a quaternion
        auto vx = splat<t>( 0.0f );
        auto vy = splat<t>( 0.0f );
        auto vz = splat<t>( 0.0f );
        auto vw = splat<t>( 1.0f );

        u = create_quaternion ( vx, vy, vz, vw );
        v = create_quaternion ( vx, vy, vz, vw );

        auto m = create_symmetric_matrix( in );
        
        //1. Compute the V matrix as a quaternion

        //4 iterations of jacobi conjugation to obtain V
        for (auto i = 0; i < 4 ; ++i)
        {
            svd::jacobi_conjugation< t, 1, 2 > ( m, v );
            svd::jacobi_conjugation< t, 2, 3 > ( m, v );
            svd::jacobi_conjugation< t, 1, 3 > ( m, v );
        }

        //normalize the quaternion. this is optional
        normalize<t>(v);

        //convert quaternion v to matrix {
        auto tmp1 = v.x * v.x;
        auto tmp2 = v.y * v.y;
        auto tmp3 = v.z * v.z;

        auto v11  = v.w * v.w;
        auto v22  = v11 - tmp1;
        auto v33  = v22 - tmp2;

        v33 = v33 + tmp3;

        v22 = v22 + tmp2;
        v22 = v22 - tmp3;

        v11 = v11 + tmp1;
        v11 = v11 - tmp2;
        v11 = v11 - tmp3;

        tmp1 = v.x + v.x;
        tmp2 = v.y + v.y;
        tmp3 = v.z + v.z;

        auto v32 = v.w * tmp1;
        auto v13 = v.w * tmp2;
        auto v21 = v.w * tmp3;

        tmp1 = v.y * tmp1;
        tmp2 = v.z * tmp2;
        tmp3 = v.x * tmp3;

        auto v12 = tmp1 - v21;
        auto v23 = tmp2 - v32;
        auto v31 = tmp3 - v13;

        v21 = v21 + tmp1;
        v32 = v32 + tmp2;
        v13 = v13 + tmp3;
        //} convert quaternion to matrix

        // compute AV

        auto a11 = dot3( in.a11, in.a12, in.a13, v11, v21, v31 );
        auto a12 = dot3( in.a11, in.a12, in.a13, v12, v22, v32 );
        auto a13 = dot3( in.a11, in.a12, in.a13, v13, v23, v33 );

        auto a21 = dot3( in.a21, in.a22, in.a23, v11, v21, v31 );
        auto a22 = dot3( in.a21, in.a22, in.a23, v12, v22, v32 );
        auto a23 = dot3( in.a21, in.a22, in.a23, v13, v23, v33 );

        auto a31 = dot3( in.a31, in.a32, in.a33, v11, v21, v31 );
        auto a32 = dot3( in.a31, in.a32, in.a33, v12, v22, v32 );
        auto a33 = dot3( in.a31, in.a32, in.a33, v13, v23, v33 );

        //2. sort the singular values

        //compute the norms of the columns for comparison
        auto rho1 = dot3( a11, a21, a31, a11, a21, a31 );
        auto rho2 = dot3( a12, a22, a32, a12, a22, a32 );
        auto rho3 = dot3( a13, a23, a33, a13, a23, a33 );

        auto c = rho1 < rho2;

        // Swap columns 1-2 if necessary
        conditional_swap( c, a11, a12 );
        conditional_swap( c, a21, a22 );
        conditional_swap( c, a31, a32 );
        
        //either -1 or 1
        auto multiplier = negative_conditional_swap_multiplier( c );

        // If columns 1-2 have been swapped, negate 2nd column of A and V so that V is still a rotation
        a12 = a12 * multiplier;
        a22 = a22 * multiplier;
        a32 = a32 * multiplier;

        // If columns 1-2 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)
        // do v*vr, where vr= (1, 0, 0, -c) -> this represents column swap as a quaternion, see the paper for more details
       
        auto half = svd::math::splat<t> ( 0.5f );
        conditional_swap<t, 3>( v, multiplier * half - half );

        c = rho1 < rho3;

        // Swap columns 1-3 if necessary
        conditional_swap( c, a11, a13 );
        conditional_swap( c, a21, a23 );
        conditional_swap( c, a31, a33 );

        multiplier = negative_conditional_swap_multiplier( c );

        // If columns 1-3 have been swapped, negate 1st column of A and V so that V is still a rotation
        a11 = a11 * multiplier;
        a21 = a21 * multiplier;
        a31 = a31 * multiplier;

        // If columns 1-3 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)
        // do v*vr, where vr= (1, 0, -c, 0) -> this represents column swap as a quaternion, see the paper for more details
        conditional_swap<t, 2>( v, multiplier * half - half );

        c = rho2 < rho3;

        // Swap columns 2-3 if necessary
        conditional_swap( c, a12, a13 );
        conditional_swap( c, a22, a23 );
        conditional_swap( c, a32, a33 );

        multiplier = negative_conditional_swap_multiplier( c );

        // If columns 2-3 have been swapped, negate 3rd column of A and V so that V is still a rotation
        a13 = a13 * multiplier;
        a23 = a23 * multiplier;
        a33 = a33 * multiplier;

        // If columns 2-3 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)
        // do v*vr, where vr= (1, -c, 0, 0) -> this represents column swap as a quaternion, see the paper for more details
        conditional_swap<t, 1>( v, multiplier * half - half );

        //normalize the quaternion, because it can get denormalized form swapping
        normalize(v);


        //3. compute qr factorization
        svd::givens_conjugation< t, 1, 2 > ( a11, a12, a13, a21, a22, a23, a31, a32, a33, u.x, u.y, u.z, u.w );
        svd::givens_conjugation< t, 1, 3 > ( a11, a12, a13, a21, a22, a23, a31, a32, a33, u.x, u.y, u.z, u.w );
        svd::givens_conjugation< t, 2, 3 > ( a11, a12, a13, a21, a22, a23, a31, a32, a33, u.x, u.y, u.z, u.w );

        s.x = a11;
        s.y = a22;
        s.z = a33;

    }

    //obtain A = USV' 
    template < typename t > inline std::tuple< quaternion<t>, vector3<t>, quaternion<t> > compute_as_quaternion( const matrix3x3<t>& in )
    {
        quaternion<t> u;
        quaternion<t> v;
        vector3<t>    s;
        compute( in, u, s, v );
        return std::make_tuple ( std::move(u), std::move(s), std::move(v) );
    }

    //obtain A = USV' 
    template < typename t > inline void compute( const matrix3x3<t>& in, matrix3x3<t>& uu, vector3<t>& s, matrix3x3<t>& vv )
    {
        using namespace svd::math;

        // initial value of v as a quaternion
        auto vx = splat<t>( 0.0f );
        auto vy = splat<t>( 0.0f );
        auto vz = splat<t>( 0.0f );
        auto vw = splat<t>( 1.0f );

        auto u = create_quaternion ( vx, vy, vz, vw );
        auto v = create_quaternion ( vx, vy, vz, vw );

        auto m = create_symmetric_matrix( in );
        
        //1. Compute the V matrix as a quaternion

        //4 iterations of jacobi conjugation to obtain V
        for (auto i = 0; i < 4 ; ++i)
        {
            svd::jacobi_conjugation< t, 1, 2 > ( m, v );
            svd::jacobi_conjugation< t, 2, 3 > ( m, v );
            svd::jacobi_conjugation< t, 1, 3 > ( m, v );
        }

        //normalize the quaternion. this is optional
        normalize<t>(v);

        //convert quaternion v to matrix {
        auto tmp1 = v.x * v.x;
        auto tmp2 = v.y * v.y;
        auto tmp3 = v.z * v.z;

        auto v11  = v.w * v.w;
        auto v22  = v11 - tmp1;
        auto v33  = v22 - tmp2;

        v33 = v33 + tmp3;

        v22 = v22 + tmp2;
        v22 = v22 - tmp3;

        v11 = v11 + tmp1;
        v11 = v11 - tmp2;
        v11 = v11 - tmp3;

        tmp1 = v.x + v.x;
        tmp2 = v.y + v.y;
        tmp3 = v.z + v.z;

        auto v32 = v.w * tmp1;
        auto v13 = v.w * tmp2;
        auto v21 = v.w * tmp3;

        tmp1 = v.y * tmp1;
        tmp2 = v.z * tmp2;
        tmp3 = v.x * tmp3;

        auto v12 = tmp1 - v21;
        auto v23 = tmp2 - v32;
        auto v31 = tmp3 - v13;

        v21 = v21 + tmp1;
        v32 = v32 + tmp2;
        v13 = v13 + tmp3;
        //} convert quaternion to matrix

        // compute AV

        auto a11 = dot3( in.a11, in.a12, in.a13, v11, v21, v31 );
        auto a12 = dot3( in.a11, in.a12, in.a13, v12, v22, v32 );
        auto a13 = dot3( in.a11, in.a12, in.a13, v13, v23, v33 );

        auto a21 = dot3( in.a21, in.a22, in.a23, v11, v21, v31 );
        auto a22 = dot3( in.a21, in.a22, in.a23, v12, v22, v32 );
        auto a23 = dot3( in.a21, in.a22, in.a23, v13, v23, v33 );

        auto a31 = dot3( in.a31, in.a32, in.a33, v11, v21, v31 );
        auto a32 = dot3( in.a31, in.a32, in.a33, v12, v22, v32 );
        auto a33 = dot3( in.a31, in.a32, in.a33, v13, v23, v33 );

        //2. sort the singular values

        //compute the norms of the columns for comparison
        auto rho1 = dot3( a11, a21, a31, a11, a21, a31 );
        auto rho2 = dot3( a12, a22, a32, a12, a22, a32 );
        auto rho3 = dot3( a13, a23, a33, a13, a23, a33 );

        auto c = rho1 < rho2;

        // Swap columns 1-2 if necessary
        conditional_swap( c, a11, a12 );
        conditional_swap( c, a21, a22 );
        conditional_swap( c, a31, a32 );
        
        //either -1 or 1
        auto multiplier = negative_conditional_swap_multiplier( c );

        // If columns 1-2 have been swapped, negate 2nd column of A and V so that V is still a rotation
        a12 = a12 * multiplier;
        a22 = a22 * multiplier;
        a32 = a32 * multiplier;

        // If columns 1-2 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)
        // do v*vr, where vr= (1, 0, 0, -c) -> this represents column swap as a quaternion, see the paper for more details
       
        auto half = svd::math::splat<t> ( 0.5f );
        conditional_swap<t, 3>( v, multiplier * half - half );

        c = rho1 < rho3;

        // Swap columns 1-3 if necessary
        conditional_swap( c, a11, a13 );
        conditional_swap( c, a21, a23 );
        conditional_swap( c, a31, a33 );

        multiplier = negative_conditional_swap_multiplier( c );

        // If columns 1-3 have been swapped, negate 1st column of A and V so that V is still a rotation
        a11 = a11 * multiplier;
        a21 = a21 * multiplier;
        a31 = a31 * multiplier;

        // If columns 1-3 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)
        // do v*vr, where vr= (1, 0, -c, 0) -> this represents column swap as a quaternion, see the paper for more details
        conditional_swap<t, 2>( v, multiplier * half - half );

        c = rho2 < rho3;

        // Swap columns 2-3 if necessary
        conditional_swap( c, a12, a13 );
        conditional_swap( c, a22, a23 );
        conditional_swap( c, a32, a33 );

        multiplier = negative_conditional_swap_multiplier( c );

        // If columns 2-3 have been swapped, negate 3rd column of A and V so that V is still a rotation
        a13 = a13 * multiplier;
        a23 = a23 * multiplier;
        a33 = a33 * multiplier;

        // If columns 2-3 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)
        // do v*vr, where vr= (1, -c, 0, 0) -> this represents column swap as a quaternion, see the paper for more details
        conditional_swap<t, 1>( v, multiplier * half - half );

        //normalize the quaternion, because it can get denormalized form swapping
        normalize(v);


        //3. compute qr factorization
        svd::givens_conjugation< t, 1, 2 > ( a11, a12, a13, a21, a22, a23, a31, a32, a33, u.x, u.y, u.z, u.w );
        svd::givens_conjugation< t, 1, 3 > ( a11, a12, a13, a21, a22, a23, a31, a32, a33, u.x, u.y, u.z, u.w );
        svd::givens_conjugation< t, 2, 3 > ( a11, a12, a13, a21, a22, a23, a31, a32, a33, u.x, u.y, u.z, u.w );

        s.x = a11;
        s.y = a22;
        s.z = a33;

    }

    //obtain A = USV' 
    template < typename t > inline std::tuple< matrix3x3<t>, vector3<t>, matrix3x3<t> > compute_as_matrix( const matrix3x3<t>& in )
    {
        matrix3x3<t> u;
        matrix3x3<t> v;
        vector3<t>    s;
        compute( in, u, s, v );
        return std::make_tuple ( std::move(u), std::move(s), std::move(v) );
    }
}

std::int32_t main(int argc, _TCHAR* argv[])
{
    
    using namespace svd;
    using namespace svd::math;

    typedef svd::cpu_scalar number;

    auto m11 = svd::math::splat<number>( 2.0f );
    auto m12 = svd::math::splat<number>( -0.2f );
    auto m13 = svd::math::splat<number>( 1.0f );

    auto m21 = svd::math::splat<number>( -0.2f);
    auto m22 = svd::math::splat<number>( 1.0f);
    auto m23 = svd::math::splat<number>( 6.0f);

    auto m31 = svd::math::splat<number>( 15.0f);
    auto m32 = svd::math::splat<number>( 0.0f);
    auto m33 = svd::math::splat<number>( 8.0f);

    auto urv = svd::compute_as_matrix<number>( svd::create_matrix ( m11, m12, m13, m21, m22, m23, m31, m32, m33 ) );
    
    return 0;
}

