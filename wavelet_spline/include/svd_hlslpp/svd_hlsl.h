#ifndef __svd_hlsl_h__
#define __svd_hlsl_h__

#if defined(__cplusplus)
#pragma warning(disable : 4189)
#pragma warning(disable : 4127)
#endif


#include "svd_hlsl_types.h"
#include "svd_hlsl_math.h"



/*
Reference implementation of paper:

Computing the Singular Value Decomposition of 3 x 3 matrices with minimal branching and elementary floating point operations 

Notes: For production usage please consider inlining all functions into one. This will eliminate parameter passing 

*/

#if defined(__cplusplus)
#define HLSLCPP_IN_OUT(v)   v&
#define HLSLCPP_IN(v)    const v&
#else
#define HLSLCPP_HLSLCPP_IN_OUT(v)   inout v
#define HLSLCPP_IN_HLSLCPP_IN(v)    const in v 
#endif




namespace svdhlslcpp
{
    struct givens_quaternion_t
    {
        float m_ch;
        float m_sh;
    };

    inline givens_quaternion_t approximate_givens_quaternion(float a11, float a12, float a22 )
    {
        float half  = splat ( 0.5f );
        float sh    = a12 * half;
        float id    = cmp_ge( sh * sh,  splat ( tiny_number ) );
        
        //if sh squared is tiny, make sh = 0 and ch = 1. this comes from the several jacobi iterations
        sh		    = blend( zero(), sh, id);
        float ch    = blend (one(), a11 - a22, id );

        float sh_2 = sh * sh;
        float ch_2 = ch * ch;

        float x = ch_2 + sh_2;
        float w = rsqrt ( x );

        //one iteration of newton rhapson.
        w = w + ( w * half ) - ( ( w * half )  *  w * w * x  );

        sh = w * sh;
        ch = w * ch;

        float b = cmp_le( ch_2, sh_2 * splat( four_gamma_squared ) ) ;

        sh = blend ( sh, splat( sine_pi_over_eight ), b );
        ch = blend ( ch, splat( cosine_pi_over_eight ), b );

        givens_quaternion_t r;
        r.m_ch = ch;
        r.m_sh = sh;
        return r;
    }

    //(1,2), (1,3), (2,3)
    //jacobi conjugation of a symmetric matrix
    inline void jacobi_conjugation
                                                        (   
                                                            const int p,
                                                            const int q,
                                                            HLSLCPP_IN_OUT(float) a11, //t&  a12, t&  a13,
                                                            HLSLCPP_IN_OUT(float) a21, HLSLCPP_IN_OUT(float) a22, //t&  a23,
                                                            HLSLCPP_IN_OUT(float) a31, HLSLCPP_IN_OUT(float) a32, HLSLCPP_IN_OUT(float)  a33,
                                                            HLSLCPP_IN_OUT(float) qx,  HLSLCPP_IN_OUT(float) qy,  HLSLCPP_IN_OUT(float)  qz, HLSLCPP_IN_OUT(float) qw
                                                        )
    {
        if ( p == 1 && q == 2 )
        {
            givens_quaternion_t r  = approximate_givens_quaternion ( a11, a21, a22 );
            float ch = r.m_ch;
            float sh = r.m_sh;

            float ch_minus_sh_2 = ch * ch - sh * sh;
            float ch_sh_2       = ch * sh + ch * sh;

            //Q matrix in the jaocobi method, formed from quaternion
            float r11 = ch_minus_sh_2;
            float r12 = ch_sh_2;             
            float r21 = zero() - ch_sh_2;
            float r22 = ch_minus_sh_2;

            float c = r11;
            float s = r12;

            float t1 = a31;
            float t2 = a32;

            a31 = c * t1 + s * t2;
            a32 = c * t2 - s * t1;
            a33 = a33;

            float t3 = a11;
            float t4 = a21;
            float t5 = a22;

            a11 = s * s * t5 + c * c * t3 + ( s * c + s * c) * t4;
            a22 = c * c * t5 + s * s * t3 - ( s * c + s * c) * t4;
            a21 = s * c * ( t5 - t3 ) + ( c * c - s * s ) * t4; 


            //now create the apply the total quaternion transformation
            float q0 = qw;
            float q1 = qx;
            float q2 = qy;
            float q3 = qz;

            float r0 = ch;
            float r1 = 0;
            float r2 = 0;
            float r3 = sh;

            qw = r0 * q0 - r3 * q3;
            qx = r0 * q1 + r3 * q2;
            qy = r0 * q2 - r3 * q1;
            qz = r0 * q3 + r3 * q0;

        }
        else if ( p == 2 && q == 3 )
        {
            givens_quaternion_t r  = approximate_givens_quaternion ( a22, a32, a33 );
            float ch = r.m_ch;
            float sh = r.m_sh;

            float ch_minus_sh_2 = ch * ch - sh * sh;
            float ch_sh_2       = ch * sh + ch * sh;

            //Q matrix in the jaocobi method, formed from quaternion
            float r11 = ch_minus_sh_2;
            float r12 = ch_sh_2;             
            float r21 = zero() - ch_sh_2;
            float r22 = ch_minus_sh_2;

            float c = r11;
            float s = r12;

            float t1 = a21;
            float t2 = a31;

            a21 = c * t1 + s * t2;
            a31 = c * t2 - s * t1;
            a11 = a11;

            float t3 = a22;
            float t4 = a32;
            float t5 = a33;

            a22 = s * s * t5 + c * c * t3 + ( s * c + s * c) * t4;
            a33 = c * c * t5 + s * s * t3 - ( s * c + s * c) * t4;
            a32 = s * c * ( t5 - t3 ) + ( c * c - s * s ) * t4; 

            //now create the apply the total quaternion transformation
            float q0 = qw;
            float q1 = qx;
            float q2 = qy;
            float q3 = qz;

            float r0 = ch;
            float r1 = sh;
            float r2 = 0;
            float r3 = 0;

            qw = r0 * q0 - r1 * q1;
            qx = r0 * q1 + r1 * q0;
            qy = r0 * q2 + r1 * q3;
            qz = r0 * q3 - r1 * q2;
        }
        else if ( p == 1 && q == 3 )
        {
            givens_quaternion_t r  = approximate_givens_quaternion ( a33, a31, a11 );
            float ch = r.m_ch;
            float sh = r.m_sh;

            float ch_minus_sh_2 = ch * ch - sh * sh;
            float ch_sh_2       = ch * sh + ch * sh;

            //Q matrix in the jaocobi method, formed from quaternion
            float r11 = ch_minus_sh_2;
            float r12 = ch_sh_2;             
            float r21 = zero() - ch_sh_2;
            float r22 = ch_minus_sh_2;

            float c = r11;
            float s = r12;

            float t1 = a32;
            float t2 = a21;

            a32 = c * t1 + s * t2;
            a21 = c * t2 - s * t1;
            a22 = a22;

            float t3 = a33;
            float t4 = a31;
            float t5 = a11;

            a33 = s * s * t5 + c * c * t3 + ( s * c + s * c) * t4;
            a11 = c * c * t5 + s * s * t3 - ( s * c + s * c) * t4;
            a31 = s * c * ( t5 - t3 ) + ( c * c - s * s ) * t4; 

            //now create the apply the total quaternion transformation
            float q0 = qw;
            float q1 = qx;
            float q2 = qy;
            float q3 = qz;

            float r0 = ch;
            float r1 = 0;
            float r2 = sh;
            float r3 = 0;

            qw = r0 * q0 - r2 * q2;
            qx = r0 * q1 - r2 * q3;
            qy = r0 * q2 + r2 * q0;
            qz = r0 * q3 + r2 * q1;
        }
    }

    //(1,2), (1,3), (2,3)
    //jacobi conjugation of a symmetric matrix
    inline void jacobi_conjugation ( const int p, const int q, HLSLCPP_IN_OUT(symmetric_matrix3x3) m, HLSLCPP_IN_OUT(quaternion) quaternion )
    {
        jacobi_conjugation ( p, q, m.a11, m.a21, m.a22, m.a31, m.a32, m.a33, quaternion.x, quaternion.y, quaternion.z, quaternion.w) ;
    }

    inline symmetric_matrix3x3 create_symmetric_matrix ( HLSLCPP_IN(matrix3x3) inp  )
    {
        float a11 = inp.a11 * inp.a11 + inp.a21 * inp.a21 + inp.a31 * inp.a31;
        float a12 = inp.a11 * inp.a12 + inp.a21 * inp.a22 + inp.a31 * inp.a32;
        float a13 = inp.a11 * inp.a13 + inp.a21 * inp.a23 + inp.a31 * inp.a33;

        float a21 = a12;
        float a22 = inp.a12 * inp.a12 + inp.a22 * inp.a22 + inp.a32 * inp.a32;
        float a23 = inp.a12 * inp.a13 + inp.a22 * inp.a23 + inp.a32 * inp.a33;

        float a31 = a13;
        float a32 = a23;
        float a33 = inp.a13 * inp.a13 + inp.a23 * inp.a23 + inp.a33 * inp.a33;

        symmetric_matrix3x3 r = { a11, a21, a22, a31, a32, a33 };
        return r;
    }

    inline quaternion normalize( const quaternion q )
    {
        float half = splat ( 0.5f );
        
        float x = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
        float w = rsqrt( x );

        //one iteration of newton rhapson.
        w = w + ( w * half ) - ( ( w * half )  *  w * w * x  );

        return create_quaternion( q.x * w, q.y * w, q.z * w, q.w * w);
    }

    inline void conditional_swap( float c, HLSLCPP_IN_OUT(float) x, HLSLCPP_IN_OUT(float) y )
    {
        float a = x;
        float b = y;
        x = blend(a, b, c);
        y = blend(a, b, 1.0f - c);
    }

    //returns -1.0f or 1.0f depending on c
    //used for conditional_negative_swap
    inline float negative_conditional_swap_multiplier( float c )
    {
        return blend(-one(), one(), 1.0f - c);
    }

    inline void conditional_swap( const int axis, HLSLCPP_IN_OUT(quaternion) v, float c )
    {
        if (axis == 3 )
        {
            // If columns 1-2 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)
            // do v*vr, where vr= (1, 0, 0, -c) -> this represents column swap as a quaternion, see the paper for more details

            float w = v.w;
            float x = v.x;
            float y = v.y;
            float z = v.z;

            v.w = w + c * z;
            v.x = x - c * y;
            v.y = y + c * x;
            v.z = z - c * w;
        }
        else if ( axis == 2 )
        {
            // If columns 1-3 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)
            // do v*vr, where vr= (1, 0, -c, 0) -> this represents column swap as a quaternion, see the paper for more details

            float w = v.w;
            float x = v.x;
            float y = v.y;
            float z = v.z;

            v.w = w + c * y;
            v.x = x + c * z;
            v.y = y - c * w;
            v.z = z - c * x;

        }
        else if ( axis == 1 )
        {
            // If columns 2-3 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)
            // do v*vr, where vr= (1, -c, 0, 0) -> this represents column swap as a quaternion, see the paper for more details

            float w = v.w;
            float x = v.x;
            float y = v.y;
            float z = v.z;

            v.w = w + c * x;
            v.x = x - c * w;
            v.y = y - c * z;
            v.z = z + c * y;
        }
    }

    inline givens_quaternion_t givens_quaternion( float a1, float a2 )
    {
        float half = splat ( 0.5f );

        float id = cmp_ge( a2 * a2,  splat ( small_number ) );
        float sh = blend(zero(), a2, id);

        float ch = max ( a1, zero() - a1 );
        float c	= cmp_le( a1, zero() );
        ch = max ( ch, splat(small_number ) );

        // compute sqrt(ch * ch + sh * sh )
        float x = ch * ch + sh * sh;
        float w = rsqrt( x );
        //one iteration of newton rhapson.
        w = w + ( w * half ) - ( ( w * half )  *  w * w * x  );
        w = x * w; 

        float rho = w;
        ch = ch + rho;

        conditional_swap(c, ch, sh );

        x = ch * ch + sh * sh;
        w = rsqrt( x );
        //one iteration of newton rhapson.
        w = w + ( w * half ) - ( ( w * half )  *  w * w * x  );

        ch = w * ch;
        sh = w * sh;

        givens_quaternion_t r;

        r.m_ch = ch;
        r.m_sh = sh;

        return r;
    }

    //(1,2), (1,3), (2,3)
    //jacobi conjugation of a symmetric matrix
    inline void givens_conjugation
                                                        (   
                                                            const int p, const int q,
                                                            HLSLCPP_IN_OUT(float) a11, HLSLCPP_IN_OUT(float) a12, HLSLCPP_IN_OUT(float) a13,
                                                            HLSLCPP_IN_OUT(float) a21, HLSLCPP_IN_OUT(float) a22, HLSLCPP_IN_OUT(float) a23,
                                                            HLSLCPP_IN_OUT(float) a31, HLSLCPP_IN_OUT(float) a32, HLSLCPP_IN_OUT(float) a33,
                                                            HLSLCPP_IN_OUT(float) qx,  HLSLCPP_IN_OUT(float) qy,  HLSLCPP_IN_OUT(float) qz, HLSLCPP_IN_OUT(float) qw
                                                        )
    {
        if ( p == 1 && q == 2 )
        {
            givens_quaternion_t r  = givens_quaternion ( a11, a21 );
            float ch = r.m_ch;
            float sh = r.m_sh;

            float ch_minus_sh_2 = ch * ch - sh * sh;
            float ch_sh_2       = ch * sh + ch * sh;

            //Q matrix in the jaocobi method, formed from quaternion
            float r11 = ch_minus_sh_2;
            float r12 = ch_sh_2;             
            float r21 = zero() - ch_sh_2; 
            float r22 = ch_minus_sh_2;

            float c = r11;
            float s = r12;

            float t11 = a11;
            float t12 = a12;
            float t13 = a13;

            float t21 = a21;
            float t22 = a22;
            float t23 = a23;

            a11 = c * t11 + s * t21;
            a21 = c * t21 - s * t11;   

            a12 = c * t12 + s * t22;
            a22 = c * t22 - s * t12;
            
            a13 = c * t13 + s * t23;
            a23 = c * t23 - s * t13;

            //now create the apply the total quaternion transformation7
            float w = qw;
            float x = qx;
            float y = qy;
            float z = qz;

            //use the fact, that x,y,z = 0
            qw = ch * w;
            qx = x;
            qy = x;
            qz = sh * w;

        }
        else if ( p == 2 && q == 3 )
        {
            givens_quaternion_t r  = givens_quaternion ( a22, a32 );
            float ch = r.m_ch;
            float sh = r.m_sh;

            float ch_minus_sh_2 = ch * ch - sh * sh;
            float ch_sh_2       = ch * sh + ch * sh;

            //Q matrix in the jaocobi method, formed from quaternion
            float r11 = ch_minus_sh_2;
            float r12 = ch_sh_2;             
            float r21 = zero() - ch_sh_2; 
            float r22 = ch_minus_sh_2;

            float c = r11;
            float s = r12;

            float t11 = a21;
            float t12 = a22;
            float t13 = a23;

            float t21 = a31;
            float t22 = a32;
            float t23 = a33;

            a21 = c * t11 + s * t21;
            a31 = c * t21 - s * t11;   

            a22 = c * t12 + s * t22;
            a32 = c * t22 - s * t12;
            
            a23 = c * t13 + s * t23;
            a33 = c * t23 - s * t13;

            //now create the apply the total quaternion transformation7
            float w = qw;
            float x = qx;
            float y = qy;
            float z = qz;

            //Quaternion[ ch, sh, 0, 0] -> q * r
            qw = ch * w - sh * x;
            qx = ch * x + sh * w;
            qy = ch * y + sh * z;
            qz = ch * z - sh * y;

        }
        else if ( p == 1 && q == 3 )
        {
            givens_quaternion_t r  = givens_quaternion ( a11, a31 );
            float ch = r.m_ch;
            float sh = r.m_sh;

            float ch_minus_sh_2 = ch * ch - sh * sh;
            float ch_sh_2       = ch * sh + ch * sh;

            //Q matrix in the jaocobi method, formed from quaternion
            float r11 = ch_minus_sh_2;
            float r12 = ch_sh_2;             
            float r21 = zero() - ch_sh_2; 
            float r22 = ch_minus_sh_2;

            float c = r11;
            float s = r12;

            float t11 = a11;
            float t12 = a12;
            float t13 = a13;

            float t21 = a31;
            float t22 = a32;
            float t23 = a33;

            a11 = c * t11 + s * t21;
            a31 = c * t21 - s * t11;   

            a12 = c * t12 + s * t22;
            a32 = c * t22 - s * t12;
            
            a13 = c * t13 + s * t23;
            a33 = c * t23 - s * t13;

            //now create the apply the total quaternion transformation
            float w = qw;
            float x = qx;
            float y = qy;
            float z = qz;

            //use the fact, that x = 0 and y = 0 from the previous iteration
            qw = ch * w;
            qx = sh * z;
            qy = zero()-sh * w;
            qz = ch * z;
        }
    }

    //obtain A = USV' 
    inline void compute( HLSLCPP_IN(matrix3x3) inp, HLSLCPP_IN_OUT(quaternion) u, HLSLCPP_IN_OUT(vector3) s, HLSLCPP_IN_OUT(quaternion) v )
    {
        // initial value of v as a quaternion
        float vx = splat( 0.0f );
        float vy = splat( 0.0f );
        float vz = splat( 0.0f );
        float vw = splat( 1.0f );

        u = create_quaternion ( vx, vy, vz, vw );
        v = create_quaternion ( vx, vy, vz, vw );

        symmetric_matrix3x3 m = create_symmetric_matrix( inp );
        
        //1. Compute the V matrix as a quaternion

        //4 iterations of jacobi conjugation to obtain V
        for (int i = 0; i < 4 ; ++i)
        {
            jacobi_conjugation( 1, 2, m, v );
            jacobi_conjugation( 2, 3, m, v );
            jacobi_conjugation( 1, 3, m, v );
        }

        //normalize the quaternion. this is optional
        v = normalize(v);

        //convert quaternion v to matrix {
        float tmp1 = v.x * v.x;
        float tmp2 = v.y * v.y;
        float tmp3 = v.z * v.z;

        float v11  = v.w * v.w;
        float v22  = v11 - tmp1;
        float v33  = v22 - tmp2;

        v33 = v33 + tmp3;

        v22 = v22 + tmp2;
        v22 = v22 - tmp3;

        v11 = v11 + tmp1;
        v11 = v11 - tmp2;
        v11 = v11 - tmp3;

        tmp1 = v.x + v.x;
        tmp2 = v.y + v.y;
        tmp3 = v.z + v.z;

        float v32 = v.w * tmp1;
        float v13 = v.w * tmp2;
        float v21 = v.w * tmp3;

        tmp1 = v.y * tmp1;
        tmp2 = v.z * tmp2;
        tmp3 = v.x * tmp3;

        float v12 = tmp1 - v21;
        float v23 = tmp2 - v32;
        float v31 = tmp3 - v13;

        v21 = v21 + tmp1;
        v32 = v32 + tmp2;
        v13 = v13 + tmp3;
        //} convert quaternion to matrix

        // compute AV

        float a11 = dot3( inp.a11, inp.a12, inp.a13, v11, v21, v31 );
        float a12 = dot3( inp.a11, inp.a12, inp.a13, v12, v22, v32 );
        float a13 = dot3( inp.a11, inp.a12, inp.a13, v13, v23, v33 );

        float a21 = dot3( inp.a21, inp.a22, inp.a23, v11, v21, v31 );
        float a22 = dot3( inp.a21, inp.a22, inp.a23, v12, v22, v32 );
        float a23 = dot3( inp.a21, inp.a22, inp.a23, v13, v23, v33 );

        float a31 = dot3( inp.a31, inp.a32, inp.a33, v11, v21, v31 );
        float a32 = dot3( inp.a31, inp.a32, inp.a33, v12, v22, v32 );
        float a33 = dot3( inp.a31, inp.a32, inp.a33, v13, v23, v33 );

        //2. sort the singular values

        //compute the norms of the columns for comparison
        float rho1 = dot3( a11, a21, a31, a11, a21, a31 );
        float rho2 = dot3( a12, a22, a32, a12, a22, a32 );
        float rho3 = dot3( a13, a23, a33, a13, a23, a33 );

        float c = rho1 < rho2;

        // Swap columns 1-2 if necessary
        conditional_swap( c, a11, a12 );
        conditional_swap( c, a21, a22 );
        conditional_swap( c, a31, a32 );
        
        //either -1 or 1
        float multiplier = negative_conditional_swap_multiplier( c );

        // If columns 1-2 have been swapped, negate 2nd column of A and V so that V is still a rotation
        a12 = a12 * multiplier;
        a22 = a22 * multiplier;
        a32 = a32 * multiplier;

        // If columns 1-2 have been swapped, also update quaternion representation of V (the quaternion may become un-normalized after this)
        // do v*vr, where vr= (1, 0, 0, -c) -> this represents column swap as a quaternion, see the paper for more details
       
        float half = splat ( 0.5f );
        conditional_swap( 3, v, multiplier * half - half );

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
        conditional_swap( 2, v, multiplier * half - half );

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
        conditional_swap( 1, v, multiplier * half - half );

        //normalize the quaternion, because it can get denormalized form swapping
        normalize(v);


        //3. compute qr factorization
        givens_conjugation( 1, 2, a11, a12, a13, a21, a22, a23, a31, a32, a33, u.x, u.y, u.z, u.w );
        givens_conjugation( 1, 3, a11, a12, a13, a21, a22, a23, a31, a32, a33, u.x, u.y, u.z, u.w );
        givens_conjugation( 2, 3, a11, a12, a13, a21, a22, a23, a31, a32, a33, u.x, u.y, u.z, u.w );

        s.x = a11;
        s.y = a22;
        s.z = a33;
    }

    struct svd_result_quaternion_usv
    {
        quaternion		m_u;
        vector3			m_s;
        quaternion		m_v;
    };

    struct svd_result_quaternion_uv
    {
        quaternion	m_u;
        quaternion	m_v;
    };

    //obtain A = USV' 
    inline svd_result_quaternion_usv compute_as_quaternion_rusv( HLSLCPP_IN(matrix3x3) inp )
    {
        quaternion u;
        quaternion v;
        vector3    s;
        compute( inp, u, s, v );
        svd_result_quaternion_usv r;
        r.m_u = u;
        r.m_s = s;
        r.m_v = v;
        return r;
    }

    //obtain A = USV' 
    inline svd_result_quaternion_uv compute_as_quaternion_ruv( HLSLCPP_IN(matrix3x3) inp )
    {
        quaternion		u;
        quaternion		v;
        vector3			s;
        compute( inp, u, s, v );
        svd_result_quaternion_uv r;
        r.m_u = u;
        r.m_v = v;
        return r;
    }

    //obtain A = USV' 
    inline  void compute_as_quaternion_usv( HLSLCPP_IN(matrix3x3) inp, HLSLCPP_IN_OUT(quaternion) u, HLSLCPP_IN_OUT(vector3) s, HLSLCPP_IN_OUT(quaternion) v )
    {
        compute( inp, u, s, v );
     }

    //obtain A = USV' 
    inline void  compute_as_quaternion_uv( HLSLCPP_IN(matrix3x3) inp, HLSLCPP_IN_OUT(quaternion) u, HLSLCPP_IN_OUT(quaternion) v )
    {
        vector3    s;
        compute( inp, u, s, v );
    }

    //(1,2), (1,3), (2,3)
    //jacobi conjugation of a symmetric matrix
    inline void givens_conjugation
                                                        ( 
                                                            const int p,
                                                            const int q,
                                                            HLSLCPP_IN_OUT(float)  a11, HLSLCPP_IN_OUT(float)  a12, HLSLCPP_IN_OUT(float)  a13,
                                                            HLSLCPP_IN_OUT(float)  a21, HLSLCPP_IN_OUT(float)  a22, HLSLCPP_IN_OUT(float)  a23,
                                                            HLSLCPP_IN_OUT(float)  a31, HLSLCPP_IN_OUT(float)  a32, HLSLCPP_IN_OUT(float)  a33,

                                                            HLSLCPP_IN_OUT(float)  u11, HLSLCPP_IN_OUT(float)  u12, HLSLCPP_IN_OUT(float)  u13,
                                                            HLSLCPP_IN_OUT(float)  u21, HLSLCPP_IN_OUT(float)  u22, HLSLCPP_IN_OUT(float)  u23,
                                                            HLSLCPP_IN_OUT(float)  u31, HLSLCPP_IN_OUT(float)  u32, HLSLCPP_IN_OUT(float)  u33
                                                        )
    {
        if ( p == 1 && q == 2 )
        {
            givens_quaternion_t r  = givens_quaternion ( a11, a21 );
            float ch = r.m_ch;
            float sh = r.m_sh;

            float ch_minus_sh_2 = ch * ch - sh * sh;
            float ch_sh_2       = ch * sh + ch * sh;

            //Q matrix in the jaocobi method, formed from quaternion
            float r11 = ch_minus_sh_2;
            float r12 = ch_sh_2;             
            float r21 = zero() - ch_sh_2; 
            float r22 = ch_minus_sh_2;

            float c = r11;
            float s = r12;

            float t11 = a11;
            float t12 = a12;
            float t13 = a13;

            float t21 = a21;
            float t22 = a22;
            float t23 = a23;

            a11 = c * t11 + s * t21;
            a21 = c * t21 - s * t11;   

            a12 = c * t12 + s * t22;
            a22 = c * t22 - s * t12;
            
            a13 = c * t13 + s * t23;
            a23 = c * t23 - s * t13;

            //u = { { c, -s, 0}, {  s, c, 0}, { 0, 0, 1 } }
            //u1.u

            float k11 = u11;
            float k12 = u12;
            float k13 = u13;

            float k21 = u21;
            float k22 = u22;
            float k23 = u23;

            float k31 = u31;
            float k32 = u32;
            float k33 = u33;

            /*
            u11 = c * k11 + s * k12;
            u12 = c * k12 - s * k11;
            u13 = k13;

            u21 = c * k21 + s * k22;
            u22 = c * k22 - s * k21;
            u23 = k23;

            u31 = c * k31 + s * k32;
            u32 = c * k32 - s * k31;
            u33 = k33;
            */

            //explore the fact that initially we have identity matrix

            u11 = c;
            u12 = zero() - s;
            u13 = zero();

            u21 = s;
            u22 = c;
            u23 = zero();

            u31 = zero();
            u32 = zero();
            u33 = splat(1.0f);
        }
        else if ( p == 2 && q == 3 )
        {
            givens_quaternion_t r  = givens_quaternion ( a22, a32 );
            float ch = r.m_ch;
            float sh = r.m_sh;

            float ch_minus_sh_2 = ch * ch - sh * sh;
            float ch_sh_2       = ch * sh + ch * sh;

            //Q matrix in the jaocobi method, formed from quaternion
            float r11 = ch_minus_sh_2;
            float r12 = ch_sh_2;             
            float r21 = zero() - ch_sh_2; 
            float r22 = ch_minus_sh_2;

            float c = r11;
            float s = r12;

            float t11 = a21;
            float t12 = a22;
            float t13 = a23;

            float t21 = a31;
            float t22 = a32;
            float t23 = a33;

            a21 = c * t11 + s * t21;
            a31 = c * t21 - s * t11;   

            a22 = c * t12 + s * t22;
            a32 = c * t22 - s * t12;
            
            a23 = c * t13 + s * t23;
            a33 = c * t23 - s * t13;

            //u = { { c, -s, 0}, {  s, c, 0}, { 0, 0, 1 } }
            //u1.u

            float k11 = u11;
            float k12 = u12;
            float k13 = u13;

            float k21 = u21;
            float k22 = u22;
            float k23 = u23;

            float k31 = u31;
            float k32 = u32;
            float k33 = u33;

            //u = { { 1, 0, 0}, {  0, c, -s}, { 0, s, c } }
            //u1.u
            u11 = k11;
            u12 = c * k12 + s * k13;
            u13 = c * k13 - s * k12;

            u21 = k21;
            u22 = c * k22 + s * k23;
            u23 = c * k23 - s * k22;

            u31 = k31;
            u32 = c * k32 + s * k33;
            u33 = c * k33 - s * k32;
        }
        else if ( p == 1 && q == 3 )
        {
            givens_quaternion_t r  = givens_quaternion ( a11, a31 );
            float ch = r.m_ch;
            float sh = r.m_sh;

            float ch_minus_sh_2 = ch * ch - sh * sh;
            float ch_sh_2       = ch * sh + ch * sh;

            //Q matrix in the jaocobi method, formed from quaternion
            float r11 = ch_minus_sh_2;
            float r12 = ch_sh_2;             
            float r21 = zero() - ch_sh_2; 
            float r22 = ch_minus_sh_2;

            float c = r11;
            float s = r12;

            float t11 = a11;
            float t12 = a12;
            float t13 = a13;

            float t21 = a31;
            float t22 = a32;
            float t23 = a33;

            a11 = c * t11 + s * t21;
            a31 = c * t21 - s * t11;   

            a12 = c * t12 + s * t22;
            a32 = c * t22 - s * t12;
            
            a13 = c * t13 + s * t23;
            a33 = c * t23 - s * t13;


            //u = { { c, -s, 0}, {  s, c, 0}, { 0, 0, 1 } }
            //u1.u

            float k11 = u11;
            float k12 = u12;
            float k13 = u13;

            float k21 = u21;
            float k22 = u22;
            float k23 = u23;

            float k31 = u31;
            float k32 = u32;
            float k33 = u33;

            
            //u = { { c, 0, -s}, {  0, 1, 0}, { s, 0, c } }
            //u1.u

            /*
            u11 = c * k11 + s * k13;
            u12 = k12;
            u13 = c * k13 - s * k11;

            u21 = c * k21 + s * k23;
            u22 = k22;
            u23 = c * k23 - s * k21;

            u31 = c * k31 + s * k33;
            u32 = k32;
            u33 = c * k33 - s * k31;
            */

            //explore the special structure from the previous iteration
            u11 = c * k11;
            u12 = k12;
            u13 = zero() - s * k11;

            u21 = zero() - c * k12;
            u22 = k11;
            u23 = s * k12;

            u31 = s;
            u32 = zero();
            u33 = c;

        }
    }

    //obtain A = USV' 
    inline void compute(HLSLCPP_IN(matrix3x3) inp, HLSLCPP_IN_OUT(matrix3x3) uu, HLSLCPP_IN_OUT(vector3) s, HLSLCPP_IN_OUT(matrix3x3) vv )
    {
        // initial value of v as a quaternion
        float vx = splat( 0.0f );
        float vy = splat( 0.0f );
        float vz = splat( 0.0f );
        float vw = splat( 1.0f );

        quaternion u = create_quaternion ( vx, vy, vz, vw );
        quaternion v = create_quaternion ( vx, vy, vz, vw );

        symmetric_matrix3x3 m = create_symmetric_matrix( inp );

        uu.a11 = splat(1.0f);
        uu.a12 = splat(0.0f);
        uu.a13 = splat(0.0f);

        uu.a21 = splat(0.0f);
        uu.a22 = splat(1.0f);
        uu.a23 = splat(0.0f);

        uu.a31 = splat(0.0f);
        uu.a32 = splat(0.0f);
        uu.a33 = splat(1.0f);
        
        //1. Compute the V matrix as a quaternion

        //4 iterations of jacobi conjugation to obtain V
        for (int i = 0; i < 4 ; ++i)
        {
            jacobi_conjugation( 1, 2, m, v );
            jacobi_conjugation( 2, 3, m, v );
            jacobi_conjugation( 1, 3, m, v );
        }

        //normalize the quaternion. this is optional
        normalize(v);

        //convert quaternion v to matrix {
        float tmp1 = v.x * v.x;
        float tmp2 = v.y * v.y;
        float tmp3 = v.z * v.z;

        float v11  = v.w * v.w;
        float v22  = v11 - tmp1;
        float v33  = v22 - tmp2;

        v33 = v33 + tmp3;

        v22 = v22 + tmp2;
        v22 = v22 - tmp3;

        v11 = v11 + tmp1;
        v11 = v11 - tmp2;
        v11 = v11 - tmp3;

        tmp1 = v.x + v.x;
        tmp2 = v.y + v.y;
        tmp3 = v.z + v.z;

        float v32 = v.w * tmp1;
        float v13 = v.w * tmp2;
        float v21 = v.w * tmp3;

        tmp1 = v.y * tmp1;
        tmp2 = v.z * tmp2;
        tmp3 = v.x * tmp3;

        float v12 = tmp1 - v21;
        float v23 = tmp2 - v32;
        float v31 = tmp3 - v13;

        v21 = v21 + tmp1;
        v32 = v32 + tmp2;
        v13 = v13 + tmp3;
        //} convert quaternion to matrix

        // compute AV
        float a11 = dot3( inp.a11, inp.a12, inp.a13, v11, v21, v31 );
        float a12 = dot3( inp.a11, inp.a12, inp.a13, v12, v22, v32 );
        float a13 = dot3( inp.a11, inp.a12, inp.a13, v13, v23, v33 );

        float a21 = dot3( inp.a21, inp.a22, inp.a23, v11, v21, v31 );
        float a22 = dot3( inp.a21, inp.a22, inp.a23, v12, v22, v32 );
        float a23 = dot3( inp.a21, inp.a22, inp.a23, v13, v23, v33 );

        float a31 = dot3( inp.a31, inp.a32, inp.a33, v11, v21, v31 );
        float a32 = dot3( inp.a31, inp.a32, inp.a33, v12, v22, v32 );
        float a33 = dot3( inp.a31, inp.a32, inp.a33, v13, v23, v33 );

        //2. sort the singular values

        //compute the norms of the columns for comparison
        float rho1 = dot3( a11, a21, a31, a11, a21, a31 );
        float rho2 = dot3( a12, a22, a32, a12, a22, a32 );
        float rho3 = dot3( a13, a23, a33, a13, a23, a33 );

        float c = rho1 < rho2;

        // Swap columns 1-2 if necessary
        conditional_swap( c, a11, a12 );
        conditional_swap( c, a21, a22 );
        conditional_swap( c, a31, a32 );

        conditional_swap( c, v11, v12 );
        conditional_swap( c, v21, v22 );
        conditional_swap( c, v31, v32 );

        //either -1 or 1
        float multiplier = negative_conditional_swap_multiplier( c );

        // If columns 1-2 have been swapped, negate 2nd column of A and V so that V is still a rotation
        a12 = a12 * multiplier;
        a22 = a22 * multiplier;
        a32 = a32 * multiplier;

        v12 = v12 * multiplier;
        v22 = v22 * multiplier;
        v32 = v32 * multiplier;

        c = rho1 < rho3;

        // Swap columns 1-3 if necessary
        conditional_swap( c, a11, a13 );
        conditional_swap( c, a21, a23 );
        conditional_swap( c, a31, a33 );

        conditional_swap( c, v11, v13 );
        conditional_swap( c, v21, v23 );
        conditional_swap( c, v31, v33 );


        multiplier = negative_conditional_swap_multiplier( c );

        // If columns 1-3 have been swapped, negate 1st column of A and V so that V is still a rotation
        a11 = a11 * multiplier;
        a21 = a21 * multiplier;
        a31 = a31 * multiplier;

        v11 = v11 * multiplier;
        v21 = v21 * multiplier;
        v31 = v31 * multiplier;

        c = rho2 < rho3;

        // Swap columns 2-3 if necessary
        conditional_swap( c, a12, a13 );
        conditional_swap( c, a22, a23 );
        conditional_swap( c, a32, a33 );

        conditional_swap( c, v12, v13 );
        conditional_swap( c, v22, v23 );
        conditional_swap( c, v32, v33 );


        multiplier = negative_conditional_swap_multiplier( c );

        // If columns 2-3 have been swapped, negate 3rd column of A and V so that V is still a rotation
        a13 = a13 * multiplier;
        a23 = a23 * multiplier;
        a33 = a33 * multiplier;

        v13 = v13 * multiplier;
        v23 = v23 * multiplier;
        v33 = v33 * multiplier;

        //3. compute qr factorization
        givens_conjugation( 1, 2, a11, a12, a13, a21, a22, a23, a31, a32, a33, uu.a11, uu.a12, uu.a13, uu.a21, uu.a22, uu.a23, uu.a31, uu.a32, uu.a33 );
        givens_conjugation( 1, 3, a11, a12, a13, a21, a22, a23, a31, a32, a33, uu.a11, uu.a12, uu.a13, uu.a21, uu.a22, uu.a23, uu.a31, uu.a32, uu.a33 );
        givens_conjugation( 2, 3, a11, a12, a13, a21, a22, a23, a31, a32, a33, uu.a11, uu.a12, uu.a13, uu.a21, uu.a22, uu.a23, uu.a31, uu.a32, uu.a33 );

        s.x = a11;
        s.y = a22;
        s.z = a33;

        vv.a11 = v11;
        vv.a12 = v12;
        vv.a13 = v13;

        vv.a21 = v21;
        vv.a22 = v22;
        vv.a23 = v23;

        vv.a31 = v31;
        vv.a32 = v32;
        vv.a33 = v33;
    }

    struct svd_result_matrix_usv
    {
        matrix3x3    m_u;
        vector3		 m_s;
        matrix3x3    m_v;
    };

    struct svd_result_matrix_uv
    {
        matrix3x3 m_u;
        matrix3x3 m_v;
    };

    //obtain A = USV' 
    svd_result_matrix_usv compute_as_matrix_rusv( HLSLCPP_IN(matrix3x3) inp )
    {
        matrix3x3 u;
        matrix3x3 v;
        vector3    s;
        compute( inp, u, s, v );

        svd_result_matrix_usv r;

        r.m_u = u;
        r.m_s = s;
        r.m_v = v;

        return r;
    }

    //obtain A = USV' 
    inline void compute_as_matrix_usv( HLSLCPP_IN(matrix3x3) inp, HLSLCPP_IN_OUT(matrix3x3) u, HLSLCPP_IN_OUT(vector3) s, HLSLCPP_IN_OUT(matrix3x3) v)
    {
        compute( inp, u, s, v );
    }

    //obtain A = USV' 
    inline svd_result_matrix_uv compute_as_matrix_ruv( HLSLCPP_IN(matrix3x3) inp )
    {
        matrix3x3 u;
        matrix3x3 v;
        vector3   s;
        compute( inp, u, s, v );

        svd_result_matrix_uv r;

        r.m_u = u;
        r.m_v = v;

        return r;
    }

    //obtain A = USV' 
    inline void compute_as_matrix_uv( HLSLCPP_IN(matrix3x3) inp, HLSLCPP_IN_OUT(matrix3x3) u, HLSLCPP_IN_OUT(matrix3x3) v)
    {
        vector3 s;
        compute( inp, u, s, v );
    }

    struct svd_result_polar
    {
        matrix3x3 m_u;
        matrix3x3 m_h;
    };


    inline svd_result_polar compute_as_matrix_polar_decomposition(HLSLCPP_IN(matrix3x3) inp)
    {
        svd_result_matrix_usv usv = compute_as_matrix_rusv(inp);

        svd_result_polar res;

        res.m_u = mul(usv.m_u, transpose(usv.m_v));
        res.m_h = mul(usv.m_v, mul(usv.m_s, transpose(usv.m_v)));

        return res;
    }


}

#if defined(__cplusplus)
#pragma warning(default : 4189)
#pragma warning(default : 4127)
#endif

#endif
