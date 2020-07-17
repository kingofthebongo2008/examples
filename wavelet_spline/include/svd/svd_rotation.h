#ifndef __svd_rotation_h__
#define __svd_rotation_h__

#include <cstdint>

#include "svd_types.h"
#include "svd_math.h"
#include "svd.h"


namespace svd
{
    //finds rotation and translation from points p to points q in least squares sense
    //p is 3 points, q is 3 points
    template <typename t> inline void rotation ( const vector3<t>* p, const vector3<t>* q, matrix3x3<t>& rotation, vector3<t>& translation )
    {
        using namespace svd::math;

        auto size = 3;
        auto cp = vector3<t>();
        auto cq = vector3<t>();
        
        cp.x = math::zero<t>();
        cp.y = math::zero<t>();
        cp.z = math::zero<t>();

        cq.x = math::zero<t>();
        cq.y = math::zero<t>();
        cq.z = math::zero<t>();

        
        for ( int32_t i = 0; i < size; ++i )
        {
            cp.x = cp.x + p[i].x;
            cp.y = cp.y + p[i].y;
            cp.z = cp.z + p[i].z;

            cq.x = cq.x + q[i].x;
            cq.y = cq.y + q[i].y;
            cq.z = cq.z + q[i].z;
        }
        
        cp.x = cp.x / math::splat<t>(3.0f);
        cp.y = cp.y / math::splat<t>(3.0f);
        cp.z = cp.z / math::splat<t>(3.0f);

        cq.x = cq.x / math::splat<t>(3.0f);
        cq.y = cq.y / math::splat<t>(3.0f);
        cq.z = cq.z / math::splat<t>(3.0f);

        vector3<t> x[3];
        vector3<t> y[3];

        for ( int32_t i = 0; i < size; ++i )
        {
            x[i].x = p[i].x - cp.x;
            x[i].y = p[i].y - cp.y;
            x[i].z = p[i].z - cp.z;

            y[i].x = q[i].x - cq.x;
            y[i].y = q[i].y - cq.y;
            y[i].z = q[i].z - cq.z;
        }

        //compute x.transpose(y)
        
        //x0x y0x + x1x y1x + x2x y2x
        auto s11 = dot3( x[0].x, x[1].x, x[2].x, y[0].x, y[1].x, y[2].x );

        //x0 x y0y + x1x y1y + x2x y2y
        auto s12 = dot3( x[0].x, x[1].x, x[2].x, y[0].y, y[1].y, y[2].y );

        //x0x y0z + x1x y1z + x2x y2z
        auto s13 = dot3( x[0].x, x[1].x, x[2].x, y[0].z, y[1].z, y[2].z );


        //x0y y0x + x1y y1x + x2y y2x
        auto s21 = dot3( x[0].y, x[1].y, x[2].y, y[0].x, y[1].x, y[2].x );
        
        //x0y y0y + x1y y1y + x2y y2y
        auto s22 = dot3( x[0].y, x[1].y, x[2].y, y[0].y, y[1].y, y[2].y );
        
        //x0y y0z + x1y y1z + x2y y2z
        auto s23 = dot3( x[0].y, x[1].y, x[2].y, y[0].z, y[1].z, y[2].z );


        //x0z y0x + x1z y1x + x2z y2x
        auto s31 = dot3( x[0].z, x[1].z, x[2].z, y[0].x, y[1].x, y[2].x );

        //x0z y0y + x1z y1y + x2z y2y
        auto s32 = dot3( x[0].z, x[1].z, x[2].z, y[0].y, y[1].y, y[2].y );
        
        //x0z y0z + x1z y1z + x2z y2z
        auto s33 = dot3( x[0].z, x[1].z, x[2].z, y[0].z, y[1].z, y[2].z );

        matrix3x3<t> u;
        matrix3x3<t> v;

        compute_as_matrix_uv( create_matrix( s11, s12, s13, s21, s22, s23, s31, s32, s33 ), u, v );

        
        //compute u.transpose(v)
        //see svd rotation paper for more details

        //u11 v11 + u12 v12 + u13 v13
        auto r11 = dot3 ( u.a11, u.a12, u.a13, v.a11, v.a12, v.a13 );

        //u21 v11 + u22 v12 + u23 v13
        auto r12 = dot3 ( u.a21, u.a22, u.a23, v.a11, v.a12, v.a13 );

        //u31 v11 + u32 v12 + u33 v13
        auto r13 = dot3 ( u.a31, u.a32, u.a33, v.a11, v.a12, v.a13 );

        //u11 v21 + u12 v22 + u13 v23
        auto r21 = dot3 ( u.a11, u.a12, u.a13, v.a21, v.a22, v.a23 );

        //u21 v21 + u22 v22 + u23 v23
        auto r22 = dot3 ( u.a21, u.a22, u.a23, v.a21, v.a22, v.a23 );

        //u31 v21 + u32 v22 + u33 v23
        auto r23 = dot3 ( u.a31, u.a32, u.a33, v.a21, v.a22, v.a23 );

        //u11 v31 + u12 v32 + u13 v33
        auto r31 = dot3 ( u.a11, u.a12, u.a13, v.a31, v.a32, v.a33 );

        //u21 v31 + u22 v32 + u23 v33
        auto r32 = dot3 ( u.a21, u.a22, u.a23, v.a31, v.a32, v.a33 );

        //u31 v31 + u32 v32 + u33 v33
        auto r33 = dot3 ( u.a31, u.a32, u.a33, v.a31, v.a32, v.a33 );


        //calculate translation
        translation.x = cq.x - dot3( cp.x, cp.y, cp.z, r11, r12, r13 );
        translation.y = cq.y - dot3( cp.x, cp.y, cp.z, r21, r22, r23 );
        translation.z = cq.z - dot3( cp.x, cp.y, cp.z, r31, r32, r33 );

        rotation.a11 = r11;
        rotation.a12 = r12;
        rotation.a13 = r13;

        rotation.a21 = r21;
        rotation.a22 = r22;
        rotation.a23 = r23;

        rotation.a31 = r31;
        rotation.a32 = r32;
        rotation.a33 = r33;
    }
}



#endif
