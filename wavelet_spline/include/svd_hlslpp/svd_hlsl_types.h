#ifndef __svd_hlsl_types_h__
#define __svd_hlsl_types_h__

#include "svd_hlsl_math.h"

#if defined(__cplusplus)
#include <cmath>
#endif

namespace svdhlslcpp
{
    struct matrix3x3
    {
        //row major
        float a11; float a12; float a13;
        float a21; float a22; float a23;
        float a31; float a32; float a33;
    };

    struct symmetric_matrix3x3
    {
        //row major
        float a11; //t a12; t a13;
        float a21; float a22; //t a23;
        float a31; float a32; float a33;
    };

    struct quaternion
    {
        float x;
        float y;
        float z;
        float w;
    };

    struct vector3
    {
        float x;
        float y;
        float z;
    };

    inline matrix3x3 create_matrix 
        ( 
            float a11, float a12, float a13,
            float a21, float a22, float a23,
            float a31, float a32, float a33
        )
    {
        matrix3x3 r =  { a11, a12, a13, a21, a22, a23, a31, a32, a33 };
        return r;
    }

    inline quaternion create_quaternion(float qx, float qy, float qz, float qw)
    {
        quaternion r = { qx, qy, qz, qw };
        return r;
    }

    matrix3x3 transpose(matrix3x3 inp)
    {
        matrix3x3 r;

        r.a11 = inp.a11;
        r.a12 = inp.a21;
        r.a13 = inp.a31;

        r.a21 = inp.a12;
        r.a22 = inp.a22;
        r.a23 = inp.a32;

        r.a31 = inp.a13;
        r.a32 = inp.a23;
        r.a33 = inp.a33;

        return r;
    }

    matrix3x3 mul(matrix3x3 a, matrix3x3 b)
    {
        matrix3x3 r;

        r.a11 = dot3( a.a11, a.a12, a.a13, b.a11, b.a21, b.a31);
        r.a12 = dot3( a.a11, a.a12, a.a13, b.a12, b.a22, b.a32);
        r.a13 = dot3( a.a11, a.a12, a.a13, b.a13, b.a23, b.a33);

        r.a21 = dot3(a.a21, a.a22, a.a23, b.a11, b.a21, b.a31);
        r.a22 = dot3(a.a21, a.a22, a.a23, b.a12, b.a22, b.a32);
        r.a23 = dot3(a.a21, a.a22, a.a23, b.a13, b.a23, b.a33);

        r.a31 = dot3(a.a31, a.a32, a.a33, b.a11, b.a21, b.a31);
        r.a32 = dot3(a.a31, a.a32, a.a33, b.a12, b.a22, b.a32);
        r.a33 = dot3(a.a31, a.a32, a.a33, b.a13, b.a23, b.a33);

        return r;
    }

    matrix3x3 mul(vector3 d, matrix3x3 b)
    {
        matrix3x3 r;

        r.a11 = dot3(d.x, 0.0f,   0.0f, b.a11, b.a21, b.a31);
        r.a12 = dot3(d.y, 0.0f,   0.0f, b.a12, b.a22, b.a32);
        r.a13 = dot3(d.z, 0.0f,   0.0f, b.a13, b.a23, b.a33);

        r.a21 = dot3(0.0f, d.y, 0.0f, b.a11, b.a21, b.a31);
        r.a22 = dot3(0.0f, d.y, 0.0f, b.a12, b.a22, b.a32);
        r.a23 = dot3(0.0f, d.y, 0.0f, b.a13, b.a23, b.a33);

        r.a31 = dot3(0.0f, 0.0f, d.z, b.a11, b.a21, b.a31);
        r.a32 = dot3(0.0f, 0.0f, d.z, b.a12, b.a22, b.a32);
        r.a33 = dot3(0.0f, 0.0f, d.z, b.a13, b.a23, b.a33);

        return r;
    }

#if defined(__cplusplus)
    float norm_inf(matrix3x3 b)
    {
        float r = 0;

        r = std::max<float>(std::abs(b.a11), r);
        r = std::max<float>(std::abs(b.a12), r);
        r = std::max<float>(std::abs(b.a13), r);

        r = std::max<float>(std::abs(b.a21), r);
        r = std::max<float>(std::abs(b.a22), r);
        r = std::max<float>(std::abs(b.a23), r);


        r = std::max<float>(std::abs(b.a31), r);
        r = std::max<float>(std::abs(b.a32), r);
        r = std::max<float>(std::abs(b.a33), r);
        return r;
    }

    matrix3x3 sub(matrix3x3 a, matrix3x3 b)
    {
        matrix3x3 r;

        r.a11 = a.a11 - b.a11;
        r.a12 = a.a12 - b.a12;
        r.a13 = a.a13 - b.a13;

        r.a21 = a.a21 - b.a21;
        r.a22 = a.a22 - b.a22;
        r.a23 = a.a23 - b.a23;

        r.a31 = a.a31 - b.a31;
        r.a32 = a.a32 - b.a32;
        r.a33 = a.a33 - b.a33;

        return r;
    }
#endif
}


#endif
