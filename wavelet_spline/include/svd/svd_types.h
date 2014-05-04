#ifndef __svd_types_h__
#define __svd_types_h__

namespace svd
{
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

}


#endif
