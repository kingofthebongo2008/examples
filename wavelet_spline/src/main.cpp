#include "precompiled.h"

#include <svd/svd.h>
#include <svd/svd_math.h>
#include <svd/svd_rotation.h>


std::int32_t main(int argc, _TCHAR* argv[])
{
    
    using namespace svd;
    using namespace svd::math;

    typedef svd::cpu_scalar number;

    const auto m11 = svd::math::splat<number>( 2.0f );
	const auto m12 = svd::math::splat<number>( -0.2f );
	const auto m13 = svd::math::splat<number>( 1.0f );

	const auto m21 = svd::math::splat<number>( -0.2f);
	const auto m22 = svd::math::splat<number>( 1.0f);
	const auto m23 = svd::math::splat<number>( 6.0f);

	const auto m31 = svd::math::splat<number>( 0.0f);
	const auto m32 = svd::math::splat<number>( 0.0f);
	const auto m33 = svd::math::splat<number>( 0.0f);

	const auto urv = svd::compute_as_matrix_rusv<number>( svd::create_matrix ( m11, m12, m13, m21, m22, m23, m31, m32, m33 ) );

	const auto urv1 = svd::compute_as_quaternion_rusv<number>( svd::create_matrix ( m11, m12, m13, m21, m22, m23, m31, m32, m33 ) );


    svd::vector3<number> p[3];
    svd::vector3<number> q[3];

    p[0].x = svd::math::splat<number>(1.0f);
    p[0].y = svd::math::splat<number>(2.0f);
    p[0].z = svd::math::splat<number>(3.0f);

    p[1].x = svd::math::splat<number>(5.0f);
    p[1].y = svd::math::splat<number>(7.0f);
    p[1].z = svd::math::splat<number>(4.0f);

    p[2].x = svd::math::splat<number>(0.0f);
    p[2].y = svd::math::splat<number>(0.0f);
    p[2].z = svd::math::splat<number>(0.0f);

    
    q[0].x = svd::math::splat<number>(1.0f);
    q[0].y = svd::math::splat<number>(2.0f);
    q[0].z = svd::math::splat<number>(3.0f);

    q[1].x = svd::math::splat<number>(5.0f);
    q[1].y = svd::math::splat<number>(7.0f);
    q[1].z = svd::math::splat<number>(4.0f);

    q[2].x = svd::math::splat<number>(0.0f);
    q[2].y = svd::math::splat<number>(0.0f);
    q[2].z = svd::math::splat<number>(0.0f);

    svd::matrix3x3<number> r;
    svd::vector3<number> t;

    svd::rotation<svd::cpu_scalar>( &p[0], &q[0],  r, t );
    
    return 0;
}

