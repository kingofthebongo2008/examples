#include "precompiled.h"

#include <svd_hlslpp/svd_hlsl.h>
#include <svd_hlslpp/svd_hlsl_math.h>

#include <cstdint>

#include <svd/svd.h>
#include <svd/svd_math.h>
#include <svd/svd_rotation.h>

std::int32_t main1(int argc, _TCHAR* argv[])
{
    argc;
    argv;

    using namespace svd;
    using namespace svd::math;

	using number = cpu_scalar;

    const auto m11 = svd::math::splat<number>( 2.0f );
	const auto m12 = svd::math::splat<number>( -0.2f );
	const auto m13 = svd::math::splat<number>( 1.0f );

	const auto m21 = svd::math::splat<number>( -0.2f);
	const auto m22 = svd::math::splat<number>( 1.0f);
	const auto m23 = svd::math::splat<number>( 6.0f);

	const auto m31 = svd::math::splat<number>( 15.0f);
	const auto m32 = svd::math::splat<number>( 0.0f);
	const auto m33 = svd::math::splat<number>( 8.0f);

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

    svd::rotation<number>( &p[0], &q[0],  r, t );
    
    return 0;
}

std::int32_t main(int argc, _TCHAR* argv[])
{
	main1(argc, argv);

	using namespace svdhlslcpp;

	typedef float number;

	const float m11  = splat(2.0f );
	const float m12  = splat(-0.2f );
	const float m13  = splat(1.0f );

	const float m21  = splat(-0.2f);
	const float m22  = splat(1.0f);
	const float m23  = splat(6.0f);

	const float m31  = splat(15.0f);
	const float m32  = splat(0.0f);
	const float m33  = splat(8.0f);

	const svd_result_matrix_usv urv  = compute_as_matrix_rusv( create_matrix ( m11, m12, m13, m21, m22, m23, m31, m32, m33 ) );
    const svd_result_polar      uh   = compute_as_matrix_polar_decomposition(create_matrix(m11, m12, m13, m21, m22, m23, m31, m32, m33));

    const auto k                     = mul( uh.m_u, transpose(uh.m_u) );
    const auto k1                    = mul(urv.m_u, transpose(urv.m_u));
    const auto k2                    = mul(urv.m_v, transpose(urv.m_v));
	


	return 0;
}

