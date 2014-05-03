#include "precompiled.h"


/*
#include <iostream>
#include <numeric>
#include <vector>
*/

#include <svd/svd.h>


std::int32_t main(int argc, _TCHAR* argv[])
{
    
    using namespace svd;
    using namespace svd::math;

    typedef svd::avx_vector number;

    auto m11 = svd::math::splat<number>( 2.0f );
    auto m12 = svd::math::splat<number>( -0.2f );
    auto m13 = svd::math::splat<number>( 1.0f );

    auto m21 = svd::math::splat<number>( -0.2f);
    auto m22 = svd::math::splat<number>( 1.0f);
    auto m23 = svd::math::splat<number>( 6.0f);

    auto m31 = svd::math::splat<number>( 15.0f);
    auto m32 = svd::math::splat<number>( 0.0f);
    auto m33 = svd::math::splat<number>( 8.0f);

    auto urv = svd::compute_as_matrix_rusv<number>( svd::create_matrix ( m11, m12, m13, m21, m22, m23, m31, m32, m33 ) );

    auto urv1 = svd::compute_as_quaternion_rusv<number>( svd::create_matrix ( m11, m12, m13, m21, m22, m23, m31, m32, m33 ) );
    
    return 0;
}

