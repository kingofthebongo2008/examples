#include "precompiled.h"

#include <cstdint>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include <thrust/iterator/zip_iterator.h>

#include <vector_functions.h>

#include <array>

#include <compression/arithmetic.h>

#include <bezier/fit_curve.h>


std::int32_t main(int argc, _TCHAR* argv[])
{

    std::vector< bezier::point3 >  fitted_cubics;
    std::vector< bezier::point3 >  data;

    data.push_back( bezier::point3( 1.0f, 0.0f, 1.22f ) );
    data.push_back( bezier::point3( 2.0f, 0.0f, 1.22f ) );
    data.push_back( bezier::point3( 5.0f, 0.0f, 1.52f ) );
    data.push_back( bezier::point3( 3.0f, 0.0f, 1.22f ) );

    data.push_back( bezier::point3( 3.0f, 0.0f, 1.22f ) );
    data.push_back( bezier::point3( 3.0f, 0.0f, 1.22f ) );
    data.push_back( bezier::point3( 3.0f, 0.0f, 1.22f ) );
    data.push_back( bezier::point3( 3.0f, 0.0f, 1.22f ) );

    data.push_back( bezier::point3( 3.0f, 0.0f, 1.22f ) );
    data.push_back( bezier::point3( 3.0f, 0.0f, 1.22f ) );
    data.push_back( bezier::point3( 3.0f, 0.0f, 1.22f ) );
    data.push_back( bezier::point3( 3.0f, 0.0f, 1.22f ) );

    data.push_back( bezier::point3( 3.0f, 0.0f, 1.22f ) );
    data.push_back( bezier::point3( 3.0f, 0.0f, 1.22f ) );
    data.push_back( bezier::point3( 3.0f, 0.0f, 1.22f ) );
    data.push_back( bezier::point3( 3.0f, 0.0f, 1.22f ) );

    data.push_back( bezier::point3( 3.0f, 0.0f, 1.22f ) );
    data.push_back( bezier::point3( 3.0f, 0.0f, 1.22f ) );
    data.push_back( bezier::point3( 3.0f, 0.0f, 1.22f ) );
    data.push_back( bezier::point3( 3.0f, 0.0f, 1.22f ) );

    bezier::fit_curve( std::begin(data), std::end(data), 0.01f, std::back_inserter(fitted_cubics) );

    float val = 0.0f;

    for (int32_t i = 0; i < fitted_cubics.size(); ++i)
    {
        std::cout<< fitted_cubics[i].x << ", " << fitted_cubics[i].y << std::endl;
        std::cout<< std::endl;
    }

    return 0;
}

