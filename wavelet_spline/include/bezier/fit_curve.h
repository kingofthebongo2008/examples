#ifndef __bezier_fit_curve_h__
#define __bezier_fit_curve_h__

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>

#include <glm/glm.hpp>

namespace bezier
{
    typedef glm::vec3   point3;
    typedef glm::vec2   point2;
    typedef float       point1;

    typedef glm::vec2   vector2;
    typedef glm::vec3   vector3;
    typedef glm::vec4   vector4;


    inline float dot3 ( const point3& a, const point3& b )
    {
        return glm::dot( a, b );
    }

    inline float dot ( const point3& a, const point3& b )
    {
        return glm::dot( a, b );
    }

    inline float dot2 ( const point2& a, const point2& b )
    {
        return glm::dot( a, b );
    }

    inline float dot ( const point2& a, const point2& b )
    {
        return glm::dot( a, b );
    }

    inline float distance ( const point3& a, const point3& b)
    {
        return glm::distance(a, b);
    }

    inline float distance ( point2 a, point2 b)
    {
        return glm::distance(a, b);
    }

    template <typename vector>
    inline float dot( vector a, vector b)
    {
        return glm::dot(a,b);
    }

    template <typename point> point zero();
    template <typename point> point one();

    template <> point3 zero<point3>()
    {
        return point3(0.0f, 0.0f, 0.0f);
    }

    template <> point2 zero<point2>()
    {
        return point2(0.0f, 0.0f);
    }

    template <> float zero<float>()
    {
        return 0.0f;
    }

    template<typename const_point_iterator>
    auto evaluate ( const_point_iterator control_points, const int32_t degree, float t ) -> typename std::iterator_traits< const_point_iterator >::value_type
    {
        typedef typename std::iterator_traits< const_point_iterator >::value_type point;

        std::vector< point > buffer_points;

        buffer_points.resize( degree + 1  );

        std::copy( control_points, control_points + degree + 1, std::begin(buffer_points) );

        for (int32_t i = 1; i <= degree; ++i)
        {
            for (int32_t j = 0; j <= degree - i; ++j )
            {
                buffer_points[j] = (1.0f - t ) * buffer_points[j] + t * buffer_points[j+1];
            }
        }

        return buffer_points[0];
    }

    template< typename const_point_iterator , int32_t degree >
    auto evaluate ( const_point_iterator control_points, float t ) -> typename std::iterator_traits< const_point_iterator >::value_type
    {
        typedef typename std::iterator_traits< const_point_iterator >::value_type point;

        std::vector< point > buffer_points;

        buffer_points.resize( degree + 1  );

        std::copy( control_points, control_points + degree + 1, std::begin(buffer_points) );

        for (int32_t i = 1; i <= degree; ++i)
        {
            for (int32_t j = 0; j <= degree - i; ++j )
            {
                buffer_points[j] = (1.0f - t ) * buffer_points[j] + t * buffer_points[j+1];
            }
        }

        return buffer_points[0];
    }


    //one iteration for ( q(u)- p ) dot (q'(u) ) = 0
    template <typename const_point_iterator>
    inline float newton_raphson ( const_point_iterator control_points, const typename std::iterator_traits< const_point_iterator >::value_type & p, float u )
    {
        typedef typename std::iterator_traits< const_point_iterator >::value_type point;

        point q_u, q1_u, q2_u;                    // u evaluated for q, q', q''
        std::tr1::array< point, 3 > q1;           // q'   control points
        std::tr1::array< point, 2 > q2;           // q''  control points

        //generate q'
        for ( int32_t i = 0 ; i <= 2; ++i )
        {
            q1[i] = ( *(control_points + i + 1) - *(control_points + i ) ) / 3.0f;
        }

        //generate q''
        for ( int32_t i = 0 ; i <= 1; ++i )
        {
            q2[i] = ( q1[i + 1 ] - q1[i] ) / 2.0f;
        }

        q_u  = evaluate<const_point_iterator, 3>  ( control_points,  u );
        q1_u = evaluate<std::tr1::array< point, 3 >::const_iterator, 2>  ( std::begin(q1), u );
        q2_u = evaluate<std::tr1::array< point, 2 >::const_iterator, 1>  ( std::begin(q2), u );

        //compute f(u) / f'(u) ( q(u) - p ) dot q'(u)
        float numerator = dot ( q_u - p, q1_u );

        // q'(u) dot q'(u) + q(u) dot q''(u)
        float denomerator = dot ( q1_u - p, q1_u - p ) + dot ( q_u - p, q2_u );

        if ( denomerator == 0.0f )
        {
            return u;
        }

        // u = u - f(u)/ f'(u) 
        return u - numerator / denomerator;
    }

    template<typename point, typename vector> vector left_tangent( const point& p0, const point& p1 )
    {
        auto  r = p1 - p0;

        if ( glm::length(r) == 0.0f )
        {
            return zero<vector>();
        }
        else
        {
            return glm::normalize( r );
        }
    }

    template<typename point, typename vector> vector right_tangent( const point& p0, const point& p1 )
    {
        auto  r = p0 - p1;

        if ( glm::length(r) == 0.0f )
        {
            return zero<vector>();
        }
        else
        {
            return glm::normalize( r );
        }
    }

    template<typename point, typename vector> vector center_tangent( const point& p0, const point& p1, const point& p2 )
    {
        auto v1 = p0 - p1;
        auto v2 = p1 - p2;

        auto center = glm::normalize( (v1 + v2) / 2.0f );

        if (glm::length ( center ) < 0.00001f )
        {
            center = glm::normalize( v1 );
        }

        return center;
    }


    template <typename point, typename vector> std::tuple<vector, vector> compute_center_tangents( const point& left, const point& center, const point& right )
    {
        auto v1n = center_tangent<point, vector>(left, center, right);
        auto v2n = -1.0f * v1n;

        return std::make_tuple( v1n, v2n );
    }

    template<typename const_iterator, typename iterator> void chord_length_parametrize( const_iterator begin, const_iterator end, iterator output )
    {
        typedef typename std::iterator_traits< iterator >::value_type point;

        auto begin_output = output;

        *output++=zero<point>();

        for ( auto it = begin + 1; it < end; ++it, ++output ) 
        {
            *output = *(output - 1) + distance( *it, *(it - 1) );
        }

        auto end_output = output;

        for ( auto it = begin_output + 1; it < end_output; ++it)
        {
            *it = *it / ( *( end_output - 1) );
        }
    }

    template<typename point>
    struct max_error
    {
        float m_error;
        point m_point_of_max_error;

        max_error( float error, const point& point_of_error ) :  
        m_error(error)
        , m_point_of_max_error( point_of_error )
        {

        }
    };

    //find maximum of squared distance of points to a curve
    template <typename const_iterator_points, typename const_iterator_bezier, typename const_iterator_curve > 
    auto compute_max_error( const_iterator_points begin, const_iterator_points end, const_iterator_bezier bezier, const_iterator_curve curve_params )
         -> std::tuple< float, typename const_iterator_points > 
    {
        typedef typename std::iterator_traits< const_iterator_points >::value_type point;

        float max_distance = 0.0f;

        const_iterator_points split = ( begin + std::distance( begin, end ) / 2 ) ;

        for ( auto it = begin; it != end; ++it, ++curve_params )
        {
            auto p = evaluate< const_iterator_bezier, 3> ( bezier, *curve_params ) ;
            auto new_distance = dot ( p - *it, p - *it );

            if (new_distance >= max_distance )
            {
                max_distance = new_distance;
                split = it;
            }
        }

        return std::make_tuple( max_distance, split );
    }

    template <typename const_iterator_points, typename const_iterator_bezier, typename const_iterator_curve, typename iterator_curve> void reparameterize( const_iterator_points begin, const_iterator_points end, const_iterator_bezier bezier, const_iterator_curve curve_params, iterator_curve out )
    {
        for ( const_iterator_points it = begin; it!=end; ++it, ++curve_params, ++out )
        {
            *out = newton_raphson( bezier, *it, *curve_params);
        }
    }

    inline float b0( float u )
    {
        auto tmp = 1.0f - u;

        return ( tmp * tmp * tmp );
    }

    inline float b1( float u )
    {
        auto tmp = 1.0f - u;

        return 3 * u * ( tmp * tmp );
    }

    inline float b2( float u )
    {
        auto tmp = 1.0f - u;

        return 3 * u * u * ( tmp );
    }

    inline float b3( float u )
    {
        return u * u * u;
    }


    template <  typename const_iterator_points, 
                typename const_iterator_curve,
                typename vector_type,
                typename out_iterator_curve
            >
    void generate_bezier
    ( 
        const_iterator_points   begin_points,
        const_iterator_points   end_points,
        const_iterator_curve    begin_curve,
        const_iterator_curve    end_curve,
        vector_type             hat1,
        vector_type             hat2,
        out_iterator_curve      out_curve
    )
    {
        typedef typename thrust::iterator_traits< const_iterator_points >::value_type point;
        typedef typename thrust::iterator_traits< const_iterator_curve >::value_type curve_point;


        std::vector< vector_type > a0;
        std::vector< vector_type > a1;

        a0.resize  ( std::distance ( begin_curve, end_curve ) );
        a1.resize  ( std::distance ( begin_curve, end_curve ) );

        std::transform( begin_curve, end_curve, std::begin(a0), [&]( const curve_point o ) -> vector_type
        {
            return hat1 * b1( o );
        });

        std::transform( begin_curve, end_curve, std::begin(a1), [&]( const curve_point o ) -> vector_type
        {
            return hat2 * b2( o );
        });

        // matrix part
        auto c00 = std::accumulate( std::begin(a0), std::end(a0), 0.0f, [&]( float val, const vector_type& v ) -> float
        {
            return val + dot(v,v);
        });

        auto c11 = std::accumulate( std::begin(a1), std::end(a1), 0.0f, [&]( float val, const vector_type& v ) -> float
        {
            return val + dot(v,v);
        });

        auto z0b = thrust::make_zip_iterator ( thrust::make_tuple( std::begin(a0), std::begin(a1) )  );
        auto z0e = thrust::make_zip_iterator ( thrust::make_tuple( std::end(a0), std::end(a1) ) );

        auto c01 = std::accumulate( z0b, z0e, 0.0f, [&]( float val, const thrust::tuple< vector_type, vector_type >&  v ) -> float
        {
            return val + dot( thrust::get<0>(v), thrust::get<1>(v) );
        });

        auto c10 = c01;

        // right vector 0
        auto z1b = thrust::make_zip_iterator ( thrust::make_tuple( begin_points, begin_curve, std::begin(a0) ) );
        auto z1e = thrust::make_zip_iterator ( thrust::make_tuple( end_points, end_curve, std::end(a0) )  );

        auto v0 = *begin_points;
        auto v3 = *(end_points - 1 );
        
        auto x0 = std::accumulate( z1b, z1e, 0.0f, [&]( float val, const thrust::tuple< point, curve_point, vector_type >&  v ) -> float
        {
            return val + dot( thrust::get<2>(v), 
                 thrust::get<0>(v) -
                 (
                    v0 * b0( thrust::get<1>(v) ) + 
                    v0 * b1( thrust::get<1>(v) ) +
                    v3 * b2( thrust::get<1>(v) ) +
                    v3 * b3( thrust::get<1>(v) ) 
                 )
                 
                 );
        });


        // right vector 1
        auto z2b = thrust::make_zip_iterator ( thrust::make_tuple( begin_points, begin_curve, std::begin(a1) ) );
        auto z2e = thrust::make_zip_iterator ( thrust::make_tuple( end_points, end_curve, std::end(a1) )  );

        auto x1 = std::accumulate( z2b, z2e, 0.0f, [&]( float val, const thrust::tuple< point, curve_point, vector_type >&  v ) -> float
        {
            return val + dot( thrust::get<2>(v), 
                 thrust::get<0>(v) -
                 (
                    v0 * b0( thrust::get<1>(v) ) + 
                    v0 * b1( thrust::get<1>(v) ) +
                    v3 * b2( thrust::get<1>(v) ) +
                    v3 * b3( thrust::get<1>(v) ) 
                 )
                 
                 );
        });
        
        // Compute the determinants of c and x 
        auto det_c0_c1 = c00 * c11 - c01 * c10;
        auto det_c0_x  = c00 * x1  - c10 * x0;
        auto det_x_c1  = c11 * x0  - c01 * x1;

        // Finally, derive alpha values
        auto alpha_l = (det_c0_c1 == 0.0f) ? 0.0f : det_x_c1 / det_c0_c1;
        auto alpha_r = (det_c0_c1 == 0.0f) ? 0.0f : det_c0_x / det_c0_c1;

        std::tr1::array< point, 4 > bezier;

        // If alpha negative, use the Wu/Barsky heuristic (see text) (if alpha is 0, you get coincident control points that lead to
        // divide by zero in any subsequent NewtonRaphsonRootFind() call. 

        auto  segment_length = distance(v0, v3);
        auto  epsilon = 1.0e-6f * segment_length;

        auto scale_l = alpha_l;
        auto scale_r = alpha_r;

        if ( alpha_l < segment_length || alpha_r < segment_length)
        {
            // fall back on standard (probably inaccurate) formula, and subdivide further if needed.
            scale_l = segment_length / 3.0f;
            scale_r = scale_l;
        }

        // First and last control points of the Bezier curve are positioned exactly at the first and last data points
        // Control points 1 and 2 are positioned an alpha distance out on the tangent vectors, left and right, respectively
        // vector part

        bezier[0] = v0;            
        bezier[1] = v0 + (scale_l * hat1);
        bezier[2] = v3 + (scale_r * hat2 );
        bezier[3] = v3;

        std::copy( std::begin(bezier), std::end(bezier), out_curve );
    }

    template <  typename const_iterator_points, 
                typename vector_type,
                typename out_iterator_curve    >
    void fit_cubic( 
            const_iterator_points   begin_points,
            const_iterator_points   end_points,
            vector_type             hat1,
            vector_type             hat2,
            float                   error,
            out_iterator_curve      out_curve )
    {

        typedef typename thrust::iterator_traits< const_iterator_points >::value_type point;
        auto point_count = std::distance( begin_points, end_points);

        //  Use heuristic if region only has two points in it
        if (point_count == 2)
        {
            auto v0 = *begin_points;
            auto v3 = *(end_points - 1 );

            auto d = distance ( v0, v3 ) / 3.0f;
            std::tr1::array< point, 4 > bezier;

            bezier[0] = v0;            
            bezier[1] = v0 + (d * hat1);
            bezier[2] = v3 + (d * hat2 );
            bezier[3] = v3;

            std::copy( std::begin( bezier ), std::end( bezier ), out_curve) ;
        }
        else
        {
            std::vector< float > u;
            u.resize( point_count );
            chord_length_parametrize( begin_points, end_points, std::begin(u) );

            std::tr1::array< point, 4 > bezier;
            bezier::generate_bezier( begin_points, end_points, std::begin(u), std::end(u), hat1, hat2, std::begin(bezier) );

            auto max_error = bezier::compute_max_error< const_iterator_points, std::tr1::array< point, 4 >::iterator> ( begin_points, end_points, std::begin(bezier), std::begin(u) );

            // Find max deviation of points to fitted curve
            if ( std::get<0>(max_error) < error )
            {
                std::copy( std::begin( bezier), std::end(bezier), out_curve);
                return;
            }

            auto iteration_error = error * error;
            auto max_iterations = 4;

            if ( std::get<0>(max_error) < iteration_error )
            {
                std::vector< float > u_prime;
                u_prime.resize( point_count );

                for ( int32_t i = 0; i < max_iterations; ++i )
                {
                    reparameterize( begin_points, end_points, std::begin(bezier), std::begin(u), std::begin(u_prime) );
                    generate_bezier( begin_points, end_points, std::begin(u_prime), std::end(u_prime), hat1, hat2, std::begin(bezier) );

                    auto max_error = bezier::compute_max_error( begin_points, end_points, std::begin(bezier), std::begin(u) );

                    // Find max deviation of points to fitted curve
                    if ( std::get<0>(max_error) < error )
                    {
                        std::copy( std::begin( bezier), std::end( bezier ), out_curve );
                        return;
                    }

                    //swap u and u_prime, todo remove the copy
                    std::copy( std::begin(u_prime), std::end(u_prime), std::begin(u) );
                }
            }


            //fitting failed -> split at max error and fit recursively
            auto split_point = std::get<1>( max_error );

            auto tangents = compute_center_tangents< point, vector_type >( *(split_point - 1), *(split_point), *(split_point + 1) );
        
            fit_cubic( begin_points, split_point + 1, hat1, std::get<0>(tangents), error, out_curve );
            fit_cubic( split_point, end_points, std::get<1>(tangents), hat2, error, out_curve );
        }
    }


    template <  typename const_iterator_points,
                typename out_iterator_curve    >
    void fit_curve ( 
            const_iterator_points   begin_points,
            const_iterator_points   end_points,
            float                   error,
            out_iterator_curve      out_curve )
    {
        typedef typename thrust::iterator_traits< const_iterator_points >::value_type point;

        auto hat1 = left_tangent<point, point>( *(begin_points), *( begin_points + 1 ) );
        auto hat2 = right_tangent<point, point>( *(end_points - 2 ), *( end_points - 1 ) );


        fit_cubic( begin_points, end_points, hat1, hat2, error, out_curve);
    }


    inline void example()
    {
        std::vector< bezier::point2 >  fitted_cubics;
        std::vector< bezier::point2 >  data;

        data.push_back( bezier::point2( 1.0f, 0.0f ) );
        data.push_back( bezier::point2( 2.0f, 0.0f ) );
        data.push_back( bezier::point2( 3.0f, 0.0f ) );
        data.push_back( bezier::point2( 4.0f, 0.0f ) ); 
        data.push_back( bezier::point2( 3.0f, 0.0f ) );

        bezier::fit_curve( std::begin(data), std::end(data), 0.001f, std::back_inserter(fitted_cubics) );

    }
}



#endif
