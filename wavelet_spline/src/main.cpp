#include "precompiled.h"

#include <cstdint>
#include <iostream>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include <thrust/iterator/zip_iterator.h>

#include <vector_functions.h>

#include <array>

#include <compression/arithmetic.h>

#include <glm/glm.hpp>
#include <glm/gtx/simd_vec4.hpp>


namespace lwt
{
    template < typename iterator, typename normalizer >
    void normalize_even(iterator begin, iterator end, normalizer n)
    {
        auto  middle    = thrust::distance( begin, end ) / 2;
        auto  half      = begin + (middle);

        for ( auto it = begin; it < half; ++it )
        {
            (*it) = n( it, begin, end );
        }
    }

    template < typename iterator, typename normalizer >
    void normalize_odd(iterator begin, iterator end, normalizer n)
    {
        auto  middle    = thrust::distance( begin, end ) / 2;
        auto  half      = begin + (middle);

        for ( auto it = half; it < end; ++it )
        {
            ( *it ) = n( it, begin, end );
        }
    }

    namespace fwd
    {
        template <typename iterator>
        void split( iterator begin, iterator end )
        {
            begin = begin + 1;
            end   = end - 1; 

            while ( begin < end )
            {
                for ( auto it = begin; it < end; it+=2)
                {
                    std::swap( *it, *(it+1) );
                }

                begin++;
                end--;
            }
        }

        template <typename iterator, typename predictor >
        void predict(iterator begin, iterator end, predictor p )
        {
            auto  middle    = thrust::distance( begin, end ) / 2;
            auto  half      = begin + (middle);

            for ( auto it = begin; it < half; ++it )
            {
                auto even   =   *it;
                auto odd    =   *(it + middle);

                auto detail = odd - p( it, begin, end );

                *(it + middle ) = detail;
            }
        }

        template < typename iterator, typename updater >
        void update(iterator begin, iterator end, updater u)
        {
            auto  middle    = thrust::distance( begin, end ) / 2;
            auto  half      = begin + (middle);

            for ( auto it = begin; it < half; ++it )
            {
                auto even   =   *it;
                auto odd    =   *(it + middle);

                auto scale = even + u( it + middle, begin, end );

                *(it) = scale;
            }
        }
    }

    namespace inv
    {
        template <typename iterator>
        void merge( iterator begin, iterator end )
        {
            auto  middle    = thrust::distance( begin, end );
            auto  half      = begin + (middle / 2);

            auto  s         = half - 1;
            auto  e         = half;
        
            while ( s > begin )
            {
                for ( auto it = s; it < e; it+=2 )
                {
                    std::swap( *it, *(it+1));
                }

                s--;
                e++;
            }
        }

        template <typename iterator, typename predictor >
        void predict(iterator begin, iterator end, predictor p )
        {
            auto  middle    = thrust::distance( begin, end ) / 2;
            auto  half      = begin + (middle);

            for ( auto it = begin; it < half; ++it )
            {
                auto even   =   *it;
                auto odd    =   *(it + middle);

                auto detail = odd + p( it, begin, end );

                *(it + middle ) = detail;
            }
        }

        template < typename iterator, typename updater >
        void update(iterator begin, iterator end, updater u)
        {
            auto  middle    = thrust::distance( begin, end ) / 2;
            auto  half      = begin + (middle);

            for ( auto it = begin; it < half; ++it )
            {
                auto even   =   *it;
                auto odd    =   *(it + middle);

                auto scale = even - u( it + middle, begin, end );

                *(it) = scale;
            }
        }
    }

    namespace haar
    {
        template <typename iterator> struct predictor
        {
            typedef typename thrust::iterator_value<iterator>::type value;
            value operator()( iterator even, iterator begin, iterator end ) const
            {
                return *even;
            }
        };

        template <typename iterator> struct updater
        {
            typedef typename thrust::iterator_value<iterator>::type value;
            value operator()( iterator odd, iterator begin, iterator end ) const
            {
                return ( *odd ) / 2.0f;
            }
        };

        namespace fwd
        {
            template < typename iterator >
            void normalize( iterator begin, iterator end )
            {
                auto  middle    = thrust::distance( begin, end ) / 2;
                auto  half      = begin + (middle);

                auto  sqrt2     = sqrtf(2.0f);

                for ( auto it = begin; it < half; ++it )
                {
                    *it *= sqrt2;
                    *(it + middle ) /= sqrt2;
                }
            }

            template < typename iterator>
            void transform( iterator begin, iterator end )
            {
                lwt::fwd::split(begin, end);
                lwt::fwd::predict( begin, end, predictor<iterator>());
                lwt::fwd::update( begin, end, updater<iterator>() );
            }
        }

        namespace inv
        {
            template < typename iterator>
            void transform( iterator begin, iterator end )
            {
                lwt::inv::update( begin, end, updater<iterator>() );
                lwt::inv::predict( begin, end, predictor<iterator>());
                lwt::inv::merge(begin, end);
            }
        }
    }

    namespace linear
    {
        template <typename iterator> struct predictor
        {
            typedef typename thrust::iterator_value<iterator>::type value;

            value operator()( iterator even, iterator begin, iterator end ) const
            {
                auto  distance  = thrust::distance( begin, end );
                auto  half      = begin + (distance / 2);

                //handle also out of bounds.
                if (even < half - 1 )
                {
                    return ( *even + *(even+1) ) / 2.0f;
                }
                else if ( distance == 2 )
                {
                    return *begin;
                }
                else
                {
                    //spawn line between the last two points on the y axis, x axis is 0 and 1, new point is on x axis 2
                    auto y2 = *even;
                    auto y1 = *(even - 1 );
                    return ( *even + 2 * y2 - y1 ) / 2;
                }
            }
        };

        template <typename iterator> struct updater
        {
            typedef typename thrust::iterator_value<iterator>::type value;
            value operator()( iterator odd, iterator begin, iterator end ) const
            {
                auto  distance  = thrust::distance( begin, end );
                auto  half      = begin + (distance / 2);

                if ( odd > half)
                {
                    return ( *odd + *(odd-1) ) / 4.0f;
                }
                else //out of bounds
                {
                    return *odd / 2.0f;
                }
            }
        };


        namespace fwd
        {
            template < typename iterator>
            void transform( iterator begin, iterator end )
            {
                lwt::fwd::split( begin, end );
                lwt::fwd::predict( begin, end, predictor<iterator>() );
                lwt::fwd::update( begin, end, updater<iterator>() );
            }
        }

        namespace inv
        {
            template < typename iterator>
            void transform( iterator begin, iterator end )
            {
                lwt::inv::update( begin, end, updater<iterator>() );
                lwt::inv::predict( begin, end, predictor<iterator>());
                lwt::inv::merge( begin, end );
            }
        }
    }

    namespace d4
    {
        template <typename iterator> struct predictor0
        {
            typedef typename thrust::iterator_value<iterator>::type value;

            value operator()( iterator even, iterator begin, iterator end ) const
            {
                const auto f0 = sqrt( 3.0 );
                const auto f1 = (sqrt(3.0) - 2);
                auto left = even - 1;

                auto  distance  = thrust::distance( begin, end );
                auto  half      = begin + (distance / 2);

                if (left < begin )
                {
                    left = half - 1;
                }

                return  ( f0 * (*even) +  f1 * (*left) ) / 4.0 ;
            }
        };

        template <typename iterator> struct updater0
        {
            typedef typename thrust::iterator_value<iterator>::type value;

            value operator()( iterator odd, iterator begin, iterator end ) const
            {
                const auto f = sqrtf( 3.0 );
                return f * ( *odd ) ;
            }
        };

        template <typename iterator> struct updater1
        {
            typedef typename thrust::iterator_value<iterator>::type value;

            value operator()( iterator odd, iterator begin, iterator end ) const
            {
                auto right = odd + 1;

                if ( right > end - 1 )
                {
                    auto  distance  = thrust::distance( begin, end );
                    auto  half      = begin + (distance / 2);

                    right = half;
                }

                return  - ( *( right ) );
            }
        };

        namespace fwd
        {
            template <typename iterator> struct scale_even
            {
                typedef typename thrust::iterator_value<iterator>::type value;

                value operator()( iterator even, iterator begin, iterator end ) const
                {
                    const auto f = ( sqrt( 3.0 ) - 1.0 ) / sqrt( 2.0 );
                    return f * (*even) ;
                }
            };

            template <typename iterator> struct scale_odd
            {
                typedef typename thrust::iterator_value<iterator>::type value;

                value operator()( iterator odd, iterator begin, iterator end ) const
                {
                    const auto f = ( sqrt(3.0) + 1.0 ) / sqrt(2.0);
                    return f * (*odd);
                }
            };

            template < typename iterator>
            void transform( iterator begin, iterator end )
            {
                lwt::fwd::split(begin, end);
                lwt::fwd::update( begin, end, updater0<iterator>() );
                lwt::fwd::predict( begin, end, predictor0<iterator>());
                lwt::fwd::update( begin, end, updater1<iterator>() );

                lwt::normalize_even( begin, end, scale_even<iterator>() );
                lwt::normalize_odd( begin, end, scale_odd<iterator>() );
            }
        }

        namespace inv
        {
            template <typename iterator> struct scale_even
            {
                typedef typename thrust::iterator_value<iterator>::type value;

                value operator()( iterator even, iterator begin, iterator end ) const
                {
                    const auto f = sqrt( 2.0 ) / ( sqrt( 3.0 ) - 1.0 ) ;
                    return f * (*even) ;
                }
            };

            template <typename iterator> struct scale_odd
            {
                typedef typename thrust::iterator_value<iterator>::type value;

                value operator()( iterator odd, iterator begin, iterator end ) const
                {
                    const auto f =  sqrt(2.0) / ( sqrt(3.0) + 1.0 ) ;
                    return f * (*odd);
                }
            };

            template < typename iterator>
            void transform( iterator begin, iterator end )
            {
                lwt::normalize_even( begin, end, scale_even<iterator>() );
                lwt::normalize_odd( begin, end, scale_odd<iterator>() );

                lwt::inv::update( begin, end, updater1<iterator>() );
                lwt::inv::predict( begin, end, predictor0<iterator>());
                lwt::inv::update( begin, end, updater0<iterator>() );
                lwt::inv::merge(begin, end);
            }
        }
    }

    namespace cdf1
    {
        template <typename iterator> struct predictor0
        {
            typedef typename thrust::iterator_value<iterator>::type value;

            value operator()( iterator even, iterator begin, iterator end ) const
            {
                return *even;
            }
        };

        template <typename iterator> struct cdf11
        {
            typedef typename thrust::iterator_value<iterator>::type value;

            value operator()( iterator odd, iterator begin, iterator end ) const
            {
                return (*odd) / 2.0 ;
            }
        };

        template <typename iterator> struct cdf13
        {
            typedef typename thrust::iterator_value<iterator>::type value;

            value operator()( iterator odd, iterator begin, iterator end ) const
            {
                auto left  = odd - 1;
                auto right = odd + 1;

                auto  distance  = thrust::distance( begin, end );
                auto  half      = begin + (distance / 2);

                if ( left < half )
                {
                    left = end - 1;
                }

                if ( right > end - 1)
                {
                    right = half;
                }

                return ( *right - *left - 8 * (*odd) ) / -16.0;
            }
        };

        template <typename iterator> struct cdf15
        {
            typedef typename thrust::iterator_value<iterator>::type value;

            value operator()( iterator odd, iterator begin, iterator end ) const
            { 
                auto left0   = odd - 2;
                auto left1   = odd - 1;

                auto right0  = odd + 1;
                auto right1  = odd + 2;

                auto  distance  = thrust::distance( begin, end );
                auto  half      = begin + ( distance / 2 );

                if ( left0 < half )
                {
                    if (left0 < half - 1 )
                    {
                        left0 = end - 2;
                    }
                    else
                    {
                        left0 = end - 1;
                    }
                }

                if ( left1 < half )
                {
                    left1 = end - 1;
                }

                if ( right1 > end - 1 )
                {
                    if (right1 > end )
                    {
                        right1 = half + 1;
                    }
                    else
                    {
                        right1 = half;
                    }
                }

                if ( right0 > end - 1 )
                {
                    right0 = half;
                }

                return ( 3 * (*left0) - 22 * (*left1) - 128 * (*odd) + 22 * (*right0) - 3 * (*right1) ) / -256.0;
            }
        };

        namespace fwd
        {
            template <typename iterator> struct scale_even
            {
                typedef typename thrust::iterator_value<iterator>::type value;

                value operator()( iterator even, iterator begin, iterator end ) const
                {
                    const auto f = sqrt(2.0);
                    return f * (*even) ;
                }
            };

            template <typename iterator> struct scale_odd
            {
                typedef typename thrust::iterator_value<iterator>::type value;

                value operator()( iterator odd, iterator begin, iterator end ) const
                {
                    const auto f = sqrt(2.0) / 2.0;
                    return f * (*odd);
                }
            };

            template < typename iterator>
            void transform( iterator begin, iterator end )
            {
                lwt::fwd::split(begin, end);

                lwt::fwd::predict( begin, end, predictor0<iterator>());                
                lwt::fwd::update( begin, end, cdf15<iterator>() );

                lwt::normalize_even( begin, end, scale_even<iterator>() );
                lwt::normalize_odd( begin, end, scale_odd<iterator>() );
            }
        }
    }

    namespace cdf4
    {
        template <typename iterator> struct predictor0
        {
            typedef typename thrust::iterator_value<iterator>::type value;

            value operator()( iterator even, iterator begin, iterator end ) const
            {
                auto right = even + 1;

                auto  distance  = thrust::distance( begin, end );
                auto  half      = begin + (distance / 2);

                if ( right > half - 1 )
                {
                    right = begin;
                }

                return (*even) + (*right);
            }
        };

        template <typename iterator> struct updater0
        {
            typedef typename thrust::iterator_value<iterator>::type value;

            value operator()( iterator odd, iterator begin, iterator end ) const
            {
               auto left = odd - 1;

               auto  distance  = thrust::distance( begin, end );
               auto  half      = begin + (distance / 2);

               
               if ( left < half )
               {
                   left = end - 1;
               }

               return ( *left + *odd ) / -4.0;
            }
        };

        template <typename iterator> struct cdf42
        {
            typedef typename thrust::iterator_value<iterator>::type value;

            value operator()( iterator odd, iterator begin, iterator end ) const
            {
                auto left  = odd - 1;

                auto  distance  = thrust::distance( begin, end );
                auto  half      = begin + (distance / 2);

                if ( left < half )
                {
                    left = end - 1;
                }

                return 3 * ( *left + *odd ) / 16.0;
            }
        };

        template <typename iterator> struct cdf44
        {
            typedef typename thrust::iterator_value<iterator>::type value;

            value operator()( iterator odd, iterator begin, iterator end ) const
            {
                auto left0  = odd - 2;
                auto left1  = odd - 1;

                auto right0 = odd + 1;
                auto right1 = odd + 2;

                auto  distance  = thrust::distance( begin, end );
                auto  half      = begin + (distance / 2);

                if ( left0 < half )
                {
                    if (left0 < half - 1 )
                    {
                        left0 = end - 2;
                    }
                    else
                    {
                        left0 = end - 1;
                    }
                }

                if ( left1 < half )
                {
                    left1 = end - 1;
                }

                if ( right1 > end - 1 )
                {
                    if (right1 > end )
                    {
                        right1 = half + 1;
                    }
                    else
                    {
                        right1 = half;
                    }
                }

                if ( right0 > end - 1 )
                {
                    right0 = half;
                }

                return ( 5 * (*left0 ) - 29 * ( *left1 ) - 29 * (*odd) + 5 * (*right0 ) ) / -128.0;
            }
        };


        namespace fwd
        {
            template <typename iterator> struct scale_even
            {
                typedef typename thrust::iterator_value<iterator>::type value;

                value operator()( iterator even, iterator begin, iterator end ) const
                {
                    const auto f = 2 * sqrt(2.0);
                    return f * (*even) ;
                }
            };

            template <typename iterator> struct scale_odd
            {
                typedef typename thrust::iterator_value<iterator>::type value;

                value operator()( iterator odd, iterator begin, iterator end ) const
                {
                    const auto f = sqrt(2.0) / 4.0;
                    return f * (*odd);
                }
            };

            template < typename iterator>
            void transform( iterator begin, iterator end )
            {
                lwt::fwd::split(begin, end);

                lwt::fwd::update( begin, end, updater0<iterator>() );
                lwt::fwd::predict( begin, end, predictor0<iterator>());                

                lwt::fwd::update( begin, end, cdf44<iterator>() );

                lwt::normalize_even( begin, end, scale_even<iterator>() );
                lwt::normalize_odd( begin, end, scale_odd<iterator>() );
            }
        }
    }
}

namespace bezier
{
    typedef glm::vec3   point3;
    typedef glm::vec2   point2;
    typedef float       point1;

    typedef glm::vec2   vector2;
    typedef glm::vec3   vector3;
    typedef glm::vec4   vector4;


    inline float dot3 ( point3 a, point3 b )
    {
        return glm::dot( a, b );
    }

    inline float dot2 ( point2 a, point2 b )
    {
        return glm::dot( a, b );
    }

    inline float distance ( point3 a, point3 b)
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

    template<typename point, int32_t degree >
    point evaluate ( const point* control_points, float t )
    {
        std::tr1::array< point, degree + 1 > buffer_points;

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


    //one iterator for ( q(u)- p ) dot (q'(u) ) = 0
    template <typename point>
    inline float newton_raphson ( const point q[4], point p, float u )
    {
        point q_u, q1_u, q2_u;  // u evaluated for q, q', q''
        point q1[3];            // q'   control points
        point q2[2];            // q''  control points

        //generate q'
        for ( int32_t i = 0 ; i <= 2; ++i )
        {
            q1[i] = ( q[i + 1 ] - q[i] ) / 3.0f;
        }


        //generate q''
        for ( int32_t i = 0 ; i <= 1; ++i )
        {
            q2[i] = ( q1[i + 1 ] - q1[i] ) / 2.0f;
        }

        q_u  = evaluate<point, 3>  ( &q[0],  u );
        q1_u = evaluate<point, 2>  ( &q1[0], u );
        q2_u = evaluate<point, 1>  ( &q2[0], u );

        //compute f(u) / f'(u) ( q(u) - p ) dot q'(u)
        float numerator = dot3 ( q_u - p, q1_u );

        // q'(u) dot q'(u) + q(u) dot q''(u)
        float denomerator = dot3 ( q1_u - p, q1_u - p ) + dot3 ( q_u - p, q2_u );

        if ( denomerator == 0.0f )
        {
            return u;
        }

        // u = u - f(u)/ f'(u) 
        return u - numerator / denomerator;
    }

    template<typename point, typename vector> vector left_tangent( const point& p0, const point& p1 )
    {
        return glm::normalize( p1 - p0 );
    }

    template<typename point, typename vector> vector right_tangent( const point& p0, const point& p1 )
    {
        return glm::normalize( p0 - p1 );
    }

    template<typename point, typename vector> vector center_tangent( const point& p0, const point& p1 )
    {
        auto v1 = p0 - p1;
        auto v2 = p1 - p0;
        return glm::normalize( (v1 + v2) / 2.0f );
    }

    template<typename point, typename const_iterator, typename iterator> void chord_length_parametrize( const_iterator begin, const_iterator end, iterator output )
    {
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

    template <typename point, typename const_iterator_points, typename const_iterator_curve> max_error<point> compute_max_error( const_iterator_points begin, const_iterator_points end, const point* bezier, const_iterator_curve curve_params )
    {
        float max_distance = 0.0f;

        point split = *( begin + std::distance( begin, end ) / 2 ) ;

        for ( auto it = begin; it != end; ++it, ++curve_params )
        {
            auto p = evaluate<point, 3> ( bezier, *curve_params );

            auto new_distance = dot ( p - *it, p - *it );

            if (new_distance > max_distance )
            {
                max_distance = new_distance;
                split = *it;
            }
        }

        return max_error<point>( max_distance, split );
    }
}

std::int32_t main(int argc, _TCHAR* argv[])
{
    bezier::point3 p[] =
    { 
        bezier::point3(0.0f, 0.0f, 0.0f),
        bezier::point3(1.0f, 0.0f, 0.0f),
        bezier::point3(2.0f, 0.0f, 0.0f),
        bezier::point3(5.0f, 0.0f, 0.0f),
    };

    bezier::point2 p1[] =
    { 
        bezier::point2(0.0f, 0.0f),
        bezier::point2(1.0f, 0.0f),
        bezier::point2(2.0f, 0.0f),
        bezier::point2(5.0f, 0.0f),
    };

    std::tr1::array< float, 4  > param;

    bezier::max_error<bezier::point3> r1 = bezier::compute_max_error( std::begin(p), std::end(p), std::begin(p), std::begin(param) );


    bezier::chord_length_parametrize<float>( std::begin(p1), std::end(p1), param.begin() );



    /*

    auto p1 = bezier::newton_raphson<bezier::point3>(p, bezier::point3(2.0, 0.0, 0.0), 1.0f);
    auto p2 = bezier::newton_raphson<bezier::point3>(p, bezier::point3(2.0, 0.0, 0.0), p1 );
    auto p3 = bezier::newton_raphson<bezier::point3>(p, bezier::point3(2.0, 0.0, 0.0), p2 );
    auto p4 = bezier::newton_raphson<bezier::point3>(p, bezier::point3(2.0, 0.0, 0.0), p3 );
    auto p5 = bezier::newton_raphson<bezier::point3>(p, bezier::point3(2.0, 0.0, 0.0), p4 );
    auto p6 = bezier::newton_raphson<bezier::point3>(p, bezier::point3(2.0, 0.0, 0.0), p5 );
    auto p7 = bezier::newton_raphson<bezier::point3>(p, bezier::point3(2.0, 0.0, 0.0), p6 );
    auto p8 = bezier::newton_raphson<bezier::point3>(p, bezier::point3(2.0, 0.0, 0.0), p7 );
    auto p9 = bezier::newton_raphson<bezier::point3>(p, bezier::point3(2.0, 0.0, 0.0), p8 );
    auto p10 = bezier::newton_raphson<bezier::point3>(p, bezier::point3(2.0, 0.0, 0.0), p9 );

    */

    return 0;
}

