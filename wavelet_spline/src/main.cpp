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

#include "liftbase.h"
#include "daub.h"

namespace lifting
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
                lifting::fwd::split(begin, end);
                lifting::fwd::predict( begin, end, predictor<iterator>());
                lifting::fwd::update( begin, end, updater<iterator>() );
            }
        }

        namespace inv
        {
            template < typename iterator>
            void transform( iterator begin, iterator end )
            {
                lifting::inv::update( begin, end, updater<iterator>() );
                lifting::inv::predict( begin, end, predictor<iterator>());
                lifting::inv::merge(begin, end);
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
                lifting::fwd::split( begin, end );
                lifting::fwd::predict( begin, end, predictor<iterator>() );
                lifting::fwd::update( begin, end, updater<iterator>() );
            }
        }

        namespace inv
        {
            template < typename iterator>
            void transform( iterator begin, iterator end )
            {
                lifting::inv::update( begin, end, updater<iterator>() );
                lifting::inv::predict( begin, end, predictor<iterator>());
                lifting::inv::merge( begin, end );
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
                lifting::fwd::split(begin, end);
                lifting::fwd::update( begin, end, updater0<iterator>() );
                lifting::fwd::predict( begin, end, predictor0<iterator>());
                lifting::fwd::update( begin, end, updater1<iterator>() );

                lifting::normalize_even( begin, end, scale_even<iterator>() );
                lifting::normalize_odd( begin, end, scale_odd<iterator>() );
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
                lifting::normalize_even( begin, end, scale_even<iterator>() );
                lifting::normalize_odd( begin, end, scale_odd<iterator>() );

                lifting::inv::update( begin, end, updater1<iterator>() );
                lifting::inv::predict( begin, end, predictor0<iterator>());
                lifting::inv::update( begin, end, updater0<iterator>() );
                lifting::inv::merge(begin, end);
            }
        }
    }
}

std::int32_t main(int argc, _TCHAR* argv[])
{
    using namespace lifting;

    //double arr[] = { 4 , 3 , 2, 1, 2, 3 , 4 , 5 };
    //double arr1[] = { 4 , 3 , 2, 1, 2, 3 , 4 , 5 };

    double arr[]    = { 2, 4, 16, 256, 65536, 6, 7, 8 };
    double arr1[]   = { 1, 2, 3, 4, 5, 6, 7, 8 };

    typedef std::tr1::array<double, 8> arr_type  ;
    std::tr1::array<double, 8>  arr2 = { 1, 2, 3, 4, 5, 6, 7, 8 };

    arr[0] = arr1[0] = arr2[0] = 0.125;

    for ( auto i = 1; i < 8 ;++i)
    {
        arr[i]  = arr[i-1] * arr[i-1];
        arr1[i] = arr1[i-1] * arr1[i-1];
        arr2[i] = arr2[i-1] * arr2[i-1];
    }

    d4::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / sizeof(arr[0]) );
    d4::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / ( 2 * sizeof(arr[0] ) ) );
    d4::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / ( 4 * sizeof(arr[0] ) ) );

    linear::fwd::transform(&arr1[0], &arr1[0] + sizeof(arr1) / ( sizeof(arr1[0]) ) );
    linear::fwd::transform(&arr1[0], &arr1[0] + sizeof(arr1) / ( 2 * sizeof(arr1[0]) ) );
    linear::fwd::transform(&arr1[0], &arr1[0] + sizeof(arr1) / ( 4* sizeof(arr1[0]) ) );

    d4::inv::transform(&arr[0], &arr[0] + sizeof(arr) / ( 4 * sizeof(arr[0] ) ) );
    d4::inv::transform(&arr[0], &arr[0] + sizeof(arr) / ( 2 * sizeof(arr[0] ) ) );
    d4::inv::transform(&arr[0], &arr[0] + sizeof(arr) / sizeof(arr[0]) );

    /*
    linear::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / ( 2* sizeof(arr[0]) ) );
    linear::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / ( 4* sizeof(arr[0]) ) );
    linear::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / ( 8* sizeof(arr[0]) ) );
    */
    /*
    linear::inv::transform(&arr[0], &arr[0] + sizeof(arr) / ( 8* sizeof(arr[0]) ) );
    linear::inv::transform(&arr[0], &arr[0] + sizeof(arr) / ( 4* sizeof(arr[0]) ) );
    linear::inv::transform(&arr[0], &arr[0] + sizeof(arr) / ( 2* sizeof(arr[0]) ) );
    linear::inv::transform(&arr[0], &arr[0] + sizeof(arr) / sizeof(arr[0]) );
    */

    
    
    Daubechies< arr_type > d;

    d.forwardTrans( arr2, 8);
   
    return 0;
}

