#include "precompiled.h"

#include <cstdint>
#include <iostream>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include <thrust/iterator/zip_iterator.h>

#include <vector_functions.h>



namespace lifting
{
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

                auto predicted = odd - p( it, begin, end );

                *(it + middle ) = predicted;
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

                auto updated = even + u( it + middle, begin, end );

                *(it) = updated;
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

                auto predicted = odd + p( it, begin, end );

                *(it + middle ) = predicted;
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

                auto updated = even - u( it + middle, begin, end );

                *(it) = updated;
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
                return (*odd) / 2.0f;
            }
        };

        namespace fwd
        {
            template <typename iterator>
            void split( iterator begin, iterator end )
            {
                lifting::fwd::split(begin, end);
            }

            template <typename iterator>
            void predict(iterator begin, iterator end)
            {
                lifting::fwd::predict( begin, end, predictor<iterator>());
            }

            template < typename iterator >
            void update( iterator begin, iterator end )
            {
                lifting::fwd::update( begin, end, updater<iterator>() );
            }

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
                split(begin, end);
                predict(begin, end);
                update(begin, end);
            }
        }

        namespace inv
        {
            template <typename iterator>
            void merge( iterator begin, iterator end )
            {
                lifting::inv::merge(begin, end);
            }

            template <typename iterator>
            void predict(iterator begin, iterator end)
            {
                lifting::inv::predict( begin, end, predictor<iterator>());
            }

            template < typename iterator >
            void update( iterator begin, iterator end )
            {
                lifting::inv::update( begin, end, updater<iterator>() );
            }

            template < typename iterator>
            void transform( iterator begin, iterator end )
            {
                update( begin, end );
                predict( begin, end );
                merge( begin, end );
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
            template <typename iterator>
            void split( iterator begin, iterator end )
            {
                lifting::fwd::split(begin, end);
            }

            template <typename iterator>
            void predict(iterator begin, iterator end)
            {
                lifting::fwd::predict( begin, end, predictor<iterator>());
            }

            template < typename iterator >
            void update( iterator begin, iterator end )
            {
                lifting::fwd::update( begin, end, updater<iterator>() );
            }

            template < typename iterator>
            void transform( iterator begin, iterator end )
            {
                split(begin, end);
                predict(begin, end);
                update(begin, end);
            }
        }

        namespace inv
        {
            template <typename iterator>
            void merge( iterator begin, iterator end )
            {
                lifting::inv::merge(begin, end);
            }

            template <typename iterator>
            void predict(iterator begin, iterator end)
            {
                lifting::inv::predict( begin, end, predictor<iterator>());
            }

            template < typename iterator >
            void update( iterator begin, iterator end )
            {
                lifting::inv::update( begin, end, updater<iterator>() );
            }

            template < typename iterator>
            void transform( iterator begin, iterator end )
            {
                update( begin, end );
                predict( begin, end );
                merge( begin, end );
            }
        }

    }
}

std::int32_t main(int argc, _TCHAR* argv[])
{
    using namespace lifting;

    double arr[] = { 4 , 3 , 2, 1, 2, 3 , 4 , 5 };
    double arr_orig[] = { arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7] };
    double arr2[] = { 4 , 3 , 2, 1, 1, 2 , 3 , 4 };
    
    haar::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / sizeof(arr[0]) );
    haar::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / ( 2* sizeof(arr[0]) ) );
    haar::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / ( 4* sizeof(arr[0]) ) );

    haar::inv::transform(&arr[0], &arr[0] + sizeof(arr) / ( 4* sizeof(arr[0]) ) );
    haar::inv::transform(&arr[0], &arr[0] + sizeof(arr) / ( 2* sizeof(arr[0]) ) );
    haar::inv::transform(&arr[0], &arr[0] + sizeof(arr) / sizeof(arr[0]) );
    

    linear::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / sizeof(arr[0]) );
    linear::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / ( 2* sizeof(arr[0]) ) );
    linear::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / ( 4* sizeof(arr[0]) ) );
    linear::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / ( 8* sizeof(arr[0]) ) );

    linear::inv::transform(&arr[0], &arr[0] + sizeof(arr) / ( 8* sizeof(arr[0]) ) );
    linear::inv::transform(&arr[0], &arr[0] + sizeof(arr) / ( 4* sizeof(arr[0]) ) );
    linear::inv::transform(&arr[0], &arr[0] + sizeof(arr) / ( 2* sizeof(arr[0]) ) );
    linear::inv::transform(&arr[0], &arr[0] + sizeof(arr) / sizeof(arr[0]) );
   
   
    return 0;
}

