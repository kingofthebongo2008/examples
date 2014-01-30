#include "precompiled.h"

#include <cstdint>
#include <iostream>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include <thrust/iterator/zip_iterator.h>

#include <vector_functions.h>

#include "haar.h"
#include "line.h"

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
                typedef typename thrust::iterator_value<iterator>::type value;

                struct identity
                {
                    value operator()( iterator even, iterator begin, iterator end ) const
                    {
                        return *even;
                    }
                };

                lifting::fwd::predict( begin, end, identity());
            }

            template < typename iterator >
            void update( iterator begin, iterator end )
            {
                typedef typename thrust::iterator_value<iterator>::type value;

                struct identity
                {
                    value operator()( iterator odd, iterator begin, iterator end ) const
                    {
                        return (*odd) / 2.0f;
                    }
                };

                lifting::fwd::update( begin, end, identity() );
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
                typedef typename thrust::iterator_value<iterator>::type value;

                struct identity
                {
                    value operator()( iterator even, iterator begin, iterator end ) const
                    {
                        return *even;
                    }
                };

                lifting::inv::predict( begin, end, identity());
            }

            template < typename iterator >
            void update( iterator begin, iterator end )
            {
                typedef typename thrust::iterator_value<iterator>::type value;

                struct identity
                {
                    value operator()( iterator odd, iterator begin, iterator end ) const
                    {
                        return (*odd) / 2.0f;
                    }
                };

                lifting::inv::update( begin, end, identity() );
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
                typedef typename thrust::iterator_value<iterator>::type value;

                struct identity
                {
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

                lifting::fwd::predict( begin, end, identity());
            }

            template < typename iterator >
            void update( iterator begin, iterator end )
            {
                typedef typename thrust::iterator_value<iterator>::type value;

                struct identity
                {
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

                lifting::fwd::update( begin, end, identity() );
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
                typedef typename thrust::iterator_value<iterator>::type value;

                struct identity
                {
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


                lifting::inv::predict( begin, end, identity());
            }

            template < typename iterator >
            void update( iterator begin, iterator end )
            {
                typedef typename thrust::iterator_value<iterator>::type value;

                struct identity
                {
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

                lifting::inv::update( begin, end, identity() );
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
    double arr[] = { 4 , 3 , 2, 1, 2, 3 , 4 , 5 };
    double arr_orig[] = { arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7] };
    double arr2[] = { 4 , 3 , 2, 1, 1, 2 , 3 , 4 };

    
    //lifting::linear::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / sizeof(arr[0]) );
    //lifting::linear::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / ( 2* sizeof(arr[0]) ) );
    //lifting::linear::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / ( 4* sizeof(arr[0]) ) );

    typedef line<double[8]> linear;

    linear l;

    l.forwardStep(arr2, 8);
    l.forwardStep(arr2, 4);
    l.forwardStep(arr2, 2);

    
    lifting::haar::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / sizeof(arr[0]) );
    lifting::haar::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / ( 2* sizeof(arr[0]) ) );
    lifting::haar::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / ( 4* sizeof(arr[0]) ) );

    lifting::haar::inv::transform(&arr[0], &arr[0] + sizeof(arr) / ( 4* sizeof(arr[0]) ) );
    lifting::haar::inv::transform(&arr[0], &arr[0] + sizeof(arr) / ( 2* sizeof(arr[0]) ) );
    lifting::haar::inv::transform(&arr[0], &arr[0] + sizeof(arr) / sizeof(arr[0]) );
    

    lifting::linear::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / sizeof(arr[0]) );
    lifting::linear::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / ( 2* sizeof(arr[0]) ) );
    lifting::linear::fwd::transform(&arr[0], &arr[0] + sizeof(arr) / ( 4* sizeof(arr[0]) ) );

    
    lifting::linear::inv::transform(&arr[0], &arr[0] + sizeof(arr) / ( 4* sizeof(arr[0]) ) );
    lifting::linear::inv::transform(&arr[0], &arr[0] + sizeof(arr) / ( 2* sizeof(arr[0]) ) );
    lifting::linear::inv::transform(&arr[0], &arr[0] + sizeof(arr) / sizeof(arr[0]) );
    
    
    
    
    float arr1[] = { 1, 2 , 3, 4};

    typedef haar< float[4] > haar_wavelet;
     
    haar_wavelet h;

    h.split( arr1, 4 );
    h.predict( arr1, 4, haar_wavelet::forward );
    h.update( arr1, 4, haar_wavelet::forward );
    h.normalize(arr1, 4, haar_wavelet::forward );


    return 0;
}

