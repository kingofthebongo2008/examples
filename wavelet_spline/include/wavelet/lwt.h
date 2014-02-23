#ifndef __wavelet_lwt_curve_h__
#define __wavelet_lwt_curve_h__

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

#endif
