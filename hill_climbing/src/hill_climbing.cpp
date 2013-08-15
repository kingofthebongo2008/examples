// hill_climbing.cpp : Defines the entry point for the console application.
//

#include "precompiled.h"
#include <cstdint>

#include <vector>
#include <iostream>

#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>

#include "morton_order.h"

struct grayscale_image
{
    uint32_t    m_width;
    uint32_t    m_height;
    uint32_t    m_pitch;

    std::vector< uint8_t > m_image;
};


template <typename Iterator>
class tiled_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct tile_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type tile_size;

        tile_functor(difference_type tile_size)
            : tile_size(tile_size) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return i % tile_size;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<tile_functor, CountingIterator>   TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the tiled_range iterator
    typedef PermutationIterator iterator;

    // construct repeated_range for the range [first,last)
    tiled_range(Iterator first, Iterator last, difference_type tiles)
        : first(first), last(last), tiles(tiles) {}
   
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), tile_functor(last - first)));
    }

    iterator end(void) const
    {
        return begin() + tiles * (last - first);
    }
    
    protected:
    Iterator first;
    Iterator last;
    difference_type tiles;
};


int main(int argc, _TCHAR* argv[])
{
    // generate 20 random numbers on the host
    thrust::host_vector<float> h_vec(16);
    thrust::host_vector<float> h_vec_out(16);
    


    h_vec[0] = 0.0f;
    h_vec[1] = 1.0f;
    h_vec[2] = 2.0f;
    h_vec[3] = 3.0f;

    h_vec[4] = 4.0f;
    h_vec[5] = 5.0f;
    h_vec[6] = 6.0f;
    h_vec[7] = 7.0f;

    h_vec[8] = 8.0f;
    h_vec[9] = 9.0f;
    h_vec[10] = 10.0f;
    h_vec[11] = 11.0f;

    h_vec[12] = 12.0f;
    h_vec[13] = 13.0f;
    h_vec[14] = 14.0f;
    h_vec[15] = 15.0f;

    // interface to CUDA code
    convert_to_morton_order_2d( h_vec, 4, 4, h_vec_out);

    // print sorted array
    thrust::copy(h_vec_out.begin(), h_vec_out.end(), std::ostream_iterator<int>(std::cout, "\n"));

    return 0;
}

