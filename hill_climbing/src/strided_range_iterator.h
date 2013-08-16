#ifndef __strided_range_h__
#define __strided_range_h__

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

// this example illustrates how to make strided access to a range of values
// examples:
//   strided_range([0, 1, 2, 3, 4, 5, 6], 1) -> [0, 1, 2, 3, 4, 5, 6] 
//   strided_range([0, 1, 2, 3, 4, 5, 6], 2) -> [0, 2, 4, 6]
//   strided_range([0, 1, 2, 3, 4, 5, 6], 3) -> [0, 3, 6]
//   ...

template <typename iterator> class strided_range
{
    public:

    typedef typename thrust::iterator_difference<iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function< difference_type, difference_type >
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                     counting_iterator;
    typedef typename thrust::transform_iterator<stride_functor, counting_iterator>  transform_iterator;
    typedef typename thrust::permutation_iterator<iterator,transform_iterator>      permutation_iterator;

    // construct strided_range for the range [first,last)
    strided_range(iterator first, iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}
   
    permutation_iterator begin(void) const
    {
        return permutation_iterator(first, transform_iterator(counting_iterator(0), stride_functor(stride)));
    }

    permutation_iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }
    
    protected:
    iterator first;
    iterator last;
    difference_type stride;
};

template <typename iterator> strided_range<iterator> make_strided_range ( iterator begin, iterator end, uint32_t stride )
{
    return strided_range<iterator>(begin, end, stride);
}


#endif