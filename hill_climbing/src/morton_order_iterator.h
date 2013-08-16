#ifndef __morton_order_h__
#define __morton_order_h__

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>


inline  __host__ __device__ uint32_t dilate_2(uint16_t t)
{
    uint32_t r = t;

    r = (r | ( r << 8) ) & 0x00FF00FF;
    r = (r | ( r << 4) ) & 0x0F0F0F0F;
    r = (r | ( r << 2) ) & 0x33333333;
    r = (r | ( r << 1) ) & 0x55555555;

    return r ;
}


//iterates on a 2d range in a morton order. rows and columns are expected to be power of 2 and up to 16536
template <typename t> struct morton_order : public thrust::unary_function<t, t>
{
    t m_rows;
    t m_columns;

    __host__ __device__
    explicit morton_order( t rows, t columns ) : m_rows ( rows ), m_columns(columns)
    {

    }

    __host__ __device__
    uint32_t operator() ( t i ) const
    {
        const uint16_t row = static_cast<uint16_t> ( i / m_rows );
        const uint16_t col = static_cast<uint16_t> ( i % m_columns );
        const uint32_t dilated_row = dilate_2 ( row );
        const uint32_t dilated_col = dilate_2 ( col );
        const uint32_t result = ( dilated_row << 1 ) + ( dilated_col );

        return  result;
    }
};

template <typename iterator> struct morton_order_2d
{
    public:

    typedef typename thrust::iterator_difference<iterator>::type difference_type;


    morton_order_2d( iterator begin, iterator end, difference_type rows, difference_type columns ) :
    m_begin(begin)
    , m_end(end)
    , m_rows(rows)
    , m_columns(columns)
    {

    }

    typedef typename thrust::counting_iterator<difference_type>                                         counting_iterator;
    typedef typename thrust::transform_iterator< morton_order<difference_type>, counting_iterator>      transform_iterator;
    typedef typename thrust::permutation_iterator<iterator, transform_iterator>                         permutation_iterator;

    permutation_iterator begin() const
    {

        typedef typename thrust::counting_iterator<difference_type>                                         counting_iterator;
        typedef typename thrust::transform_iterator< morton_order<difference_type>, counting_iterator>      transform_iterator;
        typedef typename thrust::permutation_iterator<iterator, transform_iterator>                         permutation_iterator;


        return permutation_iterator( m_begin, transform_iterator( counting_iterator(0), morton_order<difference_type>( m_rows, m_columns ) ) ) ;
    }

    permutation_iterator end() const
    {
        return begin() + m_rows * m_columns;
    }

    private:
    iterator                        m_begin;
    iterator                        m_end;

    difference_type                 m_rows;
    difference_type                 m_columns;

};

#endif