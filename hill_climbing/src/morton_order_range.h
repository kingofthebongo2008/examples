#ifndef __morton_order_range_h__
#define __morton_order_range_h__

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

namespace details
{

    inline  __host__ __device__ uint32_t dilate_2(uint16_t t)
    {
        uint32_t r = static_cast<uint32_t> ( t );

        r = (r | ( r << 8) ) & 0x00FF00FF;
        r = (r | ( r << 4) ) & 0x0F0F0F0F;
        r = (r | ( r << 2) ) & 0x33333333;
        r = (r | ( r << 1) ) & 0x55555555;

        return r ;
    }

    inline  __host__ __device__ uint16_t undilate_2 ( uint32_t t )
    {

        t = (t | (t >> 1)) & 0x33333333;
        t = (t | (t >> 2)) & 0x0F0F0F0F;
        t = (t | (t >> 4)) & 0x00FF00FF;
        t = (t | (t >> 8)) & 0x0000FFFF;

        return static_cast<uint16_t> (t);
    }


    //iterates on a 2d range in a morton order. rows and columns are expected to be power of 2 and up to 65536
    struct linear_2_morton_order : public thrust::unary_function< uint32_t, uint32_t >
    {
        uint32_t m_rows;
        uint32_t m_columns;

        __host__ __device__
        explicit linear_2_morton_order( uint32_t rows, uint32_t columns ) : m_rows ( rows ), m_columns(columns)
        {

        }

        __host__ __device__
        uint32_t operator() ( uint32_t i ) const
        {
            const uint16_t row = static_cast<uint16_t> ( i / m_rows );
            const uint16_t col = static_cast<uint16_t> ( i % m_columns );
            const uint32_t dilated_row = dilate_2 ( row );
            const uint32_t dilated_col = dilate_2 ( col );
            const uint32_t result = ( dilated_row << 1 ) + ( dilated_col );

            return  result;
        }
    };

    //iterates on a 2d range in a morton order. rows and columns are expected to be power of 2 and up to 16536
    struct morton_order_2_linear : public thrust::unary_function<  uint32_t, uint32_t >
    {
        uint32_t m_rows;
        uint32_t m_columns;

        __host__ __device__
        explicit morton_order_2_linear( uint32_t rows, uint32_t columns ) : m_rows ( rows ), m_columns(columns)
        {

        }

        __host__ __device__
        uint32_t operator() ( uint32_t i ) const
        {
            const uint32_t dilated_col = i & 0x55555555;
            const uint32_t dilated_row = (i & 0xAAAAAAAA) >> 1;

            const uint16_t col = undilate_2(dilated_col);
            const uint16_t row = undilate_2(dilated_row);

            const uint32_t result =  m_rows * row + col;

            return  result;
        }
    };


    template <typename iterator, typename transform_functor> struct morton_order_transformer_2d
    {
        public:

        typedef typename thrust::iterator_difference<iterator>::type difference_type;


        morton_order_transformer_2d( iterator begin, iterator end, difference_type rows, difference_type columns ) :
        m_begin(begin)
        , m_end(end)
        , m_rows(rows)
        , m_columns(columns)
        {

        }

        typedef typename thrust::counting_iterator<difference_type>                                                 counting_iterator;
        typedef typename thrust::transform_iterator<transform_functor, counting_iterator>                           transform_iterator;
        typedef typename thrust::permutation_iterator<iterator, transform_iterator>                                 permutation_iterator;

        permutation_iterator begin() const
        {
            return permutation_iterator( m_begin, transform_iterator( counting_iterator(0), transform_functor( m_rows, m_columns ) ) ) ;
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
}


template <typename iterator> struct linear_2_morton_order_2d_range : public details::morton_order_transformer_2d<iterator, details::linear_2_morton_order>
{

    typedef typename details::morton_order_transformer_2d<iterator, details::linear_2_morton_order>  base;
    typedef typename base::difference_type                                                           difference_type;

    linear_2_morton_order_2d_range( iterator begin, iterator end, difference_type rows, difference_type columns ) : 
    base ( begin, end, rows, columns )
    {

    }
};

template <typename iterator> struct morton_order_2d_2_linear_range : public details::morton_order_transformer_2d<iterator, details::morton_order_2_linear>
{

    typedef typename details::morton_order_transformer_2d<iterator, details::morton_order_2_linear>  base;
    typedef typename base::difference_type                                                           difference_type;


    morton_order_2d_2_linear_range( iterator begin, iterator end, difference_type rows, difference_type columns ) : 
    base ( begin, end, rows, columns )
    {

    }
};

template <typename iterator> inline linear_2_morton_order_2d_range<iterator> make_linear_2_morton_order_2d_range( iterator begin, iterator end, uint32_t rows, uint32_t columns)
{
    return linear_2_morton_order_2d_range<iterator>( begin, end, rows, columns );
}

template <typename iterator> inline morton_order_2d_2_linear_range<iterator> make_morton_order_2d_2_linear_range( iterator begin, iterator end, uint32_t rows, uint32_t columns)
{
    return morton_order_2d_2_linear_range<iterator>( begin, end, rows, columns );
}




#endif