#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "morton_order.h"
#include "morton_order_range.h"

struct a
{
    a(float f) {}
};

struct unquantize : public thrust::unary_function< uint8_t, float > 
{
    __host__ __device__
    a operator() ( uint8_t value ) const
    {
        return 0.2f;//5;//static_cast<float> (value) / 255.0f;
    }
};

struct quantize : public thrust::unary_function< float, uint8_t > 
{
    __host__ __device__
    uint8_t operator() ( float value ) const
    {
        return static_cast<uint8_t> ( value * 255.0f );
    }
};


void convert_to_morton_order_2d2( thrust::host_vector<uint8_t>& in, uint32_t rows, uint32_t columns, thrust::host_vector<float>& out )
{
    // transfer data to the device
    thrust::host_vector<uint8_t> d_vec0 = in;

    thrust::transform_iterator<unquantize, thrust::host_vector<uint8_t>::iterator, float> begin0( in.begin(), unquantize());
    thrust::transform_iterator<unquantize, thrust::host_vector<uint8_t>::iterator, float> end0( in.end(), unquantize() );

    auto morton = make_linear_2_morton_order_2d_range ( begin0, end0, rows, columns );

    //thrust::copy(morton.begin(), morton.end(), out.begin() );
    thrust::copy(begin0, end0, out.begin() );
}

void convert_to_morton_order_2d( thrust::host_vector<float>& in, uint32_t rows, uint32_t columns, thrust::host_vector<float>& out )
{
    // transfer data to the device
    thrust::device_vector<float> d_vec0 = in;

    thrust::device_vector<float> d_vec1 ( d_vec0.size() );

    auto morton = make_linear_2_morton_order_2d_range ( d_vec0.begin(), d_vec0.end(), rows, columns );

    thrust::copy(morton.begin(), morton.end(), d_vec1.begin() );

    auto linear  = make_morton_order_2d_2_linear_range( d_vec1.begin(), d_vec1.end(), rows, columns );
    
    
    thrust::copy ( linear.begin(), linear.end(), out.begin() );

}

