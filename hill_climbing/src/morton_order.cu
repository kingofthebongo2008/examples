#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "morton_order.h"
#include "morton_order_iterator.h"

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

