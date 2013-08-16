#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "morton_order.h"
#include "morton_order_iterator.h"

void convert_to_morton_order_2d( thrust::host_vector<float>& in, uint32_t rows, uint32_t columns, thrust::host_vector<float>& out )
{
    // transfer data to the device
    thrust::device_vector<float> d_vec = in;

    morton_order_2d < thrust::device_vector<float>::iterator > morton( d_vec.begin(), d_vec.end(), rows, columns );

    // transfer data back to host
    thrust::copy(morton.begin(), morton.end(), out.begin() );

}

