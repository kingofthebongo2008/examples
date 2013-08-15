#pragma once

#include <thrust/host_vector.h>

// function prototype
void convert_to_morton_order_2d( thrust::host_vector<float>& in, uint32_t rows, uint32_t columns, thrust::host_vector<float>& out );
