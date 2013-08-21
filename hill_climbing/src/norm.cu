#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

#include "norm.h"

float mse( thrust::host_vector<float>& v0, thrust::host_vector<float>& v1 )
{
    thrust::device_vector<float> v2=v0;
    thrust::device_vector<float> v3=v1;

    return cuda_rt::mse( v2.begin(), v2.end(), v3.begin(), v3.end() );
}
