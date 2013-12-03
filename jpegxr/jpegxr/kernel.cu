
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_helper.h"

#include <iostream>
#include <memory>


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cuda::throw_if_failed<cuda::exception> ( addWithCuda(c, a, b, arraySize) );

    std::cout << "{1,2,3,4,5} + {10,20,30,40,50} = "<< std::endl << c[0] << c[1] << c[2] << c[3] << c[4];

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cuda::throw_if_failed<cuda::exception> ( cudaDeviceReset() );

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    cuda::throw_if_failed<cuda::exception> (  cudaSetDevice(0) );

    // Allocate GPU buffers for three vectors (two input, one output)    .
    auto dev_a = std::make_shared< cuda::memory_buffer > ( size * sizeof( int )  );
    auto dev_b = std::make_shared< cuda::memory_buffer > ( size * sizeof( int )  );
    auto dev_c = std::make_shared< cuda::memory_buffer > ( size * sizeof( int )  );

    // Copy input vectors from host memory to GPU buffers.
    cuda::throw_if_failed<cuda::exception> ( cudaMemcpy(dev_a->get(), a, size * sizeof(int), cudaMemcpyHostToDevice) );
    cuda::throw_if_failed<cuda::exception> ( cudaMemcpy(dev_b->get(), b, size * sizeof(int), cudaMemcpyHostToDevice) );

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>( *dev_c, *dev_a, *dev_b );

    // Check for any errors launching the kernel
    cuda::throw_if_failed<cuda::exception> ( cudaGetLastError() );
   
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cuda::throw_if_failed<cuda::exception> ( cudaDeviceSynchronize() );

    // Copy output vector from GPU buffer to host memory.
    cuda::throw_if_failed<cuda::exception> ( cudaMemcpy(c, dev_c->get(), size * sizeof(int), cudaMemcpyDeviceToHost) );

    return cudaSuccess;
}
