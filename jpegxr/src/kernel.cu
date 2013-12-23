#include <cstdint>
#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

#include <jxr/cuda_helper.h>
#include <jxr/jxr_transforms.h>
#include <jxr/jxr_analysis.h>

#include <jxr/jxr_filter.h>
#include <jxr/jxr_prefilter.h>

#include <os/windows/com_initializer.h>

#include "img_loader.h"

namespace example
{
    void add_with_cuda(int32_t * c, const int32_t * a, uint32_t size);

    class cuda_initializer
    {
        public:
        cuda_initializer()
        {
            // Choose which GPU to run on, change this on a multi-GPU system.
            cuda::throw_if_failed<cuda::exception> (  cudaSetDevice(0) );
        }

        ~cuda_initializer()
        {
            // cudaDeviceReset must be called before exiting in order for profiling and
            // tracing tools such as Nsight and Visual Profiler to show complete traces.
            cuda::throw_if_failed<cuda::exception> ( cudaDeviceReset() );
        }
    };
}

namespace example
{
    __global__ void addKernel( int32_t * c, const int32_t * a )
    {
        int i = threadIdx.x;

        jpegxr::transforms::pixel v[16] =
        { 
            a[0], a[1], a[2], a[3],
            a[4], a[5], a[6], a[7],
            a[8], a[9], a[10], a[11],
            a[12], a[13], a[14], a[15]
        };

        jpegxr::transforms::analysis::pct4x4
            (
               v
            );

        c[i]   = v[0];
        c[i+1] = v[1];
        c[i+2] = v[2];
        c[i+3] = v[3];

        c[i+4] = v[4];
        c[i+5] = v[5];
        c[i+6] = v[6];
        c[i+7] = v[7];

        c[i+8] = v[8];
        c[i+9] = v[9];
        c[i+10] = v[10];
        c[i+11] = v[11];

        c[i+12] = v[12];
        c[i+13] = v[13];
        c[i+14] = v[14];
        c[i+15] = v[15];
    }

    // Helper function for using CUDA to add vectors in parallel.
    void add_with_cuda(int *c, const int *a, uint32_t size)
    {
        // Allocate GPU buffers for three vectors (two input, one output)    .
        auto dev_a = std::make_shared< cuda::memory_buffer > ( size * sizeof( int32_t )  );
        auto dev_c = std::make_shared< cuda::memory_buffer > ( size * sizeof( int32_t )  );

        // Copy input vectors from host memory to GPU buffers.
        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy(*dev_a, a, size * sizeof(int32_t), cudaMemcpyHostToDevice) );

        // Launch a kernel on the GPU with one thread for each element.
        addKernel<<<1, 1>>>( *dev_c, *dev_a );

        // Check for any errors launching the kernel
        cuda::throw_if_failed<cuda::exception> ( cudaGetLastError() );
   
        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cuda::throw_if_failed<cuda::exception> ( cudaDeviceSynchronize() );

        // Copy output vector from GPU buffer to host memory.
        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy(c, dev_c->get(), size * sizeof(int32_t), cudaMemcpyDeviceToHost) );
    }

    struct rgb 
    {
        uint8_t color[3];
    };

    __global__ void decompose_kernel( const rgb* in, uint32_t* y_color, uint32_t* co_color, uint32_t* cg_color, const uint32_t read_pitch, const uint32_t write_pitch )
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        auto row = y;
        auto col = x;
        
        auto element = reinterpret_cast<const rgb*> (  (uint8_t*) in + ( row * read_pitch )  + sizeof(rgb) * col ); 

        jpegxr::transforms::pixel r_y  = element->color[0];
        jpegxr::transforms::pixel g_co = element->color[1];
        jpegxr::transforms::pixel b_cg = element->color[2];

        jpegxr::transforms::rgb_2_ycocg(&r_y, &g_co, &b_cg );

        y_color [ row * write_pitch + col ] = r_y;
        co_color[ row * write_pitch + col ] = g_co;
        cg_color[ row * write_pitch + col ] = b_cg;
    }

    class ycocg_image
    {
        public:

        ycocg_image
            (
                std::shared_ptr< cuda::memory_buffer >  y_buffer,
                std::shared_ptr< cuda::memory_buffer >  co_buffer,
                std::shared_ptr< cuda::memory_buffer >  cg_buffer,

                uint32_t                                width,
                uint32_t                                height,
                uint32_t                                pitch
            ) :
              m_y_buffer(y_buffer)
            , m_co_buffer(co_buffer)
            , m_cg_buffer(cg_buffer)
            , m_width(width)
            , m_height(height)
            , m_pitch(pitch)
        {

        }


        std::shared_ptr< cuda::memory_buffer >  get_y() const
        {
            return m_y_buffer;
        }

        std::shared_ptr< cuda::memory_buffer >  get_co() const
        {
            return m_co_buffer;
        }

        std::shared_ptr< cuda::memory_buffer >  get_cg() const
        {
            return m_cg_buffer;
        }

        uint32_t    get_width() const
        {
            return m_width;
        }

        uint32_t    get_height() const
        {
            return m_height;
        }

        uint32_t    get_pitch() const
        {
            return m_pitch;
        }

        private:
        std::shared_ptr< cuda::memory_buffer >  m_y_buffer;
        std::shared_ptr< cuda::memory_buffer >  m_co_buffer;
        std::shared_ptr< cuda::memory_buffer >  m_cg_buffer;

        uint32_t                                m_width;
        uint32_t                                m_height;
        uint32_t                                m_pitch;

    };

    ycocg_image decompose_ycocg ( const image& image ) 
    {
        auto w         = image.get_width();
        auto h         = image.get_height();
        auto size      = w * h * sizeof(int32_t) ;
        
        auto y_buffer  = std::make_shared < cuda::memory_buffer > ( cuda::allocate<void*> ( size ) );
        auto co_buffer = std::make_shared < cuda::memory_buffer > ( cuda::allocate<void*> ( size ) );
        auto cg_buffer = std::make_shared < cuda::memory_buffer > ( cuda::allocate<void*> ( size ) );

        auto blocks = 1;
        auto threads_per_block = dim3( w, h );

        decompose_kernel<<<blocks, threads_per_block>>>( image, *y_buffer, *co_buffer, *cg_buffer, image.get_pitch(), w );

        cuda::throw_if_failed<cuda::exception> ( cudaGetLastError() );
        cuda::throw_if_failed<cuda::exception> ( cudaDeviceSynchronize() );

        auto y  = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );
        auto co = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );
        auto cg = std::unique_ptr< uint8_t[] > ( new uint8_t [ size ] );

        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( y.get(),  y_buffer->get(),  size   , cudaMemcpyDeviceToHost) );
        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( co.get(), co_buffer->get(), size   , cudaMemcpyDeviceToHost) );
        cuda::throw_if_failed<cuda::exception> ( cudaMemcpy( cg.get(), cg_buffer->get(), size   , cudaMemcpyDeviceToHost) );

        // element access into this image looks like this
        auto row = 15;
        auto col = 15;
        auto res1 = reinterpret_cast<int32_t*> ( y.get()  );
        auto res2 = reinterpret_cast<int32_t*> ( co.get() );
        auto res3 = reinterpret_cast<int32_t*> ( cg.get() );
        auto el1 = res1[ row * w + col ];

        return ycocg_image ( y_buffer, co_buffer, cg_buffer, w, h, w );
    }
}

int32_t main()
{
    try
    {
        auto com_initializer  =  os::windows::com_initializer();
        auto cuda_initializer = example::cuda_initializer();
        auto image  =  example::create_image ( L"test_32x32.png" );

        const int32_t arraySize = 16;
        const jpegxr::transforms::pixel a[arraySize] = 
        { 
            0, 0, 0, 0,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1
        };

        jpegxr::transforms::pixel c[arraySize] = { 0 };

        auto ycocg = decompose_ycocg(*image);

        jpegxr::prefilter2x2_edge( ycocg.get_y(), ycocg.get_width(), ycocg.get_height(), ycocg.get_width() );

        /*
        example::prefilter4x4_image( ycocg.get_y(), ycocg.get_width(), ycocg.get_height(), ycocg.get_width() );

        example::prefilter4_vertical_image( ycocg.get_y(), ycocg.get_width(), ycocg.get_height(), ycocg.get_width() );( ycocg.get_y(), ycocg.get_width(), ycocg.get_height(), ycocg.get_width() );

        example::prefilter4_horizontal_image( ycocg.get_y(), ycocg.get_width(), ycocg.get_height(), ycocg.get_width() );( ycocg.get_y(), ycocg.get_width(), ycocg.get_height(), ycocg.get_width() );


        example::prefilter2x2( ycocg.get_y(), ycocg.get_width(), ycocg.get_height(), ycocg.get_width() );

        */
        std::cout << std::endl << c[0] << ", " << c[1] << ", " << c[2] << ", " << c[3] << ", " << std::endl <<  c[4] << ", " << c[5] << ", " << c[6] << ", " << c[7] << ", " << std::endl << c[8] << ", " << c[9] << ", " << c[10] << ", "  << c[11] << ", " << std::endl << c[12] << ", " << c[13] << ", " << c[14] << ", " << c[15] << std::endl;
    }

    catch (const cuda::exception& e)
    {
        std::cerr<<e.what()<<std::endl;
        return 1;
    }

    return 0;
}
