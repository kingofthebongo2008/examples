#include <cstdint>
#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

#include "cuda_helper.h"


namespace jpegxr
{
	namespace transforms
	{
		struct pixel4
		{
			struct names
			{
				int32_t a;
				int32_t b;
				int32_t c;
				int32_t d;
			};

			struct dimensions
			{
				int32_t v[4];
			};

			union
			{
				names n;
				dimensions v;
			} u;

			__device__ __host__ pixel4( int32_t a1, int32_t b1, int32_t c1, int32_t d1 )
			{
				u.n.a = a1;
				u.n.b = b1;
				u.n.c = c1;
				u.n.d = d1;
			}

		};
		

		struct pixel2
		{
			int32_t a;
			int32_t b;

			__device__ __host__ pixel2( int32_t a1, int32_t b1) :
			a(a1), b(b1) {}
		};

		struct block
		{
			struct names
			{
				int32_t a;	//0
				int32_t b;	//1
				int32_t c;	//2
				int32_t d;	//3
				
				int32_t e;	//4
				int32_t f;	//5
				int32_t g;	//6
				int32_t h;	//7

				int32_t i;	//8
				int32_t j;	//9
				int32_t k;	//10
				int32_t l;	//11

				int32_t m;	//12
				int32_t n;	//13
				int32_t o;	//14
				int32_t p;	//15
			};

			struct dimensions
			{
				int32_t v[16];
			};

			union
			{
				names	   n;
				dimensions p;
			} u;

			__device__ __host__ block( pixel4 r1, pixel4 r2, pixel4 r3, pixel4 r4 )
			{
				u.n.a = r1.u.n.a;
				u.n.b = r1.u.n.b;
				u.n.c = r1.u.n.c;
				u.n.d = r1.u.n.d;

				u.n.e = r2.u.n.a;
				u.n.f = r2.u.n.b;
				u.n.g = r2.u.n.c;
				u.n.h = r2.u.n.d;

				u.n.i = r3.u.n.a;
				u.n.j = r3.u.n.b;
				u.n.k = r3.u.n.c;
				u.n.l = r3.u.n.d;

				u.n.m = r4.u.n.a;
				u.n.n = r4.u.n.b;
				u.n.o = r4.u.n.c;
				u.n.p = r4.u.n.d;
			}

			__device__ __host__ block( const int32_t v[16] )
			{
				u.n.a = v[0];
				u.n.b = v[1];
				u.n.c = v[2];
				u.n.d = v[3];

				u.n.e = v[4];
				u.n.f = v[5];
				u.n.g = v[6];
				u.n.h = v[7];

				u.n.i = v[8];
				u.n.j = v[9];
				u.n.k = v[10];
				u.n.l = v[11];

				u.n.m = v[12];
				u.n.n = v[13];
				u.n.o = v[14];
				u.n.p = v[15];
			}
		};

		struct macro_block
		{
			block a;
			block b;
			block c;
			block d;

			block e;
			block f;
			block g;
			block h;

			block i;
			block j;
			block k;
			block l;

			block m;
			block n;
			block o;
			block p;
		};

		enum mode
		{
			truncate = 0,
			round = 1
		};


		template <int32_t mode> 			
		__device__ __host__ pixel4 t2x2h( const pixel4 in )
		{
			auto a = in.u.n.a;
			auto b = in.u.n.b;
			auto c = in.u.n.c;
			auto d = in.u.n.d;

			a += d;
			b -= c;

			int32_t val_round = 0;

			if ( mode == round )
			{
				val_round = 1;
			}
			else if (mode == truncate)
			{
				val_round = 0;
			}

			auto val_t1 = ( ( a - b ) + val_round ) >> 1;
			auto val_t2 = c;
			
			c = val_t1 - d;
			d = val_t1 - val_t2;

			a -= d;
			b += c;

			return pixel4( a, b, c, d  );
		}

		namespace forward
		{	

			__device__ __host__ pixel4 todd( const pixel4 in )
			{
				auto a = in.u.n.a;
				auto b = in.u.n.b;
				auto c = in.u.n.c;
				auto d = in.u.n.d;

				b -= c;
				a += d;
				c += (b + 1) >> 1;
				d = ((a + 1) >> 1) - d;

				b -= (a * 3 + 4) >> 3;
				a += (b * 3 + 4) >> 3;
				d -= (c * 3 + 4) >> 3;
				c += (d * 3 + 4) >> 3;

				d += b >> 1;
				c -= (a + 1) >> 1;
				b -= d;
				a += c;

				return pixel4( a, b, c, d  );
			}

			__device__ __host__ pixel4 todd_odd( const pixel4 in )
			{
				auto a = in.u.n.a;
				auto b = in.u.n.b;
				auto c = in.u.n.c;
				auto d = in.u.n.d;

				b = -b;
				c = -c;

				d += a;
				c -= b;

				int32_t t1 = d >> 1;
				int32_t t2 = c >> 1;
				a -= t1;
				b += t2;
    
				a += (b * 3 + 4) >> 3;
				b -= (a * 3 + 3) >> 2;

				a += (b * 3 + 3) >> 3;
				b -= t2;

				a += t1;
				c += b;
				d -= a;

				return pixel4( a, b, c, d  );
			}

			__host__ __device__ pixel2 scale( const pixel2 in)
			{
				auto a = in.a;
				auto b = in.b;

				b -= (a * 3 + 0) >> 4;
				b -= a >> 7;
				b += a >> 10;
				a -= (b * 3 + 0) >> 3;
				b = (a >> 1) - b;
				a -= b;

				return pixel2( a, b );
			}

			__host__ __device__ pixel2 rotate( const pixel2 in)
			{
				auto a = in.a;
				auto b = in.b;

				b -= (a + 1) >> 1;
				a += (b + 1) >> 1;

				return pixel2( a, b );
			}

			__host__ __device__ block permute( const block in )
			{
				const int32_t fwd[16] = 
				{
					0,	8,	4,	6,
					2,	10,	14,	12,
					1,	11,	15,	13,
					9,	3,	7,	5
				};

				int32_t t[16];

				t[ fwd [ 0 ]  ] = in.u.p.v[0];
				t[ fwd [ 1 ]  ] = in.u.p.v[1];
				t[ fwd [ 2 ]  ] = in.u.p.v[2];
				t[ fwd [ 3 ]  ] = in.u.p.v[3];

				t[ fwd [ 4 ]  ] = in.u.p.v[4];
				t[ fwd [ 5 ]  ] = in.u.p.v[5];
				t[ fwd [ 6 ]  ] = in.u.p.v[6];
				t[ fwd [ 7 ]  ] = in.u.p.v[7];

				t[ fwd [ 8 ]  ] = in.u.p.v[8];
				t[ fwd [ 9 ]  ] = in.u.p.v[9];
				t[ fwd [ 10 ]  ] = in.u.p.v[10];
				t[ fwd [ 11 ]  ] = in.u.p.v[11];

				t[ fwd [ 12 ]  ] = in.u.p.v[12];
				t[ fwd [ 13 ]  ] = in.u.p.v[13];
				t[ fwd [ 14 ]  ] = in.u.p.v[14];
				t[ fwd [ 15 ]  ] = in.u.p.v[15];

				return block ( t );
			}

			__host__ __device__ block fct4x4 ( const block in )
			{
				auto r1 = t2x2h<truncate> ( pixel4( in.u.n.a, in.u.n.d, in.u.n.m, in.u.n.p  ) );
				auto r2 = t2x2h<truncate> ( pixel4( in.u.n.b, in.u.n.c, in.u.n.n, in.u.n.o  ) );
				auto r3 = t2x2h<truncate> ( pixel4( in.u.n.e, in.u.n.h, in.u.n.i, in.u.n.l  ) );
				auto r4 = t2x2h<truncate> ( pixel4( in.u.n.f, in.u.n.g, in.u.n.j, in.u.n.k  ) );

				auto r5 = t2x2h< round >	( pixel4( r1.u.n.a, r1.u.n.b, r2.u.n.a, r2.u.n.b ) );
				auto r6 = todd				( pixel4( r1.u.n.c, r1.u.n.d, r2.u.n.c, r2.u.n.d ) );
				auto r7 = todd				( pixel4( r3.u.n.a, r4.u.n.a, r3.u.n.b, r4.u.n.b ) );
				auto r8 = todd_odd			( pixel4( r3.u.n.c, r3.u.n.d, r4.u.n.c, r4.u.n.d ) );

				auto b  = block(r5, r6, r7, r8 );

				return permute( b );
			}
		}

		namespace inverse
		{
			__device__ __host__ pixel4 todd( const pixel4 in )
			{
				auto a = in.u.n.a;
				auto b = in.u.n.b;
				auto c = in.u.n.c;
				auto d = in.u.n.d;

				a -= c;
				b += d;
				c += (a + 1) >> 1;
				d -= b >> 1;

				c -= (d * 3 + 4) >> 3;
				d += (c * 3 + 4) >> 3;
				a -= (b * 3 + 4) >> 3;
				b += (a * 3 + 4) >> 3;
				
				d = ((a + 1) >> 1) - d;
				c -= (b + 1) >> 1;
				a -= d;
				b += c;

				return pixel4( a, b, c, d  );
			}

			__device__ __host__ pixel4 todd_odd( const pixel4 in )
			{
				auto a = in.u.n.a;
				auto b = in.u.n.b;
				auto c = in.u.n.c;
				auto d = in.u.n.d;


				d += a;
				c -= b;

				int32_t t1 = d >> 1;
				int32_t t2 = c >> 1;

				a -= t1;
				b += t2;

				a -= (b * 3 + 3) >> 3;
				b += (a * 3 + 3) >> 2;
				a -= (b * 3 + 4) >> 3;

				b -= t2;
				a += t1;

				c += b;
				d -= a;

				c = -c;
				b = -b;

				return pixel4( a, b, c, d  );
			}

			__host__ __device__ pixel2 scale( const pixel2 in)
			{

				auto a = in.a;
				auto b = in.b;

				a += b;
				b = (a >> 1) - b;
				a += (b * 3 + 0) >> 3;
				b -= a >> 10;
				b += a >> 7;
				b += (a * 3 + 0) >> 4;
				
				return pixel2( a, b );
			}

			__host__ __device__ pixel2 rotate( const pixel2 in)
			{
				auto a = in.a;
				auto b = in.b;

				a -= (b + 1) >> 1;
				b += (a + 1) >> 1;
			
				return pixel2( a, b );
			}

			__host__ __device__ block permute( const block in )
			{
				const int32_t inverse[16] = 
				{
					0,	8,	4,	13,
					2,	15,	3,	14,
					1,	12,	5,	9,
					7,	11,	6,	10
				};

				int32_t t[16];

				t[ inverse [ 0 ]  ] = in.u.p.v[0];
				t[ inverse [ 1 ]  ] = in.u.p.v[1];
				t[ inverse [ 2 ]  ] = in.u.p.v[2];
				t[ inverse [ 3 ]  ] = in.u.p.v[3];

				t[ inverse [ 4 ]  ] = in.u.p.v[4];
				t[ inverse [ 5 ]  ] = in.u.p.v[5];
				t[ inverse [ 6 ]  ] = in.u.p.v[6];
				t[ inverse [ 7 ]  ] = in.u.p.v[7];

				t[ inverse [ 8 ]  ] = in.u.p.v[8];
				t[ inverse [ 9 ]  ] = in.u.p.v[9];
				t[ inverse [ 10 ]  ] = in.u.p.v[10];
				t[ inverse [ 11 ]  ] = in.u.p.v[11];

				t[ inverse [ 12 ]  ] = in.u.p.v[12];
				t[ inverse [ 13 ]  ] = in.u.p.v[13];
				t[ inverse [ 14 ]  ] = in.u.p.v[14];
				t[ inverse [ 15 ]  ] = in.u.p.v[15];

				return block ( t );
			}

			/*
			0	1	2	3 
			4	5	6	7
			8	9	10	11
			12	13	14	15
			*/

			__host__ __device__ block fct4x4 ( const block in )
			{
				auto b = permute( in );

				auto r1 = t2x2h< round >	( pixel4( b.u.p.v[0], b.u.p.v[1], b.u.p.v[4], b.u.p.v[5] ) );
				auto r2 = todd				( pixel4( b.u.p.v[2], b.u.p.v[3], b.u.p.v[6], b.u.p.v[7] ) );
				auto r3 = todd				( pixel4( b.u.p.v[8], b.u.p.v[12], b.u.p.v[9], b.u.p.v[13] ) );
				auto r4 = todd_odd			( pixel4( b.u.p.v[10], b.u.p.v[11], b.u.p.v[14], b.u.p.v[15] ) );

				auto r5 = t2x2h<truncate> ( pixel4( r1.u.n.a, r1.u.n.d, r4.u.n.a, r4.u.n.d ) );
				auto r6 = t2x2h<truncate> ( pixel4( r2.u.n.b, r2.u.n.c, r3.u.n.b, r3.u.n.c ) );
				auto r7 = t2x2h<truncate> ( pixel4( r1.u.n.b, r1.u.n.c, r4.u.n.b, r4.u.n.c ) );
				auto r8 = t2x2h<truncate> ( pixel4( r2.u.n.a, r2.u.n.d, r3.u.n.a, r3.u.n.d ) );

				return block ( r5, r6, r7, r8 );
			}
		}
	}
}

namespace example
{
	void addWithCuda(int *c, const int *a, const int *b, unsigned int size);

}

int main()
{
	int32_t test [16] =
	{
		255, 1, 2, 3,
		4, 5, 6, 7,
		13, 9, 127, 11,
		12, 13, 14, 15
	};

	auto p1 = jpegxr::transforms::pixel4(0,1,2,3);
	
	auto p2 = jpegxr::transforms::forward::todd(p1);
	auto p3 = jpegxr::transforms::inverse::todd(p2);

	auto p4 = jpegxr::transforms::forward::todd_odd(p1);
	auto p5 = jpegxr::transforms::inverse::todd_odd(p4);

	auto b1 = jpegxr::transforms::block( test );
	auto b2 = jpegxr::transforms::forward::permute(b1);
	auto b3 = jpegxr::transforms::inverse::permute(b2);


	jpegxr::transforms::pixel2 p ( 5, 5 );
    auto r1 = jpegxr::transforms::forward::scale(p);

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    example::addWithCuda(c, a, b, arraySize);

    std::cout << "{1,2,3,4,5} + {10,20,30,40,50} = "<< std::endl << c[0] << c[1] << c[2] << c[3] << c[4];

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cuda::throw_if_failed<cuda::exception> ( cudaDeviceReset() );

    return 0;
}

static const uint32_t macro_block_in [] =
{
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11, 
    12, 13,14, 15
};



namespace example
{

	__global__ void addKernel(int *c, const int *a, const int *b)
	{
		int i = threadIdx.x;
		c[i] = a[i] + b[i];
	}

	// Helper function for using CUDA to add vectors in parallel.
	void addWithCuda(int *c, const int *a, const int *b, unsigned int size)
	{
		// Choose which GPU to run on, change this on a multi-GPU system.
		cuda::throw_if_failed<cuda::exception> (  cudaSetDevice(0) );

		// Allocate GPU buffers for three vectors (two input, one output)    .
		auto dev_a = std::make_shared< cuda::memory_buffer > ( size * sizeof( int )  );
		auto dev_b = std::make_shared< cuda::memory_buffer > ( size * sizeof( int )  );
		auto dev_c = std::make_shared< cuda::memory_buffer > ( size * sizeof( int )  );

		// Copy input vectors from host memory to GPU buffers.
		cuda::throw_if_failed<cuda::exception> ( cudaMemcpy(*dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice) );
		cuda::throw_if_failed<cuda::exception> ( cudaMemcpy(*dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice) );

		// Launch a kernel on the GPU with one thread for each element.
		addKernel<<<1, size>>>( *dev_c, *dev_a, *dev_b );

		// Check for any errors launching the kernel
		cuda::throw_if_failed<cuda::exception> ( cudaGetLastError() );
   
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cuda::throw_if_failed<cuda::exception> ( cudaDeviceSynchronize() );

		// Copy output vector from GPU buffer to host memory.
		cuda::throw_if_failed<cuda::exception> ( cudaMemcpy(c, dev_c->get(), size * sizeof(int), cudaMemcpyDeviceToHost) );
	}
}
