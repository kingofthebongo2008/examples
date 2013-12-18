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
        typedef int16_t pixel;

		enum mode : uint32_t
		{
			truncate = 0,
			round = 1
		};

        enum indexer : uint32_t
        {
            indexer_a = 0,      indexer_b = 1,    indexer_c = 2,      indexer_d = 3,
            indexer_e = 4,      indexer_f = 5,    indexer_g = 6,      indexer_h = 7,
            indexer_i = 8,      indexer_j = 9,    indexer_k = 10,     indexer_l = 11,
            indexer_m = 12,     indexer_n = 13,   indexer_o = 14,     indexer_p = 15
        };

		template <int32_t mode> 			
		__host__ __device__ inline void t2x2h( pixel* __restrict a, pixel* __restrict b, pixel* __restrict c, pixel* __restrict d)
		{
			*a += *d;
			*b -= *c;

			int32_t val_round = 0;

			if ( mode == round )
			{
				val_round = 1;
			}
			else if (mode == truncate)
			{
				val_round = 0;
			}

			auto val_t1 = ( ( *a - *b ) + val_round ) >> 1;
			auto val_t2 = *c;
			
			*c = val_t1 - *d;
			*d = val_t1 - val_t2;

			*a -= *d;
			*b += *c;
		}

        //analysis stage
        __host__ __device__ void t2x2h_pre( pixel* __restrict a, pixel* __restrict b, pixel* __restrict c, pixel* __restrict d )
		{
            *a += *d;
            *b -= *c;

            auto t1 = *d;
            auto t2 = *c;

            *c = ((*a - *b) >> 1) - t1;
            *d = t2 + (*b >> 1);
            *b += *c;
            *a -= (*d * 3 + 4) >> 3;

        }

        //synthesis stage
        __host__ __device__ void t2x2h_post( pixel* __restrict a, pixel* __restrict b, pixel* __restrict c, pixel* __restrict d )
		{
            *a += (*d * 3 + 4) >> 3;
            *b -= *c;

            *d -= *b >> 1;

            auto t1 = ((*a - *b) >> 1) - *c;

            *c = *d;
            *d = t1;
            *a -= *d;
            *b += *c;

        }

		namespace analysis
		{	
			__host__ __device__ inline void scale( pixel* __restrict a, pixel* __restrict b )
			{
				*b -= (*a * 3 + 0) >> 4;
				*b -= *a >> 7;
				*b += *a >> 10;
				*a -= (*b * 3 + 0) >> 3;
				*b = (*a >> 1) - *b;
				*a -= *b;
			}

			__host__ __device__ inline void rotate( pixel* __restrict a, pixel* __restrict b )
			{
				*b -= (*a + 1) >> 1;
				*a += (*b + 1) >> 1;
			}

			__host__ __device__ inline void todd( pixel* __restrict a, pixel* __restrict b, pixel* __restrict c, pixel* __restrict d )
			{
				*b -= *c;
				*a += *d;
				*c += (*b + 1) >> 1;
				*d = ((*a + 1) >> 1) - *d;

				*b -= (*a * 3 + 4) >> 3;
				*a += (*b * 3 + 4) >> 3;
				*d -= (*c * 3 + 4) >> 3;
				*c += (*d * 3 + 4) >> 3;

				*d += *b >> 1;
				*c -= (*a + 1) >> 1;
				*b -= *d;
				*a += *c;
			}

			__host__ __device__ inline void todd_odd( pixel* __restrict a, pixel* __restrict b, pixel* __restrict c, pixel* __restrict d )
			{
				*b = -*b;
				*c = -*c;

				*d += *a;
				*c -= *b;

				auto t1 = *d >> 1;
				auto t2 = *c >> 1;
				*a -= t1;
				*b += t2;
    
				*a += (*b * 3 + 4) >> 3;
				*b -= (*a * 3 + 3) >> 2;

				*a += (*b * 3 + 3) >> 3;
				*b -= t2;

				*a += t1;
				*c += *b;
				*d -= *a;
			}

            __host__ __device__ inline void prefilter_todd_odd( pixel* __restrict a, pixel* __restrict b, pixel* __restrict c, pixel* __restrict d )
			{
                *d += *a;
                *c -= *b;
                auto t1 = *d >> 1;
                auto t2 = *c >> 1;
                *a -= t1;
                *b += t2;
                
                *a += (*b * 3 + 4) >> 3;
                *b -= (*a * 3 + 2) >> 2;
                
                *a += (*b * 3 + 6) >> 3;
                *b -= t2;
                
                *a += t1;
                *c += *b;
                *d -= *a;
            }

			__host__ __device__ inline void permute( pixel* in )
			{
				const int32_t fwd[16] = 
				{
					0,	8,	4,	6,
					2,	10,	14,	12,
					1,	11,	15,	13,
					9,	3,	7,	5
				};

				int32_t t[16];

				t[ fwd [ 0 ]  ] = in[0];
				t[ fwd [ 1 ]  ] = in[1];
				t[ fwd [ 2 ]  ] = in[2];
				t[ fwd [ 3 ]  ] = in[3];

				t[ fwd [ 4 ]  ] = in[4];
				t[ fwd [ 5 ]  ] = in[5];
				t[ fwd [ 6 ]  ] = in[6];
				t[ fwd [ 7 ]  ] = in[7];

				t[ fwd [ 8 ]  ] = in[8];
				t[ fwd [ 9 ]  ] = in[9];
				t[ fwd [ 10 ]  ] = in[10];
				t[ fwd [ 11 ]  ] = in[11];

				t[ fwd [ 12 ]  ] = in[12];
				t[ fwd [ 13 ]  ] = in[13];
				t[ fwd [ 14 ]  ] = in[14];
				t[ fwd [ 15 ]  ] = in[15];

                in[0] = t[0];
                in[1] = t[1];
                in[2] = t[2];
                in[3] = t[3];

                in[4] = t[4];
                in[5] = t[5];
                in[6] = t[6];
                in[7] = t[7];

                in[8] = t[8];
                in[9] = t[9];
                in[10] = t[10];
                in[11] = t[11];

                in[12] = t[12];
                in[13] = t[13];
                in[14] = t[14];
                in[15] = t[15];
			}

            __host__ __device__ inline void prefilter2( pixel* __restrict a, pixel* __restrict b  )
            {
                *b -= ((*a + 2) >> 2);
                *a -= (*b >> 13);
            
                *a -= (*b >> 9);
            
                *a -= (*b >> 5);
            
                *a -= ((*b + 1) >> 1);
                *b -= ((*a + 2) >> 2);
            }

            __host__ __device__ inline void prefilter2x2( pixel* __restrict a, pixel* __restrict b, pixel* __restrict c, pixel* __restrict d )
			{
                *a += *d;
                *b += *c;
                *d -= ((*a + 1) >> 1);
                *c -= ((*b + 1) >> 1);

                *b -= ((*a + 2) >> 2);
                *a -= (*b >> 5);

                *a -= (*b >> 9);

                *a -= (*b >> 13);

                *a -= ((*b + 1) >> 1);
                *b -= ((*a + 2) >> 2);
                *d += ((*a + 1) >> 1);
                *c += ((*b + 1) >> 1);

                *a -= *d;
                *b -= *c;
            }

            __host__ __device__ inline void prefilter4( pixel* __restrict a, pixel* __restrict b, pixel* __restrict c, pixel* __restrict d )
			{
                *a += *d;
                *b += *c;
                *d -= ((*a + 1) >> 1);
                *c -= ((*b + 1) >> 1);
                
                rotate(c, d );

                *d *= -1;
                *c *= -1;
                *a -= *d;
                *b -= *c;
                
                *d += (*a >> 1);
                *c += (*b >> 1);
                *a -= ((*d * 3 + 4) >> 3);
                *b -= ((*c * 3 + 4) >> 3);
                
                scale( a, d );
                scale( b, c );

                *d += ((*a + 1) >> 1);
                *c += ((*b + 1) >> 1);
                *a -= *d;
                *b -= *c;
            }

            /*
                a b c d 
                e f g h
                i j k l
                m n o p
            */

            __host__ __device__ inline void prefilter4x4
                ( 
                       pixel * __restrict a, pixel * __restrict b, pixel * __restrict c, pixel * __restrict d,
                       pixel * __restrict e, pixel * __restrict f, pixel * __restrict g, pixel * __restrict h,
                       pixel * __restrict i, pixel * __restrict j, pixel * __restrict k, pixel * __restrict l,
                       pixel * __restrict m, pixel * __restrict n, pixel * __restrict o, pixel * __restrict p                
                )
			{
                t2x2h_pre ( a, d, m, p);
                t2x2h_pre ( b, c, n, o);
                t2x2h_pre ( e, h, i, l);
                t2x2h_pre ( f, g, j, k);

                scale ( a, p );
                scale ( b, o );
                scale ( e, l );
                scale ( f, k );
                
                rotate ( n, m );
                rotate ( j, i );
                rotate ( h, d );
                rotate ( g, c );

                prefilter_todd_odd( k, l, o, p);

                t2x2h<truncate> ( a, m, d, p );
                t2x2h<truncate> ( b, c, n, o );
                t2x2h<truncate> ( e, h, i, l );
                t2x2h<truncate> ( f, g, j, k );
            }

            __host__ __device__ inline void prefilter4x4 ( pixel* pixels)
            {
                    prefilter4x4
                    ( 
                        pixels + 0,     pixels + 1,     pixels + 2,     pixels + 3,
                        pixels + 4,     pixels + 5,     pixels + 6,     pixels + 7,
                        pixels + 8,     pixels + 9,     pixels + 10,    pixels + 11,
                        pixels + 12,    pixels + 13,    pixels + 14,    pixels + 15
                    );
            }

            /*
                a b c d     0   1   2   3
                e f g h     4   5   6   7
                i j k l     8   9   10  11  
                m n o p     12  13  14  15
            */
			__host__ __device__ inline void pct4x4 ( pixel* in )
			{
                t2x2h<truncate> ( in + indexer_a, in + indexer_d, in + indexer_m, in + indexer_p  );
                t2x2h<truncate> ( in + indexer_f, in + indexer_g, in + indexer_j, in + indexer_k  );
                t2x2h<truncate> ( in + indexer_b, in + indexer_c, in + indexer_n, in + indexer_o  );
                t2x2h<truncate> ( in + indexer_e, in + indexer_h, in + indexer_i, in + indexer_l  );

				t2x2h< round >	( in + indexer_a, in + indexer_b, in + indexer_e, in + indexer_f  );
				todd			( in + indexer_c, in + indexer_d, in + indexer_g, in + indexer_h  );
				todd			( in + indexer_i, in + indexer_m, in + indexer_j, in + indexer_n  );
				todd_odd		( in + indexer_k, in + indexer_l, in + indexer_o, in + indexer_p  );

				permute( in );
			}

            __host__ __device__ inline void pct2x2 ( pixel* __restrict a, pixel* __restrict b, pixel* __restrict c, pixel* __restrict d )
            {
                return t2x2h<truncate>(a, b, c, d);
            }

            __host__ __device__ inline void pt2 ( pixel* __restrict a, pixel* __restrict b )
            {
                *b -= *a;
                *a += (*b + 1) >> 1;
            }
        }

		namespace synthesis
		{
			__host__ __device__ inline void todd( pixel* __restrict a, pixel* __restrict b, pixel* __restrict c, pixel* __restrict d )
			{
				*a -= *c;
				*b += *d;
				*c += (*a + 1) >> 1;
				*d -= *b >> 1;

				*c -= (*d * 3 + 4) >> 3;
				*d += (*c * 3 + 4) >> 3;
				*a -= (*b * 3 + 4) >> 3;
				*b += (*a * 3 + 4) >> 3;
				
				*d = ((*a + 1) >> 1) - *d;
				*c -= (*b + 1) >> 1;
				*a -= *d;
				*b += *c;
			}

			__host__ __device__ inline void todd_odd( pixel* __restrict a, pixel* __restrict b, pixel* __restrict c, pixel* __restrict d )
			{
				*d += *a;
				*c -= *b;

				auto t1 = *d >> 1;
				auto t2 = *c >> 1;

				*a -= t1;
				*b += t2;

				*a -= (*b * 3 + 3) >> 3;
				*b += (*a * 3 + 3) >> 2;
				*a -= (*b * 3 + 4) >> 3;

				*b -= t2;
				*a += t1;

				*c += *b;
				*d -= *a;

				*c = -*c;
				*b = -*b;
			}

            __host__ __device__ inline void overlap_todd_odd( pixel* __restrict a, pixel* __restrict b, pixel* __restrict c, pixel* __restrict d )
			{
                *d += *a;
                *c -= *b;
                auto t1 = *d >> 1;
                auto t2 = *c >> 1;
                *a -= t1;
                *b += t2;
                
                *a -= (*b * 3 + 6) >> 3;
                *b += (*a * 3 + 2) >> 2;
                *a -= (*b * 3 + 4) >> 3;
                

                *b -= t2;
                
                *a += t1;
                *c += *b;
                *d -= *a;
            }

			__host__ __device__ inline void scale( pixel* __restrict a, pixel* __restrict b  )
			{
				*a += *b;
				*b = (*a >> 1) - *b;
				*a += (*b * 3 + 0) >> 3;
				*b -= *a >> 10;
				*b += *a >> 7;
				*b += (*a * 3 + 0) >> 4;
			}

			__host__ __device__ inline void rotate( pixel* __restrict a, pixel* __restrict b  )
			{
				*a -= (*b + 1) >> 1;
				*b += (*a + 1) >> 1;
			}

			__host__ __device__ inline void permute( pixel* in )
			{
				const int32_t inverse[16] = 
				{
					0,	8,	4,	13,
					2,	15,	3,	14,
					1,	12,	5,	9,
					7,	11,	6,	10
				};

				int32_t t[16];

				t[ inverse [ 0 ]  ] = in[0];
				t[ inverse [ 1 ]  ] = in[1];
				t[ inverse [ 2 ]  ] = in[2];
				t[ inverse [ 3 ]  ] = in[3];

				t[ inverse [ 4 ]  ] = in[4];
				t[ inverse [ 5 ]  ] = in[5];
				t[ inverse [ 6 ]  ] = in[6];
				t[ inverse [ 7 ]  ] = in[7];

				t[ inverse [ 8 ]  ] = in[8];
				t[ inverse [ 9 ]  ] = in[9];
				t[ inverse [ 10 ]  ] = in[10];
				t[ inverse [ 11 ]  ] = in[11];

				t[ inverse [ 12 ]  ] = in[12];
				t[ inverse [ 13 ]  ] = in[13];
				t[ inverse [ 14 ]  ] = in[14];
				t[ inverse [ 15 ]  ] = in[15];

                in[0] = t[0];
                in[1] = t[1];
                in[2] = t[2];
                in[3] = t[3];

                in[4] = t[4];
                in[5] = t[5];
                in[6] = t[6];
                in[7] = t[7];

                in[8] = t[8];
                in[9] = t[9];
                in[10] = t[10];
                in[11] = t[11];

                in[12] = t[12];
                in[13] = t[13];
                in[14] = t[14];
                in[15] = t[15];

			}


            /*
                a b c d     0   1   2   3
                e f g h     4   5   6   7
                i j k l     8   9   10  11  
                m n o p     12  13  14  15
            */

			__host__ __device__ inline void pct4x4 ( pixel* in )
			{
				permute( in );

				t2x2h< round >	( in + indexer_a, in + indexer_b, in + indexer_e, in + indexer_f  );
				todd			( in + indexer_c, in + indexer_d, in + indexer_g, in + indexer_h  );
				todd			( in + indexer_i, in + indexer_m, in + indexer_j, in + indexer_n  );
				todd_odd		( in + indexer_k, in + indexer_l, in + indexer_o, in + indexer_p  );

				t2x2h<truncate> ( in + indexer_a, in + indexer_d, in + indexer_m, in + indexer_p  );
                t2x2h<truncate> ( in + indexer_f, in + indexer_g, in + indexer_j, in + indexer_k  );
                t2x2h<truncate> ( in + indexer_b, in + indexer_c, in + indexer_n, in + indexer_o  );
                t2x2h<truncate> ( in + indexer_e, in + indexer_h, in + indexer_i, in + indexer_l  );                
			}

            __host__ __device__ inline void pct2x2 ( pixel* __restrict a, pixel* __restrict b, pixel* __restrict c, pixel* __restrict d )
            {
                return t2x2h<truncate>(a, b, c, d);
            }

            __host__ __device__ inline void pt2 ( pixel* __restrict a, pixel* __restrict b  )
            {
                *a -= (*b + 1) >> 1;
                *b += *a;
            }

            __host__ __device__ inline void overlapfilter2 ( pixel* __restrict a, pixel* __restrict b  )
            {
                *b += ((*a + 2) >> 2);
                *a += ((*b + 1) >> 1);
                *a += (*b >> 5);
                *a += (*b >> 9);

                *a += (*b >> 13);
                *b += ((*a + 2) >> 2);
            }

            __host__ __device__ inline void overlapfilter2x2( pixel* __restrict a, pixel* __restrict b, pixel* __restrict c, pixel* __restrict d )
			{
                *a += *d;
                *b += *c;
                *d -= (*a + 1) >> 1;
                *c -= (*b + 1) >> 1;

                *b += (*a + 2) >> 2;
                *a += (*b + 1) >> 1;

                *a += (*b >> 5);
                *a += (*b >> 9);
                *a += (*b >> 13);

                *b += (*a + 2) >> 2;

                *d += (*a + 1) >> 1;
                *c += (*b + 1) >> 1;
                *a -= *d;

                *b -= *c;
            }

            __host__ __device__ inline void overlapfilter4( pixel* __restrict a, pixel* __restrict b, pixel* __restrict c, pixel* __restrict d )
			{
                *a += *d;
                *b += *c;
                *d -= ((*a + 1) >> 1);
                *c -= ((*b + 1) >> 1);

                scale( a, d );
                scale( b, c );

                *a += ((*d * 3 + 4) >> 3);
                *b += ((*c * 3 + 4) >> 3);
                *d -= (*a >> 1);
                *c -= (*b >> 1);

                *a += *d;
                *b += *c;
                *d *= -1;
                *c *= -1;

                rotate( c, d );

                *d += ((*a + 1) >> 1);
                *c += ((*b + 1) >> 1);
                *a -= *d;
                *b -= *c;
            }

            __host__ __device__ inline void overlapfilter4x4
                ( 
                       pixel * __restrict a, pixel * __restrict b, pixel * __restrict c, pixel * __restrict d,
                       pixel * __restrict e, pixel * __restrict f, pixel * __restrict g, pixel * __restrict h,
                       pixel * __restrict i, pixel * __restrict j, pixel * __restrict k, pixel * __restrict l,
                       pixel * __restrict m, pixel * __restrict n, pixel * __restrict o, pixel * __restrict p                
                )
			{
                t2x2h<truncate> ( a, m, d, p );
                t2x2h<truncate> ( b, c, n, o );
                t2x2h<truncate> ( e, h, i, l );
                t2x2h<truncate> ( f, g, j, k );

                overlap_todd_odd ( k, l, o, p );

                rotate ( n, m );
                rotate ( j, i );
                rotate ( h, d );
                rotate ( g, c );

                scale ( a, p );
                scale ( b, o );
                scale ( e, l );
                scale ( f, k );

                jpegxr::transforms::t2x2h_post( a, m, d, p );
                jpegxr::transforms::t2x2h_post( b, c, n, o );
                jpegxr::transforms::t2x2h_post( e, h, i, l );
                jpegxr::transforms::t2x2h_post( f, g, j, k );
            }

            __host__ __device__ inline void overlapfilter4x4 (  pixel* pixels )
            {
                overlapfilter4x4
                    ( 
                        pixels + 0,     pixels + 1,     pixels + 2,     pixels + 3,
                        pixels + 4,     pixels + 5,     pixels + 6,     pixels + 7,
                        pixels + 8,     pixels + 9,     pixels + 10,    pixels + 11,
                        pixels + 12,    pixels + 13,    pixels + 14,    pixels + 15
                    );
            }
		}
	}
}

namespace example
{
	void addWithCuda(int32_t * c, const int32_t * a, const int32_t * b, uint32_t size);

}

int32_t main()
{
	jpegxr::transforms::pixel test [16] =
	{
		0, 0, 0, 0,
		1, 1, 1, 1,
		1, 5, 1, 1,
		1, 1, 1, 1
	};

    
    jpegxr::transforms::analysis::pct4x4(test);
    jpegxr::transforms::synthesis::pct4x4(test);

    jpegxr::transforms::pixel test1 [4] =
	{
		0, 1, 2, 3
	};

    jpegxr::transforms::analysis::prefilter4(test1 + 0, test1 + 1, test1 + 2, test1 + 3);
    jpegxr::transforms::synthesis::overlapfilter4(test1 + 0, test1 + 1, test1 + 2, test1 + 3);

    jpegxr::transforms::analysis::prefilter2x2(test1 + 0, test1 + 1, test1 + 2, test1 + 3);
    jpegxr::transforms::synthesis::overlapfilter2x2(test1 + 0, test1 + 1, test1 + 2, test1 + 3);

    jpegxr::transforms::analysis::prefilter4x4( test );
    jpegxr::transforms::synthesis::overlapfilter4x4(test );
    

    const int32_t arraySize = 5;
    const int32_t a[arraySize] = { 1, 2, 3, 4, 5 };
    const int32_t b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    example::addWithCuda(c, a, b, arraySize);

    std::cout << "{1,2,3,4,5} + {10,20,30,40,50} = "<< std::endl << c[0] << c[1] << c[2] << c[3] << c[4];

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cuda::throw_if_failed<cuda::exception> ( cudaDeviceReset() );

    return 0;
}

namespace example
{

	__global__ void addKernel(int32_t * c, const int32_t * a, const int32_t * b)
	{
		int i = threadIdx.x;

        jpegxr::transforms::pixel v[16] =
        { 
            a[0], a[1], a[2], a[3],
            a[4], a[5], a[6], a[7],
            a[8], a[9], a[10], a[11],
            a[12], a[13], a[14], a[15]
        };

        //jpegxr::transforms::forward::pct4x4(v);

        jpegxr::transforms::analysis::prefilter4x4
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
	void addWithCuda(int *c, const int *a, const int *b, unsigned int size)
	{
		// Choose which GPU to run on, change this on a multi-GPU system.
		cuda::throw_if_failed<cuda::exception> (  cudaSetDevice(0) );

		// Allocate GPU buffers for three vectors (two input, one output)    .
		auto dev_a = std::make_shared< cuda::memory_buffer > ( size * sizeof( int32_t )  );
		auto dev_b = std::make_shared< cuda::memory_buffer > ( size * sizeof( int32_t )  );
		auto dev_c = std::make_shared< cuda::memory_buffer > ( size * sizeof( int32_t )  );

		// Copy input vectors from host memory to GPU buffers.
		cuda::throw_if_failed<cuda::exception> ( cudaMemcpy(*dev_a, a, size * sizeof(int32_t), cudaMemcpyHostToDevice) );
		cuda::throw_if_failed<cuda::exception> ( cudaMemcpy(*dev_b, b, size * sizeof(int32_t), cudaMemcpyHostToDevice) );

		// Launch a kernel on the GPU with one thread for each element.
		addKernel<<<1, size>>>( *dev_c, *dev_a, *dev_b );

		// Check for any errors launching the kernel
		cuda::throw_if_failed<cuda::exception> ( cudaGetLastError() );
   
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cuda::throw_if_failed<cuda::exception> ( cudaDeviceSynchronize() );

		// Copy output vector from GPU buffer to host memory.
		cuda::throw_if_failed<cuda::exception> ( cudaMemcpy(c, dev_c->get(), size * sizeof(int32_t), cudaMemcpyDeviceToHost) );
	}
}
