#ifndef __jxr_synthesis_h__
#define __jxr_synthesis_h__

#include <jxr/jxr_transforms.h>

namespace jpegxr
{
    namespace transforms
    {
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

#endif
