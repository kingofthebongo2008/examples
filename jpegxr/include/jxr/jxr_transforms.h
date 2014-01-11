#ifndef __jxr_transforms_h__
#define __jxr_transforms_h__

#include <cstdint>

namespace jpegxr
{
    namespace transforms
    {
        typedef int32_t pixel;

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

        enum scale : uint32_t
        {
            no_scale   = 0,
            scaled     = 3
        };

        enum bias : uint32_t
        {
            no_bias = 0,
            bd5 = (1 << 4),
            bd565 = (1 << 5),
            bd8 = (1 << 7),
            bd10 = (1 << 9 ),
            bd16 = (1 << 15)
        };

        template <int32_t scale, int32_t bias>
        __host__ __device__ inline void scale_bias_bd8_analysis ( pixel* __restrict a, pixel* __restrict b, pixel* __restrict c )
        {
            *a =  ( *a -  bias) << scale;
            *b =  ( *b -  bias) << scale;
            *c =  ( *c -  bias) << scale;
        }

        template <int32_t scale, int32_t bias>
        __host__ __device__ inline void scale_bias_bd8_synthesis ( pixel* __restrict a, pixel* __restrict b, pixel* __restrict c )
        {
            auto round = 0;
            auto shift_bits = 0;

            *a =  ( *a + ( ( bias >> shift_bits ) << scale ) + round ) >> scale;
            *b =  ( *b + ( ( bias >> shift_bits ) << scale ) + round ) >> scale;
            *c =  ( *c + ( ( bias >> shift_bits ) << scale ) + round ) >> scale;
        }

        __host__ __device__ inline void rgb_2_ycocg(pixel* __restrict r_y, pixel* __restrict  g_co, pixel* __restrict b_cg )
        {
            auto co = *r_y - *b_cg;
            auto t = *b_cg + (co >> 1);

            auto cg = *g_co - t;

            auto y = t + (cg >> 1);

            *r_y  = y;
            *g_co = co;
            *b_cg = cg;
        }

        __host__ __device__ inline void ycocg_2_rgb(pixel* __restrict r_y, pixel* __restrict  g_co, pixel* __restrict b_cg )
        {
            auto t = *r_y - (*b_cg >> 1);
            auto g = *b_cg + t;
            auto b = t - (*g_co >> 1);
            auto r = b + *g_co;

            *r_y = r;
            *g_co = g;
            *b_cg = b;
        }

        __host__ __device__ inline void rgb_2_yuv(pixel* __restrict r_y, pixel* __restrict  g_u, pixel* __restrict b_v )
        {
          auto v = *b_v - *r_y;
          auto t = *r_y - *g_u +  ( ( v + 1 ) >> 1 );
          auto y = *g_u + ( t >> 1 );

          auto u = -t;

          *r_y = y;
          *g_u = u;
          *b_v = v;
        }

        __host__ __device__ inline void yuv_2_rgb(pixel* __restrict r_y, pixel* __restrict  g_u, pixel* __restrict b_v )
        {
          auto t = -*g_u;

          auto g = *r_y  - ( t >> 1);
          auto r = t + g - ( ( *b_v + 1 ) >> 1 ) ;
          auto b = *b_v + r;

          *r_y = r;
          *g_u = g;
          *b_v = b;
        }

        
        __host__ __device__ inline void gamma( pixel* __restrict r, pixel* __restrict  g, pixel* __restrict b )
        {
            //todo
        }

        __host__ __device__ inline void degamma( pixel* __restrict r, pixel* __restrict  g, pixel* __restrict b )
        {
            //todo
        }

        __host__ __device__ inline void chroma_subsample( pixel* __restrict r, pixel* __restrict  g, pixel* __restrict b )
        {
            //todo
        }
    }
}

#endif

