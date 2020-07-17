#ifndef __svd_hlsl_math_h__
#define __svd_hlsl_math_h__

#ifdef __cplusplus
#include <cmath>
#pragma warning( disable : 4244 )
#endif

namespace svdhlslcpp
{
    static const float four_gamma_squared 	    = sqrt(8.0f) + 3.0f;
    static const float sine_pi_over_eight 	    = .5f*sqrt(2.0f - sqrt(2.0f));
    static const float cosine_pi_over_eight     = .5f*sqrt(2. + sqrt(2.0f));
    static const float tiny_number              =  1.e-20f;
    static const float small_number             =  1.e-12f;


    inline float add(float a, float b);
    inline float sub(float a, float b);
    inline float div(float a, float b);
    inline float mul(float a, float b);
    inline float madd(float a, float b, float c);
    inline float max(float  a, float b);

    inline float rsqrt(float a);

    inline float cmp_ge(float a, float b);
    inline float cmp_le(float a, float b);
    inline float blend(float  a, float b, float mask);

    inline float zero();
    inline float one();

    inline float splat(float f);


    inline float dot3(float a1, float a2, float a3, float b1, float b2, float b3)
    {
        return a1 * b1 + a2 * b2 + a3 * b3;
    }

    inline float splat(float f)
    {
        return f;
    }

    inline float zero()
    {
        return 0.0f;
    }

    inline float one()
    {
        return 1.0f;
    }

    float add(float a, float b)
    {
        float r;
        r = a + b;
        return r;
    }

    inline float sub(float a, float b)
    {
        float r;
        r = a - b;
        return r;
    }

    inline float mul(float a, float b)
    {
        float r;
        r = a * b;
        return r;
    }

    inline float div(float a, float b)
    {
        float r;
        r = a / b;
        return r;
    }

    inline float madd(float a, float b, float c)
    {
        float r;
        r = a * b + c;
        return r;
    }

    inline float max(float a, float b)
    {
        float r;
        r = a > b ? a : b;
        return r;
    }

    inline float rsqrt(float a)
    {
        return 1.0f / sqrt(a);
    }

    inline float cmp_ge(float a, float b)
    {
        return a < b ? 0.0f : 1.0f;
    }

    inline float cmp_le(float a, float b)
    {
        return a > b ? 0.0f : 1.0f;
    }

    // r = (mask == 0) ? a : b;
    inline float blend(float a, float b, float mask)
    {
        return a * (1 - mask) + b * mask;
    }
}

#if defined(__cplusplus)
#pragma warning( default : 4244 )
#endif
#endif
