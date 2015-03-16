#ifndef PIXEL_SHADER
    #define PIXEL_SHADER
#endif

#include "samples_shaders.hlsl"

struct ps_input
{
    float4 position_ps     : SV_Position;
    uint   sample_class    : SAMPLE_CLASS;
};

Texture2D       image : register(t0);

SamplerState	default_sampler : register(s0);

float4  pixel_main(in  ps_input input) : sv_target
{
    const float4 colors[] =
    {
        float4(1.0f, 0.0f, 0.0f, 1.0f),
        float4(0.0f, 1.0f, 0.0f, 1.0f),
        float4(0.0f, 0.0f, 1.0f, 1.0f),
        float4(0.5f, 0.5f, 0.0f, 1.0f),
        float4(0.0f, 0.5f, 0.5f, 1.0f)
    };

    float4 c = image.Sample(default_sampler, float2(0.0, 0.0));

    float4 c2 = colors[input.sample_class];

    return c2;
}


