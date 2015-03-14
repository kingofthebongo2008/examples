#ifndef PIXEL_SHADER
    #define PIXEL_SHADER
#endif

#include "samples_shaders.hlsl"

struct ps_input
{
    float4 position_ps     : SV_Position;
    uint   sample_class    : SAMPLE_CLASS;
};

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

    return colors[input.sample_class];
}


