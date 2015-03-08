#ifndef PIXEL_SHADER
    #define PIXEL_SHADER
#endif

#include "samples_shaders.hlsl"

struct ps_input
{
    float4 position_ps : sv_position;
};

float4  pixel_main(in  ps_input input) : sv_target
{
    return float4 (1.0f, 0.0f, 0.0f, 1.0f);
}


