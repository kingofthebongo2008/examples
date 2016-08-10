#include "mip_map_utils.h"

struct interpolants
{
    float4 position     : SV_POSITION0;
    float2 uv           : texcoord0;
};

SamplerState g_sampler
{
    Filter = MIN_MAG_MIP_POINT;
    AddressU = Clamp;
    AddressV = Clamp;
};

Texture2D         g_texture : register(t0);

float4 main(interpolants input) : SV_TARGET
{
    float max_lod       = mip_map_level( input.uv );
    uint int_max_lod    = (uint) min(max_lod, 7); //7 is 128x128 texels tile

    return g_texture.SampleLevel(g_sampler, input.uv, int_max_lod);
    //return g_texture.Sample(g_sampler, input.uv);//, (uint)max_lod);
}