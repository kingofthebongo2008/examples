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

uint4 main(interpolants input) : SV_TARGET
{
    float2 scale        = float2( 200.0f / 1600.0f, 113.0f / 900.0f);
    float max_lod       = mip_map_level(input.uv, scale);
    uint int_max_lod    = (uint) max( 0, min(max_lod, 7)); //7 is 128x128 texels tile
    uint2 tile          = (uint2) floor( input.uv * 128 / exp2(int_max_lod) );



    return uint4( (uint) tile.x, (uint) tile.y, int_max_lod, 1 );

}