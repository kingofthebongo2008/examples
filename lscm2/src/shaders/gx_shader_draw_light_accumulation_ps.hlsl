#include "gx_shader_geometry_pass_common.hlsl"

typedef uint uint32_t;

cbuffer per_frame : register(SLOT_PER_FRAME)
{
    uint2    light_accumulation_buffer_dimensions;
}

struct vs_output
{
    float4	position_ps : sv_position;
    float2	uv          : texcoord; 
};

StructuredBuffer<uint2>     sampled_texture;


float4 decode_light ( uint2 r )
{
    uint2 a = ( r >> 16 );
    uint2 b =  r;
    return float4( f16tof32(a), f16tof32(b) );
}

uint linear_address(uint2 address)
{
    return address.y * light_accumulation_buffer_dimensions.x + address.x;
}


float4 main( in  vs_output input) : sv_target
{
    uint2 address = uint2 (input.uv * float2(light_accumulation_buffer_dimensions));

    uint2  samples  = sampled_texture[ linear_address ( address )  ];   
    uint2  t        = samples;
    float4 k        = decode_light(t);

    return k;
}
