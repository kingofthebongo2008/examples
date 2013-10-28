#include "gx_shader_geometry_pass_common.hlsl"

typedef uint uint32_t;

RWStructuredBuffer<uint2>   light_buffer;

cbuffer per_frame : register(SLOT_PER_FRAME)
{
    uint2    light_accumulation_buffer_dimensions;
}

uint linear_address(uint2 address)
{
    return address.y * light_accumulation_buffer_dimensions.x + address.x;
}

[numthreads(1, 1, 1)]
void main( uint3 dispatch_thread_id : SV_DispatchThreadID ) 
{
    uint address = linear_address(dispatch_thread_id.xy);
    light_buffer[address] = uint2(0,0);
}