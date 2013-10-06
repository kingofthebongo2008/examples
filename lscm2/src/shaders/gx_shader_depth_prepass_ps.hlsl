#include "gx_shader_geometry_pass_common.hlsl"

typedef uint uint32_t;

cbuffer per_object : register(SLOT_PER_DRAW_CALL)
{
    uint32_t    instance_id;    //number of draw call, tree, house, character, etc.
}

float4 main( in uint32_t primitive_id : SV_PrimitiveID) : sv_target
{
    const uint32_t instance_id_bits = 10;
    uint32_t visibility = (primitive_id << instance_id_bits) | instance_id;
    return  float4( 1.0f, 0.0f, 0.0f, 1.0f );//( 0xFF000000 );
}