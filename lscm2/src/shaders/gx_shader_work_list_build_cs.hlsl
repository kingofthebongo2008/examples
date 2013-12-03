#include "gx_shader_geometry_pass_common.hlsl"

typedef uint uint32_t;

static const uint32_t closure_count = 256;  //256 materials per frame maximum

static const uint32_t tile_size_x = 16;
static const uint32_t tile_size_y = 8;
static const uint32_t sample_count = 8;

struct draw_instance_info
{
    uint32_t m_closure_id;      //shader 
    uint32_t m_vertex_offset;   //offset into a big vertex buffer with geometry
    uint32_t m_index_offset;    //offset into a big index buffer with geometry
};

RWStructuredBuffer<draw_instance_info>   instance_info;
Texture2DMS<uint, sample_count>          visibility_buffer;

groupshared uint closures[ closure_count ];

cbuffer per_frame : register(SLOT_PER_FRAME)
{
    uint2    light_accumulation_buffer_dimensions;
}

uint linear_address(uint2 address)
{
    return address.y * light_accumulation_buffer_dimensions.x + address.x;
}

[numthreads(tile_size_x, tile_size_y, 1)]
void main( uint3 group_thread_id : SV_GroupThreadID , uint3 dispatch_thread_id : SV_DispatchThreadID ) 
{
    const uint group_index = group_thread_id.x + group_thread_id.y * tile_size_x;

    if ( group_index == 0 )
    { 
        for (int i = 0; i < closure_count; ++i)
        {
            closures[i] = 0;
        }
    }

    //GroupMemoryBarrierWithSync();

    const uint32_t instance_id_bits = 10;
    const uint32_t mask = ((1 << instance_id_bits) - 1);

    float4 light = float4(0.0, 0.0f, 0.0f, 0.0f);

    for (uint32_t i = 0; i < sample_count; ++i)
    {
        uint    s = visibility_buffer.sample[i][dispatch_thread_id.xy];
        uint    instance_id = s & mask;

        //fetch draw call info
        draw_instance_info info = instance_info[instance_id];

        //mark the shader as used
        closures[ info.m_closure_id ] = 1;
    }
}