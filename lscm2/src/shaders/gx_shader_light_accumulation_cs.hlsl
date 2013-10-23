#include "gx_shader_geometry_pass_common.hlsl"

typedef uint uint32_t;

static const uint32_t sample_count = 8;

Texture2DMS<uint,       sample_count>         visibility_buffer;
RWStructuredBuffer<uint2>                     light_buffer;

cbuffer per_frame : register(SLOT_PER_FRAME)
{
    uint2    light_accumulation_buffer_dimensions;
}

float4 decode_light ( uint2 r )
{
    uint2 a = ( r >> 16 );
    uint2 b =  r;
    return float4( f16tof32(a), f16tof32(b) );
}

uint2 encode_light ( float4 r )
{
    uint2 a = f32tof16(r.xy);
    uint2 b = f32tof16(r.zw);
    return (a << 16) | b;
}

void write_light ( uint index, float4 r )
{
    light_buffer[index] = encode_light(r);
}

void blend_light ( uint index, float4 r )
{
    //float4 s = decode_light( light_buffer[index] );
    float4 s = float4(0.0, 0.0, 0.0, 0.0);
    write_light ( index, r + s );
}

uint linear_address(uint2 address)
{
    return address.y * light_accumulation_buffer_dimensions.x + address.x;
}

[numthreads(1, 1, 1)]
void main( uint3 dispatch_thread_id : SV_DispatchThreadID ) 
{
    const uint32_t instance_id_bits = 10;
    const uint32_t mask = ((1 << instance_id_bits) - 1);
   
    float4 light = float4(0.0, 0.0f, 0.0f, 0.0f);

    for(uint32_t i = 0; i < sample_count; ++i)
    {
        uint    s = visibility_buffer.sample[i][dispatch_thread_id.xy];
        uint    instance_id = s & mask;
        float   o = 0.0f;

        float4 r = float4( (instance_id / (float)mask), o, o, 1.0f );

        light += r;
    }

    blend_light( linear_address ( dispatch_thread_id.xy ), light * rcp(sample_count) );
}