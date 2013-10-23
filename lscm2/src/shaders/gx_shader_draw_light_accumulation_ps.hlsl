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


float4 main( in  vs_output input) : sv_target
{
    uint2 address = uint2 (input.uv * float2(1280, 720));

    uint  linear_address = address.y * 1280 + address.x;

    uint2  samples  = sampled_texture[ linear_address ];   
    uint2  t        = samples;
    float4 k        = decode_light(t);

    return k;
}
