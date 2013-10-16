Texture2D<uint>              source;
RWTexture2D<uint2>           destination;

typedef uint uint32_t;

//LUV, YCbCr or YCoCg can do also, but now, just pack in smaller buffer

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
    return (a << 16) | b ;
}

void write_light ( uint2 index, float4 r )
{
    destination[ index.xy ] = encode_light(r);
}

void blend_light ( uint2 index, float4 r )
{
    float4 s = decode_light ( destination[ index ] );

    write_light ( index, r + s );
}


[numthreads(1, 1, 1)]
void main( uint3 dispatch_thread_id : SV_DispatchThreadID ) 
{
    uint s  = source[dispatch_thread_id.xy];

    const uint32_t instance_id_bits = 10;
    uint  instance_id = s & ( (1 << instance_id_bits) - 1 );
    float o = 0.0f;
    destination[ dispatch_thread_id.xy ] =   encode_light(float4(1.0, 1.0, 1.0, 1.0));     //float4( ( instance_id / 1023.0f), o , o , 1.0f);
}