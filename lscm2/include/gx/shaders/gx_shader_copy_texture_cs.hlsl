Texture2D<uint>   source;
RWTexture2D<float4>   destination;

typedef uint uint32_t;

[numthreads(1, 1, 1)]
void main( uint3 dispatch_thread_id : SV_DispatchThreadID ) 
{
    uint s  = source[dispatch_thread_id.xy];

    const uint32_t instance_id_bits = 10;
    uint  instance_id = s & ( (1 << instance_id_bits) - 1 );
    float o = 0.0f;
    destination[ dispatch_thread_id.xy ].xyzw  =   float4( ( instance_id / 1023.0f), o , o , 1.0f);
}