struct vs_output
{
    float4	position_ps : sv_position;
    float2	uv          : texcoord; 
};

Texture2D<float2>     sampled_texture;
SamplerState          default_sampler;


float4 decode_light ( uint2 r )
{
    uint2 a = ( r >> 16 );
    uint2 b =  r;
    return float4( f16tof32(a), f16tof32(b) );
}


float4 main( in  vs_output input) : sv_target
{
    //read (sample) what is in a texture and output it on the render target

    float2 samples = sampled_texture.Sample(default_sampler, input.uv).xy;
    uint2  t = asuint(samples);
    float4 k = decode_light(t);

	return float4( k.xy, 0.0f, 1.0f);

    //return float4(1.0f, 0.0f, 0.0f, 1.0f);
}
