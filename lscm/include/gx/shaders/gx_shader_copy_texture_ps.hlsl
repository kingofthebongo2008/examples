struct vs_output
{
    float4	position_ps : sv_position;
    float2	uv          : texcoord; 
};

Texture2D       sampled_texture;
SamplerState    default_sampler;

float4 main( in  vs_output input) : sv_target
{
	return float4(1.0f, 0.0f, 0.0f, 1.0f ); //sampled_texture.Sample(default_sampler, input.uv).xyzw;
}