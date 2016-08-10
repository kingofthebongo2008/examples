struct interpolants
{
    float4 position     : SV_POSITION0;
    float2 uv           : texcoord0;
};

SamplerState g_sampler
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Clamp;
    AddressV = Clamp;
};

Texture2D         g_texture : register(t0);

float4 main(interpolants input) : SV_TARGET
{
    return g_texture.Sample(g_sampler, input.uv);
}