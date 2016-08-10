float mip_map_level(float2 uv, float2 scale)
{
    float2 texcoords_texels = uv * 16384.0f * scale;

    float2 dx       = ddx(texcoords_texels);
    float2 dy       = ddy(texcoords_texels); 
    float px        = dot(dx, dx);
    float py        = dot(dy, dy);
    float max_lod   = 0.5f * log2(max(px, py));

    return max_lod;
}

float mip_map_level(float2 uv)
{
    return mip_map_level(uv, float2(1.0f, 1.0f));
}
