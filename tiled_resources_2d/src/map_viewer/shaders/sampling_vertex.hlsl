#include "main_pass_vertex.h"

struct interpolants
{
    float4 position     : SV_POSITION0;
    float2 uv           : texcoord0;
};

struct input
{
    float3 position : position;
    float2 uv: texcoord0;
};

float4 project_vertex(float4 v_os, float4x4 world, float4x4 view, float4x4 perspective)
{
    //three muls for greater accuracy
    float4 result = mul(mul(mul(v_os, world), view), perspective);
    return result;
}

interpolants main(input i)
{
    interpolants r;

    r.uv = i.uv;
    r.position = project_vertex(float4(i.position, 1.0f), m_world, m_view, m_perspective);

    return r;
}
