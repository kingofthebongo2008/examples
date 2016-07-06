#include "gx_shader_geometry_pass_common.hlsl"

cbuffer per_object : register(SLOT_PER_DRAW_CALL)
{
    row_major float4x4 m_w;
}

struct vs_input
{
    float3   position_os : position;
};

struct vs_output
{
    float4	position_ps : sv_position;
};

vs_output main( in  vs_input input)
{
    vs_output output;
    output.position_ps = project_vertex(float4( input.position_os.xyz,1.0f ), m_w, m_v, m_p);
    return output;
}