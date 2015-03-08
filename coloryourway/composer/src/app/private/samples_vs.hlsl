struct vs_input
{
    float4 position_ps : position;
};

struct vs_output
{
    float4 position_ps : sv_position;
};

vs_output vertex_main(vs_input i)
{
    vs_output a;
    a.position_ps = i.position_ps;

    return a;
}
