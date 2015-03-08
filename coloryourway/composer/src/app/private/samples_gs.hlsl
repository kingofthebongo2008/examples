
struct gs_input
{
    float4	center_position_ps	: sv_position;
};

struct gs_output
{
    float4  position_ps	    : sv_position;
};

[maxvertexcount(3)]
void geometry_main(point gs_input input[1], inout TriangleStream<gs_output> stream)
{
 
    gs_output o;
    o.position_ps = float4(0.0f, 0.0f, 0.0f, 1.0f);
    stream.Append(o);
    stream.RestartStrip();
}

