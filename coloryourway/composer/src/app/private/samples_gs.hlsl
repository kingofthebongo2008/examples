
struct gs_input
{
    //float4	center_position_ps	: sv_position;
};

struct gs_output
{
    float4  position_ps	    : sv_position;
};

[maxvertexcount(3)]
void geometry_main(triangle gs_input input[3], inout TriangleStream<gs_output> stream)
{
    gs_output o;


    o.position_ps = float4(1.0f, 1.0f, 0.0f, 1.0f);
    stream.Append(o);

    
    o.position_ps = float4(1.0f, -1.0f, 0.3f, 1.0f);
    stream.Append(o);


    o.position_ps = float4(-1.0f, -1.0f, 0.0f, 1.0f);
    stream.Append(o);

    stream.RestartStrip();
}

