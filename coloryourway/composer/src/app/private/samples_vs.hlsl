struct sample
{
    float       m_x;        //x coordinate
    float       m_y;        //y coordinate
    uint        m_c;        //sample class
};

struct vertex_out
{   
    float4 position_ps     : SV_Position;
    uint   sample_class    : SAMPLE_CLASS;
};

StructuredBuffer<sample> buffer;

vertex_out vertex_main(uint id : SV_VertexID, uint instance : SV_InstanceID)
{
    vertex_out o;

    float4 v = float4(0.0,0.0,0.0, 0.0 );

    float  scale = 0.015f;
    float2 offset = { 0.11f * instance, 0.0f };

    sample s = buffer.Load(instance);

    offset = float2(s.m_x, s.m_y) - float2( +0.5f, + 0.5f);

    float4x4 model =
    {
        scale, 0.0f, 0.0f, 0.0f,
        0.0f, scale, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        offset.x, offset.y, 0.0f, 1.0f,
    };

    switch (id)
    {
        case 0:
        {   v = float4 (0, 1, 0, 1);
            break;
        }

        case 1:
        {
            v = float4 (1, 0, 0, 1);
            break;
        }

        case 2:
        {
            v = float4 (-1, 0, 0, 1);
            break;
        }
    }

    o.position_ps   = mul(v, model);
    o.sample_class  = s.m_c;
    return o;
}