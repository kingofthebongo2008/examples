struct sample
{
    float       m_x;        //x coordinate
    float       m_y;        //y coordinate
    uint        m_c;        //sample class
};

StructuredBuffer<sample> buffer;

float4 vertex_main(uint id : SV_VertexID, uint instance : SV_InstanceID) : SV_Position
{
    float4 v;

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

        default: return float4 (0, 0, 0, 0);
    }

    return mul(v, model);
}