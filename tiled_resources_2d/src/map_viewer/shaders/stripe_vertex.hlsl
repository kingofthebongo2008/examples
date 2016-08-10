struct interpolants
{
    float4 position     : SV_POSITION0;
};

interpolants main( in uint VertID : SV_VertexID )
{
    interpolants r;

    float pixel_size = 1600;

    if (VertID > 5)
    {
        pixel_size = 900.0f;
    }

    switch ( VertID % 6 )
    {
        case 0:
        {
            r.position  = float4( -1.0f / pixel_size, 1.0f, 0.0f, 1.0f );
            break;
        }

        case 1:
        {
            r.position = float4( -1.0f / pixel_size,  -1.0f, 0.0f, 1.0f);
            break;
        }

        case 2:
        {
            r.position = float4( +1.0f / pixel_size, -1.0f, 0.0f, 1.0f);
            break;
        }

        case 3:
        {
            r.position = float4( +1.0f / pixel_size, -1.0f, 0.0f, 1.0f);
            break;
        }

        case 4:
        {
            r.position = float4(+1.0f / pixel_size,  1.0f, 0.0f, 1.0f);
            break;
        }

        case 5:
        {
            r.position = float4(-1.0f / pixel_size, 1.0f, 0.0f, 1.0f);
            break;
        }
    }

    if ( VertID > 5)
    {
        float t = r.position.x;
        r.position.x = r.position.y;
        r.position.y = t;
    }

    return r;
}
