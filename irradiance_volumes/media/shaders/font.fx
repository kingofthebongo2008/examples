
BlendState blendState
{
	BlendEnable[0] = true;
	SrcBlend = SRC_ALPHA;
	DestBlend = INV_SRC_ALPHA;
	RenderTargetWriteMask[0] = 0xF;
};

DepthStencilState dsState
{
	DepthEnable = false;
	DepthWriteMask = false;
	StencilEnable = false;
};

RasterizerState rsState
{
	CullMode = None;
	DepthClipEnable = true;
};

Texture2D tex;
SamplerState smpState
{
	Filter = MIN_MAG_LINEAR_MIP_POINT;
	AddressU = CLAMP;
	AddressV = CLAMP;
};

cbuffer perPass
{
	float4 color;
};

struct InOut
{
	float4 pos: SV_Position;
	float2 texCoord: texCoord;
};

InOut vsMain(InOut In)
{
	return In;
}

float4 psMain(InOut In): SV_Target
{
	return color * tex.Sample(smpState, In.texCoord);
}

/////////////////////////////////////////////////////////////////////////////

technique10 t0
{
	pass font
	{
		SetBlendState(blendState, float4(0, 0, 0, 0), 0xFFFFFFFF);
		SetDepthStencilState(dsState, 0x0);
		SetRasterizerState(rsState);

		SetVertexShader(CompileShader(vs_4_0, vsMain()));
		SetGeometryShader(NULL);
		SetPixelShader(CompileShader(ps_4_0, psMain()));
	}
}
