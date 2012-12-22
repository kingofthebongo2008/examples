
BlendState blendDisabled
{
	BlendEnable[0] = false;
};

BlendState blendTranslucent
{
	BlendEnable[0] = true;
	SrcBlend = SRC_ALPHA;
	DestBlend = INV_SRC_ALPHA;
};

DepthStencilState dsNoTest
{
	DepthEnable = false;
	DepthWriteMask = true;
	StencilEnable = false;
};

DepthStencilState dsTest
{
	DepthEnable = true;
	DepthWriteMask = true;
	StencilEnable = false;
};

DepthStencilState dsTestNoWrite
{
	DepthEnable = true;
	DepthWriteMask = false;
	StencilEnable = false;
};

RasterizerState rsState
{
	CullMode = None;
	DepthClipEnable = true;
};

cbuffer MainVS
{
	float4x4 mvp;
};

cbuffer MainPS
{
	float4 color;
};

/////////////////////////////////////////////////////////////////////////////

struct InOutLine
{
	float4 pos: SV_Position;
};

InOutLine vsLine(InOutLine In)
{
	In.pos = mul(mvp, In.pos);

	return In;
}

float4 psLine(InOutLine In): SV_Target
{
	return float4(0, 1, 0, 1);
}

//-------------------------------------------------------------------------//

technique10 Line
{
	pass Main
	{
		SetBlendState(blendDisabled, float4(0, 0, 0, 0), 0xFFFFFFFF);
		SetDepthStencilState(dsTest, 0x0);
		SetRasterizerState(rsState);

		SetVertexShader(CompileShader(vs_4_0, vsLine()));
		SetGeometryShader(NULL);
		SetPixelShader(CompileShader(ps_4_0, psLine()));
	}
}

/////////////////////////////////////////////////////////////////////////////

struct VsInBillboard
{
	float4 pos: SV_Position;
};

struct PsInBillboard
{
	float4 pos: SV_Position;
	float4 texCoord: TexCoord;
};

PsInBillboard vsBillboard(VsInBillboard In, uint VertexID: SV_VertexID)
{
	PsInBillboard Out;

	Out.pos = mul(mvp, In.pos);
	Out.texCoord.xy = float2(VertexID & 0x1, (VertexID & 0x2) * 0.5) * 2.0 - 1.0;
	Out.texCoord.z = 1.0;
	Out.texCoord.w = -1.0;

	return Out;
}

float4 psBillboard(PsInBillboard In): SV_Target
{
	float4 a = saturate(dot(In.texCoord.xyz, -In.texCoord.xyw));
	a.rgb *= rsqrt(a);

	return a * color;
}

//-------------------------------------------------------------------------//

technique10 Billboard
{
	pass Particle
	{
		SetBlendState(blendTranslucent, float4(0, 0, 0, 0), 0xFFFFFFFF);
		SetDepthStencilState(dsTestNoWrite, 0x0);
		SetRasterizerState(rsState);

		SetVertexShader(CompileShader(vs_4_0, vsBillboard()));
		SetGeometryShader(NULL);
		SetPixelShader(CompileShader(ps_4_0, psBillboard()));
	}
}

/////////////////////////////////////////////////////////////////////////////

Texture2D tex2d;
Texture2DArray texArray;
Texture3D tex3d;

SamplerState bilinear
{
	Filter = MIN_MAG_LINEAR_MIP_POINT;
	MaxLOD = 0;
	AddressU = CLAMP;
	AddressV = CLAMP;
	AddressW = CLAMP;
};

struct InOutTex
{
	float4 pos: SV_Position;
	float3 texCoord: TexCoord;
};

InOutTex vsTex(InOutTex In)
{
	return In;
}

float4 psTex2D(InOutTex In): SV_Target
{
	return tex2d.Sample(bilinear, In.texCoord);
}

float4 psTex2DArray(InOutTex In): SV_Target
{
	return texArray.Sample(bilinear, In.texCoord);
}

float4 psTex3D(InOutTex In): SV_Target
{
	return tex3d.Sample(bilinear, In.texCoord);
}

//-------------------------------------------------------------------------//

technique10 Debug
{
	pass Tex2D
	{
		SetBlendState(blendDisabled, float4(0, 0, 0, 0), 0xFFFFFFFF);
		SetDepthStencilState(dsNoTest, 0x0);
		SetRasterizerState(rsState);

		SetVertexShader(CompileShader(vs_4_0, vsTex()));
		SetGeometryShader(NULL);
		SetPixelShader(CompileShader(ps_4_0, psTex2D()));
	}

	pass Tex2DArray
	{
		SetBlendState(blendDisabled, float4(0, 0, 0, 0), 0xFFFFFFFF);
		SetDepthStencilState(dsNoTest, 0x0);
		SetRasterizerState(rsState);

		SetVertexShader(CompileShader(vs_4_0, vsTex()));
		SetGeometryShader(NULL);
		SetPixelShader(CompileShader(ps_4_0, psTex2DArray()));
	}

	pass Tex3D
	{
		SetBlendState(blendDisabled, float4(0, 0, 0, 0), 0xFFFFFFFF);
		SetDepthStencilState(dsNoTest, 0x0);
		SetRasterizerState(rsState);

		SetVertexShader(CompileShader(vs_4_0, vsTex()));
		SetGeometryShader(NULL);
		SetPixelShader(CompileShader(ps_4_0, psTex3D()));
	}
}
