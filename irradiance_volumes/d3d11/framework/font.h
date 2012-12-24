//-----------------------------------------------------------------------------
// File: Framework\Font.h
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------





#ifndef FONT_H
#define FONT_H

#include <windows.h>
#include <tchar.h>
#include <d3d11.h>
#include <D3DX11tex.h>

#include <d3dx11effect.h>
#include "Math/Vector.h"

struct Character
{
	float s0, t0;
	float s1, t1;
	float ratio;
	float y0, y1;
};

enum HAlign
{
	HA_LEFT,
	HA_CENTER,
	HA_RIGHT,
};

enum VAlign
{
	VA_TOP,
	VA_CENTER,
	VA_BOTTOM,
};

class TexFont
{
public:
	TexFont();
	~TexFont();

	HRESULT LoadFont(ID3D11Device *pDevice, const TCHAR *textureFile, const TCHAR *fontFile);
	void Release();

	void DrawText(ID3D11DeviceContext* pDeviceContext, const char *str, float x, float y, const float charWidth, const float charHeight, const HAlign hAlign = HA_LEFT, const VAlign vAlign = VA_TOP, const float4 *color = NULL);

	float getTextWidth(const char *str, int length) const;
	float getLineWidth(const char *str) const;



private:
	Character m_chars[256];

	ID3D11Device *m_pDevice;
	ID3DX11Effect *m_pEffect;
	ID3DX11EffectPass *m_pPass;

	ID3D11ShaderResourceView *m_pTexSRV;
	ID3D11Texture2D *m_pTexture;

	ID3D11Buffer *m_pVertexBuffer;
	ID3D11Buffer *m_pIndexBuffer;
	ID3D11InputLayout *m_pInputLayout;
	ID3D11Buffer *m_pConstantBuffer;

	float4 m_color;
};

#endif // FONT_H
