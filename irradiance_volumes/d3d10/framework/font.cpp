//-----------------------------------------------------------------------------
// File: Framework\Font.cpp
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------





#include "font.h"
#include <stdio.h>



#define BATCH_CHARS 64

struct Vertex
{
	float x, y;
	float s, t;
};

struct PerPass
{
	float4 color;
};

TexFont::TexFont()
{
	m_pDevice = NULL;
	m_pEffect = NULL;

	m_pTexture = NULL;
	m_pTexSRV  = NULL;

	m_pVertexBuffer = NULL;
	m_pIndexBuffer  = NULL;
	m_pInputLayout  = NULL;
	m_pConstantBuffer = NULL;

	memset(m_chars, 0, sizeof(m_chars));

	m_color = float4(0, 0, 0, 0);
}

TexFont::~TexFont()
{


}

HRESULT TexFont::LoadFont(ID3D11Device *pDevice, const TCHAR *textureFile, const TCHAR *fontFile)
{
	TCHAR str[256];

	m_pDevice = pDevice;

	// Load the font file
	FILE *file = _tfopen(fontFile, _T("rb"));
	if (file == NULL)
	{
		_stprintf(str, _T("Couldn't load \"%s\""), fontFile);
		MessageBox(NULL, str, _T("Error"), MB_OK | MB_ICONERROR);
		return E_FAIL;
	}

	unsigned int version = 0;
	fread(&version, sizeof(version), 1, file);
	if (version != 2)
	{
		fclose(file);
		MessageBoxA(NULL, "Unsupported font file version", "Error", MB_OK | MB_ICONERROR);
		return E_FAIL;
	}
	fread(m_chars, sizeof(m_chars), 1, file);
	fclose(file);


	// Load the effect
	HRESULT hr;
	ID3D10Blob *pErr = NULL;
	if (FAILED(hr = D3DX11CompileEffectFromFile(SHADER_PATH _T("Font.fx"), NULL, NULL, 0, 0, m_pDevice, &m_pEffect, &pErr)))
	{
		if (pErr)
		{
			const char *err = (LPCSTR) pErr->GetBufferPointer();
			OutputDebugStringA(err);
			MessageBoxA(NULL, err, "Error", MB_OK | MB_ICONERROR);
		}
		return hr;
	}
	m_pPass = m_pEffect->GetTechniqueByIndex(0)->GetPassByIndex(0);


	// Replace effect constant buffer with our own so we can Map() it.
	ID3DX11EffectConstantBuffer *cbVar = m_pEffect->GetConstantBufferByName("perPass");
	ID3D11Buffer *cb;
	cbVar->GetConstantBuffer(&cb);
	D3D11_BUFFER_DESC cbDesc;
	cb->GetDesc(&cbDesc);
	cb->Release();
	cbDesc.Usage = D3D11_USAGE_DYNAMIC;
	cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	if (FAILED(pDevice->CreateBuffer(&cbDesc, NULL, &m_pConstantBuffer)))
	{
		MessageBoxA(NULL, "Constant buffer creation failed", "Error", MB_OK | MB_ICONERROR);
		return E_FAIL;
	}
	cbVar->SetConstantBuffer(m_pConstantBuffer);



	// Create input layout
	const D3D11_INPUT_ELEMENT_DESC inputLayout[] =
	{
		{ "SV_Position", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0 * sizeof(float), D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "texCoord",    0, DXGI_FORMAT_R32G32_FLOAT, 0, 2 * sizeof(float), D3D11_INPUT_PER_VERTEX_DATA, 0 },
	};

	D3DX11_PASS_DESC pd;
	if (FAILED(hr = m_pPass->GetDesc(&pd))) return hr;
	if (FAILED(hr = m_pDevice->CreateInputLayout(inputLayout, sizeof(inputLayout) / sizeof(inputLayout[0]), pd.pIAInputSignature, pd.IAInputSignatureSize, &m_pInputLayout)))
	{
		MessageBoxA(NULL, "Input layout creation failed", "Error", MB_OK | MB_ICONERROR);
		return hr;
	}

	// Load the texture
	if (FAILED(hr = D3DX11CreateTextureFromFile(m_pDevice, textureFile, NULL, NULL, (ID3D11Resource **) &m_pTexture, NULL)))
	{
		_stprintf(str, _T("Couldn't load \"%s\""), textureFile);
		MessageBox(NULL, str, _T("Error"), MB_OK | MB_ICONERROR);
		return hr;
	}

	D3D11_TEXTURE2D_DESC desc;
	m_pTexture->GetDesc(&desc);

	// Create a shader resource view for the texture
	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
	ZeroMemory(&srvDesc, sizeof(srvDesc));
	srvDesc.Format = desc.Format;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MipLevels = desc.MipLevels;
	if (FAILED(hr = m_pDevice->CreateShaderResourceView(m_pTexture, &srvDesc, &m_pTexSRV)))
	{
		MessageBoxA(NULL, "CreateShaderResourceView failed", "Error", MB_OK | MB_ICONERROR);
		return hr;
	}

	m_pEffect->GetVariableByName("tex")->AsShaderResource()->SetResource(m_pTexSRV);


	// Create a vertex buffer
	D3D11_BUFFER_DESC bd;
	bd.ByteWidth = BATCH_CHARS * 4 * sizeof(Vertex);
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	bd.Usage = D3D11_USAGE_DYNAMIC;
	bd.MiscFlags = 0;
	if (FAILED(hr = m_pDevice->CreateBuffer(&bd, NULL, &m_pVertexBuffer)))
	{
		MessageBoxA(NULL, "Vertex buffer creation failed", "Error", MB_OK | MB_ICONERROR);
		return hr;
	}


	// Create an index buffer
	unsigned short indices[BATCH_CHARS * 6];
	for (int i = 0; i < BATCH_CHARS; i++)
	{
		indices[6 * i + 0] = 4 * i + 0;
		indices[6 * i + 1] = 4 * i + 1;
		indices[6 * i + 2] = 4 * i + 2;
		indices[6 * i + 3] = 4 * i + 2;
		indices[6 * i + 4] = 4 * i + 1;
		indices[6 * i + 5] = 4 * i + 3;
	}

	bd.ByteWidth = sizeof(indices);
	bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	bd.CPUAccessFlags = 0;
	bd.Usage = D3D11_USAGE_IMMUTABLE;

	D3D11_SUBRESOURCE_DATA data;
	data.pSysMem = indices;
	data.SysMemPitch = 0;
	data.SysMemSlicePitch = 0;
	if (FAILED(hr = m_pDevice->CreateBuffer(&bd, &data, &m_pIndexBuffer)))
	{
		MessageBoxA(NULL, "Index buffer creation failed", "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}


	return S_OK;
}

void TexFont::Release()
{
	if (m_pEffect) m_pEffect->Release();

	if (m_pTexture) m_pTexture->Release();
	if (m_pTexSRV) m_pTexSRV->Release();

	if (m_pVertexBuffer) m_pVertexBuffer->Release();
	if (m_pIndexBuffer) m_pIndexBuffer->Release();
	if (m_pInputLayout) m_pInputLayout->Release();
	if (m_pConstantBuffer) m_pConstantBuffer->Release();
}

void TexFont::DrawText(ID3D11DeviceContext* pDeviceContext, const char *str, float x, float y, const float charWidth, const float charHeight, const HAlign hAlign, const VAlign vAlign, const float4 *color)
{
	m_pPass->Apply(0, pDeviceContext);

	float4 c = color? *color : float4(1, 1, 1, 1);

	if (c != m_color)
	{
		// Update per frame constants
		PerPass *pp;
		D3D11_MAPPED_SUBRESOURCE resource;
		pDeviceContext->Map ( m_pConstantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &resource);
			pp = reinterpret_cast<PerPass*> ( resource.pData );
			pp->color = (m_color = c);
		pDeviceContext->Unmap( m_pConstantBuffer, 0 );
	}

	pDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	pDeviceContext->IASetInputLayout(m_pInputLayout);
	pDeviceContext->IASetIndexBuffer(m_pIndexBuffer, DXGI_FORMAT_R16_UINT, 0);


	float startx = x;

	// Adjust horizontal position for first line
	if (hAlign != HA_LEFT)
	{
		float lineWidth = charWidth * getLineWidth(str);
		if (hAlign == HA_RIGHT)
			x -= lineWidth;
		else
			x -= 0.5f * lineWidth;
	}

	// Adjust vertical position
	if (vAlign != VA_TOP)
	{
		// Count number of lines
		int i = 0, n = 1;
		while (str[i])
		{
			if (str[i] == '\n') n++;
			i++;
		}
		if (vAlign == VA_BOTTOM)
			y += n * charHeight;
		else
			y += n * 0.5f * charHeight;
	}



	int count = 0;
	Vertex *dest;

	while (true)
	{
		// Flush if buffer is full or we've reached the end of the string
		if (*str == '\0' || count >= BATCH_CHARS)
		{
			pDeviceContext->Unmap ( m_pVertexBuffer, 0 );

			UINT stride = sizeof(Vertex), offset = 0;
			pDeviceContext->IASetVertexBuffers(0, 1, &m_pVertexBuffer, &stride, &offset);
			pDeviceContext->DrawIndexed(6 * count, 0, 0);

			if (*str == '\0') break;

			count = 0;
		}

		if (*str == '\n')
		{
			// New line
			y -= charHeight;
			x = startx;

			// Adjust horizontal position for next line
			if (hAlign != HA_LEFT)
			{
				float lineWidth = charWidth * getLineWidth(str + 1);
				if (hAlign == HA_RIGHT)
					x -= lineWidth;
				else
					x -= 0.5f * lineWidth;
			}
		}
		else
		{
			Character chr = m_chars[*(unsigned char *) str];
			float cw = charWidth * chr.ratio;

			if (*str != ' ') // No need to add space chars to the vertex buffer
			{

				// First char needs to map the buffer
				if (count == 0)
				{
					ID3D11Buffer *null = NULL;
					UINT so = 0;
					pDeviceContext->IASetVertexBuffers(0, 1, &null, &so, &so);

					D3D11_MAPPED_SUBRESOURCE resource;
					pDeviceContext->Map( m_pVertexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &resource) ;
					dest = reinterpret_cast<Vertex*> ( resource .pData );
				}

				// Insert this char in the vertex buffer
				dest[0].x = x;
				dest[0].y = y - chr.y0 * charHeight;
				dest[0].s = chr.s0;
				dest[0].t = chr.t0;
				dest[1].x = x + cw;
				dest[1].y = y - chr.y0 * charHeight;
				dest[1].s = chr.s1;
				dest[1].t = chr.t0;
				dest[2].x = x;
				dest[2].y = y - chr.y1 * charHeight;
				dest[2].s = chr.s0;
				dest[2].t = chr.t1;
				dest[3].x = x + cw;
				dest[3].y = y - chr.y1 * charHeight;
				dest[3].s = chr.s1;
				dest[3].t = chr.t1;
				dest += 4;
				count++;
			}

			x += cw;
		}

		str++;
	}
}

float TexFont::getTextWidth(const char *str, int length) const
{
	if (length < 0) length = (int) strlen(str);

	float len = 0;
	for (int i = 0; i < length; i++){
		len += m_chars[(unsigned char) str[i]].ratio;
	}

	return len;
}

float TexFont::getLineWidth(const char *str) const
{
	float len = 0;
	int i = 0;
	while (str[i] && str[i] != '\n')
	{
		len += m_chars[(unsigned char) str[i]].ratio;
		i++;
	}

	return len;
}
