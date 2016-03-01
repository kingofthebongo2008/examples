//--------------------------------------------------------------------------------------
// PageDebugRender.h
//
// A system that iterates through the tiles in the title residency manager, and
// draws a graphical representation of tile activity on screen.
// Also supports drawing textured quads on screen to visualize the residency sample pass.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#include <windows.h>
#include <d3d11.h>
#include <xnamath.h>

#include <list>

#include "d3d11tiled.h"

#include "TitleResidencyManager.h"

class TileDebugRender
{
protected:
    ID3D11InputLayout* m_pInputLayoutTiles;
    ID3D11VertexShader* m_pVertexShaderTiles;
    ID3D11PixelShader* m_pPixelShaderTiles;

    ID3D11InputLayout* m_pInputLayoutTexture;
    ID3D11VertexShader* m_pVertexShaderTexture;
    ID3D11PixelShader* m_pPixelShaderTexture;

    ID3D11BlendState* m_pAlphaBlendState;
    ID3D11BlendState* m_pSolidBlendState;
    ID3D11DepthStencilState* m_pDepthStencilState;

    ID3D11SamplerState* m_pSamplerPoint;

    ID3D11Buffer* m_pDynamicVB;
    ID3D11Buffer* m_pVertexCB;

    D3D11_MAPPED_SUBRESOURCE m_MappedData;
    UINT m_QuadIndex;
    UINT m_MaxQuadCount;

    INT m_MaxTileScore;
    INT m_MinTileScore;

    POINT* m_pMipLocationCache;

public:
    TileDebugRender()
    {
        ZeroMemory( this, sizeof(TileDebugRender) );
    }

    VOID Initialize( ID3D11Device* pd3dDevice );
    VOID Terminate();

    VOID Render( ID3D11DeviceContext* pd3dContext, TitleResidencyManager* pTRM, ID3D11TiledTexture2D* pTexture2D, INT XPos, INT YPos, INT* pTotalHeight = NULL, INT* pSliceHeight = NULL );

    VOID RenderTexture( ID3D11DeviceContext* pd3dContext, ID3D11ShaderResourceView* pSRView, RECT ScreenRect );

protected:
    VOID AddQuad( ID3D11DeviceContext* pd3dDeviceContext, INT X, INT Y, INT Width, INT Height, CXMVECTOR Color );
    VOID StartQuads( ID3D11DeviceContext* pd3dDeviceContext );
    VOID FlushQuads( ID3D11DeviceContext* pd3dDeviceContext );
};

