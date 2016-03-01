//--------------------------------------------------------------------------------------
// TerrainView.h
//
// Draws a heightmap-driven terrain, using tiled textures for the heightmap and the
// surface normal and diffuse maps.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#include <windows.h>
#include <d3d11.h>
#include "TitleResidencyManager.h"
#include "PageLoaders.h"
#include "SceneObject.h"
#include "DXUTcamera.h"

//--------------------------------------------------------------------------------------
// Name: TerrainView
// Desc: Implements a height mapped 3D terrain, using three tiled textures to draw the terrain.
//       The terrain view implements not only its scene render, but also its residency
//       sample render as well.
//--------------------------------------------------------------------------------------
class TerrainView
{
protected:
    // D3D11 device and immediate context:
    ID3D11Device* m_pd3dDevice;
    ID3D11TiledResourceDevice* m_pd3dDeviceEx;

    // Pointer to the title residency manager:
    TitleResidencyManager* m_pTRM;

    // Bound tiled textures:
    BoundTiledTexture m_DiffuseMapTexture;
    BoundTiledTexture m_NormalMapTexture;
    BoundTiledTexture m_HeightMapTexture;

    // Resource set ID for the combination of the height map, diffuse map, and normal map:
    ResourceSetID m_RSID;

    // Vertex and index buffers for a section of the terrain:
    ID3D11Buffer* m_pVBGrid;
    ID3D11Buffer* m_pIBGrid;

    // Input layout for the vertex shader and grid VB:
    ID3D11InputLayout* m_pInputLayout;

    // Constant buffers for terrain rendering:
    ID3D11Buffer* m_pCBVertex;
    ID3D11Buffer* m_pCBPixel;

    // Shaders for terrain rendering:
    ID3D11VertexShader* m_pVSTerrain;
    ID3D11PixelShader* m_pPSRender;

    // Renderstate members:
    ID3D11RasterizerState* m_pRasterizerState;
    ID3D11BlendState* m_pBlendState;
    ID3D11DepthStencilState* m_pDepthStencilState;

    // Width of the terrain render view, in pixels:
    FLOAT m_RenderWidth;

    // Camera members for viewing the terrain:
    CFirstPersonCamera m_Camera;
    XMFLOAT4X4 m_matWorld;
    XMFLOAT4X4 m_matWorldViewProjection;

    // Number of vertices on the edge of a single grid mesh:
    UINT m_MeshGridSize;

    // Number of X axis repetitions of the grid mesh:
    UINT m_LayoutGridWidth;

    // Number of Z axis repetitions of the grid mesh:
    UINT m_LayoutGridHeight;

public:
    TerrainView( ID3D11Device* pd3dDevice, ID3D11TiledResourceDevice* pd3dDeviceEx, TitleResidencyManager* pTRM );
    ~TerrainView();

    CFirstPersonCamera* GetCamera() { return &m_Camera; }

    ID3D11TiledTexture2D* GetInspectionTexture() const { return m_DiffuseMapTexture.pTexture; }

    BOOL IsLoaded() const;

    VOID Update( FLOAT DeltaTime );
    VOID PreFrameRender( ID3D11DeviceContext* pd3dContext, FLOAT DeltaTime );
    VOID Render( ID3D11DeviceContext* pd3dContext );

protected:
    VOID CreateGeometry();
    HRESULT LoadTiledTextureFile( BoundTiledTexture* pTexture, const WCHAR* strFileName );
    VOID SetupVSAndRender( ID3D11DeviceContext* pd3dContext );
};

