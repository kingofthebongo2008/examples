//--------------------------------------------------------------------------------------
// SceneObject.h
//
// Represents a scene object that is rendered with tiled textures as surface textures.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#include <windows.h>
#include <d3d11.h>
#include <xnamath.h>
#include <vector>
#include "d3d11tiled.h"
#include "SamplingQualityManager.h"
#include "TitleResidencyManager.h"

//--------------------------------------------------------------------------------------
// Name: BoundTiledTexture
// Desc: Represents a combination of a tiled texture, its associated tile loader, its
//       sampling quality manager, and its shader resource view.
//--------------------------------------------------------------------------------------
struct BoundTiledTexture
{
    ITileLoader* pTileLoader;
    BOOL DeleteTileLoader;

    ID3D11TiledTexture2D* pTexture;
    ID3D11TiledShaderResourceView* pTextureSRV;
    SamplingQualityManager* pSamplingQualityManager;
    D3D11_TILED_SURFACE_DESC BaseLevelDesc;

    //--------------------------------------------------------------------------------------
    // Name: BoundTiledTexture constructor
    //--------------------------------------------------------------------------------------
    BoundTiledTexture()
    {
        ZeroMemory( this, sizeof(BoundTiledTexture) );
    }

    //--------------------------------------------------------------------------------------
    // Name: BoundTiledTexure::Release
    // Desc: Releases all D3D11 objects associated with this tiled texture, and deletes other
    //       associated objects.
    //--------------------------------------------------------------------------------------
    VOID Release()
    {
        SAFE_RELEASE( pTexture );
        SAFE_RELEASE( pTextureSRV );

        delete pSamplingQualityManager;

        if( DeleteTileLoader )
        {
            delete pTileLoader;
        }
        pTileLoader = NULL;
    }

    //--------------------------------------------------------------------------------------
    // Name: BoundTiledTexture::Loaded
    // Desc: Returns TRUE if this texture is loaded and valid, FALSE otherwise.
    //--------------------------------------------------------------------------------------
    BOOL Loaded() const
    {
        return pTexture != NULL && pSamplingQualityManager != NULL;
    }
};

// The maximum amount of tiled textures that can be sampled from in a single pass:
static const UINT MAX_TEXTURES_PER_OBJECT = 4;

//--------------------------------------------------------------------------------------
// Name: SceneObject
// Desc: Represents a single textured object that uses tiled resources for its surface
//       textures.
//--------------------------------------------------------------------------------------
struct SceneObject
{
    // World transform matrix:
    XMFLOAT4X4 matWorld;

    // Vertex and index buffer members for the object:
    UINT VertexStrideBytes;
    ID3D11InputLayout* m_pLayoutResidencySample;
    ID3D11InputLayout* m_pLayoutSceneRender;
    ID3D11Buffer* pVertexBuffer;
    ID3D11Buffer* pIndexBuffer;

    // Pixel shader used for scene rendering of the object:
    ID3D11PixelShader* pPixelShader;

    // Primitive topology members for the object:
    D3D11_PRIMITIVE_TOPOLOGY PrimitiveType;
    UINT IndexCount;
    UINT VertexCount;

    // Tiled texture members for the object:
    UINT TextureCount;
    BoundTiledTexture Textures[MAX_TEXTURES_PER_OBJECT];

    // Resource set ID that represents the set of textures mapped onto this object:
    ResourceSetID RSID;

    //--------------------------------------------------------------------------------------
    // Name: SceneObject constructor
    //--------------------------------------------------------------------------------------
    SceneObject()
    {
        ZeroMemory( this, sizeof(SceneObject) );
        XMStoreFloat4x4( &matWorld, XMMatrixIdentity() );
    }
};

typedef std::vector<SceneObject*> SceneObjectVector;
