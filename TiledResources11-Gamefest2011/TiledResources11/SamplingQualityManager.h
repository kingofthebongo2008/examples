//--------------------------------------------------------------------------------------
// SamplingQualityManager.h
//
// The sampling quality manager maintains a "sampling quality map" for a single tiled 
// resource.  The sampling quality map is sized so that one texel represents one tile of
// the tiled texture's base mip level (level 0).  Each texel's value in the sampling
// quality map represents a fractional minimum mip LOD level allowed at that region of
// the texture.  The value is animated over time as tiles are mapped and unmapped in the
// tiled texture, which creates smooth mip level transitions during streaming.
//
// During scene rendering, the scene render pixel shader samples a value from the
// sampling quality map, returning a min LOD value.  Then, when sampling from the tiled
// texture, the sampling quality value is passed as a min LOD clamp value.  This prevents
// the tiled resource sample from sampling from a nonresident mip level of the texture.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#include <windows.h>
#include <d3d11.h>
#include <xnamath.h>

#include "d3d11tiled.h"
#include "TitleResidencyManager.h"

// When a tile is loaded that populates a lower mip level, the new mip value at that location
// is lerped to over this amount of time.
#define MIP_TRANSITION_TIME_SECONDS 0.25f

//--------------------------------------------------------------------------------------
// Name: SamplingQualityManager
// Desc: Class that tracks per-tile mip residency for a single tiled texture.  2D and
//       2D array textures are supported.  The sampling quality manager listens to tile
//       activity from the title residency manager to know when the sampling quality map
//       needs to be updated.
//--------------------------------------------------------------------------------------
class SamplingQualityManager : public ITileActivityHandler
{
protected:
    // Sampler state object for sampling from the sampling quality map:
    static ID3D11SamplerState* s_pSamplerState;

    // The tiled texture2D that this sampling quality map tracks:
    ID3D11TiledTexture2D* m_pResource;

    // The shader resource view for the tiled texture:
    ID3D11TiledShaderResourceView* m_pResourceSRV;

    // The sampling quality map from the previous update:
    ID3D11Texture2D* m_pTileLODTextureSample;

    // The sampling quality map from the current update:
    ID3D11Texture2D* m_pTileLODTextureRender;

    // The sampling quality map shader resource view for the previous update:
    ID3D11ShaderResourceView* m_pTileLODTextureSRV;

    // The non-array sampling quality map render target view for the current update:
    ID3D11RenderTargetView* m_pTileLODSurface;

    // The array sampling quality map render target views for the current update:
    ID3D11RenderTargetView** m_ppTileLODSurfaceArray;

    // The number of array slices in the tiled texture:
    UINT m_ArraySize;

    // Flag that is TRUE on the first update, FALSE otherwise:
    BOOL m_FirstFrame;

    // The viewport for rendering to the sampling quality map rendertarget view:
    D3D11_VIEWPORT m_Viewport;

    // The amount of time that it takes for a section of a new mip level to be lerped into view:
    FLOAT m_MipTransitionDuration;

    // A countdown timer for each array slice, representing the length of time from now that the
    // sampling quality map for each array slice needs to be continuously updated.
    // This allows the sampling quality manager to pause updating the sampling quality maps when
    // there is no tile activity on the tiled texture:
    FLOAT* m_pSliceChangingTime;

public:
    SamplingQualityManager( ID3D11TiledTexture2D* pResource, ID3D11Device* pd3dDevice, ID3D11TiledResourceDevice* pd3dDeviceEx );
    ~SamplingQualityManager();

    VOID Render( ID3D11DeviceContext* pd3dDeviceContext, ID3D11TiledResourceDevice* pd3dDeviceEx, FLOAT fDeltaTime );

    ID3D11ShaderResourceView* GetLODQualityTextureSRV() const { return m_pTileLODTextureSRV; }
    static ID3D11SamplerState* GetSamplerState() { return s_pSamplerState; }

    virtual VOID TileLoaded( const TrackedTileID* pTileID );
    virtual VOID TileUnloaded( const TrackedTileID* pTileID );
};
