//--------------------------------------------------------------------------------------
// ResidencySampleRender.h
//
// A series of methods to assist in creating a residency sample rendering of a vector of 
// scene objects.  A residency sample render is a special view of the scene that records
// per-pixel UV, UV gradient, and resource ID information that is used to determine which 
// virtual tiles of which resources are currently visible.  This information is processed
// by the title residency manager, which streams tiles in and out of memory as needed to
// render the scene at the proper texel resolution.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#include <windows.h>
#include <d3d11.h>
#include <xnamath.h>

#include "d3d11tiled.h"

#include "SceneObject.h"
#include "TitleResidencyManager.h"
#include "SamplingQualityManager.h"

//--------------------------------------------------------------------------------------
// Name: ResidencySampleRender
// Desc: Namespace that includes the residency sample render methods.
//--------------------------------------------------------------------------------------
namespace ResidencySampleRender
{
    VOID Initialize( ID3D11Device* pd3dDevice );
    VOID Terminate();

    VOID ResizeRenderView( UINT RenderWidth, UINT RenderHeight );

    ID3D11InputLayout* CreateInputLayout( ID3D11Device* pd3dDevice, const D3D11_INPUT_ELEMENT_DESC* pElements, const UINT ElementCount );

    UINT Render( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dDeviceContext, TitleResidencyManager* pResidencyManager, SceneObjectVector& SceneObjects, const XMMATRIX& matView, const XMMATRIX& matProjection );
    VOID SetPixelShader( ID3D11DeviceContext* pd3dDeviceContext, TitleResidencyManager* pResidencyManager, const UINT ResourceSetID );
}
