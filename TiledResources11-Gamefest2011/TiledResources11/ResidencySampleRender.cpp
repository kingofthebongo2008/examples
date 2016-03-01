//--------------------------------------------------------------------------------------
// ResidencySampleRender.cpp
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "ResidencySampleRender.h"
#include "Util.h"

namespace ResidencySampleRender
{
    // D3D11 objects that are used by the residency sample render methods:
    ID3D11VertexShader* g_pVertexShaderTransform = NULL;
    ID3D10Blob* g_pVertexShaderBlob = NULL;
    ID3D11PixelShader* g_pPixelShaderResidencySample = NULL;
    ID3D11Buffer* g_pVertexCB = NULL;
    ID3D11Buffer* g_pPixelCB = NULL;

    // The default render width is 1280 pixels:
    FLOAT g_RenderWidth = 1280.0f;

    //--------------------------------------------------------------------------------------
    // Name: CB_VertexShader
    // Desc: Struct that matches the constant buffer for the residency sample rendering
    //       vertex shader.
    //--------------------------------------------------------------------------------------
    struct CB_VertexShader
    {
        XMFLOAT4X4 matWVP;
    };

    //--------------------------------------------------------------------------------------
    // Name: CB_PixelShader
    // Desc: Struct that matches the constant buffer for the residency sample rendering
    //       pixel shader.
    //--------------------------------------------------------------------------------------
    struct CB_PixelShader
    {
        XMFLOAT4 vResidencyConstant;
    };

    //--------------------------------------------------------------------------------------
    // Name: Initialize
    // Desc: Creates the D3D11 objects for residency sample rendering.
    //--------------------------------------------------------------------------------------
    VOID Initialize( ID3D11Device* pd3dDevice )
    {
        // Load vertex and pixel shaders:
        g_pVertexShaderTransform = CompileVertexShader( pd3dDevice, L"TiledResources.hlsl", "VSResidencyTransform", &g_pVertexShaderBlob );
        g_pPixelShaderResidencySample = CompilePixelShader( pd3dDevice, L"TiledResources.hlsl", "PSResidencySampleTex0" );

        // Create the constant buffers:
        g_pVertexCB = CreateConstantBuffer( pd3dDevice, sizeof(CB_VertexShader) );
        g_pPixelCB = CreateConstantBuffer( pd3dDevice, sizeof(CB_PixelShader) );
    }

    //--------------------------------------------------------------------------------------
    // Name: Terminate
    // Desc: Releases all of the D3D11 objects:
    //--------------------------------------------------------------------------------------
    VOID Terminate()
    {
        SAFE_RELEASE( g_pVertexShaderTransform );
        SAFE_RELEASE( g_pPixelShaderResidencySample );
        SAFE_RELEASE( g_pVertexShaderBlob );
        SAFE_RELEASE( g_pVertexCB );
        SAFE_RELEASE( g_pPixelCB );
    }

    //--------------------------------------------------------------------------------------
    // Name: ResizeRenderView
    // Desc: This method is called when the render view is resized.  The width is stored for
    //       per-frame computations.
    //--------------------------------------------------------------------------------------
    VOID ResizeRenderView( UINT RenderWidth, UINT RenderHeight )
    {
        g_RenderWidth = (FLOAT)RenderWidth;
    }

    //--------------------------------------------------------------------------------------
    // Name: CreateInputLayout
    // Desc: Creates an input layout for the given input elements, based on the residency
    //       sample render vertex shader.
    //--------------------------------------------------------------------------------------
    ID3D11InputLayout* CreateInputLayout( ID3D11Device* pd3dDevice, const D3D11_INPUT_ELEMENT_DESC* pElements, const UINT ElementCount )
    {
        ID3D11InputLayout* pInputLayout = NULL;
        pd3dDevice->CreateInputLayout( pElements, ElementCount, g_pVertexShaderBlob->GetBufferPointer(), g_pVertexShaderBlob->GetBufferSize(), &pInputLayout );
        return pInputLayout;
    }

    //--------------------------------------------------------------------------------------
    // Name: Render
    // Desc: Renders a residency sample view of the given scene objects, using the given
    //       view and projection matrices.
    //--------------------------------------------------------------------------------------
    UINT Render( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dDeviceContext, TitleResidencyManager* pResidencyManager, SceneObjectVector& SceneObjects, const XMMATRIX& matView, const XMMATRIX& matProjection )
    {
        DXUT_BeginPerfEvent( 0, L"Residency Sample View" );

        // Get a new residency sample view from the title residency manager, which also sets the rendertargets:
        UINT ViewID = pResidencyManager->BeginView( pd3dDeviceContext, g_RenderWidth );

        // Compute a view projection matrix:
        const FLOAT fFOVScaling = 1.0f;
        XMMATRIX matCameraVP = matView * matProjection;
        matCameraVP = matCameraVP * XMMatrixScaling( fFOVScaling, fFOVScaling, 1.0f );

        // Iterate through scene objects:
        DWORD dwVisibleCount = (DWORD)SceneObjects.size();
        for( DWORD dwModelIndex = 0; dwModelIndex < dwVisibleCount; ++dwModelIndex )
        {
            SceneObject* pSceneObject = SceneObjects[dwModelIndex];

            // Compute world * view * projection matrix for this model and set into constants.
            XMMATRIX matWVP = XMLoadFloat4x4( &pSceneObject->matWorld ) * matCameraVP;
            matWVP = XMMatrixTranspose( matWVP );

            // Update the vertex constant buffer:
            D3D11_MAPPED_SUBRESOURCE MapData;
            pd3dDeviceContext->Map( g_pVertexCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MapData );
            CB_VertexShader* pCBVertex = (CB_VertexShader*)MapData.pData;
            XMStoreFloat4x4( &pCBVertex->matWVP, matWVP );
            pd3dDeviceContext->Unmap( g_pVertexCB, 0 );

            pd3dDeviceContext->VSSetConstantBuffers( 0, 1, &g_pVertexCB );

            // Set the input layout and vertex shader:
            pd3dDeviceContext->IASetInputLayout( pSceneObject->m_pLayoutResidencySample );
            pd3dDeviceContext->VSSetShader( g_pVertexShaderTransform, NULL, 0 );

            // Set the pixel shader and pixel shader constant buffer, using the scene object's resource set ID:
            SetPixelShader( pd3dDeviceContext, pResidencyManager, pSceneObject->RSID );

            // Set the vertex buffer and primitive topology:
            UINT Stride = pSceneObject->VertexStrideBytes;
            UINT Offset = 0;
            pd3dDeviceContext->IASetVertexBuffers( 0, 1, &pSceneObject->pVertexBuffer, &Stride, &Offset );
            pd3dDeviceContext->IASetPrimitiveTopology( pSceneObject->PrimitiveType );

            // Draw the object:
            if( pSceneObject->pIndexBuffer != NULL )
            {
                pd3dDeviceContext->IASetIndexBuffer( pSceneObject->pIndexBuffer, DXGI_FORMAT_R16_UINT, 0 );
                pd3dDeviceContext->DrawIndexed( pSceneObject->IndexCount, 0, 0 );
            }
            else
            {
                pd3dDeviceContext->Draw( pSceneObject->VertexCount, 0 );
            }
        }

        // End the residency sample view, releasing it to be processed next frame:
        pResidencyManager->EndView( pd3dDeviceContext, ViewID );

        DXUT_EndPerfEvent();

        return ViewID;
    }

    //--------------------------------------------------------------------------------------
    // Name: SetPixelShader
    // Desc: Sets the residency sample view pixel shader and the pixel shader constant buffer,
    //       which incorporates the given resource set ID:
    //--------------------------------------------------------------------------------------
    VOID SetPixelShader( ID3D11DeviceContext* pd3dDeviceContext, TitleResidencyManager* pResidencyManager, const UINT ResourceSetID )
    {
        // Create a resource shader constant using the current resource set ID:
        XMFLOAT4 ResidencyConstant;
        pResidencyManager->CreateResourceConstant( ResourceSetID, &ResidencyConstant );

        // Update the pixel constant buffer:
        D3D11_MAPPED_SUBRESOURCE MapData;
        pd3dDeviceContext->Map( g_pPixelCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MapData );
        CB_PixelShader* pCBPixel = (CB_PixelShader*)MapData.pData;
        pCBPixel->vResidencyConstant = ResidencyConstant;
        pd3dDeviceContext->Unmap( g_pPixelCB, 0 );
        pd3dDeviceContext->PSSetConstantBuffers( 0, 1, &g_pPixelCB );
        
        // Set the pixel shader:
        pd3dDeviceContext->PSSetShader( g_pPixelShaderResidencySample, NULL, 0 );
    }
}
