//--------------------------------------------------------------------------------------
// SamplingQualityManager.cpp
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "SamplingQualityManager.h"
#include "Util.h"

// D3D11 resources for sampling quality manager rendering:
ID3D11VertexShader* g_pVSQualityPassThru = NULL;
ID3D10Blob* g_pVSQualityPassThruBlob = NULL;
ID3D11PixelShader* g_pPSQualitySample = NULL;
ID3D11PixelShader* g_pPSQualitySampleArray = NULL;
ID3D11InputLayout* g_pQualityInputLayout = NULL;
ID3D11Buffer* g_pPixelCB = NULL;
ID3D11Buffer* g_pQualityVB = NULL;

ID3D11BlendState* g_pQualityBlendState = NULL;
ID3D11DepthStencilState* g_pQualityDepthStencilState = NULL;

struct CB_Pixel
{
    XMFLOAT4 vLODConstant;
};

// The texture format used for the sampling quality map:
const DXGI_FORMAT g_QualityTextureFormat = DXGI_FORMAT_R16_UNORM;

// The sampler state used by the scene rendering shaders to sample from the sampling quality map:
ID3D11SamplerState* SamplingQualityManager::s_pSamplerState = NULL;

//--------------------------------------------------------------------------------------
// Name: SamplingQualityManager constructor
//--------------------------------------------------------------------------------------
SamplingQualityManager::SamplingQualityManager( ID3D11TiledTexture2D* pResource, ID3D11Device* pd3dDevice, ID3D11TiledResourceDevice* pd3dDeviceEx )
{
    HRESULT hr;

    // Store a pointer to the tiled resource and increase its refcount:
    m_pResource = pResource;
    m_pResource->AddRef();

    // Create a shader resource view for the tiled resource:
    hr = pd3dDeviceEx->CreateShaderResourceView( m_pResource, &m_pResourceSRV );
    assert( SUCCEEDED(hr) );

    // Get the texture desc for the tiled texture:
    D3D11_TILED_TEXTURE2D_DESC TiledTexDesc;
    m_pResource->GetDesc( &TiledTexDesc );

    // Get the tile dimensions of the tiled resource's base level:
    D3D11_TILED_SURFACE_DESC BaseDesc;
    m_pResource->GetSubresourceDesc( 0, &BaseDesc );

    // Compute the width and height of the sampling quality map:
    const UINT TexWidth = BaseDesc.TileWidth;
    const UINT TexHeight = BaseDesc.TileHeight;

    D3D11_TEXTURE2D_DESC TexDesc;
    ZeroMemory( &TexDesc, sizeof(TexDesc) );
    TexDesc.ArraySize = TiledTexDesc.ArraySize;
    TexDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    TexDesc.CPUAccessFlags = 0;
    TexDesc.Format = g_QualityTextureFormat;
    TexDesc.Width = TexWidth;
    TexDesc.Height = TexHeight;
    TexDesc.MipLevels = 1;
    TexDesc.SampleDesc.Count = 1;
    TexDesc.Usage = D3D11_USAGE_DEFAULT;

    m_ArraySize = TexDesc.ArraySize;

    // Create the sampling quality map texture:
    pd3dDevice->CreateTexture2D( &TexDesc, NULL, &m_pTileLODTextureSample );

    // Create a shader resource view for the sampling quality map:
    pd3dDevice->CreateShaderResourceView( m_pTileLODTextureSample, NULL, &m_pTileLODTextureSRV );

    // Create another sampling quality map texture for rendering:
    pd3dDevice->CreateTexture2D( &TexDesc, NULL, &m_pTileLODTextureRender );

    // Create render target views for the sampling quality map texture:
    if( TexDesc.ArraySize <= 1 )
    {
        pd3dDevice->CreateRenderTargetView( m_pTileLODTextureRender, NULL, &m_pTileLODSurface );
        m_ppTileLODSurfaceArray = NULL;
    }
    else
    {
        m_pTileLODSurface = NULL;
        m_ppTileLODSurfaceArray = new ID3D11RenderTargetView*[TexDesc.ArraySize];
        for( UINT i = 0; i < TexDesc.ArraySize; ++i )
        {
            D3D11_RENDER_TARGET_VIEW_DESC RTDesc;
            ZeroMemory( &RTDesc, sizeof(RTDesc) );
            RTDesc.Format = DXGI_FORMAT_UNKNOWN;
            RTDesc.Texture2DArray.ArraySize = 1;
            RTDesc.Texture2DArray.FirstArraySlice = i;
            RTDesc.Texture2DArray.MipSlice = 0;
            RTDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;

            hr = pd3dDevice->CreateRenderTargetView( m_pTileLODTextureRender, &RTDesc, &m_ppTileLODSurfaceArray[i] );
            assert( SUCCEEDED(hr) );
        }
    }

    m_MipTransitionDuration = MIP_TRANSITION_TIME_SECONDS;

    // Create the countdown timers for slice updates:
    m_pSliceChangingTime = new FLOAT[m_ArraySize];

    // Initialize the countdown timers for initial activity:
    for( UINT i = 0; i < m_ArraySize; ++i )
    {
        m_pSliceChangingTime[i] = m_MipTransitionDuration * 10.0f;
    }

    // Create a viewport for rendering to the sampling quality map:
    m_Viewport.TopLeftX = 0;
    m_Viewport.TopLeftY = 0;
    m_Viewport.Width = (FLOAT)TexWidth;
    m_Viewport.Height = (FLOAT)TexHeight;
    m_Viewport.MinDepth = 0;
    m_Viewport.MaxDepth = 1;

    // Create D3D11 resources for rendering:
    if( g_pVSQualityPassThru == NULL )
    {
        g_pVSQualityPassThru = CompileVertexShader( pd3dDevice, L"TiledResources.hlsl", "VSQualityPassThru", &g_pVSQualityPassThruBlob );
        g_pPSQualitySample = CompilePixelShader( pd3dDevice, L"TiledResources.hlsl", "PSQualitySample" );
        g_pPSQualitySampleArray = CompilePixelShader( pd3dDevice, L"TiledResources.hlsl", "PSQualitySampleArray" );

        const D3D11_INPUT_ELEMENT_DESC Layout[] =
        {
            { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT,    0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0,  8, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        };
        pd3dDevice->CreateInputLayout( Layout, ARRAYSIZE(Layout), g_pVSQualityPassThruBlob->GetBufferPointer(), g_pVSQualityPassThruBlob->GetBufferSize(), &g_pQualityInputLayout );
        SAFE_RELEASE( g_pVSQualityPassThruBlob );

        g_pPixelCB = CreateConstantBuffer( pd3dDevice, sizeof(CB_Pixel) );

        const FLOAT RectVertices[] =
        {
            -1,  1, 0, 0,
             1,  1, 1, 0,
            -1, -1, 0, 1,
             1, -1, 1, 1
        };

        g_pQualityVB = CreateVertexBuffer( pd3dDevice, sizeof(RectVertices), RectVertices );
    }

    // Create the sampler state for scene rendering to sample from the sampling quality map:
    if( s_pSamplerState == NULL )
    {
        D3D11_SAMPLER_DESC samDesc;
        ZeroMemory( &samDesc, sizeof(samDesc) );
        samDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
        samDesc.AddressU = samDesc.AddressV = samDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        samDesc.MaxAnisotropy = 1;
        samDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
        samDesc.MaxLOD = D3D11_FLOAT32_MAX;
        pd3dDevice->CreateSamplerState( &samDesc, &s_pSamplerState );
    }

    m_FirstFrame = TRUE;
}

//--------------------------------------------------------------------------------------
// Name: SamplingQualityManager destructor
// Desc: Release all D3D11 objects related to sampling quality rendering.
//--------------------------------------------------------------------------------------
SamplingQualityManager::~SamplingQualityManager()
{
    SAFE_RELEASE( m_pResource );
    SAFE_RELEASE( m_pResourceSRV );
    SAFE_RELEASE( m_pTileLODTextureSRV );
    SAFE_RELEASE( m_pTileLODSurface );
    SAFE_RELEASE( m_pTileLODTextureSample );
    SAFE_RELEASE( m_pTileLODTextureRender );
    if( m_ppTileLODSurfaceArray != NULL )
    {
        for( UINT i = 0; i < m_ArraySize; ++i )
        {
            SAFE_RELEASE( m_ppTileLODSurfaceArray[i] );
        }
        delete[] m_ppTileLODSurfaceArray;
    }
    delete[] m_pSliceChangingTime;

    SAFE_RELEASE( g_pVSQualityPassThru );
    SAFE_RELEASE( g_pPSQualitySample );
    SAFE_RELEASE( g_pPSQualitySampleArray );
    SAFE_RELEASE( g_pQualityInputLayout );
    SAFE_RELEASE( g_pPixelCB );
    SAFE_RELEASE( g_pQualityVB );
    SAFE_RELEASE( s_pSamplerState );
    SAFE_RELEASE( g_pQualityDepthStencilState );
    SAFE_RELEASE( g_pQualityBlendState );
}

//--------------------------------------------------------------------------------------
// Name: SamplingQualityManager::Render
// Desc: Renders a single update of the sampling quality map.
//--------------------------------------------------------------------------------------
VOID SamplingQualityManager::Render( ID3D11DeviceContext* pd3dDeviceContext, ID3D11TiledResourceDevice* pd3dDeviceEx, FLOAT fDeltaTime )
{
    DXUT_BeginPerfEvent( 0, L"Sampling Quality Manager" );

    // Compute the amount of LOD change that will be made during this update, using the delta time.
    FLOAT LODIncrement = fDeltaTime / m_MipTransitionDuration;
    LODIncrement = min( 0.5f, LODIncrement );
    LODIncrement = max( LODIncrement, 1.0f / 65536.0f );
    
    D3D11_TILED_TEXTURE2D_DESC TexDesc;
    m_pResource->GetDesc( &TexDesc );

    // Determine the maximum LOD for this resource:
    FLOAT MaxLOD = (FLOAT)( TexDesc.MipLevels - 1 );

    // Loop over the array slices in the tiled texture:
    for( UINT i = 0; i < m_ArraySize; ++i )
    {
        // If the countdown timer has expired, skip this slice:
        if( m_pSliceChangingTime[i] <= 0.0f && !m_FirstFrame )
        {
            continue;
        }

        // Decrement the countdown timer for this slice:
        m_pSliceChangingTime[i] = max( 0.0f, m_pSliceChangingTime[i] - fDeltaTime ); 

        // Build the LOD constant that will be passed to the pixel shader:
        XMFLOAT4 LODConstant( LODIncrement, MaxLOD, 0, 0 );

        // Select the proper pixel shader and render target view:
        ID3D11RenderTargetView* pSurface = m_pTileLODSurface;
        ID3D11PixelShader* pPixelShader = g_pPSQualitySample;
        if( m_ArraySize > 1 )
        {
            // We need to use a different shader and RT view for array textures:
            assert( m_ppTileLODSurfaceArray != NULL );
            pSurface = m_ppTileLODSurfaceArray[i];
            LODConstant.z = (FLOAT)i;
            pPixelShader = g_pPSQualitySampleArray;
        }

        // Set render target and viewport:
        pd3dDeviceContext->OMSetRenderTargets( 1, &pSurface, NULL );
        pd3dDeviceContext->RSSetViewports( 1, &m_Viewport );

        // If this is our very first update, clear the RT view with the maximum possible LOD value, and copy that
        // to the sampling texture:
        if( m_FirstFrame )
        {
            FLOAT ClearColor[4] = { 1, 1, 1, 1 };
            pd3dDeviceContext->ClearRenderTargetView( pSurface, ClearColor );
            pd3dDeviceContext->CopySubresourceRegion( m_pTileLODTextureSample, i, 0, 0, 0, m_pTileLODTextureRender, i, NULL );
        }

        // Set state for rendering:
        pd3dDeviceContext->OMSetBlendState( g_pQualityBlendState, NULL, 0xFFFFFFFF );
        pd3dDeviceContext->OMSetDepthStencilState( g_pQualityDepthStencilState, 0 );

        pd3dDeviceContext->IASetInputLayout( g_pQualityInputLayout );
        UINT Stride = sizeof(FLOAT) * 4;
        UINT Offset = 0;
        pd3dDeviceContext->IASetVertexBuffers( 0, 1, &g_pQualityVB, &Stride, &Offset );
        pd3dDeviceContext->IASetIndexBuffer( NULL, DXGI_FORMAT_UNKNOWN, 0 );
        pd3dDeviceContext->VSSetShader( g_pVSQualityPassThru, NULL, 0 );
        pd3dDeviceContext->PSSetShader( pPixelShader, NULL, 0 );

        pd3dDeviceContext->PSSetShaderResources( 0, 1, &m_pTileLODTextureSRV );
        pd3dDeviceContext->PSSetSamplers( 0, 1, &s_pSamplerState );

        pd3dDeviceEx->PSSetShaderResources( 0, 1, &m_pResourceSRV );

        // Update the pixel constant buffer:
        D3D11_MAPPED_SUBRESOURCE MapData;
        pd3dDeviceContext->Map( g_pPixelCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MapData );
        CB_Pixel* pCBPixel = (CB_Pixel*)MapData.pData;
        pCBPixel->vLODConstant = LODConstant;
        pd3dDeviceContext->Unmap( g_pPixelCB, 0 );
        pd3dDeviceContext->PSSetConstantBuffers( 0, 1, &g_pPixelCB );

        // Draw a quad that fills the viewport:
        pd3dDeviceContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );
        pd3dDeviceContext->Draw( 4, 0 );
        
        // Copy this slice of the rendering texture to the sampling texture:
        pd3dDeviceContext->CopySubresourceRegion( m_pTileLODTextureSample, i, 0, 0, 0, m_pTileLODTextureRender, i, NULL );
    }

    // Clear the first update flag:
    m_FirstFrame = FALSE;

    DXUT_EndPerfEvent();
}

//--------------------------------------------------------------------------------------
// Name: SamplingQualityManager::TileLoaded
// Desc: Listens for tile updates to the tiled texture that is attached to this sampling
//       quality manager.  If this tiled texture is being updated, increment the countdown
//       timer for the affected slice of this tiled texture.
//--------------------------------------------------------------------------------------
VOID SamplingQualityManager::TileLoaded( const TrackedTileID* pTileID )
{
    if( pTileID->pResource == m_pResource )
    {
        UINT SliceIndex = pTileID->ArraySlice;
        ASSERT( SliceIndex < m_ArraySize );
        m_pSliceChangingTime[SliceIndex] = max( m_pSliceChangingTime[SliceIndex], m_MipTransitionDuration * 3.0f );
    }
}

//--------------------------------------------------------------------------------------
// Name: SamplingQualityManager::TileUnloaded
// Desc: Listens for tile updates to the tiled texture that is attached to this sampling
//       quality manager.  If this tiled texture is being updated, increment the countdown
//       timer for the affected slice of this tiled texture.
//--------------------------------------------------------------------------------------
VOID SamplingQualityManager::TileUnloaded( const TrackedTileID* pTileID )
{
    if( pTileID->pResource == m_pResource )
    {
        UINT SliceIndex = pTileID->ArraySlice;
        ASSERT( SliceIndex < m_ArraySize );
        m_pSliceChangingTime[SliceIndex] = max( m_pSliceChangingTime[SliceIndex], m_MipTransitionDuration * 1.0f );
    }
}
