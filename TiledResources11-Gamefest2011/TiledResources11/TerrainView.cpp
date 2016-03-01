//--------------------------------------------------------------------------------------
// TerrainView.cpp
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "TerrainView.h"
#include "Util.h"
#include "ResidencySampleRender.h"

//--------------------------------------------------------------------------------------
// Name: TerrainVertex
// Desc: Struct that describes a single vertex in the grid vertex buffer.
//--------------------------------------------------------------------------------------
struct TerrainVertex
{
    XMFLOAT2 Position;
    XMFLOAT2 Tex;
};

//--------------------------------------------------------------------------------------
// Name: CBVertex
// Desc: Struct that matches the vertex shader constant buffer.
//--------------------------------------------------------------------------------------
struct CBVertex
{
    XMFLOAT4X4 matWVP;
    XMFLOAT4 PositionTexScaleOffset;
    XMFLOAT4 HeightmapConstants;
};

//--------------------------------------------------------------------------------------
// Name: CBPixel
// Desc: Struct that matches the pixel shader constant buffer.
//--------------------------------------------------------------------------------------
struct CBPixel
{
    XMFLOAT4 LightDirectionWorld;
    XMFLOAT4 AmbientLight;
};

//--------------------------------------------------------------------------------------
// Name: TerrainView constructor
//--------------------------------------------------------------------------------------
TerrainView::TerrainView( ID3D11Device* pd3dDevice, ID3D11TiledResourceDevice* pd3dDeviceEx, TitleResidencyManager* pTRM )
{
    m_pd3dDevice = pd3dDevice;
    m_pd3dDevice->AddRef();
    m_pd3dDeviceEx = pd3dDeviceEx;
    m_pd3dDeviceEx->AddRef();

    m_pTRM = pTRM;

    // The default render width is 1280 pixels:
    m_RenderWidth = 1280.0f;

    HRESULT hr;

    // Compile shaders:
    ID3D10Blob* pVSBlob = NULL;
    m_pVSTerrain = CompileVertexShader( pd3dDevice, L"SceneRender.hlsl", "VSTerrain", &pVSBlob );
    m_pPSRender = CompilePixelShader( pd3dDevice, L"SceneRender.hlsl", "PSTerrainRender" );

    // Create input layout:
    const D3D11_INPUT_ELEMENT_DESC VertexLayout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT,    0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0,  8, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    hr = pd3dDevice->CreateInputLayout( VertexLayout, ARRAYSIZE(VertexLayout), pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), &m_pInputLayout );
    ASSERT( SUCCEEDED(hr) );

    SAFE_RELEASE( pVSBlob );

    // Create constant buffers:
    m_pCBVertex = CreateConstantBuffer( pd3dDevice, sizeof(CBVertex) );
    m_pCBPixel = CreateConstantBuffer( pd3dDevice, sizeof(CBPixel) );

    // Create geometry for a single grid mesh:
    CreateGeometry();

    // Create renderstate:
    D3D11_BLEND_DESC BlendDesc;
    ZeroMemory( &BlendDesc, sizeof( D3D11_BLEND_DESC ) );
    BlendDesc.RenderTarget[0].BlendEnable = FALSE;
    BlendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    BlendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    BlendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    BlendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_SRC_ALPHA;
    BlendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    BlendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    BlendDesc.RenderTarget[0].RenderTargetWriteMask = D3D10_COLOR_WRITE_ENABLE_ALL;
    hr = pd3dDevice->CreateBlendState( &BlendDesc, &m_pBlendState );
    ASSERT( SUCCEEDED(hr) );

    D3D11_DEPTH_STENCIL_DESC DSDesc;
    ZeroMemory( &DSDesc, sizeof(D3D11_DEPTH_STENCIL_DESC) );
    DSDesc.DepthEnable = TRUE;
    DSDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    DSDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
    DSDesc.StencilEnable = FALSE;
    DSDesc.StencilReadMask = D3D11_DEFAULT_STENCIL_READ_MASK;
    DSDesc.StencilWriteMask = D3D11_DEFAULT_STENCIL_WRITE_MASK;
    hr = pd3dDevice->CreateDepthStencilState( &DSDesc, &m_pDepthStencilState );
    ASSERT( SUCCEEDED(hr) );

    D3D11_RASTERIZER_DESC RSDesc;
    ZeroMemory( &RSDesc, sizeof(RSDesc) );
    RSDesc.AntialiasedLineEnable = FALSE;
    RSDesc.CullMode = D3D11_CULL_NONE;
    RSDesc.DepthBias = 0;
    RSDesc.DepthBiasClamp = 0.0f;
    RSDesc.DepthClipEnable = TRUE;
    RSDesc.FillMode = D3D11_FILL_SOLID;
    RSDesc.FrontCounterClockwise = FALSE;
    RSDesc.MultisampleEnable = TRUE;
    RSDesc.ScissorEnable = FALSE;
    RSDesc.SlopeScaledDepthBias = 0.0f;
    hr = pd3dDevice->CreateRasterizerState( &RSDesc, &m_pRasterizerState );
    ASSERT( SUCCEEDED(hr) );

    // Load the tiled textures from files on disk:
    LoadTiledTextureFile( &m_DiffuseMapTexture, L"s_diffuse.sp" );
    LoadTiledTextureFile( &m_NormalMapTexture, L"s_normalmap.sp" );
    LoadTiledTextureFile( &m_HeightMapTexture, L"s_heightmap.sp" );

    if( IsLoaded() )
    {
        // Create a resource set ID from the three tiled textures:
        const ID3D11TiledTexture2D* ppResources[] = { m_DiffuseMapTexture.pTexture, m_NormalMapTexture.pTexture, m_HeightMapTexture.pTexture };
        ITileLoader* ppTileLoaders[] = { m_DiffuseMapTexture.pTileLoader, m_NormalMapTexture.pTileLoader, m_HeightMapTexture.pTileLoader };
        m_RSID = m_pTRM->CreateResourceSet( ppResources, ppTileLoaders, ARRAYSIZE(ppResources) );
    }
    else
    {
        m_RSID = 0;
    }

    // Create the world transform matrix, scaling one centimeter in the terrain to one texel:
    FLOAT WorldScaling = 163.84f;
    XMStoreFloat4x4( &m_matWorld, XMMatrixScalingFromVector( XMVectorReplicate( WorldScaling ) ) );

    // Create the default camera angle and default camera settings:
    D3DXVECTOR3 EyePos( 50.0f, 70.0f, 250.0f );
    D3DXVECTOR3 TargetPos( WorldScaling * 0.5f, 0.0f, WorldScaling * 0.5f );
    m_Camera.SetRotateButtons( true, false, false );
    m_Camera.SetEnablePositionMovement( true );
    m_Camera.SetViewParams( &EyePos, &TargetPos );
    m_Camera.SetScalers( 0.005f, 20.0f );
    m_Camera.SetInvertPitch( true );
}

//--------------------------------------------------------------------------------------
// Name: TerrainView::CreateGeometry
// Desc: Creates a grid vertex buffer and index buffer.
//--------------------------------------------------------------------------------------
VOID TerrainView::CreateGeometry()
{
    // Set the mesh 1D size:
    m_MeshGridSize = 32;

    // Set the overall terrain layout in grid tiles:
    m_LayoutGridWidth = 8;
    m_LayoutGridHeight = 8;

    const UINT MeshVertexCount = ( m_MeshGridSize + 1 ) * ( m_MeshGridSize + 1 );
    const UINT MeshIndexCount = m_MeshGridSize * m_MeshGridSize * 6;

    const FLOAT VertexDelta = 1.0f / (FLOAT)m_MeshGridSize;

    // Create the terrain vertices:
    TerrainVertex* pVerts = new TerrainVertex[MeshVertexCount];

    for( UINT y = 0; y < m_MeshGridSize + 1; ++y )
    {
        FLOAT YPos = (FLOAT)y * VertexDelta;
        YPos = min( 1.0f, YPos );

        for( UINT x = 0; x < m_MeshGridSize + 1; ++x )
        {
            UINT Index = y * (m_MeshGridSize + 1) + x;

            FLOAT XPos = (FLOAT)x * VertexDelta;
            XPos = min( 1.0f, XPos );

            pVerts[Index].Position.x = XPos;
            pVerts[Index].Position.y = YPos;
            pVerts[Index].Tex.x = XPos;
            pVerts[Index].Tex.y = YPos;
        }
    }

    // Create vertex buffer:
    m_pVBGrid = CreateVertexBuffer( m_pd3dDevice, MeshVertexCount * sizeof(TerrainVertex), pVerts );
    delete[] pVerts;

    // Create terrain indices:
    WORD* pIndices = new WORD[ MeshIndexCount ];
    WORD* pWriteIndex = pIndices;
    for( UINT y = 0; y < m_MeshGridSize; ++y )
    {
        for( UINT x = 0; x < m_MeshGridSize; ++x )
        {
            UINT UpperLeftIndex = y * ( m_MeshGridSize + 1 ) + x;
            UINT UpperRightIndex = UpperLeftIndex + 1;
            UINT LowerLeftIndex = UpperLeftIndex + m_MeshGridSize + 1;
            UINT LowerRightIndex = LowerLeftIndex + 1;

            *pWriteIndex++ = (WORD)UpperLeftIndex;
            *pWriteIndex++ = (WORD)UpperRightIndex;
            *pWriteIndex++ = (WORD)LowerLeftIndex;
            *pWriteIndex++ = (WORD)LowerLeftIndex;
            *pWriteIndex++ = (WORD)UpperRightIndex;
            *pWriteIndex++ = (WORD)LowerRightIndex;
        }
    }

    // Create index buffer:
    m_pIBGrid = CreateIndexBuffer( m_pd3dDevice, MeshIndexCount * sizeof(WORD), pIndices );
    delete[] pIndices;
}

//--------------------------------------------------------------------------------------
// Name: TerrainView::LoadTiledTextureFile
// Desc: Loads a single tiled texture resource from disk, and creates a bound tiled texture
//       from the resource file.
//--------------------------------------------------------------------------------------
HRESULT TerrainView::LoadTiledTextureFile( BoundTiledTexture* pTexture, const WCHAR* strFileName )
{
    // Create the tiled file loader:
    TiledFileLoader* pFileLoader = new TiledFileLoader();
    if( pFileLoader == NULL )
    {
        return E_OUTOFMEMORY;
    }

    // Load the file from disk:
    HRESULT hr = pFileLoader->LoadFile( strFileName );
    if( FAILED(hr) )
    {
        WCHAR strText[100];
        swprintf_s( strText, L"Could not load tiled texture resource file \"%s\".\n", strFileName );
        OutputDebugString( strText );

        delete pFileLoader;
        return hr;
    }

    // Create a tiled texture2D that corresponds to the file data:
    ID3D11TiledTexture2D* pTiledTexture = NULL;
    hr = pFileLoader->CreateTiledTexture2D( m_pd3dDeviceEx, &pTiledTexture );
    if( FAILED(hr) || pTiledTexture == NULL )
    {
        return E_FAIL;
    }

    // Create a shader resource view for the texture:
    ID3D11TiledShaderResourceView* pTiledSRV = NULL;
    hr = m_pd3dDeviceEx->CreateShaderResourceView( pTiledTexture, &pTiledSRV );
    if( FAILED(hr) || pTiledSRV == NULL )
    {
        return E_FAIL;
    }

    // Fill in the bound tiled texture struct:
    pTexture->pTexture = pTiledTexture;
    pTexture->pTextureSRV = pTiledSRV;
    pTexture->pTileLoader = pFileLoader;
    pTexture->DeleteTileLoader = TRUE;

    // Create a sampling quality manager for the tiled texture:
    pTexture->pSamplingQualityManager = new SamplingQualityManager( pTiledTexture, m_pd3dDevice, m_pd3dDeviceEx );

    // Register the sampling quality manager with the title residency manager:
    m_pTRM->RegisterTileActivityHandler( pTexture->pSamplingQualityManager );

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Name: TerrainView destructor
// Desc: Releases D3D11 objects and other members.
//--------------------------------------------------------------------------------------
TerrainView::~TerrainView(void)
{
    m_DiffuseMapTexture.Release();
    m_HeightMapTexture.Release();
    m_NormalMapTexture.Release();

    SAFE_RELEASE( m_pVBGrid );
    SAFE_RELEASE( m_pIBGrid );
    SAFE_RELEASE( m_pInputLayout );

    SAFE_RELEASE( m_pDepthStencilState );
    SAFE_RELEASE( m_pRasterizerState );
    SAFE_RELEASE( m_pBlendState );

    SAFE_RELEASE( m_pCBVertex );
    SAFE_RELEASE( m_pCBPixel );
    SAFE_RELEASE( m_pVSTerrain );
    SAFE_RELEASE( m_pPSRender );

    SAFE_RELEASE( m_pd3dDeviceEx );
    SAFE_RELEASE( m_pd3dDevice );
}

//--------------------------------------------------------------------------------------
// Name: TerrainView::IsLoaded
// Desc: Returns TRUE if all three tiled textures were loaded succcessfully, FALSE otherwise.
//--------------------------------------------------------------------------------------
BOOL TerrainView::IsLoaded() const
{
    return m_DiffuseMapTexture.pTexture != NULL &&
           m_HeightMapTexture.pTexture != NULL &&
           m_NormalMapTexture.pTexture != NULL;
}

//--------------------------------------------------------------------------------------
// Name: TerrainView::Update
// Desc: Updates the camera for the terrain view, and builds camera matrices.
//--------------------------------------------------------------------------------------
VOID TerrainView::Update( FLOAT DeltaTime )
{
    // Update the camera:
    m_Camera.FrameMove( DeltaTime );

    // Get the camera matrices:
    XMMATRIX matCameraView = XMLoadFloat4x4( (const XMFLOAT4X4*)m_Camera.GetViewMatrix() );
    XMMATRIX matCameraProj = XMLoadFloat4x4( (const XMFLOAT4X4*)m_Camera.GetProjMatrix() );

    // Compose a world view projection for the terrain view:
    XMMATRIX matTerrainWorld = XMLoadFloat4x4( &m_matWorld );
    XMMATRIX matWVP = matTerrainWorld * matCameraView * matCameraProj;
    XMStoreFloat4x4( &m_matWorldViewProjection, XMMatrixTranspose( matWVP ) );
}

//--------------------------------------------------------------------------------------
// Name: TerrainView::PreFrameRender
// Desc: Renders the residency sample view for the terrain, as well as updating the
//       sampling quality managers.
//--------------------------------------------------------------------------------------
VOID TerrainView::PreFrameRender( ID3D11DeviceContext* pd3dContext, FLOAT DeltaTime )
{
    // Exit early if the tiled textures are not loaded:
    if( !IsLoaded() )
    {
        return;
    }

    DXUT_BeginPerfEvent( 0, L"Residency Sample View" );

    // Render the residency sample view:
    UINT ViewID = m_pTRM->BeginView( pd3dContext, m_RenderWidth );

    // Set up the pixel shader, using the residency sample render module:
    ResidencySampleRender::SetPixelShader( pd3dContext, m_pTRM, m_RSID );

    // Set up the vertex shader and state and draw the terrain:
    SetupVSAndRender( pd3dContext );

    m_pTRM->EndView( pd3dContext, ViewID );

    DXUT_EndPerfEvent();

    // Update the sampling quality managers for each tiled texture:
    if( m_DiffuseMapTexture.pSamplingQualityManager != NULL )
    {
        m_DiffuseMapTexture.pSamplingQualityManager->Render( pd3dContext, m_pd3dDeviceEx, DeltaTime );
    }
    if( m_NormalMapTexture.pSamplingQualityManager != NULL )
    {
        m_NormalMapTexture.pSamplingQualityManager->Render( pd3dContext, m_pd3dDeviceEx, DeltaTime );
    }
    if( m_HeightMapTexture.pSamplingQualityManager != NULL )
    {
        m_HeightMapTexture.pSamplingQualityManager->Render( pd3dContext, m_pd3dDeviceEx, DeltaTime );
    }
}

//--------------------------------------------------------------------------------------
// Name: TerrainView::Render
// Desc: Renders the terrain, using the tiled textures.
//--------------------------------------------------------------------------------------
VOID TerrainView::Render( ID3D11DeviceContext* pd3dContext )
{
    DXUT_BeginPerfEvent( 0, L"Terrain Render" );

    // Set the default rendertarget and depth stencil views:
    DXUTSetupD3D11Views( pd3dContext );

    // Get the viewport and record the render width:
    D3D11_VIEWPORT Viewport;
    UINT NumViewports = 1;
    pd3dContext->RSGetViewports( &NumViewports, &Viewport );
    m_RenderWidth = Viewport.Width;

    ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
    ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();

    // Clear the views:
    float ClearColor[4] = { 0, 0, 0, 1 };
    pd3dContext->ClearRenderTargetView( pRTV, ClearColor );
    pd3dContext->ClearDepthStencilView( pDSV, D3D11_CLEAR_DEPTH, 1.0, 0 );

    // Exit early if the tiled textures are not loaded:
    if( !IsLoaded() )
    {
        DXUT_EndPerfEvent();
        return;
    }

    // Update the pixel constant buffer:
    D3D11_MAPPED_SUBRESOURCE MapData;
    pd3dContext->Map( m_pCBPixel, 0, D3D11_MAP_WRITE_DISCARD, 0, &MapData );
    CBPixel* pCBPixel = (CBPixel*)MapData.pData;
    pCBPixel->AmbientLight = XMFLOAT4( 0.01f, 0.01f, 0.01f, 0.01f );
    XMStoreFloat4( &pCBPixel->LightDirectionWorld, XMVector3Normalize( XMVectorSet( 1, -2, 0, 0 ) ) );
    pd3dContext->Unmap( m_pCBPixel, 0 );
    pd3dContext->PSSetConstantBuffers( 0, 1, &m_pCBPixel );

    // Set renderstate:
    pd3dContext->RSSetState( m_pRasterizerState );
    const FLOAT BlendFactor[] = { 1, 1, 1, 1 };
    pd3dContext->OMSetBlendState( m_pBlendState, BlendFactor, 0xFFFFFFFF );
    pd3dContext->OMSetDepthStencilState( m_pDepthStencilState, 0 );

    // Set the pixel shader:
    pd3dContext->PSSetShader( m_pPSRender, NULL, 0 );

    // Set the diffuse and normal maps to the tiled pixel shader resource views:
    ID3D11TiledShaderResourceView* pPixelSRVs[] = { m_DiffuseMapTexture.pTextureSRV, m_NormalMapTexture.pTextureSRV };
    m_pd3dDeviceEx->PSSetShaderResources( 0, 2, pPixelSRVs );

    // Set the sampling quality map shader resource views to the pixel shader resource views:
    ID3D11ShaderResourceView* pQualitySRVs[] = { m_DiffuseMapTexture.pSamplingQualityManager->GetLODQualityTextureSRV(), m_NormalMapTexture.pSamplingQualityManager->GetLODQualityTextureSRV() };
    pd3dContext->PSSetShaderResources( 1, 2, pQualitySRVs );

    // Set the pixel sampler state for the sampling quality maps:
    ID3D11SamplerState* pQualitySS = SamplingQualityManager::GetSamplerState();
    ID3D11SamplerState* pQualitySSPS[] = { pQualitySS, pQualitySS };
    pd3dContext->PSSetSamplers( 1, 2, pQualitySSPS );

    // Set up vertex shader state and render the terrain:
    SetupVSAndRender( pd3dContext );

    DXUT_EndPerfEvent();
}

//--------------------------------------------------------------------------------------
// Name: TerrainView::SetupVSAndRender
// Desc: Sets up vertex shader state for rendering, and renders the terrain.  This
//       codepath is shared by the residency sample render and the scene render.
//--------------------------------------------------------------------------------------
VOID TerrainView::SetupVSAndRender( ID3D11DeviceContext* pd3dContext )
{
    // Set tiled shader resource views for the height map in the vertex shader:
    m_pd3dDeviceEx->VSSetShaderResources( 0, 1, &m_HeightMapTexture.pTextureSRV );

    // Set the shader resource view for the height map sampling quality map in the vertex shader:
    ID3D11ShaderResourceView* pSRV = m_HeightMapTexture.pSamplingQualityManager->GetLODQualityTextureSRV();
    pd3dContext->VSSetShaderResources( 1, 1, &pSRV );

    // Set the sampler state for the sampling quality manager in the vertex shader:
    ID3D11SamplerState* pQualitySS = SamplingQualityManager::GetSamplerState();
    pd3dContext->VSSetSamplers( 1, 1, &pQualitySS );

    // Set the vertex shader that samples the heightmap:
    pd3dContext->VSSetShader( m_pVSTerrain, NULL, 0 );

    // Set the vertex and index buffers, input layout, and primitive topology:
    UINT Stride = sizeof(TerrainVertex);
    UINT Offset = 0;
    pd3dContext->IASetVertexBuffers( 0, 1, &m_pVBGrid, &Stride, &Offset );
    pd3dContext->IASetIndexBuffer( m_pIBGrid, DXGI_FORMAT_R16_UINT, 0 );
    pd3dContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );
    pd3dContext->IASetInputLayout( m_pInputLayout );

    // Compute the width and height of a single grid mesh in UV space:
    XMFLOAT4 PositionScaleOffset;
    PositionScaleOffset.x = 1.0f / (FLOAT)m_LayoutGridWidth;
    PositionScaleOffset.y = 1.0f / (FLOAT)m_LayoutGridHeight;

    // Determine the number of indices in a single grid mesh:
    const UINT IndexCount = m_MeshGridSize * m_MeshGridSize * 6;

    D3D11_MAPPED_SUBRESOURCE MapData;

    // Loop over the grid mesh layout in 2 dimensions:
    for( UINT y = 0; y < m_LayoutGridHeight; ++y )
    {
        // The W component of the position scale offset is the column position in UV space:
        PositionScaleOffset.w = (FLOAT)y / (FLOAT)m_LayoutGridHeight;

        for( UINT x = 0; x < m_LayoutGridWidth; ++x )
        {
            // The Z component of the position scale offset is the row position in UV space:
            PositionScaleOffset.z = (FLOAT)x / (FLOAT)m_LayoutGridWidth;

            // Update the vertex constant buffer:
            pd3dContext->Map( m_pCBVertex, 0, D3D11_MAP_WRITE_DISCARD, 0, &MapData );
            CBVertex* pCBVertex = (CBVertex*)MapData.pData;

            pCBVertex->matWVP = m_matWorldViewProjection;
            pCBVertex->PositionTexScaleOffset = PositionScaleOffset;
            pCBVertex->HeightmapConstants = XMFLOAT4( 0.25f, 0, 0, 0 );

            pd3dContext->Unmap( m_pCBVertex, 0 );
            pd3dContext->VSSetConstantBuffers( 0, 1, &m_pCBVertex );

            // Draw a single grid mesh:
            pd3dContext->DrawIndexed( IndexCount, 0, 0 );
        }
    }
}