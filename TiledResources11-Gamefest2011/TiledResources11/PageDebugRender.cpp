//--------------------------------------------------------------------------------------
// PageDebugRender.cpp
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "PageDebugRender.h"
#include "Util.h"

// Vertex structures

struct VertexPos2Color
{
    XMFLOAT2 Position;
    XMFLOAT4 Color;
};

struct VertexPos2Tex2
{
    XMFLOAT2 Position;
    XMFLOAT2 TexCoord;
};

// Constant buffer struct for the vertex shader

struct VertexCB
{
    XMFLOAT4X4 matWVP;
};

#define MAX_ARRAY_SIZE 64
#define MAX_MIP_SIZE 10

//--------------------------------------------------------------------------------------
// Name: TileDebugRender::Initialize
// Desc: Sets up the tile debug render instance, including a lot of D3D11 resource
//       creation.
//--------------------------------------------------------------------------------------
VOID TileDebugRender::Initialize( ID3D11Device* pd3dDevice )
{
    m_QuadIndex = 0;
    m_MaxQuadCount = 400;
    m_MappedData.pData = NULL;
    m_MaxTileScore = 1024;
    m_MinTileScore = 0;

    m_pDynamicVB = CreateVertexBuffer( pd3dDevice, 6 * m_MaxQuadCount * sizeof(VertexPos2Color) );
    m_pVertexCB = CreateConstantBuffer( pd3dDevice, sizeof(VertexCB) );

    D3D11_BLEND_DESC BlendDesc;
    ZeroMemory( &BlendDesc, sizeof( D3D11_BLEND_DESC ) );
    BlendDesc.RenderTarget[0].BlendEnable = TRUE;
    BlendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    BlendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    BlendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    BlendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_SRC_ALPHA;
    BlendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    BlendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    BlendDesc.RenderTarget[0].RenderTargetWriteMask = D3D10_COLOR_WRITE_ENABLE_ALL;
    HRESULT hr = pd3dDevice->CreateBlendState( &BlendDesc, &m_pAlphaBlendState );
    ASSERT( SUCCEEDED(hr) );

    BlendDesc.RenderTarget[0].BlendEnable = FALSE;
    hr = pd3dDevice->CreateBlendState( &BlendDesc, &m_pSolidBlendState );
    ASSERT( SUCCEEDED(hr) );

    D3D11_DEPTH_STENCIL_DESC DSDesc;
    ZeroMemory( &DSDesc, sizeof(D3D11_DEPTH_STENCIL_DESC) );
    DSDesc.DepthEnable = FALSE;
    DSDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    DSDesc.DepthFunc = D3D11_COMPARISON_LESS;
    DSDesc.StencilEnable = FALSE;
    DSDesc.StencilReadMask = D3D11_DEFAULT_STENCIL_READ_MASK;
    DSDesc.StencilWriteMask = D3D11_DEFAULT_STENCIL_WRITE_MASK;
    hr = pd3dDevice->CreateDepthStencilState( &DSDesc, &m_pDepthStencilState );
    ASSERT( SUCCEEDED(hr) );

    D3D11_SAMPLER_DESC SamplerDesc;
    ZeroMemory( &SamplerDesc, sizeof(SamplerDesc) );
    SamplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    SamplerDesc.AddressU = SamplerDesc.AddressV = SamplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    SamplerDesc.MaxAnisotropy = 1;
    SamplerDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
    SamplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
    hr = pd3dDevice->CreateSamplerState( &SamplerDesc, &m_pSamplerPoint );
    ASSERT( SUCCEEDED(hr) );

    ID3D10Blob* pVSBlob = NULL;
    m_pVertexShaderTiles = CompileVertexShader( pd3dDevice, L"TiledResources.hlsl", "VSDebugRenderPages", &pVSBlob );
    m_pPixelShaderTiles = CompilePixelShader( pd3dDevice, L"TiledResources.hlsl", "PSDebugRenderPages" );

    const D3D11_INPUT_ELEMENT_DESC VertexLayout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT,       0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0,  8, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    hr = pd3dDevice->CreateInputLayout( VertexLayout, ARRAYSIZE(VertexLayout), pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), &m_pInputLayoutTiles );
    ASSERT( SUCCEEDED(hr) );

    SAFE_RELEASE( pVSBlob );

    m_pVertexShaderTexture = CompileVertexShader( pd3dDevice, L"TiledResources.hlsl", "VSDebugRenderTexture", &pVSBlob );
    m_pPixelShaderTexture = CompilePixelShader( pd3dDevice, L"TiledResources.hlsl", "PSDebugRenderTexture" );

    const D3D11_INPUT_ELEMENT_DESC VertexLayoutTexture[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0,  8, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    hr = pd3dDevice->CreateInputLayout( VertexLayoutTexture, ARRAYSIZE(VertexLayoutTexture), pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), &m_pInputLayoutTexture );
    ASSERT( SUCCEEDED(hr) );

    SAFE_RELEASE( pVSBlob );

    m_pMipLocationCache = new POINT[ MAX_ARRAY_SIZE * MAX_MIP_SIZE ];
}

//--------------------------------------------------------------------------------------
// Name: TileDebugRender::Terminate
// Desc: Releases all of the D3D11 resources.
//--------------------------------------------------------------------------------------
VOID TileDebugRender::Terminate()
{
    SAFE_RELEASE( m_pVertexShaderTiles );
    SAFE_RELEASE( m_pPixelShaderTiles );
    SAFE_RELEASE( m_pInputLayoutTiles );
    SAFE_RELEASE( m_pVertexShaderTexture );
    SAFE_RELEASE( m_pPixelShaderTexture );
    SAFE_RELEASE( m_pInputLayoutTexture );
    SAFE_RELEASE( m_pDynamicVB );
    SAFE_RELEASE( m_pVertexCB );
    SAFE_RELEASE( m_pAlphaBlendState );
    SAFE_RELEASE( m_pSolidBlendState );
    SAFE_RELEASE( m_pDepthStencilState );
    SAFE_RELEASE( m_pSamplerPoint );

    SAFE_DELETE_ARRAY( m_pMipLocationCache );
}

//--------------------------------------------------------------------------------------
// Name: TileDebugRender::StartQuads
// Desc: Locks the dynamic vertex buffer for writing, and stores the data pointer for
//       writing by AddQuad. Note that this method makes sure that we're starting our
//       write at the beginning of the buffer.
//--------------------------------------------------------------------------------------
VOID TileDebugRender::StartQuads( ID3D11DeviceContext* pd3dDeviceContext )
{
    ASSERT( m_MappedData.pData == NULL );
    ASSERT( m_pDynamicVB != NULL );
    ASSERT( m_QuadIndex == 0 );

    pd3dDeviceContext->Map( m_pDynamicVB, 0, D3D11_MAP_WRITE_DISCARD, 0, &m_MappedData );
    ASSERT( m_MappedData.pData != NULL );

    m_QuadIndex = 0;
}

//--------------------------------------------------------------------------------------
// Name: TileDebugRender::FlushQuads
// Desc: Draws the accumulated list of quads that is in the dynamic VB.
//--------------------------------------------------------------------------------------
VOID TileDebugRender::FlushQuads( ID3D11DeviceContext* pd3dDeviceContext )
{
    ASSERT( m_MappedData.pData != NULL );
    ASSERT( m_pDynamicVB != NULL );

    UINT QuadsToDraw = m_QuadIndex;

    pd3dDeviceContext->Unmap( m_pDynamicVB, 0 );
    m_MappedData.pData = NULL;
    m_QuadIndex = 0;

    if( QuadsToDraw == 0 )
    {
        return;
    }

    D3D11_VIEWPORT Viewport;
    UINT NumViewports = 1;
    pd3dDeviceContext->RSGetViewports( &NumViewports, &Viewport );

    // Set up shader constants
    D3D11_MAPPED_SUBRESOURCE MappedCB;
    pd3dDeviceContext->Map( m_pVertexCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedCB );
    VertexCB* pCB = (VertexCB*)MappedCB.pData;
    XMMATRIX Scaling = XMMatrixScaling( 2.0f / (FLOAT)Viewport.Width, -2.0f / (FLOAT)Viewport.Height, 0 );
    Scaling *= XMMatrixTranslation( -1.0f, 1.0f, 0 );
    XMStoreFloat4x4( &pCB->matWVP, XMMatrixTranspose( Scaling ) );
    pd3dDeviceContext->Unmap( m_pVertexCB, 0 );
    pd3dDeviceContext->VSSetConstantBuffers( 0, 1, &m_pVertexCB );

    FLOAT BlendFactor[4] = { 1, 1, 1, 1 };
    pd3dDeviceContext->OMSetBlendState( m_pAlphaBlendState, BlendFactor, 0xFFFFFFFF );

    pd3dDeviceContext->OMSetDepthStencilState( m_pDepthStencilState, 0 );

    pd3dDeviceContext->VSSetShader( m_pVertexShaderTiles, NULL, 0 );
    pd3dDeviceContext->PSSetShader( m_pPixelShaderTiles, NULL, 0 );
    pd3dDeviceContext->IASetInputLayout( m_pInputLayoutTiles );
    UINT Strides[] = { sizeof(VertexPos2Color) };
    UINT Offsets[] = { 0 };
    pd3dDeviceContext->IASetVertexBuffers( 0, 1, &m_pDynamicVB, Strides, Offsets );
    pd3dDeviceContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );

    pd3dDeviceContext->Draw( QuadsToDraw * 6, 0 );
}

//--------------------------------------------------------------------------------------
// Name: TileDebugRender::AddQuad
// Desc: Adds a single quad of the given size and color to the dynamic VB.  If the
//       dynamic VB is full, the current list of quads is flushed and the VB is re-locked
//       for adding fresh contents.
//--------------------------------------------------------------------------------------
VOID TileDebugRender::AddQuad( ID3D11DeviceContext* pd3dDeviceContext, INT X, INT Y, INT Width, INT Height, CXMVECTOR Color )
{
    if( m_QuadIndex >= m_MaxQuadCount )
    {
        FlushQuads( pd3dDeviceContext );
    }

    if( m_MappedData.pData == NULL )
    {
        StartQuads( pd3dDeviceContext );
    }

    VertexPos2Color* pVerts = (VertexPos2Color*)m_MappedData.pData;
    ASSERT( pVerts != NULL );
    ASSERT( m_QuadIndex < m_MaxQuadCount );

    pVerts += ( m_QuadIndex * 6 );

    const FLOAT Left = (FLOAT)X;
    const FLOAT Top = (FLOAT)Y;
    const FLOAT Right = Left + (FLOAT)Width;
    const FLOAT Bottom = Top + (FLOAT)Height;

    pVerts[0].Position = XMFLOAT2( Left, Top );
    pVerts[1].Position = XMFLOAT2( Right, Top );
    pVerts[2].Position = XMFLOAT2( Left, Bottom );
    pVerts[3].Position = XMFLOAT2( Left, Bottom );
    pVerts[4].Position = XMFLOAT2( Right, Top );
    pVerts[5].Position = XMFLOAT2( Right, Bottom );

    for( UINT i = 0; i < 6; ++i )
    {
        XMStoreFloat4( &pVerts[i].Color, Color );
    }

    ++m_QuadIndex;
}

//--------------------------------------------------------------------------------------
// Name: TileDebugRender::Render
// Desc: Draws a view of the given tiled texture's tile activity, at the given X and Y
//       coordinates onscreen.  Returns the height of the view and the height of one
//       slice within the view.
//--------------------------------------------------------------------------------------
VOID TileDebugRender::Render( ID3D11DeviceContext* pd3dContext, TitleResidencyManager* pTRM, ID3D11TiledTexture2D* pTexture2D, INT XPos, INT YPos, INT* pTotalHeight, INT* pSliceHeight )
{
    DXUT_BeginPerfEvent( 0, L"Tile Debug Render" );

    static const XMVECTOR ColorWhite = { 1, 1, 1, 1 };
    static const XMVECTOR ColorMipBackground = { 1, 1, 1, 0.1f };
    static const XMVECTOR ColorTileSeen = { 0.5f, 0.5f, 0.5f, 1.0f };
    static const XMVECTOR ColorTileLoading = { 1.0f, 1.0f, 0.0f, 1.0f };
    static const XMVECTOR ColorTileLoaded = { 0.5f, 1.0f, 0.5f, 1.0f };
    static const XMVECTOR ColorTileQueuedForUnmap = { 1.0f, 0.5f, 0.5f, 1.0f };
    static const XMVECTOR ColorTileUnmapped = { 1.0f, 0.0f, 0.0f, 1.0f };

    D3D11_TILED_TEXTURE2D_DESC TexDesc;
    pTexture2D->GetDesc( &TexDesc );

    INT SliceHeight = 0;

    const INT TileSize = 4;
    const INT StartMipSpacing = 10;

    for( UINT ArrayIndex = 0; ArrayIndex < TexDesc.ArraySize; ++ArrayIndex )
    {
        INT CursorY = YPos;
        INT CursorX = XPos;

        INT MipSpacing = StartMipSpacing;

        for( UINT MipIndex = 0; MipIndex < TexDesc.MipLevels; ++MipIndex )
        {
            UINT SubresourceIndex = ArrayIndex * TexDesc.MipLevels + MipIndex;

            D3D11_TILED_SURFACE_DESC SurfDesc;
            pTexture2D->GetSubresourceDesc( SubresourceIndex, &SurfDesc );

            if( MipIndex == 0 )
            {
                SliceHeight = SurfDesc.TileHeight * TileSize + MipSpacing;
                YPos += SliceHeight;
            }

            m_pMipLocationCache[SubresourceIndex].x = CursorX;
            m_pMipLocationCache[SubresourceIndex].y = CursorY;

            AddQuad( pd3dContext, CursorX, CursorY, SurfDesc.TileWidth * TileSize, SurfDesc.TileHeight * TileSize, ColorMipBackground );

            switch( MipIndex % 4 )
            {
            case 0:
                // next mip is to the right
                CursorX += ( SurfDesc.TileWidth * TileSize ) + MipSpacing;
                break;
            case 1:
                // next mip is below and left aligned with h center of current mip
                CursorX += ( SurfDesc.TileWidth * TileSize ) / 2;
                CursorY += ( SurfDesc.TileHeight * TileSize ) + MipSpacing;
                break;
            case 2:
                // next mip is to the left and top aligned v center of current mip
                CursorX -= ( SurfDesc.TileWidth * TileSize ) / 2 + MipSpacing;
                CursorY += ( SurfDesc.TileHeight * TileSize ) / 2;
                break;
            case 3:
                // next mip is above and left aligned with current mip
                CursorY -= ( SurfDesc.TileHeight * TileSize ) / 2 + MipSpacing;
                break;
            }

            MipSpacing = max( StartMipSpacing / 3, MipSpacing - 1 );
        }
    }

    INT CurrentMinScore = INT_MAX;
    INT CurrentMaxScore = INT_MIN;

    TitleResidencyManager::TrackedTileSortList::const_iterator iter = pTRM->GetTrackedTilesBegin();
    TitleResidencyManager::TrackedTileSortList::const_iterator end = pTRM->GetTrackedTilesEnd();
    while( iter != end )
    {
        const TitleResidencyManager::TrackedTile* pTP = *iter;
        ++iter;

        if( pTP->ID.pResource != pTexture2D )
        {
            continue;
        }

        UINT SubresourceIndex = pTP->ID.ArraySlice * TexDesc.MipLevels + pTP->ID.MipLevel;
        INT CursorX = m_pMipLocationCache[SubresourceIndex].x;
        INT CursorY = m_pMipLocationCache[SubresourceIndex].y;
        
        D3D11_TILED_SURFACE_DESC SurfDesc;
        pTexture2D->GetSubresourceDesc( SubresourceIndex, &SurfDesc );

        CursorX += (UINT)( pTP->ID.U * (FLOAT)SurfDesc.TileWidth ) * TileSize;
        CursorY += (UINT)( pTP->ID.V * (FLOAT)SurfDesc.TileHeight ) * TileSize;

        XMVECTOR Color = ColorTileSeen;
        switch( pTP->State )
        {
        case TitleResidencyManager::TPS_QueuedForLoad:
            {
                INT Score = (INT)pTP->CurrentPriority;
                if( Score >= 100000 )
                {
                    Score -= 100000;
                }

                CurrentMaxScore = max( CurrentMaxScore, Score );
                CurrentMinScore = min( CurrentMinScore, Score );

                Score -= m_MinTileScore;
                FLOAT PriorityIntensity = (FLOAT)Score / (FLOAT)( m_MaxTileScore - m_MinTileScore );

                Color = XMVectorSet( PriorityIntensity * 0.5f, PriorityIntensity * 0.5f, PriorityIntensity, 1.0f );
                break;                
            }
        case TitleResidencyManager::TPS_Loading:
        case TitleResidencyManager::TPS_Unmapping:
            Color = ColorTileLoading;
            break;
        case TitleResidencyManager::TPS_LoadedAndMapped:
            Color = ColorTileLoaded;
            break;
        case TitleResidencyManager::TPS_QueuedForUnmap:
            Color = ColorTileQueuedForUnmap;
            break;
        case TitleResidencyManager::TPS_Unmapped:
            Color = ColorTileUnmapped;
            break;
        }

        AddQuad( pd3dContext, CursorX, CursorY, TileSize, TileSize, Color );
    }

    m_MaxTileScore = CurrentMaxScore;
    m_MinTileScore = CurrentMinScore;

    FlushQuads( pd3dContext );

    DXUT_EndPerfEvent();

    if( pTotalHeight != NULL )
    {
        *pTotalHeight = YPos;
    }

    if( pSliceHeight != NULL )
    {
        *pSliceHeight = SliceHeight;
    }
}

//--------------------------------------------------------------------------------------
// Name: TileDebugRender::RenderTexture
// Desc: Draws the given shader resource on screen using ScreenRect as the coordinates.
//--------------------------------------------------------------------------------------
VOID TileDebugRender::RenderTexture( ID3D11DeviceContext* pd3dDeviceContext, ID3D11ShaderResourceView* pSRView, RECT ScreenRect )
{
    D3D11_MAPPED_SUBRESOURCE MapData;
    pd3dDeviceContext->Map( m_pDynamicVB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MapData );
    VertexPos2Tex2* pVerts = (VertexPos2Tex2*)MapData.pData;

    const FLOAT Left = (FLOAT)ScreenRect.left;
    const FLOAT Top = (FLOAT)ScreenRect.top;
    const FLOAT Right = Left + (FLOAT)( ScreenRect.right - ScreenRect.left );
    const FLOAT Bottom = Top + (FLOAT)( ScreenRect.bottom - ScreenRect.top );

    pVerts[0].Position = XMFLOAT2( Left, Top );
    pVerts[1].Position = XMFLOAT2( Right, Top );
    pVerts[2].Position = XMFLOAT2( Left, Bottom );
    pVerts[3].Position = XMFLOAT2( Right, Bottom );

    pVerts[0].TexCoord = XMFLOAT2( 0, 0 );
    pVerts[1].TexCoord = XMFLOAT2( 1, 0 );
    pVerts[2].TexCoord = XMFLOAT2( 0, 1 );
    pVerts[3].TexCoord = XMFLOAT2( 1, 1 );

    pd3dDeviceContext->Unmap( m_pDynamicVB, 0 );

    D3D11_VIEWPORT Viewport;
    UINT NumViewports = 1;
    pd3dDeviceContext->RSGetViewports( &NumViewports, &Viewport );

    // Set up shader constants
    D3D11_MAPPED_SUBRESOURCE MappedCB;
    pd3dDeviceContext->Map( m_pVertexCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedCB );
    VertexCB* pCB = (VertexCB*)MappedCB.pData;
    XMMATRIX Scaling = XMMatrixScaling( 2.0f / (FLOAT)Viewport.Width, -2.0f / (FLOAT)Viewport.Height, 0 );
    Scaling *= XMMatrixTranslation( -1.0f, 1.0f, 0 );
    XMStoreFloat4x4( &pCB->matWVP, XMMatrixTranspose( Scaling ) );
    pd3dDeviceContext->Unmap( m_pVertexCB, 0 );
    pd3dDeviceContext->VSSetConstantBuffers( 0, 1, &m_pVertexCB );

    FLOAT BlendFactor[4] = { 1, 1, 1, 1 };
    pd3dDeviceContext->OMSetBlendState( m_pSolidBlendState, BlendFactor, 0xFFFFFFFF );

    pd3dDeviceContext->OMSetDepthStencilState( m_pDepthStencilState, 0 );

    pd3dDeviceContext->VSSetShader( m_pVertexShaderTexture, NULL, 0 );
    pd3dDeviceContext->PSSetShader( m_pPixelShaderTexture, NULL, 0 );

    pd3dDeviceContext->PSSetShaderResources( 0, 1, &pSRView );
    pd3dDeviceContext->PSSetSamplers( 0, 1, &m_pSamplerPoint );

    pd3dDeviceContext->IASetInputLayout( m_pInputLayoutTexture );
    UINT Strides[] = { sizeof(VertexPos2Tex2) };
    UINT Offsets[] = { 0 };
    pd3dDeviceContext->IASetVertexBuffers( 0, 1, &m_pDynamicVB, Strides, Offsets );
    pd3dDeviceContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );

    pd3dDeviceContext->Draw( 4, 0 );
}