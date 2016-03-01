//--------------------------------------------------------------------------------------
// File: TiledResources11.cpp
//
// A Direct3D 11 sample that demonstrates an implementation of tiled resources, also
// known as virtual textures, megatextures, or partially resident textures.
//
// The sample has three pieces:
//
// 1. A software implementation of tiled resources, including new functionality implemented
// to look like the D3D runtime.  Under the hood, traditional D3D textures, constant buffers,
// and render state is used to simulate the presence of large and/or partially resident
// tiled resources.  The front-end to this system can be found in d3d11tiled.h.
//
// 2. A software implementation of tiled texture sampling, implemented in a library of
// HLSL subroutines.  These subroutines are found in TiledResourceEmulationLib.hlsl.
//
// 3. Title-space systems that serve as an example of the types of systems that a game
// developer would have to write and customize for their own titles, if they were to use
// tiled resources.  These systems perform the streaming and resource management for the
// tiled resources and their contents.  These systems include the title residency manager,
// the sampling quality manager, the residency view renderer, and the page loaders.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "DXUTgui.h"
#include "DXUTmisc.h"
#include "DXUTCamera.h"
#include "DXUTSettingsDlg.h"
#include "SDKmisc.h"
#include "SDKmesh.h"
#include "resource.h"

#include "d3d11tiled.h"
#include "TitleResidencyManager.h"
#include "ResidencySampleRender.h"
#include "SamplingQualityManager.h"
#include "PageLoaders.h"
#include "SceneObject.h"
#include "PageDebugRender.h"

#include "TerrainView.h"

#include "TiledResourceRuntimeTest.h"

#include "Util.h"

#pragma warning( disable: 4800 )

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
XMFLOAT4X4A g_matView;
XMFLOAT4X4A g_matProjection;
XMFLOAT4A   g_CameraPos = XMFLOAT4A( 0, 5.0f, 0.5f, 1 );

FLOAT g_HalfClientWidthPixels = 0;
FLOAT g_HalfClientHeightPixels = 0;

CDXUTDialogResourceManager  g_DialogResourceManager; // manager for shared resources of dialogs
CD3DSettingsDlg             g_SettingsDlg;          // Device settings dialog
CDXUTTextHelper*            g_pTxtHelper = NULL;
CDXUTDialog                 g_HUD;                  // dialog for standard controls
CDXUTDialog                 g_SampleUI;             // dialog for sample specific controls

// Direct3D 11 resources
ID3D11VertexShader*         g_pVSTransform = NULL;
ID3D11PixelShader*          g_pPSSceneRender = NULL;
ID3D11PixelShader*          g_pPSSceneRenderArray = NULL;
ID3D11PixelShader*          g_pPSSceneRenderQuilt = NULL;

ID3D11InputLayout*          g_pDefaultInputLayout = NULL;
ID3D11SamplerState*         g_pSamLinear = NULL;

ID3D11BlendState*           g_pBlendState = NULL;
ID3D11DepthStencilState*    g_pDepthStencilState = NULL;
ID3D11RasterizerState*      g_pRasterizerState = NULL;

ID3D11TiledResourceDevice*  g_pd3dDeviceEx = NULL;
ID3D11TilePool*             g_pTilePool = NULL;

TitleResidencyManager*      g_pTitleResidencyManager = NULL;
MandelbrotTileLoader        g_MandelbrotPageLoader;
ColorTileLoader             g_ColorTileLoader;

SceneObjectVector           g_SceneObjects;
ID3D11Buffer*               g_pQuadVB = NULL;
ID3D11Buffer*               g_pQuiltQuadVB = NULL;
ID3D11Buffer*               g_pArrayQuadsVB = NULL;

TerrainView*                g_pTerrainView = NULL;
BOOL                        g_bDrawTerrain = FALSE;

TileDebugRender             g_PageDebugRender;
ID3D11TiledTexture2D*       g_pInspectionTexture = NULL;
INT                         g_InspectionSliceIndex = 0;
UINT                        g_LastResidencySampleViewID = 0;
BOOL                        g_bDrawResidencySampleViews = FALSE;
BOOL                        g_bPauseStreaming = FALSE;

//--------------------------------------------------------------------------------------
// Constant buffers
//--------------------------------------------------------------------------------------
#pragma pack(push,1)
struct CB_VS_PER_OBJECT
{
    D3DXMATRIX  m_mWorldViewProjection;
};
#pragma pack(pop)

ID3D11Buffer*                       g_pcbVSPerObject11 = NULL;

//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
#define IDC_TOGGLEFULLSCREEN        1
#define IDC_TOGGLEREF               2
#define IDC_CHANGEDEVICE            3
#define IDC_TOGGLERESIDENCYVIEWS    4
#define IDC_TOGGLETERRAINVIEW       5
#define IDC_PAUSESTREAMING          6


//--------------------------------------------------------------------------------------
// Forward declarations 
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
                          void* pUserContext );
void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext );
void CALLBACK OnMouse( bool bLeftButtonDown, bool bRightButtonDown, bool bMiddleButtonDown, bool bSideButton1Down, bool bSideButton2Down, int nMouseWheelDelta, int xPos, int yPos, void* pUserContext );
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext );
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext );
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext );

bool CALLBACK IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo,
                                       DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext );
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc,
                                     void* pUserContext );
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
                                         const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext );
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext );
void CALLBACK OnD3D11DestroyDevice( void* pUserContext );
void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime,
                                 float fElapsedTime, void* pUserContext );

void InitApp();
VOID CreateSceneGeometry( ID3D11Device* pd3dDevice );
void RenderText();


//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
int WINAPI wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow )
{
    // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF | _CRTDBG_CHECK_ALWAYS_DF );
#endif

    // DXUT will create and use the best device (either D3D9 or D3D11) 
    // that is available on the system depending on which D3D callbacks are set below

    // Set DXUT callbacks
    DXUTSetCallbackMsgProc( MsgProc );
    DXUTSetCallbackKeyboard( OnKeyboard );
    DXUTSetCallbackMouse( OnMouse, true );
    DXUTSetCallbackFrameMove( OnFrameMove );
    DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );

    DXUTSetCallbackD3D11DeviceAcceptable( IsD3D11DeviceAcceptable );
    DXUTSetCallbackD3D11DeviceCreated( OnD3D11CreateDevice );
    DXUTSetCallbackD3D11SwapChainResized( OnD3D11ResizedSwapChain );
    DXUTSetCallbackD3D11SwapChainReleasing( OnD3D11ReleasingSwapChain );
    DXUTSetCallbackD3D11DeviceDestroyed( OnD3D11DestroyDevice );
    DXUTSetCallbackD3D11FrameRender( OnD3D11FrameRender );

    InitApp();
    DXUTInit( true, true, NULL ); // Parse the command line, show msgboxes on error, no extra command line params
    DXUTSetCursorSettings( true, true );

    WCHAR strWindowTitle[128] = L"TiledResources11";
#ifdef _DEBUG
    wcscat_s( strWindowTitle, L" [DEBUG]" );
#endif

    DXUTCreateWindow( strWindowTitle );

    // set the update & render thread to the first hardware thread
    SetThreadAffinityMask( GetCurrentThread(), 0x1 );

    // Only require 10-level hardware, change to D3D_FEATURE_LEVEL_11_0 to require 11-class hardware
    // Switch to D3D_FEATURE_LEVEL_9_x for 10level9 hardware
    DXUTCreateDevice( D3D_FEATURE_LEVEL_10_0, true, 1280, 720 );

    DXUTMainLoop(); // Enter into the DXUT render loop

    return DXUTGetExitCode();
}


VOID UpdateViewMatrix()
{
    XMVECTOR vCameraPos = XMLoadFloat4A( &g_CameraPos );
    XMVECTOR vCameraTarget = vCameraPos * XMVectorSet( 1, 0, 1, 1 );
    XMMATRIX matView = XMMatrixLookAtLH( vCameraPos, vCameraTarget, XMVectorSet( 0, 0, 1, 0 ) );
    XMStoreFloat4x4A( &g_matView, matView );
}


//--------------------------------------------------------------------------------------
// Initialize the app 
//--------------------------------------------------------------------------------------
void InitApp()
{
    g_SettingsDlg.Init( &g_DialogResourceManager );
    g_HUD.Init( &g_DialogResourceManager );
    g_SampleUI.Init( &g_DialogResourceManager );

    g_HUD.SetCallback( OnGUIEvent );
    int iY = 30;
    int iYo = 26;
    g_HUD.AddButton( IDC_TOGGLEFULLSCREEN, L"Toggle full screen", 0, iY, 170, 22 );
    g_HUD.AddButton( IDC_TOGGLEREF, L"Toggle REF (F3)", 0, iY += iYo, 170, 22, VK_F3 );
    g_HUD.AddButton( IDC_CHANGEDEVICE, L"Change device (F2)", 0, iY += iYo, 170, 22, VK_F2 );

    iY = 0;
    g_SampleUI.AddCheckBox( IDC_TOGGLERESIDENCYVIEWS, L"Show residency view render", 0, iY += iYo, 0, 22, g_bDrawResidencySampleViews );
    g_SampleUI.AddCheckBox( IDC_TOGGLETERRAINVIEW, L"Terrain render", 0, iY += iYo, 0, 22, g_bDrawTerrain );
    g_SampleUI.AddCheckBox( IDC_PAUSESTREAMING, L"Pause streaming", 0, iY += iYo, 0, 22, g_bPauseStreaming );

    g_SampleUI.SetCallback( OnGUIEvent );
}


//--------------------------------------------------------------------------------------
// Render the help and statistics text. This function uses the ID3DXFont interface for 
// efficient text rendering.
//--------------------------------------------------------------------------------------
void RenderText()
{
    g_pTxtHelper->Begin();
    g_pTxtHelper->SetInsertionPos( 5, 5 );
    g_pTxtHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 0.0f, 1.0f ) );
    g_pTxtHelper->DrawTextLine( DXUTGetFrameStats( DXUTIsVsyncEnabled() ) );
    g_pTxtHelper->DrawTextLine( DXUTGetDeviceStats() );

    WCHAR strText[200];

    const ResidencyStats& RS = g_pTitleResidencyManager->GetStats();
    swprintf_s( strText, L"Tile Tracker: %d tracked, %d loaded, %d queued for load, %d unused, %d loader threads%s", RS.NumTilesTracked, RS.NumTilesLoaded, RS.NumTilesQueuedForLoad, RS.NumTilesUnused, RS.LoaderThreadCount, RS.OutOfPhysicalTiles ? L", out of physical pages" : L"" );
    g_pTxtHelper->DrawTextLine( strText );

    D3D11_TILED_MEMORY_USAGE MemUsage;
    g_pTilePool->GetMemoryUsage( &MemUsage );
    swprintf_s( strText, L"%u resources, %0.3f GB resource VA space, %0.3f MB resource video mem, %d tile pools, %0.1f MB page video mem", 
        MemUsage.ResourceCount,
        (DOUBLE)MemUsage.ResourceVirtualBytesAllocated / (1073741824.0),
        (DOUBLE)MemUsage.ResourceTextureMemoryBytesAllocated / (1048576.0),
        MemUsage.FormatPoolsActive,
        (DOUBLE)MemUsage.TileTextureMemoryBytesAllocated / (1048576.0) );
    g_pTxtHelper->DrawTextLine( strText );

    if( g_pInspectionTexture != NULL )
    {
        D3D11_TILED_TEXTURE2D_DESC TexDesc;
        g_pInspectionTexture->GetDesc( &TexDesc );

        swprintf_s( strText, L"Tiled texture: format %S width %d height %d mip levels %d array size %d quilt width %d quilt height %d",
            TiledRuntimeTest::GetFormatName( TexDesc.Format ),
            TexDesc.Width,
            TexDesc.Height,
            TexDesc.MipLevels,
            TexDesc.ArraySize,
            TexDesc.QuiltWidth,
            TexDesc.QuiltHeight );

        g_pTxtHelper->DrawTextLine( strText );
    }

    if( g_bDrawTerrain && !g_pTerrainView->IsLoaded() )
    {
        g_pTxtHelper->DrawTextLine( L"Missing content for terrain view.  Please ensure s_diffuse.sp, s_heightmap.sp, and s_normalmap.sp are in the working directory." );
    }

    g_pTxtHelper->End();
}


//--------------------------------------------------------------------------------------
// Reject any D3D11 devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo,
                                       DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
{
    return true;
}


//--------------------------------------------------------------------------------------
// Create any D3D11 resources that aren't dependant on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc,
                                     void* pUserContext )
{
    DXUT_SetDebugName( pd3dDevice, "Main Device" );

    HRESULT hr;

    ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();
    V_RETURN( g_DialogResourceManager.OnD3D11CreateDevice( pd3dDevice, pd3dImmediateContext ) );
    V_RETURN( g_SettingsDlg.OnD3D11CreateDevice( pd3dDevice ) );
    g_pTxtHelper = new CDXUTTextHelper( pd3dDevice, pd3dImmediateContext, &g_DialogResourceManager, 15 );

    IDXGIDevice* pDXGIDevice;
    hr = pd3dDevice->QueryInterface( __uuidof(IDXGIDevice), (VOID**)&pDXGIDevice );
    if( SUCCEEDED(hr) )
    {
        IDXGIAdapter* pAdapter;
        hr = pDXGIDevice->GetAdapter( &pAdapter );
        if( SUCCEEDED(hr) )
        {
            DXGI_ADAPTER_DESC AdapterDesc;
            pAdapter->GetDesc( &AdapterDesc );
            SetAdapterInfoForShaderCompilation( AdapterDesc.Description );
            SAFE_RELEASE( pAdapter );
        }
        SAFE_RELEASE( pDXGIDevice );
    }

    ID3D10Blob* pVSBlob = NULL;
    g_pVSTransform = CompileVertexShader( pd3dDevice, L"SceneRender.hlsl", "VSTransform", &pVSBlob );
    g_pPSSceneRender = CompilePixelShader( pd3dDevice, L"SceneRender.hlsl", "PSSceneRender" );
    g_pPSSceneRenderArray = CompilePixelShader( pd3dDevice, L"SceneRender.hlsl", "PSSceneRenderArray" );
    g_pPSSceneRenderQuilt = CompilePixelShader( pd3dDevice, L"SceneRender.hlsl", "PSSceneRenderQuilt" );

    // Create a layout for the object data
    const D3D11_INPUT_ELEMENT_DESC layout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };

    V_RETURN( pd3dDevice->CreateInputLayout( layout, ARRAYSIZE( layout ), pVSBlob->GetBufferPointer(),
                                             pVSBlob->GetBufferSize(), &g_pDefaultInputLayout ) );

    // No longer need the shader blobs
    SAFE_RELEASE( pVSBlob );

    // Create state objects
    D3D11_SAMPLER_DESC samDesc;
    ZeroMemory( &samDesc, sizeof(samDesc) );
    samDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    samDesc.AddressU = samDesc.AddressV = samDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    samDesc.MaxAnisotropy = 1;
    samDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
    samDesc.MaxLOD = D3D11_FLOAT32_MAX;
    V_RETURN( pd3dDevice->CreateSamplerState( &samDesc, &g_pSamLinear ) );
    DXUT_SetDebugName( g_pSamLinear, "Linear" );

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
    hr = pd3dDevice->CreateBlendState( &BlendDesc, &g_pBlendState );
    ASSERT( SUCCEEDED(hr) );

    D3D11_DEPTH_STENCIL_DESC DSDesc;
    ZeroMemory( &DSDesc, sizeof(D3D11_DEPTH_STENCIL_DESC) );
    DSDesc.DepthEnable = FALSE;
    DSDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    DSDesc.DepthFunc = D3D11_COMPARISON_LESS;
    DSDesc.StencilEnable = FALSE;
    DSDesc.StencilReadMask = D3D11_DEFAULT_STENCIL_READ_MASK;
    DSDesc.StencilWriteMask = D3D11_DEFAULT_STENCIL_WRITE_MASK;
    hr = pd3dDevice->CreateDepthStencilState( &DSDesc, &g_pDepthStencilState );
    ASSERT( SUCCEEDED(hr) );

    D3D11_RASTERIZER_DESC RSDesc;
    ZeroMemory( &RSDesc, sizeof(RSDesc) );
    RSDesc.AntialiasedLineEnable = FALSE;
    RSDesc.CullMode = D3D11_CULL_BACK;
    RSDesc.DepthBias = 0;
    RSDesc.DepthBiasClamp = 0.0f;
    RSDesc.DepthClipEnable = TRUE;
    RSDesc.FillMode = D3D11_FILL_SOLID;
    RSDesc.FrontCounterClockwise = FALSE;
    RSDesc.MultisampleEnable = TRUE;
    RSDesc.ScissorEnable = FALSE;
    RSDesc.SlopeScaledDepthBias = 0.0f;
    hr = pd3dDevice->CreateRasterizerState( &RSDesc, &g_pRasterizerState );
    ASSERT( SUCCEEDED(hr) );

    g_pcbVSPerObject11 = CreateConstantBuffer( pd3dDevice, sizeof(CB_VS_PER_OBJECT) );
    DXUT_SetDebugName( g_pcbVSPerObject11, "CB_VS_PER_OBJECT" );

    // Create other render resources here
    D3D11_TILED_EMULATION_PARAMETERS EmulationParams;
    EmulationParams.DefaultPhysicalTileFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
    EmulationParams.MaxPhysicalTileCount = 1000;
    D3D11CreateTiledResourceDevice( pd3dDevice, pd3dImmediateContext, &EmulationParams, &g_pd3dDeviceEx );

    g_pd3dDeviceEx->CreateTilePool( &g_pTilePool );

    g_pTitleResidencyManager = new TitleResidencyManager( pd3dDevice, pd3dImmediateContext, 1, EmulationParams.MaxPhysicalTileCount, g_pTilePool );

    ResidencySampleRender::Initialize( pd3dDevice );

    g_PageDebugRender.Initialize( pd3dDevice );

    CreateSceneGeometry( pd3dDevice );

    // Setup the camera's view parameters
    XMMATRIX matProjection = XMMatrixPerspectiveFovLH( XM_PIDIV4, (FLOAT)pBackBufferSurfaceDesc->Width / (FLOAT)pBackBufferSurfaceDesc->Height, 0.001f, 100.0f );
    XMStoreFloat4x4A( &g_matProjection, matProjection );

    UpdateViewMatrix();

    g_pTitleResidencyManager->StartThreads();

    return S_OK;
}

struct SceneVertex
{
    XMFLOAT3 Pos;
    XMFLOAT3 TexCoord;
};

VOID CreateArrayVB( const UINT ArrayWidth, const UINT ArrayHeight, D3D11_SUBRESOURCE_DATA* pInitialData )
{
    const UINT ArraySize = ArrayWidth * ArrayHeight;

    SceneVertex* pVerts = new SceneVertex[ ArraySize * 4 ];
    pInitialData->pSysMem = pVerts;
    pInitialData->SysMemPitch = sizeof(SceneVertex) * ArraySize * 4;

    const FLOAT QuadWidth = 1.0f / (FLOAT)ArrayWidth;
    const FLOAT QuadHeight = 1.0f / (FLOAT)ArrayHeight;
    const FLOAT QuadXOrigin = -0.5f;
    const FLOAT QuadYOrigin = -0.5f;

    UINT ArrayIndex = 0;
    for( UINT y = 0; y < ArrayHeight; ++y )
    {
        FLOAT YOrigin = QuadYOrigin + (FLOAT)y * QuadHeight;

        for( UINT x = 0; x < ArrayWidth; ++x )
        {
            FLOAT ArraySlice = (FLOAT)ArrayIndex;

            UINT VertexIndex = ArrayIndex * 4;

            FLOAT XOrigin = QuadXOrigin + (FLOAT)x * QuadWidth;

            pVerts[VertexIndex + 0].Pos = XMFLOAT3( XOrigin + QuadWidth, 0, YOrigin );
            pVerts[VertexIndex + 0].TexCoord = XMFLOAT3( 0, 0, ArraySlice );
            pVerts[VertexIndex + 1].Pos = XMFLOAT3( XOrigin, 0, YOrigin );
            pVerts[VertexIndex + 1].TexCoord = XMFLOAT3( 1, 0, ArraySlice );
            pVerts[VertexIndex + 2].Pos = XMFLOAT3( XOrigin + QuadWidth, 0, YOrigin + QuadHeight );
            pVerts[VertexIndex + 2].TexCoord = XMFLOAT3( 0, 1, ArraySlice );
            pVerts[VertexIndex + 3].Pos = XMFLOAT3( XOrigin, 0, YOrigin + QuadHeight );
            pVerts[VertexIndex + 3].TexCoord = XMFLOAT3( 1, 1, ArraySlice );

            ++ArrayIndex;
        }
    }
}


VOID CreateSceneGeometry( ID3D11Device* pd3dDevice )
{
    const D3D11_INPUT_ELEMENT_DESC SceneVertexLayout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };

    const SceneVertex QuadVerts[] = 
    {
        { XMFLOAT3(  0.5f, 0, -0.5f ), XMFLOAT3( 0, 0, 0 ) },
        { XMFLOAT3( -0.5f, 0, -0.5f ), XMFLOAT3( 1, 0, 0 ) },
        { XMFLOAT3(  0.5f, 0,  0.5f ), XMFLOAT3( 0, 1, 0 ) },
        { XMFLOAT3( -0.5f, 0,  0.5f ), XMFLOAT3( 1, 1, 0 ) }
    };

    const SceneVertex QuiltQuadVerts[] = 
    {
        { XMFLOAT3(  0.5f, 0, -0.5f ), XMFLOAT3( 0, 0, 0 ) },
        { XMFLOAT3( -0.5f, 0, -0.5f ), XMFLOAT3( 8, 0, 0 ) },
        { XMFLOAT3(  0.5f, 0,  0.5f ), XMFLOAT3( 0, 8, 0 ) },
        { XMFLOAT3( -0.5f, 0,  0.5f ), XMFLOAT3( 8, 8, 0 ) }
    };

    g_pQuadVB = CreateVertexBuffer( pd3dDevice, sizeof(QuadVerts), QuadVerts );
    g_pQuiltQuadVB = CreateVertexBuffer( pd3dDevice, sizeof(QuiltQuadVerts), QuiltQuadVerts );

    D3D11_SUBRESOURCE_DATA InitialData;
    CreateArrayVB( 8, 8, &InitialData );
    g_pArrayQuadsVB = CreateVertexBuffer( pd3dDevice, InitialData.SysMemPitch, InitialData.pSysMem );
    delete[] InitialData.pSysMem;

    struct QuadDesc
    {
        DXGI_FORMAT Format;
        UINT TextureWidth;
        UINT TextureHeight;
        UINT ArraySliceCount;
        XMFLOAT2 CenterPosXZ;
        UINT QuiltWidth;
        UINT QuiltHeight;
        const WCHAR* strFileName;
    };

    const QuadDesc QuadDescs[] =
    {
        { DXGI_FORMAT_R8G8B8A8_UNORM,          16384, 16384,  1, XMFLOAT2(  0,  0 ) },
        { DXGI_FORMAT_R16G16B16A16_UNORM,      16384, 16384,  1, XMFLOAT2( -1,  0 ) },
        { DXGI_FORMAT_R32G32B32A32_FLOAT,      16384, 16384,  1, XMFLOAT2( -2,  0 ) },
        { DXGI_FORMAT_R8G8B8A8_UNORM,          16384, 16384,  1, XMFLOAT2(  1,  0 ), 8, 8 },
        { DXGI_FORMAT_R8G8B8A8_UNORM,          16384, 16384, 64, XMFLOAT2(  2,  0 ) },
        { DXGI_FORMAT_BC1_UNORM,               16384, 16384,  1, XMFLOAT2( -2,  1 ) },
        { DXGI_FORMAT_BC2_UNORM,               16384, 16384,  1, XMFLOAT2( -1,  1 ) },
        { DXGI_FORMAT_BC3_UNORM,               16384, 16384,  1, XMFLOAT2(  0,  1 ) },
        { DXGI_FORMAT_BC4_UNORM,               16384, 16384,  1, XMFLOAT2(  1,  1 ) },
        { DXGI_FORMAT_BC5_UNORM,               16384, 16384,  1, XMFLOAT2(  2,  1 ) },

        // BC6H and BC7 work, but they compress very slowly, so their swatches are disabled by default:
//         { DXGI_FORMAT_BC6H_UF16,               16384, 16384,  1, XMFLOAT2(  0,  2 ) },
//         { DXGI_FORMAT_BC7_UNORM,               16384, 16384,  1, XMFLOAT2(  1,  2 ) },
    };

    const FLOAT QuadSpacing = 1.25f;

    for( UINT i = 0; i < ARRAYSIZE(QuadDescs); ++i )
    {
        const QuadDesc& QD = QuadDescs[i];

        SceneObject* pQuad = new SceneObject();

        HRESULT hr;

        ID3D11TiledTexture2D* pTiledTexture = NULL;
        if( QD.strFileName != NULL )
        {
            TiledFileLoader* pFileLoader = new TiledFileLoader();
            HRESULT hr = pFileLoader->LoadFile( QD.strFileName );
            if( FAILED(hr) )
            {
                delete pFileLoader;
                delete pQuad;
                continue;
            }

            hr = pFileLoader->CreateTiledTexture2D( g_pd3dDeviceEx, &pTiledTexture );
            ASSERT( SUCCEEDED(hr) );

            pQuad->Textures[0].pTileLoader = pFileLoader;
            pQuad->Textures[0].DeleteTileLoader = TRUE;
        }
        else
        {
            // create a tiled texture
            D3D11_TILED_TEXTURE2D_DESC TexDesc;
            ZeroMemory( &TexDesc, sizeof(TexDesc) );
            TexDesc.Width = QD.TextureWidth;
            TexDesc.Height = QD.TextureHeight;
            TexDesc.ArraySize = max( 1, QD.ArraySliceCount );
            TexDesc.MipLevels = 0;
            TexDesc.Format = QD.Format;
            TexDesc.Usage = D3D11_USAGE_DEFAULT;
            TexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

            if( QD.QuiltWidth > 1 || QD.QuiltHeight > 1 )
            {
                TexDesc.MiscFlags |= D3D11_RESOURCE_MISC_TEXTUREQUILT;
                TexDesc.QuiltWidth = QD.QuiltWidth;
                TexDesc.QuiltHeight = QD.QuiltHeight;
                TexDesc.ArraySize = TexDesc.QuiltWidth * TexDesc.QuiltHeight;
            }

            hr = g_pd3dDeviceEx->CreateTexture2D( g_pTilePool, &TexDesc, &pTiledTexture );
            ASSERT( SUCCEEDED(hr) );

            pQuad->Textures[0].pTileLoader = &g_MandelbrotPageLoader;
            pQuad->Textures[0].DeleteTileLoader = FALSE;
        }

        hr = g_pd3dDeviceEx->CreateShaderResourceView( pTiledTexture, &pQuad->Textures[0].pTextureSRV );
        ASSERT( SUCCEEDED(hr) );

        if( QD.QuiltWidth > 1 || QD.QuiltHeight > 1 )
        {
            pQuad->pVertexBuffer = g_pQuiltQuadVB;
            pQuad->pVertexBuffer->AddRef();
            pQuad->VertexCount = 4;

            pQuad->pPixelShader = g_pPSSceneRenderQuilt;
        }
        else if( QD.ArraySliceCount > 1 )
        {
            pQuad->pVertexBuffer = g_pArrayQuadsVB;
            pQuad->pVertexBuffer->AddRef();
            pQuad->VertexCount = 4 * min( 64, QD.ArraySliceCount );

            pQuad->pPixelShader = g_pPSSceneRenderArray;
        }
        else
        {
            pQuad->pVertexBuffer = g_pQuadVB;
            pQuad->pVertexBuffer->AddRef();
            pQuad->VertexCount = 4;

            pQuad->pPixelShader = g_pPSSceneRender;
        }

        XMMATRIX matWorld = XMMatrixTranslation( QD.CenterPosXZ.x * QuadSpacing, 0, QD.CenterPosXZ.y * QuadSpacing );
        XMStoreFloat4x4( &pQuad->matWorld, matWorld );
        pQuad->m_pLayoutResidencySample = ResidencySampleRender::CreateInputLayout( pd3dDevice, SceneVertexLayout, ARRAYSIZE(SceneVertexLayout) );
        pQuad->VertexStrideBytes = sizeof(SceneVertex);
        pQuad->pIndexBuffer = NULL;
        pQuad->PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;

        // create a sampling quality manager for the tiled texture
        SamplingQualityManager* pSQM = new SamplingQualityManager( pTiledTexture, pd3dDevice, g_pd3dDeviceEx );
        g_pTitleResidencyManager->RegisterTileActivityHandler( pSQM );

        // register the tiled texture with the residency manager
        const ID3D11TiledTexture2D* pResourceSet[] = { pTiledTexture };
        ITileLoader* pLoaderSet[] = { pQuad->Textures[0].pTileLoader };
        UINT ResourceSetSize = ARRAYSIZE( pResourceSet );
        ResourceSetID RSID = g_pTitleResidencyManager->CreateResourceSet( pResourceSet, pLoaderSet, ResourceSetSize );

        pQuad->Textures[0].pTexture = pTiledTexture;
        pQuad->Textures[0].pSamplingQualityManager = pSQM;
        pQuad->RSID = RSID;
        pQuad->TextureCount = 1;

        g_SceneObjects.push_back( pQuad );
    }

    g_pTerrainView = new TerrainView( pd3dDevice, g_pd3dDeviceEx, g_pTitleResidencyManager );
}

//--------------------------------------------------------------------------------------
// Create any D3D11 resources that depend on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
                                         const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
    HRESULT hr;

    V_RETURN( g_DialogResourceManager.OnD3D11ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );
    V_RETURN( g_SettingsDlg.OnD3D11ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );

    FLOAT fAspect = (FLOAT)pBackBufferSurfaceDesc->Width / (FLOAT)pBackBufferSurfaceDesc->Height;

    // Setup the camera's projection parameters
    XMMATRIX matProjection = XMMatrixPerspectiveFovLH( XM_PIDIV4, fAspect, 0.001f, 100.0f );
    XMStoreFloat4x4A( &g_matProjection, matProjection );

    g_pTerrainView->GetCamera()->SetProjParams( XM_PIDIV4, fAspect, 0.01f, 1000.0f );

    g_HUD.SetLocation( pBackBufferSurfaceDesc->Width - 170, 0 );
    g_HUD.SetSize( 170, 170 );

    INT UIWidth = 250;
    INT UIHeight = 110;
    g_SampleUI.SetLocation( pBackBufferSurfaceDesc->Width - UIWidth, pBackBufferSurfaceDesc->Height - UIHeight );
    g_SampleUI.SetSize( UIWidth, UIHeight );

    g_HalfClientWidthPixels = (FLOAT)pBackBufferSurfaceDesc->Width * 0.5f;
    g_HalfClientHeightPixels = (FLOAT)pBackBufferSurfaceDesc->Height * 0.5f;

    ResidencySampleRender::ResizeRenderView( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height );

    return S_OK;
}


VOID RenderPreFrame( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, CXMMATRIX matView, CXMMATRIX matProj, FLOAT fElapsedTime )
{
    DXUT_BeginPerfEvent( 0, L"Pre Frame Render" );

    g_pd3dDeviceEx->PreFrameRender();

    if( !g_bPauseStreaming )
    {
        if( g_bDrawTerrain )
        {
            g_pTerrainView->PreFrameRender( pd3dImmediateContext, fElapsedTime );
        }
        else
        {
            g_LastResidencySampleViewID = ResidencySampleRender::Render( pd3dDevice, pd3dImmediateContext, g_pTitleResidencyManager, g_SceneObjects, matView, matProj );

            const UINT SceneObjectCount = (UINT)g_SceneObjects.size();
            for( UINT i = 0; i < SceneObjectCount; ++i )
            {
                SamplingQualityManager* pSQM = g_SceneObjects[i]->Textures[0].pSamplingQualityManager;
                pSQM->Render( pd3dImmediateContext, g_pd3dDeviceEx, fElapsedTime );
            }
        }
    }

    DXUT_EndPerfEvent();
}


VOID RenderTestObjects( ID3D11DeviceContext* pd3dImmediateContext, CXMMATRIX matVP )
{
    DXUT_BeginPerfEvent( 0, L"Frame Render" );

    DXUTSetupD3D11Views( pd3dImmediateContext );

    ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
    ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();

    float ClearColor[4] = { 0, 0, 0, 1 };
    pd3dImmediateContext->ClearRenderTargetView( pRTV, ClearColor );
    pd3dImmediateContext->ClearDepthStencilView( pDSV, D3D11_CLEAR_DEPTH, 1.0, 0 );

    HRESULT hr;

    pd3dImmediateContext->IASetInputLayout( g_pDefaultInputLayout );
    pd3dImmediateContext->VSSetShader( g_pVSTransform, NULL, 0 );
    pd3dImmediateContext->PSSetSamplers( 0, 1, &g_pSamLinear );
    FLOAT BlendFactor[4] = { 1, 1, 1, 1 };
    pd3dImmediateContext->OMSetBlendState( g_pBlendState, BlendFactor, 0xFFFFFFFF );
    pd3dImmediateContext->OMSetDepthStencilState( g_pDepthStencilState, 0 );
    pd3dImmediateContext->RSSetState( g_pRasterizerState );

    const UINT SceneObjectCount = (UINT)g_SceneObjects.size();
    for( UINT i = 0; i < SceneObjectCount; ++i )
    {
        DXUT_BeginPerfEvent( 0, L"Scene Object Render" );

        SceneObject* pSO = g_SceneObjects[i];

        pd3dImmediateContext->PSSetShader( pSO->pPixelShader, NULL, 0 );

        D3D11_MAPPED_SUBRESOURCE MappedResource;
        V( pd3dImmediateContext->Map( g_pcbVSPerObject11, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource ) );
        CB_VS_PER_OBJECT* pVSPerObject = ( CB_VS_PER_OBJECT* )MappedResource.pData;
        XMMATRIX matWorld = XMLoadFloat4x4( &pSO->matWorld );
        XMMATRIX matWVP = XMMatrixTranspose( matWorld * matVP );
        XMStoreFloat4x4( (XMFLOAT4X4*)&pVSPerObject->m_mWorldViewProjection, matWVP );
        pd3dImmediateContext->Unmap( g_pcbVSPerObject11, 0 );
        pd3dImmediateContext->VSSetConstantBuffers( 0, 1, &g_pcbVSPerObject11 );

        UINT Strides = { pSO->VertexStrideBytes };
        UINT Offsets = { 0 };
        pd3dImmediateContext->IASetVertexBuffers( 0, 1, &pSO->pVertexBuffer, &Strides, &Offsets );
        pd3dImmediateContext->IASetPrimitiveTopology( pSO->PrimitiveType );

        g_pd3dDeviceEx->PSSetShaderResources( 0, 1, &pSO->Textures[0].pTextureSRV );

        UINT QualityShaderResourceSlot = 1;
        if( pSO->pPixelShader != g_pPSSceneRender )
        {
            QualityShaderResourceSlot = 2;
        }
        ID3D11ShaderResourceView* pQualitySRV = pSO->Textures[0].pSamplingQualityManager->GetLODQualityTextureSRV();
        pd3dImmediateContext->PSSetShaderResources( QualityShaderResourceSlot, 1, &pQualitySRV );
        ID3D11SamplerState* pQualitySS = pSO->Textures[0].pSamplingQualityManager->GetSamplerState();
        pd3dImmediateContext->PSSetSamplers( 1, 1, &pQualitySS );

        if( pSO->pIndexBuffer != NULL )
        {
            pd3dImmediateContext->IASetIndexBuffer( pSO->pIndexBuffer, DXGI_FORMAT_R16_UINT, 0 );
            pd3dImmediateContext->DrawIndexed( pSO->IndexCount, 0, 0 );
        }
        else
        {
            pd3dImmediateContext->Draw( pSO->VertexCount, 0 );
        }

        DXUT_EndPerfEvent();
    }

    DXUT_EndPerfEvent();
}

VOID RenderHUD( ID3D11DeviceContext* pd3dImmediateContext, FLOAT fElapsedTime )
{
    DXUT_BeginPerfEvent( DXUT_PERFEVENTCOLOR, L"HUD / Stats" );
    g_HUD.OnRender( fElapsedTime );
    g_SampleUI.OnRender( fElapsedTime );

    static FLOAT InspectionYOffset = 0;
    if( g_pInspectionTexture != NULL )
    {
        const INT ViewportHeight = (INT)g_HalfClientHeightPixels * 2;
        INT TotalHeight = 0;
        INT SliceHeight = 0;
        g_PageDebugRender.Render( pd3dImmediateContext, g_pTitleResidencyManager, g_pInspectionTexture, 10, ViewportHeight - (INT)InspectionYOffset, &TotalHeight, &SliceHeight );

        FLOAT fLerp = min( 1.0f, fElapsedTime * 8.0f );
        FLOAT TargetOffset = (FLOAT)( SliceHeight * ( g_InspectionSliceIndex + 1 ) );
        if( fabsf( TargetOffset - InspectionYOffset ) < 2.0f )
        {
            InspectionYOffset = TargetOffset;
        }
        else
        {
            InspectionYOffset = ( TargetOffset * fLerp + (FLOAT)InspectionYOffset * ( 1.0f - fLerp ) );
        }
    }
    else
    {
        InspectionYOffset = 0;
        g_InspectionSliceIndex = 0;
    }

    if( g_bDrawResidencySampleViews )
    {
        ID3D11ShaderResourceView* pSRViewUVGradientID = NULL;
        ID3D11ShaderResourceView* pSRViewExtendedUVSlice = NULL;

        g_pTitleResidencyManager->GetViewShaderResources( g_LastResidencySampleViewID, &pSRViewUVGradientID, &pSRViewExtendedUVSlice );

        if( pSRViewExtendedUVSlice != NULL && pSRViewUVGradientID != NULL )
        {
            RECT ClientRect;
            GetClientRect( DXUTGetHWND(), &ClientRect );

            UINT Width = 256;
            UINT Height = 144;
            UINT Margin = 10;
            UINT BottomMargin = 60 + Margin;

            RECT Rect2 = { ClientRect.right - ( Width + Margin ), ClientRect.bottom - ( Height + BottomMargin ), ClientRect.right - Margin, ClientRect.bottom - BottomMargin };
            RECT Rect1 = { Rect2.left, Rect2.top - ( Height + Margin ), Rect2.right, Rect2.top - Margin };

            g_PageDebugRender.RenderTexture( pd3dImmediateContext, pSRViewUVGradientID, Rect1 );
            g_PageDebugRender.RenderTexture( pd3dImmediateContext, pSRViewExtendedUVSlice, Rect2 );
        }
    }

    RenderText();

    DXUT_EndPerfEvent();
}

//--------------------------------------------------------------------------------------
// Render the scene using the D3D11 device
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime,
                                 float fElapsedTime, void* pUserContext )
{
    XMMATRIX matView = XMLoadFloat4x4A( &g_matView );
    XMMATRIX matProj = XMLoadFloat4x4A( &g_matProjection );
    XMMATRIX matVP = matView * matProj;

    RenderPreFrame( pd3dDevice, pd3dImmediateContext, matView, matProj, fElapsedTime );

    // If the settings dialog is being shown, then render it instead of rendering the app's scene
    if( g_SettingsDlg.IsActive() )
    {
        g_SettingsDlg.OnRender( fElapsedTime );
        return;
    }

    if( g_bDrawTerrain )
    {
        g_pTerrainView->Render( pd3dImmediateContext );
    }
    else
    {
        RenderTestObjects( pd3dImmediateContext, matVP );
    }

    RenderHUD( pd3dImmediateContext, fElapsedTime );

    static DWORD dwTimefirst = GetTickCount();
    if ( GetTickCount() - dwTimefirst > 5000 )
    {    
        OutputDebugString( DXUTGetFrameStats( DXUTIsVsyncEnabled() ) );
        OutputDebugString( L"\n" );
        dwTimefirst = GetTickCount();
    }
}


//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D11ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext )
{
    g_DialogResourceManager.OnD3D11ReleasingSwapChain();
}


//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D11CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11DestroyDevice( void* pUserContext )
{
    g_DialogResourceManager.OnD3D11DestroyDevice();
    g_SettingsDlg.OnD3D11DestroyDevice();
    DXUTGetGlobalResourceCache().OnDestroyDevice();
    SAFE_DELETE( g_pTxtHelper );

    SAFE_RELEASE( g_pVSTransform );
    SAFE_RELEASE( g_pPSSceneRender );
    SAFE_RELEASE( g_pPSSceneRenderArray );
    SAFE_RELEASE( g_pPSSceneRenderQuilt );

    SAFE_RELEASE( g_pDefaultInputLayout );
    SAFE_RELEASE( g_pSamLinear );

    SAFE_RELEASE( g_pBlendState );
    SAFE_RELEASE( g_pDepthStencilState );
    SAFE_RELEASE( g_pRasterizerState );

    SAFE_RELEASE( g_pcbVSPerObject11 );
    SAFE_RELEASE( g_pQuadVB );
    SAFE_RELEASE( g_pQuiltQuadVB );
    SAFE_RELEASE( g_pArrayQuadsVB );

    g_PageDebugRender.Terminate();
    ResidencySampleRender::Terminate();

    SAFE_DELETE( g_pTitleResidencyManager );

    g_pInspectionTexture = NULL;

    for( UINT i = 0; i < g_SceneObjects.size(); ++i )
    {
        SceneObject* pSO = g_SceneObjects[i];
        SAFE_RELEASE( pSO->m_pLayoutResidencySample );
        SAFE_RELEASE( pSO->m_pLayoutSceneRender );
        SAFE_RELEASE( pSO->pVertexBuffer );
        SAFE_RELEASE( pSO->pIndexBuffer );

        for( UINT j = 0; j < pSO->TextureCount; ++j )
        {
            pSO->Textures[j].Release();
        }

        delete pSO;
    }
    g_SceneObjects.clear();

    SAFE_DELETE( g_pTerrainView );

    SAFE_RELEASE( g_pTilePool );
    SAFE_RELEASE( g_pd3dDeviceEx );
}


//--------------------------------------------------------------------------------------
// Called right before creating a D3D9 or D3D11 device, allowing the app to modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
{
    if( pDeviceSettings->ver == DXUT_D3D9_DEVICE )
    {
        IDirect3D9* pD3D = DXUTGetD3D9Object();
        D3DCAPS9 Caps;
        pD3D->GetDeviceCaps( pDeviceSettings->d3d9.AdapterOrdinal, pDeviceSettings->d3d9.DeviceType, &Caps );

        // If device doesn't support HW T&L or doesn't support 1.1 vertex shaders in HW 
        // then switch to SWVP.
        if( ( Caps.DevCaps & D3DDEVCAPS_HWTRANSFORMANDLIGHT ) == 0 ||
            Caps.VertexShaderVersion < D3DVS_VERSION( 1, 1 ) )
        {
            pDeviceSettings->d3d9.BehaviorFlags = D3DCREATE_SOFTWARE_VERTEXPROCESSING;
        }

        // Debugging vertex shaders requires either REF or software vertex processing 
        // and debugging pixel shaders requires REF.  
#ifdef DEBUG_VS
        if( pDeviceSettings->d3d9.DeviceType != D3DDEVTYPE_REF )
        {
            pDeviceSettings->d3d9.BehaviorFlags &= ~D3DCREATE_HARDWARE_VERTEXPROCESSING;
            pDeviceSettings->d3d9.BehaviorFlags &= ~D3DCREATE_PUREDEVICE;
            pDeviceSettings->d3d9.BehaviorFlags |= D3DCREATE_SOFTWARE_VERTEXPROCESSING;
        }
#endif
#ifdef DEBUG_PS
        pDeviceSettings->d3d9.DeviceType = D3DDEVTYPE_REF;
#endif
    }

    // For the first device created if its a REF device, optionally display a warning dialog box
    static bool s_bFirstTime = true;
    if( s_bFirstTime )
    {
        s_bFirstTime = false;
        if( ( DXUT_D3D9_DEVICE == pDeviceSettings->ver && pDeviceSettings->d3d9.DeviceType == D3DDEVTYPE_REF ) ||
            ( DXUT_D3D11_DEVICE == pDeviceSettings->ver &&
            pDeviceSettings->d3d11.DriverType == D3D_DRIVER_TYPE_REFERENCE ) )
        {
            DXUTDisplaySwitchingToREFWarning( pDeviceSettings->ver );
        }

    }

    return true;
}


//--------------------------------------------------------------------------------------
// Handle updates to the scene.  This is called regardless of which D3D API is used
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext )
{
    if( g_bDrawTerrain )
    {
        g_pTerrainView->Update( fElapsedTime );
    }

    float fStreamingDeltaTime = fElapsedTime;
    if( g_bPauseStreaming )
    {
        fStreamingDeltaTime = 0;
    }
    g_pTitleResidencyManager->Update( fStreamingDeltaTime );
}


//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
                          void* pUserContext )
{
    // Pass messages to dialog resource manager calls so GUI state is updated correctly
    *pbNoFurtherProcessing = g_DialogResourceManager.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;

    // Pass messages to settings dialog if its active
    if( g_SettingsDlg.IsActive() )
    {
        g_SettingsDlg.MsgProc( hWnd, uMsg, wParam, lParam );
        return 0;
    }

    // Give the dialogs a chance to handle the message first
    *pbNoFurtherProcessing = g_HUD.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;
    *pbNoFurtherProcessing = g_SampleUI.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;

    if( g_bDrawTerrain )
    {
        g_pTerrainView->GetCamera()->HandleMessages( hWnd, uMsg, wParam, lParam );
    }

    return 0;
}


//--------------------------------------------------------------------------------------
// Handle key presses
//--------------------------------------------------------------------------------------
void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{
    if( !bKeyDown || g_bDrawTerrain )
    {
        return;
    }

    const FLOAT LogMoveSpeed = 0.05f;

    FLOAT CameraY = g_CameraPos.y;
    FLOAT CameraYMoved = expf( logf( CameraY ) + LogMoveSpeed );
    
    FLOAT LinearMoveSpeed = CameraYMoved - CameraY;

    switch( nChar )
    {
    case 'S':
        {
            float LogYPos = logf( CameraY );
            LogYPos -= LogMoveSpeed;
            g_CameraPos.y = max( 0.001f, expf( LogYPos ) );
            break;
        }
    case 'W':
        {
            float LogYPos = logf( CameraY );
            LogYPos += LogMoveSpeed;
            g_CameraPos.y = expf( LogYPos );
            break;
        }
    case VK_LEFT:
        g_CameraPos.x += LinearMoveSpeed;
        break;
    case VK_RIGHT:
        g_CameraPos.x -= LinearMoveSpeed;
        break;
    case VK_UP:
        g_CameraPos.z -= LinearMoveSpeed;
        break;
    case VK_DOWN:
        g_CameraPos.z += LinearMoveSpeed;
        break;
    case VK_BACK:
        g_pInspectionTexture = NULL;
        break;
    case VK_PRIOR:
        if( g_InspectionSliceIndex > 0 )
        {
            --g_InspectionSliceIndex;
        }
        break;
    case VK_NEXT:
        if( g_pInspectionTexture != NULL )
        {
            D3D11_TILED_TEXTURE2D_DESC TexDesc;
            g_pInspectionTexture->GetDesc( &TexDesc );
            g_InspectionSliceIndex = min( (INT)TexDesc.ArraySize - 1, g_InspectionSliceIndex + 1 );
        }
        break;
    }

    UpdateViewMatrix();
}


// inline VOID PrintVector( const WCHAR* strLabel, CXMVECTOR v )
// {
//     WCHAR strText[128];
//     swprintf_s( strText, L"%s: < %0.3f %0.3f %0.3f >\n", strLabel, XMVectorGetX(v), XMVectorGetY(v), XMVectorGetZ(v) );
//     OutputDebugString( strText );
// }


//--------------------------------------------------------------------------------------
// Handles mouse input
//--------------------------------------------------------------------------------------
void CALLBACK OnMouse( bool bLeftButtonDown, bool bRightButtonDown, bool bMiddleButtonDown, bool bSideButton1Down, bool bSideButton2Down, int nMouseWheelDelta, int xPos, int yPos, void* pUserContext )
{
    if( g_bDrawTerrain )
    {
        if( bRightButtonDown )
        {
            if( g_pInspectionTexture != NULL )
            {
                g_pInspectionTexture = NULL;
            }
            else
            {
                g_pInspectionTexture = g_pTerrainView->GetInspectionTexture();
            }
            g_InspectionSliceIndex = 0;
        }
        return;
    }

    static BOOL MouseDragging = FALSE;
    static XMFLOAT4 DragMouseCursorPos( 0, 0, 0, 0 );
    static XMFLOAT4 DragCameraPos( 0, 0, 0, 0 );

    // vMousePos is the mouse cursor's location on the Z far and near planes in homogenous space
    XMVECTOR vMousePosFar = XMVectorSet( ( (FLOAT)xPos - g_HalfClientWidthPixels ) / g_HalfClientWidthPixels, ( (FLOAT)yPos - g_HalfClientHeightPixels ) / -g_HalfClientHeightPixels, 1, 1 );

    XMMATRIX matView = XMLoadFloat4x4A( &g_matView );
    XMMATRIX matViewProj = matView * XMLoadFloat4x4A( &g_matProjection );
    XMVECTOR vDet;
    XMMATRIX matInvViewProj = XMMatrixInverse( &vDet, matViewProj );

    // vMouseWorldPos is the location in world space of the mouse cursor on the near plane
    XMVECTOR vMouseWorldPosFar = XMVector3TransformCoord( vMousePosFar, matInvViewProj );

    XMVECTOR vCameraPosWorld = XMLoadFloat4A( &g_CameraPos );

    const XMVECTOR vPlaneNormal = XMVectorSet( 0, 1, 0, 0 );
    const XMVECTOR vPlaneDistance = XMVectorZero();

    XMVECTOR vCameraToMouseWorldFar = vMouseWorldPosFar - vCameraPosWorld;

    XMVECTOR t = ( vPlaneDistance - XMVector3Dot( vPlaneNormal, vCameraPosWorld ) ) / XMVector3Dot( vPlaneNormal, vCameraToMouseWorldFar );
    XMVECTOR vMouseCursorWorld = vCameraPosWorld + t * vCameraToMouseWorldFar;

    if( bLeftButtonDown )
    {
        if( !MouseDragging )
        {
            MouseDragging = TRUE;
            XMStoreFloat4( &DragMouseCursorPos, vMouseCursorWorld );
            XMStoreFloat4( &DragCameraPos, vCameraPosWorld );
        }
        else
        {
            XMVECTOR vDragPos = XMLoadFloat4( &DragMouseCursorPos );
            XMVECTOR vCameraDelta = vMouseCursorWorld - vDragPos;

            vCameraPosWorld -= vCameraDelta;

            XMStoreFloat4A( &g_CameraPos, vCameraPosWorld );
            UpdateViewMatrix();
            XMStoreFloat4( &DragCameraPos, vCameraPosWorld );
        }
    }
    else if( MouseDragging )
    {
        MouseDragging = FALSE;
    }

    if( bRightButtonDown )
    {
        ID3D11TiledTexture2D* pCurrentTexture = g_pInspectionTexture;
        g_pInspectionTexture = NULL;

        UINT SceneObjectCount = (UINT)g_SceneObjects.size();
        for( UINT i = 0; i < SceneObjectCount; ++i )
        {
            const SceneObject* pSO = g_SceneObjects[i];
            FLOAT X = pSO->matWorld._41;
            FLOAT Y = pSO->matWorld._43;

            FLOAT DeltaX = fabsf( X - XMVectorGetX( vMouseCursorWorld ) );
            FLOAT DeltaY = fabsf( Y - XMVectorGetZ( vMouseCursorWorld ) );
            if( DeltaX <= 0.5f && DeltaY <= 0.5f )
            {
                g_pInspectionTexture = pSO->Textures[0].pTexture;
                break;
            }
        }

        if( g_pInspectionTexture != pCurrentTexture )
        {
            g_InspectionSliceIndex = 0;
        }
    }

    if( nMouseWheelDelta != 0 )
    {
        const FLOAT LogMoveSpeed = 0.05f * ( (FLOAT)nMouseWheelDelta / 120.0f );

        FLOAT CameraY = XMVectorGetY( vCameraPosWorld );
        float LogYPos = logf( CameraY );
        LogYPos -= LogMoveSpeed;
        
        vCameraPosWorld = XMVectorSetY( vCameraPosWorld, max( 0.001f, expf( LogYPos ) ) );
        XMStoreFloat4A( &g_CameraPos, vCameraPosWorld );
        UpdateViewMatrix();
    }
}

//--------------------------------------------------------------------------------------
// Handles the GUI events
//--------------------------------------------------------------------------------------
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext )
{
    switch( nControlID )
    {
        case IDC_TOGGLEFULLSCREEN:
            DXUTToggleFullScreen();
            break;
        case IDC_TOGGLEREF:
            DXUTToggleREF();
            break;
        case IDC_CHANGEDEVICE:
            g_SettingsDlg.SetActive( !g_SettingsDlg.IsActive() );
            break;
        case IDC_TOGGLERESIDENCYVIEWS:
            g_bDrawResidencySampleViews = !g_bDrawResidencySampleViews;
            break;
        case IDC_TOGGLETERRAINVIEW:
            g_bDrawTerrain = !g_bDrawTerrain;
            break;
        case IDC_PAUSESTREAMING:
            g_bPauseStreaming = !g_bPauseStreaming;
            break;
    }
}


