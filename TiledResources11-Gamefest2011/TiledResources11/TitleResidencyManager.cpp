//--------------------------------------------------------------------------------------
// TitleResidencyManager.cpp
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include <assert.h>
#include "TitleResidencyManager.h"
#include "Util.h"

// The residency view has a fixed size (small multiple of 720p):
static const UINT g_ResidencyViewWidth = 256;
static const UINT g_ResidencyViewHeight = 144;
static FLOAT g_TexelPitchMultiplier = (FLOAT)g_ResidencyViewWidth / (FLOAT)1280;
static const DXGI_FORMAT g_ResidencyViewFormat = DXGI_FORMAT_R8G8B8A8_UNORM;

// We can only have 255 resource sets:
static const ResourceSetID g_MaxResourceSetID = 255;

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager constructor
//--------------------------------------------------------------------------------------
TitleResidencyManager::TitleResidencyManager( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dDeviceContext, UINT MaxViewsPerFrame, UINT MaxPhysicalTiles, ID3D11TilePool* pTilePool )
{
    m_pd3dDeviceContext = pd3dDeviceContext;
    m_pd3dDeviceContext->AddRef();

    m_pTilePool = pTilePool;
    m_pTilePool->AddRef();

    m_FrameIndex = 0;
    m_CurrentFrameTime = GetTickCount();
    m_bPaused = FALSE;

    // Create residency sample view buffers:
    D3D11_TEXTURE2D_DESC TexDesc;
    ZeroMemory( &TexDesc, sizeof(TexDesc) );
    TexDesc.Width = g_ResidencyViewWidth;
    TexDesc.Height = g_ResidencyViewHeight;
    TexDesc.ArraySize = 1;
    TexDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    TexDesc.CPUAccessFlags = 0;
    TexDesc.Format = g_ResidencyViewFormat;
    TexDesc.MipLevels = 1;
    TexDesc.SampleDesc.Count = 1;
    TexDesc.Usage = D3D11_USAGE_DEFAULT;

    D3D11_TEXTURE2D_DESC StagingDesc = TexDesc;
    StagingDesc.BindFlags = 0;
    StagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    StagingDesc.Usage = D3D11_USAGE_STAGING;

    MaxViewsPerFrame *= 2;
    for( UINT i = 0; i < MaxViewsPerFrame; ++i )
    {
        SamplingView View;
        View.RenderFrameIndex = (UINT)-1;

        pd3dDevice->CreateTexture2D( &TexDesc, NULL, &View.pRenderTargetTextureUVGradientID );
        pd3dDevice->CreateRenderTargetView( View.pRenderTargetTextureUVGradientID, NULL, &View.pRenderTargetViewUVGradientID );
        pd3dDevice->CreateShaderResourceView( View.pRenderTargetTextureUVGradientID, NULL, &View.pSRViewUVGradientID );
        pd3dDevice->CreateTexture2D( &StagingDesc, NULL, &View.pStagingTextureUVGradientID );

        pd3dDevice->CreateTexture2D( &TexDesc, NULL, &View.pRenderTargetTextureExtendedUVSlice );
        pd3dDevice->CreateRenderTargetView( View.pRenderTargetTextureExtendedUVSlice, NULL, &View.pRenderTargetViewExtendedUVSlice );
        pd3dDevice->CreateShaderResourceView( View.pRenderTargetTextureExtendedUVSlice, NULL, &View.pSRViewExtendedUVSlice );
        pd3dDevice->CreateTexture2D( &StagingDesc, NULL, &View.pStagingTextureExtendedUVSlice );

        m_SamplingViews.push_back( View );
    }
    
    TexDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    TexDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;

    pd3dDevice->CreateTexture2D( &TexDesc, NULL, &m_pDepthStencilTexture );
    pd3dDevice->CreateDepthStencilView( m_pDepthStencilTexture, NULL, &m_pDepthStencil );

    m_Viewport.TopLeftX = 0;
    m_Viewport.TopLeftY = 0;
    m_Viewport.Width = g_ResidencyViewWidth;
    m_Viewport.Height = g_ResidencyViewHeight;
    m_Viewport.MinDepth = 0.0f;
    m_Viewport.MaxDepth = 1.0f;

    // Create renderstate for residency sample views:
    D3D11_BLEND_DESC BlendDesc;
    ZeroMemory( &BlendDesc, sizeof(BlendDesc) );
    BlendDesc.AlphaToCoverageEnable = FALSE;
    BlendDesc.IndependentBlendEnable = FALSE;
    BlendDesc.RenderTarget[0].BlendEnable = FALSE;
    BlendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    BlendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    BlendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    BlendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    BlendDesc.RenderTarget[0].RenderTargetWriteMask = 0xF;
    BlendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    BlendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_SRC_ALPHA;
    pd3dDevice->CreateBlendState( &BlendDesc, &m_pBlendState );

    D3D11_DEPTH_STENCIL_DESC DSDesc;
    ZeroMemory( &DSDesc, sizeof(DSDesc) );
    DSDesc.DepthEnable = TRUE;
    DSDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
    DSDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    DSDesc.StencilEnable = FALSE;
    pd3dDevice->CreateDepthStencilState( &DSDesc, &m_pDepthStencilState );

    m_CurrentViewIndex = (UINT)-1;
    m_NextViewIndex = 0;

    // Resource set ID 0 is reserved:
    ResourceSet EmptySet = { 0 };
    m_ResourceSets.push_back( EmptySet );

    m_MaxPhysicalTiles = MaxPhysicalTiles;
    m_AllocatedTileCount = 0;

    // Compute the gradient scaling factor.
    // The gradient scaling factor maps a range of ln(1) to ln(1/16384) into 0..1.
    m_fGradientScalingFactor = log( 1.0f / 16384.0f );

    m_NeedPhysicalTilesNow = FALSE;

    // Create the loader and unloader signalling events:
    m_hLoaderEvent = CreateEvent( NULL, FALSE, FALSE, NULL );
    m_hUnloaderEvent = CreateEvent( NULL, FALSE, FALSE, NULL );

    // Determine how many cores and HW threads we have:
    UINT CoreCount = 1;
    UINT ThreadCount = 1;
    GetCoreAndHWThreadCount( &CoreCount, &ThreadCount );

    // Make one loader thread for each HW thread, minus one:
    const UINT LoaderThreadCount = max( 1, min( MAX_LOADER_THREAD_COUNT, ThreadCount - 1 ) );
    const UINT UnloaderThreadCount = max( 1, min( MAX_UNLOADER_THREAD_COUNT, ThreadCount - 1 ) );

    m_ResidencyStats.LoaderThreadCount = LoaderThreadCount;
    m_ResidencyStats.UnloaderThreadCount = UnloaderThreadCount;

    ZeroMemory( m_hLoaderThreads, sizeof(m_hLoaderThreads) );
    ZeroMemory( m_hUnloaderThreads, sizeof(m_hUnloaderThreads) );

    // Create the loader threads.  Note that all threads start suspended:
    m_ThreadCount = 0;
    m_LoaderRunning = TRUE;
    for( UINT i = 0; i < LoaderThreadCount; ++i )
    {
        ThreadEntryContext* pEntryContext = new ThreadEntryContext();
        pEntryContext->pTRM = this;
        pEntryContext->ThreadIndex = i;
        HANDLE hLoaderThread = CreateThread( NULL, 0, LoaderThreadProc, pEntryContext, CREATE_SUSPENDED, NULL ); 
        UINT HWThreadIndex = ( ThreadCount - 1 ) - i;
        SetThreadAffinityMask( hLoaderThread, 1 << HWThreadIndex );
        m_hLoaderThreads[i] = hLoaderThread;
    }

    // Create the unloader threads.  Note that all threads start suspended:
    for( UINT i = 0; i < UnloaderThreadCount; ++i )
    {
        ThreadEntryContext* pEntryContext = new ThreadEntryContext();
        pEntryContext->pTRM = this;
        pEntryContext->ThreadIndex = i;
        m_hUnloaderThreads[i] = CreateThread( NULL, 0, UnloaderThreadProc, pEntryContext, CREATE_SUSPENDED, NULL );
    }
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager destructor
//--------------------------------------------------------------------------------------
TitleResidencyManager::~TitleResidencyManager()
{
    // Shut down threads:
    m_LoaderRunning = FALSE;
    while( m_ThreadCount > 0 )
    {
        SetEvent( m_hLoaderEvent );
        SetEvent( m_hUnloaderEvent );
    }
    CloseHandle( m_hLoaderEvent );
    CloseHandle( m_hUnloaderEvent );

    // Clear the tracked tile map; all of the contents will be deleted next:
    m_TrackedTileMap.clear();

    // Clear tracked tiles:
    TrackedTileSortList::iterator iter = m_TrackedTileSortList.begin();
    TrackedTileSortList::iterator end = m_TrackedTileSortList.end();
    while( iter != end )
    {
        delete *iter;
        ++iter;
    }
    m_TrackedTileSortList.clear();

    // Clear tracked tile free list:
    while( !m_TrackedTileFreeList.empty() )
    {
        TrackedTile* pTP = m_TrackedTileFreeList.top();
        m_TrackedTileFreeList.pop();
        delete pTP;
    }

    // Clear resource sets:
    for( UINT i = 0; i < m_ResourceSets.size(); ++i )
    {
        delete[] m_ResourceSets[i].ppResources;
        delete[] m_ResourceSets[i].ppTileLoaders;
    }
    m_ResourceSets.clear();

    // Release D3D objects for the residency sample views:
    for( UINT i = 0; i < m_SamplingViews.size(); ++i )
    {
        SamplingView& SV = m_SamplingViews[i];

        SAFE_RELEASE( SV.pRenderTargetViewUVGradientID );
        SAFE_RELEASE( SV.pStagingTextureUVGradientID );
        SAFE_RELEASE( SV.pSRViewUVGradientID );
        SAFE_RELEASE( SV.pRenderTargetTextureUVGradientID );

        SAFE_RELEASE( SV.pRenderTargetViewExtendedUVSlice );
        SAFE_RELEASE( SV.pSRViewExtendedUVSlice );
        SAFE_RELEASE( SV.pStagingTextureExtendedUVSlice );
        SAFE_RELEASE( SV.pRenderTargetTextureExtendedUVSlice );
    }
    SAFE_RELEASE( m_pDepthStencil );
    SAFE_RELEASE( m_pDepthStencilTexture );

    SAFE_RELEASE( m_pDepthStencilState );
    SAFE_RELEASE( m_pBlendState );

    // Release the core D3D objects:
    SAFE_RELEASE( m_pTilePool );
    SAFE_RELEASE( m_pd3dDeviceContext );
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::StartThreads
// Desc: Resume all of the loader and unloader threads - this is done at the very end of
//       app initialization.
//--------------------------------------------------------------------------------------
VOID TitleResidencyManager::StartThreads()
{
    for( UINT i = 0; i < ARRAYSIZE(m_hLoaderThreads); ++i )
    {
        if( m_hLoaderThreads[i] != NULL )
        {
            ResumeThread( m_hLoaderThreads[i] );
        }
    }
    for( UINT i = 0; i < ARRAYSIZE(m_hUnloaderThreads); ++i )
    {
        if( m_hUnloaderThreads[i] != NULL )
        {
            ResumeThread( m_hUnloaderThreads[i] );
        }
    }
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::LoaderThreadProc
// Desc: Static entry point for the loader thread.  Passes execution to a private method
//       on the title residency manager.
//--------------------------------------------------------------------------------------
DWORD WINAPI TitleResidencyManager::LoaderThreadProc( VOID* pParam )
{
    ThreadEntryContext* pEntryContext = (ThreadEntryContext*)pParam;
    UINT ThreadIndex = pEntryContext->ThreadIndex;
    
    TitleResidencyManager* pTRM = pEntryContext->pTRM;
    delete pEntryContext;

    InterlockedIncrement( &pTRM->m_ThreadCount );
    pTRM->LoaderEntryPoint( ThreadIndex );
    InterlockedDecrement( &pTRM->m_ThreadCount );

    return 0;
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::UnloaderThreadProc
// Desc: Static entry point for the unloader thread.  Passes execution to a private method
//       on the title residency manager.
//--------------------------------------------------------------------------------------
DWORD WINAPI TitleResidencyManager::UnloaderThreadProc( VOID* pParam )
{
    ThreadEntryContext* pEntryContext = (ThreadEntryContext*)pParam;
    UINT ThreadIndex = pEntryContext->ThreadIndex;

    TitleResidencyManager* pTRM = pEntryContext->pTRM;
    delete pEntryContext;

    InterlockedIncrement( &pTRM->m_ThreadCount );
    pTRM->UnloaderEntryPoint( ThreadIndex );
    InterlockedDecrement( &pTRM->m_ThreadCount );

    return 0;
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::UnloaderEntryPoint
// Desc: Main loop for a single unloader thread.  It pops a TrackedTile* off the
//       unloader queue and passes it to the appropriate tile loader class for processing.
//--------------------------------------------------------------------------------------
VOID TitleResidencyManager::UnloaderEntryPoint( const UINT ThreadIndex )
{
    while( m_LoaderRunning )
    {
        BOOL Success = TRUE;
        while( Success && m_LoaderRunning )
        {
            while( m_bPaused )
            {
                YieldProcessor();
            }

            // Get a tracked tile from the queue:
            TrackedTile* pTP = NULL;
            Success = m_UnmapQueue.SafeTryGet( &pTP );
            if( !Success )
            {
                continue;
            }

            // Validate the tracked tile is ready for unmapping:
            ASSERT( pTP->State == TPS_QueuedForUnmap );
            ASSERT( pTP->ID.PinnedTile == FALSE );
            ASSERT( pTP->ID.VTileID != D3D11_TILED_INVALID_VIRTUAL_ADDRESS );

            // Update the state:
            pTP->State = TPS_Unmapping;

            // Pass the tracked tile ID to the tile loader:
            ITileLoader* pTileLoader = pTP->pTileLoader;
            ASSERT( pTileLoader != NULL );
            ASSERT( ThreadIndex < ARRAYSIZE(pTileLoader->m_pUnloaderContexts) );
            VOID* pUnloaderContext = pTileLoader->m_pUnloaderContexts[ThreadIndex];

            pTileLoader->UnmapTile( &pTP->ID, pUnloaderContext );

            // Recycle the physical tile if one was used for this entry:
            if( pTP->ID.PTileID != D3D11_TILED_INVALID_PHYSICAL_ADDRESS )
            {
                m_TileRecycleQueue.SafeAddItem( pTP->ID.PTileID );

                pTP->ID.PTileID = D3D11_TILED_INVALID_PHYSICAL_ADDRESS;
            }

            // Notify the tile activity handlers:
            NotifyTileActivity( &pTP->ID, FALSE );

            // Update the state:
            pTP->State = TPS_Unmapped;
        }

        // Wait for the unloader queue signalling event:
        WaitForSingleObject( m_hUnloaderEvent, INFINITE );
    }
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::LoaderEntryPoint
// Desc: Main loop for a single loader thread.  It pops a TrackedTile* off the loading
//       queue, determines if it needs a new physical tile, and then passes the tracked
//       tile ID to a tile loader for processing.
//--------------------------------------------------------------------------------------
VOID TitleResidencyManager::LoaderEntryPoint( const UINT ThreadIndex )
{
    while( m_LoaderRunning )
    {
        BOOL Success = TRUE;
        while( Success && m_LoaderRunning )
        {
            while( m_bPaused )
            {
                YieldProcessor();
            }

            // Get a tracked tile from the queue:
            TrackedTile* pTP = NULL;
            Success = m_LoadQueue.SafeTryGet( &pTP );
            if( !Success )
            {
                continue;
            }

            // Check if the load request has already been loaded.
            // Since we can re-queue a load request if its priority changes, we can often
            // see entries in the queue that have already been loaded:
            if( pTP->State != TPS_QueuedForLoad )
            {
                continue;
            }
            
            // Update tile state:
            pTP->State = TPS_Loading;

            ITileLoader* pTileLoader = pTP->pTileLoader;
            ASSERT( pTileLoader != NULL );
            ASSERT( ThreadIndex < ARRAYSIZE(pTileLoader->m_pLoaderContexts) );
            VOID* pLoaderContext = pTileLoader->m_pLoaderContexts[ThreadIndex];

            // Check if the load request is out of date.  The load request can become out of date
            // if the tile hasn't been seen for a while and we need new tiles right now for other loads:
            if( IsTrackedTileExpired( pTP ) )
            {
                // This tile was never mapped, nor was a physical tile assigned; reassign to the unmapped state:
                ASSERT( pTP->ID.PTileID == D3D11_TILED_INVALID_PHYSICAL_ADDRESS );
                pTP->State = TPS_Unmapped;
            }
            else
            {
                // Check if the priority has changed for the worse since this tile was queued.
                // This can happen if a tile was originally observed in a place that deemed it high priority, but
                // while it was queued, the camera moved away and thus the request became lower priority.
                if( pTP->InsertPriority > 10 && pTP->CurrentPriority >= ( pTP->InsertPriority * 2 ) )
                {
                    // Re-queue the request at the current priority; we'll deal with it later when it comes off the
                    // queue again.
                    pTP->InsertPriority = pTP->CurrentPriority;
                    pTP->State = TPS_QueuedForLoad;
                    m_LoadQueue.SafeAddItem( pTP->InsertPriority, pTP );
                    SetEvent( m_hLoaderEvent );
                    continue;
                }

                DXUT_BeginPerfEvent( 0, L"Tile Loader Task" );

                // Determine if the virtual tile needs to be mapped to a unique physical tile:
                D3D11_TILED_PHYSICAL_ADDRESS TileID = D3D11_TILED_INVALID_PHYSICAL_ADDRESS;
                
                BOOL NeedsUniquePhysicalAddress = pTileLoader->TileNeedsUniquePhysicalTile( &pTP->ID );

                if( NeedsUniquePhysicalAddress )
                {
                    // We need a unique physical tile.

                    ASSERT( TileID == D3D11_TILED_INVALID_PHYSICAL_ADDRESS );

                    // Try to get a physical tile from the tile recycle queue:
                    m_TileRecycleQueue.SafeTryGet( &TileID );

                    // Try to allocate a new physical tile from the tile pool:
                    if( TileID == D3D11_TILED_INVALID_PHYSICAL_ADDRESS )
                    {
                        D3D11_TILED_SURFACE_DESC BaseDesc;
                        pTP->ID.pResource->GetSubresourceDesc( 0, &BaseDesc );

                        HRESULT hr = m_pTilePool->AllocatePhysicalTile( &TileID, BaseDesc.Format );
                        if( !SUCCEEDED(hr) )
                        {
                            TileID = D3D11_TILED_INVALID_PHYSICAL_ADDRESS;
                        }
                    }
                }

                if( TileID != D3D11_TILED_INVALID_PHYSICAL_ADDRESS || !NeedsUniquePhysicalAddress )
                {
                    // We have a physical tile; assign it to the tracked tile entry and pass it to the loader.

                    // Update tile state:
                    pTP->State = TPS_Loading;

                    // Set the selected physical tile to the tile ID.  If the virtual tile doesn't require a
                    // unique physical tile, this physical tile will be INVALID:
                    pTP->ID.PTileID = TileID;

                    // Pass the tile ID to the loader:
                    pTileLoader->LoadAndMapTile( &pTP->ID, pLoaderContext );

                    // Update tile state:
                    pTP->State = TPS_LoadedAndMapped;

                    // Reset the tiles now state:
                    m_NeedPhysicalTilesNow = FALSE;

                    // Notify listeners that a tile has been loaded:
                    NotifyTileActivity( &pTP->ID, TRUE );
                }
                else
                {
                    // We are out of physical tiles; re-add request to load queue so we can try again later:
                    pTP->InsertPriority = pTP->CurrentPriority;
                    pTP->State = TPS_QueuedForLoad;
                    m_LoadQueue.SafeAddItem( pTP->InsertPriority, pTP );
                    SetEvent( m_hLoaderEvent );
                    m_NeedPhysicalTilesNow = TRUE;
                }

                DXUT_EndPerfEvent();
            }
        }

        // Wait for the unloader queue signalling event:
        WaitForSingleObject( m_hLoaderEvent, INFINITE );
    }
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::Update
// Desc: Perform per-frame CPU tasks, including scanning the residency sample views for
//       samples, and updating tile states to move tiles through the streaming queues.
//--------------------------------------------------------------------------------------
VOID TitleResidencyManager::Update( FLOAT fDeltaTime )
{
    if( fDeltaTime <= 0.0f )
    {
        m_bPaused = TRUE;
        return;
    }
    else
    {
        m_bPaused = FALSE;
    }

    UINT LastFrameIndex = m_FrameIndex;
    ++m_FrameIndex;
    m_CurrentFrameTime = GetTickCount();

    // Iterate through the residency sample views and process each view that was updated
    // last frame:
    UINT ViewCount = (UINT)m_SamplingViews.size();
    for( UINT i = 0; i < ViewCount; ++i )
    {
        if( m_SamplingViews[i].RenderFrameIndex != LastFrameIndex )
        {
            CollectViewSamples( m_SamplingViews[i] );
        }
    }

    // Update the tracked tile collection, scheduling them for loading and unmapping as necessary:
    UpdateTileStates();

    m_ResidencyStats.NumTilesUnused = m_TileRecycleQueue.Size();
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::CreateResourceConstant
// Desc: Creates a single shader constant for drawing residency samples for a given
//       resource ID.
//--------------------------------------------------------------------------------------
VOID TitleResidencyManager::CreateResourceConstant( ResourceSetID RSID, XMFLOAT4* pConstant )
{
    pConstant->x = (FLOAT)RSID / 255.0f;
    pConstant->y = 1.0f / m_fGradientScalingFactor;
    pConstant->z = 0.0f;
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::BeginView
// Desc: Sets up a D3D device context for a residency sample render pass.  A corresponding
//       render width is specified that correlates the fixed size residency render view
//       (256x144) with a variable sized scene render (1280x720, 1920x1080, etc).
//--------------------------------------------------------------------------------------
UINT TitleResidencyManager::BeginView( ID3D11DeviceContext* pd3dDeviceContext, FLOAT FinalRenderWidth )
{
    // Compute the texel pitch multiplier that correlates a UV gradient in residency sample space to
    // a UV gradient in scene render space.
    g_TexelPitchMultiplier = (FLOAT)g_ResidencyViewWidth / FinalRenderWidth;

    // Select the next view index:
    m_CurrentViewIndex = m_NextViewIndex;
    m_NextViewIndex = ( m_NextViewIndex + 1 ) % m_SamplingViews.size();

    // Set up the rendertargets and render state:
    SamplingView& View = m_SamplingViews[m_CurrentViewIndex];
    ID3D11RenderTargetView* pViews[] = { View.pRenderTargetViewUVGradientID, View.pRenderTargetViewExtendedUVSlice };
    pd3dDeviceContext->OMSetRenderTargets( 2, pViews, m_pDepthStencil );
    pd3dDeviceContext->RSSetViewports( 1, &m_Viewport );

    pd3dDeviceContext->OMSetBlendState( m_pBlendState, NULL, 0xFFFFFFFF );
    pd3dDeviceContext->OMSetDepthStencilState( m_pDepthStencilState, 0 );

    FLOAT ClearColor[4] = { 0, 0, 0, 0 };
    pd3dDeviceContext->ClearRenderTargetView( View.pRenderTargetViewUVGradientID, ClearColor );
    pd3dDeviceContext->ClearRenderTargetView( View.pRenderTargetViewExtendedUVSlice, ClearColor );
    pd3dDeviceContext->ClearDepthStencilView( m_pDepthStencil, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0 );

    return m_CurrentViewIndex;
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::EndView
// Desc: Performs end of view operations for a residency sample render pass.
//--------------------------------------------------------------------------------------
VOID TitleResidencyManager::EndView( ID3D11DeviceContext* pd3dDeviceContext, UINT ViewID )
{
    ASSERT( ViewID == m_CurrentViewIndex );
    SamplingView& View = m_SamplingViews[ViewID];
    View.RenderFrameIndex = m_FrameIndex;

    // Copy the rendertarget textures to staging textures:
    pd3dDeviceContext->CopyResource( View.pStagingTextureUVGradientID, View.pRenderTargetTextureUVGradientID );
    pd3dDeviceContext->CopyResource( View.pStagingTextureExtendedUVSlice, View.pRenderTargetTextureExtendedUVSlice );

    // Unset the rendertargets:
    ID3D11RenderTargetView* NullViews[] = { NULL, NULL };
    pd3dDeviceContext->OMSetRenderTargets( ARRAYSIZE(NullViews), NullViews, NULL );

    m_CurrentViewIndex = (UINT)-1;
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::GetViewShaderResources
// Desc: Exposes shader resource views for a residency sample view.  These SR views are
//       used for debug rendering the residency views.
//--------------------------------------------------------------------------------------
VOID TitleResidencyManager::GetViewShaderResources( UINT ViewID, ID3D11ShaderResourceView** ppSRVUVGradientID, ID3D11ShaderResourceView** ppSRVExtendedUVSlice )
{
    if( ViewID >= (UINT)m_SamplingViews.size() )
    {
        return;
    }

    const SamplingView& View = m_SamplingViews[ViewID];

    if( ppSRVUVGradientID != NULL )
    {
        *ppSRVUVGradientID = View.pSRViewUVGradientID;
    }
    if( ppSRVExtendedUVSlice != NULL )
    {
        *ppSRVExtendedUVSlice = View.pSRViewExtendedUVSlice;
    }
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::CreateResourceSet
// Desc: Given arrays of tiled resources and corresponding tile loaders, this method
//       creates a resource set.  The resulting resource set ID is used to identify these
//       resources in a residency sample view.
//--------------------------------------------------------------------------------------
ResourceSetID TitleResidencyManager::CreateResourceSet( const ID3D11TiledTexture2D** ppResources, ITileLoader** ppTileLoaders, UINT ResourceCount )
{
    if( ResourceCount == 0 || ppResources == NULL )
    {
        return 0;
    }

    // Cannot have more than 255 resource sets:
    if( m_ResourceSets.size() > g_MaxResourceSetID )
    {
        return 0;
    }

    for( UINT i = 0; i < ResourceCount; ++i )
    {
        ASSERT( ppResources[i] != NULL );
    }

    // Create a new resource set:
    ResourceSet NewSet = { 0 };
    NewSet.ID = (ResourceSetID)m_ResourceSets.size();
    NewSet.ResourceCount = ResourceCount;

    NewSet.ppResources = new ID3D11TiledTexture2D*[ResourceCount];
    memcpy( NewSet.ppResources, ppResources, sizeof(ID3D11TiledTexture2D*) * ResourceCount );

    NewSet.ppTileLoaders = new ITileLoader*[ResourceCount];
    memcpy( NewSet.ppTileLoaders, ppTileLoaders, sizeof(ITileLoader*) * ResourceCount );

    for( UINT i = 0; i < ResourceCount; ++i )
    {
        ITileLoader* pTileLoader = NewSet.ppTileLoaders[i];

        // Ensure that each tile loader has been initialized:
        if( pTileLoader != NULL )
        {
            pTileLoader->m_pTilePool = m_pTilePool;
            pTileLoader->m_pRunning = &m_LoaderRunning;

            if( pTileLoader->m_pLoaderContexts[0] == NULL )
            {
                for( UINT i = 0; i < ARRAYSIZE(pTileLoader->m_pLoaderContexts); ++i )
                {
                    pTileLoader->m_pLoaderContexts[i] = pTileLoader->CreateThreadContext();
                }
                for( UINT i = 0; i < ARRAYSIZE(pTileLoader->m_pUnloaderContexts); ++i )
                {
                    pTileLoader->m_pUnloaderContexts[i] = pTileLoader->CreateThreadContext();
                }
            }
        }

        // Make sure the smallest mip level(s) of the resource never leaves residency:
        ID3D11TiledTexture2D* pResource = NewSet.ppResources[i];

        if( pResource != NULL )
        {
            // Compute the highest mip level for the tiled resource:
            D3D11_TILED_TEXTURE2D_DESC TexDesc;
            pResource->GetDesc( &TexDesc );
            UINT LastMipLevel = TexDesc.MipLevels - 1;

            // Loop through the array slices:
            for( UINT Slice = 0; Slice < TexDesc.ArraySize; ++Slice )
            {
                UINT SubresourceIndex = ( Slice * TexDesc.MipLevels ) + LastMipLevel;

                // Find the virtual address of the single tile for this slice:
                D3D11_TILED_VIRTUAL_ADDRESS VTileID = pResource->GetTileVirtualAddress( SubresourceIndex, 0.5f, 0.5f );

                // Create or retrieve the tracked tile entry for this virtual address:
                TrackedTile* pTP = IncrementSampleCount( VTileID, 0.5f, 0.5f, Slice, LastMipLevel, pResource, pTileLoader, 0 );

                // Set the pinned tile flag to TRUE, so the tile will be loaded high priority and will never be unmapped:
                pTP->ID.PinnedTile = TRUE;
            }
        }
    }

    m_ResourceSets.push_back( NewSet );
    return NewSet.ID;
}

//--------------------------------------------------------------------------------------
// Name: ComputeViewPositionScore
// Desc: Computes a score that is based on how close the X and Y coordinates are to the
//       center of the given viewport width and height.
//--------------------------------------------------------------------------------------
inline UINT ComputeViewPositionScore( UINT X, UINT Y, UINT Width, UINT Height )
{
    INT XCenter = (INT)X - (INT)( Width >> 1 );
    INT YCenter = (INT)Y - (INT)( Height >> 1 );

    XCenter >>= 3;
    YCenter >>= 3;

    UINT Score = ( XCenter * XCenter ) + ( YCenter * YCenter );

    return Score;
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::CollectViewSamples
// Desc: Locks a pair of residency view staging textures for reading, and selects a 
//       statistical sample of the texels to convert into residency samples.  Each selected
//       sample is processed and possibly converted into a tracked tile.
//--------------------------------------------------------------------------------------
VOID TitleResidencyManager::CollectViewSamples( SamplingView& View )
{
    DXUT_BeginPerfEvent( 0, L"Process Residency Samples" );

    // Map the staging textures for reading:
    D3D11_MAPPED_SUBRESOURCE MapData;
    m_pd3dDeviceContext->Map( View.pStagingTextureUVGradientID, 0, D3D11_MAP_READ, 0, &MapData );
    D3D11_MAPPED_SUBRESOURCE ExtMapData;
    m_pd3dDeviceContext->Map( View.pStagingTextureExtendedUVSlice, 0, D3D11_MAP_READ, 0, &ExtMapData );

    // Get the read pointers:
    const BYTE* pBits = (const BYTE*)MapData.pData;
    const BYTE* pExtBits = (const BYTE*)ExtMapData.pData;

    // Set up a hashing function to compute a "random" but repeating scan pattern
    // within a 64x4 window.
    // These constants were selected to grab 1/256 of the samples each frame, and to 
    // move the selection around as much as possible to get uniform coverage over
    // time.
    const UINT XFrequency = 64;
    const UINT XMultiplier = 27;
    const UINT YFrequency = 4;

    UINT Y = m_FrameIndex % YFrequency;

    for( ; Y < g_ResidencyViewHeight; Y += YFrequency )
    {
        // Compute the horizontal offset using the hashing constants:
        UINT Offset = ( ( m_FrameIndex + Y ) * XMultiplier ) % XFrequency;

        for( UINT X = Offset; X < g_ResidencyViewWidth; X += XFrequency )
        {
            // Compute a view position score from the original X and Y coordinates of the sample relative to
            // the dimensions of the residency sample view:
            UINT ViewPositionScore = ComputeViewPositionScore( X, Y, g_ResidencyViewWidth, g_ResidencyViewHeight );

            // Determine the byte offset to the sample:
            UINT ByteOffset = ( Y * MapData.RowPitch + X ) * sizeof(XMUBYTEN4);

            // Send the sample for processing:
            ProcessSample( *(const XMUBYTEN4*)( pBits + ByteOffset ), *(const XMUBYTEN4*)( pExtBits + ByteOffset ), ViewPositionScore );
        }
    }

    // Unmap the staging textures:
    m_pd3dDeviceContext->Unmap( View.pStagingTextureUVGradientID, 0 );
    m_pd3dDeviceContext->Unmap( View.pStagingTextureExtendedUVSlice, 0 );

    DXUT_EndPerfEvent();
}

//--------------------------------------------------------------------------------------
// Name: ComputeTextureLOD
// Desc: Given a tiled texture and a UV gradient from the residency sample render, this 
//       method computes a fractional mip LOD value.
//--------------------------------------------------------------------------------------
inline FLOAT ComputeTextureLOD( ID3D11TiledTexture2D* pResource, const FLOAT fGradient )
{
    // Compute the maximum dimension of the texture's base level (in texels):
    D3D11_TILED_SURFACE_DESC SurfDesc;
    pResource->GetSubresourceDesc( 0, &SurfDesc );
    FLOAT BaseMaxSize = (FLOAT)max( SurfDesc.TexelWidth, SurfDesc.TexelHeight );

    // Compute the amount of texels covered by the gradient value:
    FLOAT fTexelPitch = fGradient * BaseMaxSize * g_TexelPitchMultiplier;

    // Convert the texel count to a LOD value, using a base 2 logarithm:
    static const FLOAT fInvLog2 = 1.0f / logf( 2.0f );
    FLOAT fLOD = logf( fTexelPitch ) * fInvLog2;

    // Clamp the LOD value to 0 (no negative values allowed):
    return max( 0.0f, fLOD );
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::ProcessSample
// Desc: Given two raw 32 bit samples from the residency sample views, convert the bits
//       into (U, V, mip level, array slice) locations within a set of resources.  For
//       each of those combinations, pass the data along to be converted into tracked
//       tiles.
//--------------------------------------------------------------------------------------
VOID TitleResidencyManager::ProcessSample( const XMUBYTEN4& UVGradientIDSample, const XMUBYTEN4& ExtendedUVSliceSample, const UINT ViewPositionScore )
{
    // The W value on the UVGradientID sample is the resource set ID:
    ResourceSetID RSID = (ResourceSetID)UVGradientIDSample.w;
    if( RSID == 0 || RSID >= m_ResourceSets.size() )
    {
        return;
    }

    // Retrieve the resource set:
    ResourceSet& RSet = m_ResourceSets[RSID];
    ASSERT( RSet.ID == RSID );

    // Decode the UV gradient from the Z component of the UVGradientID sample:
    const FLOAT fEncodedGradient = (FLOAT)UVGradientIDSample.z / 255.0f;
    const FLOAT fLogGradient = fEncodedGradient * m_fGradientScalingFactor;
    const FLOAT fGradient = expf( fLogGradient );

    // Decode the UV whole number components from the XY components of the ExtendedUVSlice sample:
    const FLOAT fTexWholeU = (FLOAT)( (INT)ExtendedUVSliceSample.x - 128 );
    const FLOAT fTexWholeV = (FLOAT)( (INT)ExtendedUVSliceSample.y - 128 );

    // Compute the whole + fractional texture UV from the XY components of the UVGradientID sample:
    FLOAT fTexU = fTexWholeU + (FLOAT)UVGradientIDSample.x / 255.0f;
    FLOAT fTexV = fTexWholeV + (FLOAT)UVGradientIDSample.y / 255.0f;

    // Loop over each resource in the resource set: 
    UINT ResourceCount = RSet.ResourceCount;
    for( UINT i = 0; i < ResourceCount; ++i )
    {
        // Retrieve the resource pointer.
        ID3D11TiledTexture2D* pResource = RSet.ppResources[i];
        ASSERT( pResource != NULL );
        ITileLoader* pTileLoader = RSet.ppTileLoaders[i];

        // Compute the texture LOD from the UV gradient and the resource:
        FLOAT fLOD = ComputeTextureLOD( pResource, fGradient );

        // The Low LOD is the largest whole number less than the fractional LOD:
        UINT LowLOD = (UINT)floorf( fLOD );

        // Determine the number of mip levels in the resource:
        D3D11_TILED_TEXTURE2D_DESC TexDesc;
        pResource->GetDesc( &TexDesc );
        const UINT MipLevelCount = TexDesc.MipLevels;
        UINT SliceIndex = 0;

        // Decode the array slice if the texture has array slices:
        if( TexDesc.ArraySize > 1 )
        {
            SliceIndex = ExtendedUVSliceSample.z;

            // If the resource is quilted, convert the extended (0..N) UV coordinates to
            // normalized (0..1) UV and array slice index from the quilting configuration:
            pResource->ConvertQuiltUVToArrayUVSlice( &fTexU, &fTexV, &SliceIndex );
        }

        // Loop over the LOD levels up to the maximum LOD index:
        for( UINT LOD = LowLOD; LOD < MipLevelCount; ++LOD )
        {
            // Determine the subresource index for the slice index and the mip LOD:
            UINT Subresource = ( SliceIndex * MipLevelCount ) + LOD;

            // Determine the virtual address of the tile, given the subresource index and the normalized UV coordinates:
            D3D11_TILED_VIRTUAL_ADDRESS VTileID = pResource->GetTileVirtualAddress( Subresource, fTexU, fTexV );

            // If we have an invalid result, skip this resource.
            if( VTileID == D3D11_TILED_INVALID_VIRTUAL_ADDRESS )
            {
                break;
            }

            // Pass the virtual address and all other computed information on to the next step for processing:
            IncrementSampleCount( VTileID, fTexU, fTexV, SliceIndex, LOD, pResource, pTileLoader, ViewPositionScore );
        }
    }
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::IncrementSampleCount
// Desc: Given a virtual tile address and all of the sample data that led to that virtual
//       address, find or create a tracked tile entry, populate it, and increment its
//       sample count.
//--------------------------------------------------------------------------------------
TitleResidencyManager::TrackedTile* TitleResidencyManager::IncrementSampleCount( D3D11_TILED_VIRTUAL_ADDRESS VTileID, FLOAT TexU, FLOAT TexV, UINT SliceIndex, UINT MipLevel, ID3D11TiledTexture2D* pResource, ITileLoader* pTileLoader, const UINT ViewPositionScore )
{
    // Validate the virtual tile address:
    if( VTileID == D3D11_TILED_INVALID_VIRTUAL_ADDRESS )
    {
        return NULL;
    }

    TrackedTile* pTP = NULL;

    // Use a one-element cache to reduce hashtable lookups:
    static D3D11_TILED_VIRTUAL_ADDRESS s_LastID = D3D11_TILED_INVALID_VIRTUAL_ADDRESS;
    static TrackedTile* s_pLastTile = NULL;

    if( VTileID == s_LastID )
    {
        pTP = s_pLastTile;
    }
    else
    {
        // Search for tile in hash table:
        TrackedTileMap::iterator iter = m_TrackedTileMap.find( VTileID );
        if( iter != m_TrackedTileMap.end() )
        {
            pTP = iter->second;
        }
        else
        {
            // Tile not found; create a new tile:
            pTP = AddVirtualTile( VTileID, TexU, TexV, SliceIndex, MipLevel, pResource, pTileLoader, ViewPositionScore );
        }
    }

    ASSERT( pTP != NULL );

    // Increment & update counters for this tile:
    pTP->SampleCount++;
    pTP->LastTimeSeen = m_CurrentFrameTime;
    pTP->ViewPositionScore = ViewPositionScore;

    // If the tile is scheduled for unmapping, change it back to Seen:
    if( pTP->State == TPS_Unmapped )
    {
        pTP->State = TPS_Seen;
    }

    // Update our one-element cache:
    s_LastID = VTileID;
    s_pLastTile = pTP;

    return pTP;
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::AddVirtualTile
// Desc: Adds a new virtual tile entry to the title residency manager tile tracking lists.
//       This method does not check if the tile is already being tracked; that is the
//       caller's responsibility.
//--------------------------------------------------------------------------------------
TitleResidencyManager::TrackedTile* TitleResidencyManager::AddVirtualTile( D3D11_TILED_VIRTUAL_ADDRESS VTileID, FLOAT TexU, FLOAT TexV, UINT SliceIndex, UINT MipLevel, ID3D11TiledTexture2D* pResource, ITileLoader* pTileLoader, const UINT ViewPositionScore )
{
    ASSERT( VTileID != D3D11_TILED_INVALID_VIRTUAL_ADDRESS );

    // See if we have a tracked tile entry to reuse, instead of allocating a new one:
    TrackedTile* pTP = NULL;
    if( !m_TrackedTileFreeList.empty() )
    {
        // We can reuse a tracked tile entry:
        pTP = m_TrackedTileFreeList.top();
        m_TrackedTileFreeList.pop();
    }
    else
    {
        // We must allocate a new tracked tile entry:
        pTP = new TrackedTile();
    }
    ASSERT( pTP != NULL );

    // Fill in the tracked tile entry:
    pTP->ID.pResource = pResource;
    pTP->ID.VTileID = VTileID;
    pTP->ID.PTileID = D3D11_TILED_INVALID_PHYSICAL_ADDRESS;
    pTP->ID.U = TexU;
    pTP->ID.V = TexV;
    pTP->ID.MipLevel = (USHORT)MipLevel;
    pTP->ID.ArraySlice = (USHORT)SliceIndex;
    pTP->ID.PinnedTile = FALSE;
    pTP->LastTimeSeen = 0;
    pTP->SampleCount = 0;
    pTP->ViewPositionScore = ViewPositionScore;
    pTP->State = TPS_Seen;
    pTP->pTileLoader = pTileLoader;

    pTP->InsertPriority = 0;
    pTP->CurrentPriority = 0;

    // Add the tracked tile entry to the map, using the virtual address:
    m_TrackedTileMap[VTileID] = pTP;

    // Add the tracked tile entry to the end of the sorted list (the list will be re-sorted once per frame):
    m_TrackedTileSortList.push_back( pTP );

    return pTP;
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::TrackedTileSortPredicate
// Desc: Sorting predicate for sorting the tracked tile list.  The sort predicate uses
//       the last time seen and the sample count for sorting.
//--------------------------------------------------------------------------------------
bool TitleResidencyManager::TrackedTileSortPredicate( const TrackedTile* pA, const TrackedTile* pB )
{
    if( pA->LastTimeSeen != pB->LastTimeSeen )
    {
        return pA->LastTimeSeen < pB->LastTimeSeen;
    }
    return pA->SampleCount < pB->SampleCount;
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::ComputePriority
// Desc: Computes a priority score for a tracked tile, given a variety of factors.
//       Lower scores cause tiles to be loaded before tiles with higher scores.
//--------------------------------------------------------------------------------------
UINT TitleResidencyManager::ComputePriority( TrackedTile* pTP ) const
{
    // Tiles that are not currently visible are given a huge penalty:
    const UINT CurrentlyVisibleScore = ( pTP->LastTimeSeen == m_CurrentFrameTime ) ? 0 : 100000;

    // Higher mip levels are loaded first (this causes less popping onscreen):
    const UINT MipScore = max( 0, ( 8 - (INT)pTP->ID.MipLevel ) ) * 1000;

    // Tiles closer to the center of the render view are loaded before tiles on the edges
    // of the render view:
    const UINT ViewPositionScore = pTP->ViewPositionScore * 10;

    // Accumulate the scores and return:
    return MipScore + ViewPositionScore + CurrentlyVisibleScore;
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::UpdateTileStates
// Desc: This method is executed once a frame to look over all tracked tiles and determine
//       if each tile needs to be pushed forward in the state machine.  Seen tiles are
//       queued for load, loaded tiles are queued for unmapping, and unmapped tiles are
//       destroyed.  This is also where most of the per-frame statistics are gathered.
//--------------------------------------------------------------------------------------
VOID TitleResidencyManager::UpdateTileStates()
{
    // Sort the tracked tile list:
    m_TrackedTileSortList.sort( TrackedTileSortPredicate );

    // Reset some of the stats counters:
    m_ResidencyStats.NumTilesQueuedForLoad = 0;
    m_ResidencyStats.NumTilesLoaded = 0;
    m_ResidencyStats.NumTilesTracked = 0;
    m_ResidencyStats.NumTilesUnused = 0;

    TrackedTileSortList::iterator iter = m_TrackedTileSortList.begin();
    TrackedTileSortList::iterator end = m_TrackedTileSortList.end();

    // UnmapCount is a circuit breaker that allows us to crudely reclaim a certain amount
    // of physical tiles per frame when we really, really need them:
    UINT UnmapCount = 0;
    if( m_NeedPhysicalTilesNow )
    {
        UnmapCount = 10;
    }

    // Loop through the sorted list:
    while( iter != end )
    {
        TrackedTile* pTP = *iter;
        
        // Obtain the next iterator position, so we can delete from the list while we are iterating through it:
        TrackedTileSortList::iterator next_iter = iter;
        ++next_iter;

        // Increment the tiles tracked counter:
        ++m_ResidencyStats.NumTilesTracked;

        switch( pTP->State )
        {
        case TPS_Seen:
            // The residency sampler has seen this tile.  Once the sample count reaches 10, queue it for loading:
            if( pTP->SampleCount > 10 || pTP->ID.PinnedTile )
            {
                QueueTileForLoadAndMap( pTP, TRUE );
            }
            else if( ( m_CurrentFrameTime - pTP->LastTimeSeen ) >= ( 5 * 1000 ) )
            {
                // The residency sampler saw this tile, but we haven't seen it since for 5 seconds.
                // Set it to unmapped so it will be removed from tracking next frame:
                pTP->State = TPS_Unmapped;
            }
            break;
        case TPS_QueuedForLoad:
            ++m_ResidencyStats.NumTilesQueuedForLoad;

            // Determine if we need to requeue the request due to a lower priority.
            // This happens often; a tile is initially observed at the edge of the screen, giving
            // it a relatively low priority (high priority value) at the time that the request
            // is added to the load queue.  However, the camera can move while the request is 
            // queued, elevating the request's priority (lowering the CurrentPriority value).
            // If the current priority drops to less than half of the priority value at insertion
            // time, we will re-queue the request at the updated priority value.
            // Note that this will be a duplicate request in the queue for the same tile; care must
            // be taken in the loader thread to discard duplicate requests when they are encountered.
            pTP->CurrentPriority = ComputePriority( pTP );
            if( pTP->CurrentPriority <= ( pTP->InsertPriority >> 1 ) )
            {
                pTP->InsertPriority = pTP->CurrentPriority;
                QueueTileForLoadAndMap( pTP, FALSE );
            }
            break;
        case TPS_Loading:
            ++m_ResidencyStats.NumTilesQueuedForLoad;
            break;
        case TPS_LoadedAndMapped:
            ++m_ResidencyStats.NumTilesLoaded;

            // Determine if we need to unload a currently loaded tile.  UnmapCount is a circuit breaker that
            // limits the amount of unload requests per frame.
            if( IsTrackedTileExpired( pTP ) && UnmapCount > 0 )
            {
                ASSERT( pTP->ID.PinnedTile == FALSE );
                QueueTileForUnmap( pTP );
                --UnmapCount;
            }
            break;
        case TPS_Unmapped:
            {
                // This tile is fully unmapped.  Remove it from the hash table and the sort list:
                ASSERT( pTP->ID.PTileID == D3D11_TILED_INVALID_PHYSICAL_ADDRESS );
                ASSERT( pTP->ID.PinnedTile == FALSE );

                TrackedTileMap::iterator tpmiter = m_TrackedTileMap.find( pTP->ID.VTileID );
                if( tpmiter != m_TrackedTileMap.end() )
                {
                    m_TrackedTileMap.erase( tpmiter );
                }
                m_TrackedTileSortList.erase( iter );

                m_TrackedTileFreeList.push( pTP );
                pTP = NULL;
            }
            break;
        }

        // Go to the next item in the list:
        iter = next_iter;
    }

    // The residency stats includes a boolean that reflects whether we are currently waiting 
    // for physical tiles to be freed:
    m_ResidencyStats.OutOfPhysicalTiles = m_NeedPhysicalTilesNow;
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::QueueTileForLoadAndMap
// Desc: Adds a tracked tile to the loader priority queue, after optionally recomputing 
//       its load priority.
//--------------------------------------------------------------------------------------
VOID TitleResidencyManager::QueueTileForLoadAndMap( TrackedTile* pTP, BOOL RecomputePriority )
{
    // a tile ready for load queuing should not have an assigned physical tile ID
    if( pTP->ID.PTileID != D3D11_TILED_INVALID_PHYSICAL_ADDRESS )
    {
        return;
    }

    // Optionally recompute load priority:
    if( RecomputePriority )
    {
        pTP->State = TPS_QueuedForLoad;
        pTP->CurrentPriority = ComputePriority( pTP );
        pTP->InsertPriority = pTP->CurrentPriority;
    }

    // Add tile to the loader queue:
    m_LoadQueue.SafeAddItem( pTP->InsertPriority, pTP );

    // Signal to the loader threads that the queue is not empty:
    SetEvent( m_hLoaderEvent );
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::QueueTileForUnmap
// Desc: Adds a tracked tile to the unmap queue.
//--------------------------------------------------------------------------------------
VOID TitleResidencyManager::QueueTileForUnmap( TrackedTile* pTP )
{
    // Change tile state:
    pTP->State = TPS_QueuedForUnmap;

    // Add tile to unmap queue:
    m_UnmapQueue.SafeAddItem( pTP );

    // Signal to unloader thread that the queue is not empty:
    SetEvent( m_hUnloaderEvent );
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::IsTrackedTileExpired
// Desc: Determines if a tracked tile can be unmapped immediately if necessary.
//--------------------------------------------------------------------------------------
BOOL TitleResidencyManager::IsTrackedTileExpired( TrackedTile* pTP ) const
{
    if( pTP->ID.PinnedTile )
    {
        return FALSE;
    }
    if( m_NeedPhysicalTilesNow )
    {
        // Emergency expire tiles that haven't been seen for a half second or more
        return ( m_CurrentFrameTime - pTP->LastTimeSeen ) >= 500;
    }
    else
    {
        return FALSE;
    }
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::RegisterTileActivityHandler
// Desc: Adds a tile activity handler to the list.
//--------------------------------------------------------------------------------------
VOID TitleResidencyManager::RegisterTileActivityHandler( ITileActivityHandler* pHandler )
{
    m_TileActivityHandlers.push_back( pHandler );
}

//--------------------------------------------------------------------------------------
// Name: TitleResidencyManager::NotifyTileActivity
// Desc: Sends a loaded or unloaded notification to each of the tile activity handlers.
//--------------------------------------------------------------------------------------
VOID TitleResidencyManager::NotifyTileActivity( const TrackedTileID* pTileID, BOOL Loaded ) const
{
    UINT HandlerCount = (UINT)m_TileActivityHandlers.size();
    for( UINT i = 0; i < HandlerCount; ++i )
    {
        if( Loaded )
        {
            m_TileActivityHandlers[i]->TileLoaded( pTileID );
        }
        else
        {
            m_TileActivityHandlers[i]->TileUnloaded( pTileID );
        }
    }
}
