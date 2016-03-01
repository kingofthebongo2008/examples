//--------------------------------------------------------------------------------------
// PageRenderer.cpp
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "PageRenderer.h"
#include "TypedPagePool.h"
#include "PhysicalPageManager.h"
#include "TiledResourceBase.h"

#include "TiledResourceRuntimeTest.h"
using namespace TiledRuntimeTest;

namespace TiledRuntime
{
    //--------------------------------------------------------------------------------------
    // Name: PageRenderer constructor
    //--------------------------------------------------------------------------------------
    PageRenderer::PageRenderer( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dDeviceContext )
    {
        m_pd3dDevice = pd3dDevice;
        m_pd3dDevice->AddRef();
		m_pd3dDeviceContext = pd3dDeviceContext;
        m_pd3dDeviceContext->AddRef();

		InitializeCriticalSection( &m_QueueCritSec );

        ZeroMemory( m_pIntermediateTextures, sizeof(m_pIntermediateTextures) );
    }

    //--------------------------------------------------------------------------------------
    // Name: PageRenderer destructor
    // Desc: Frees all D3D11 objects held by the page renderer.
    //--------------------------------------------------------------------------------------
    PageRenderer::~PageRenderer()
    {
        while( !m_PendingOperations.empty() )
        {
            RenderOperation& Op = m_PendingOperations.front();
            if( Op.pSrcBuffer != NULL )
            {
                delete[] Op.pSrcBuffer;
            }
            m_PendingOperations.pop_front();
        }

        DeleteCriticalSection( &m_QueueCritSec );
        SAFE_RELEASE( m_pd3dDevice );
        SAFE_RELEASE( m_pd3dDeviceContext );

        for( UINT i = 0; i < ARRAYSIZE(m_pIntermediateTextures); ++i )
        {
            SAFE_RELEASE( m_pIntermediateTextures[i] );
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: PageRenderer::QueuePageUpdate
    // Desc: Queues an operation that copies an untyped page buffer into an emulated
    //       physical page, which is a rectangular region within the given typed page pool.
    //--------------------------------------------------------------------------------------
    HRESULT PageRenderer::QueuePageUpdate( TypedPagePool* pPagePool, INT PageIndex, const VOID* pPageBuffer )
    {
        if( PageIndex == -1 )
        {
            return E_FAIL;
        }
        const AtlasEntry* pAtlasEntry = pPagePool->GetAtlasEntry( PageIndex );
        ASSERT( pAtlasEntry->PageID != INVALID_PHYSICAL_PAGE_ID );

        // compute the destination rectangle within the array texture slice
        RECT PageRect = pPagePool->GetPageRect( pAtlasEntry );

        RenderOperation Op;

		// set the resolve texture
        Op.pResolveTexture = pPagePool->GetArrayTexture();
        Op.ResolveSliceIndex = pAtlasEntry->Slice;

        // set the temp linear texture as the source texture
		SIZE PageSize = GetPageSizeTexels( pPagePool->GetFormat() );
		RECT SrcRect = { 0, 0, PageSize.cx, PageSize.cy };
        Op.SrcRect = SrcRect;
        Op.SrcSliceIndex = 0;
        Op.pSrcArrayTexture = NULL;

		Op.pSrcBuffer = new BYTE[PAGE_SIZE_BYTES];
		memcpy( Op.pSrcBuffer, pPageBuffer, PAGE_SIZE_BYTES );
		Op.SrcRowPitchBytes = PageSize.cx * GetBytesPerTexel( pPagePool->GetFormat() );
		switch( pPagePool->GetFormat() )
		{
        case DXGI_FORMAT_BC1_TYPELESS: 
        case DXGI_FORMAT_BC1_UNORM : 
        case DXGI_FORMAT_BC1_UNORM_SRGB: 
        case DXGI_FORMAT_BC4_TYPELESS: 
        case DXGI_FORMAT_BC4_UNORM : 
        case DXGI_FORMAT_BC4_SNORM : 
			Op.SrcRowPitchBytes = ( PageSize.cx / 4 ) * 8;
			break;
        case DXGI_FORMAT_BC2_TYPELESS: 
        case DXGI_FORMAT_BC2_UNORM : 
        case DXGI_FORMAT_BC2_UNORM_SRGB: 
        case DXGI_FORMAT_BC3_TYPELESS: 
        case DXGI_FORMAT_BC3_UNORM : 
        case DXGI_FORMAT_BC3_UNORM_SRGB: 
        case DXGI_FORMAT_BC5_TYPELESS: 
        case DXGI_FORMAT_BC5_UNORM : 
        case DXGI_FORMAT_BC5_SNORM : 
        case DXGI_FORMAT_BC6H_TYPELESS : 
        case DXGI_FORMAT_BC6H_UF16 : 
        case DXGI_FORMAT_BC6H_SF16 : 
        case DXGI_FORMAT_BC7_TYPELESS: 
        case DXGI_FORMAT_BC7_UNORM : 
        case DXGI_FORMAT_BC7_UNORM_SRGB: 
			Op.SrcRowPitchBytes = ( PageSize.cx / 4 ) * 16;
			break;
		}

        // draw to the page rect within the atlas
        Op.DrawRect = PageRect;

        QueueOperation( Op );

        Trace::FillPage( pAtlasEntry->PageID, pPagePool->GetFormat() );

        return S_OK;
    }

    //--------------------------------------------------------------------------------------
    // Name: PageRenderer::QueueBorderUpdate
    // Desc: Queues an operation that copies edge texels from one page into the border
    //       texels of another page.
    //--------------------------------------------------------------------------------------
    HRESULT PageRenderer::QueueBorderUpdate( TypedPagePool* pPagePool, PhysicalPageID CenterPageID, PhysicalPageID BorderPageID, PageNeighbors RelationshipToCenterPage, BOOL InvertSourceRelationship )
    {
        INT CenterPageIndex = pPagePool->FindPage( CenterPageID );
        INT BorderPageIndex = -1;
        if( BorderPageID != INVALID_PHYSICAL_PAGE_ID ) 
        {
            BorderPageIndex = pPagePool->FindPage( BorderPageID );
        }

        if( CenterPageIndex == -1 )
        {
            return E_FAIL;
        }

        const AtlasEntry* pCenterAtlasEntry = pPagePool->GetAtlasEntry( CenterPageIndex );
        ASSERT( pCenterAtlasEntry != NULL );

        const AtlasEntry* pBorderAtlasEntry = NULL;
        if( BorderPageIndex != -1 )
        {
            pBorderAtlasEntry = pPagePool->GetAtlasEntry( BorderPageIndex );
        }

        UINT BorderTexelCount = GetPageBorderTexelCount( pPagePool->GetFormat() );

        RECT SourceRect = { 0, 0, 0, 0 };
        if( pBorderAtlasEntry != NULL )
        {
            // compute source rectangle from border page and relationship of border page to center page
            const RECT BorderPageRect = pPagePool->GetPageRect( pBorderAtlasEntry );
            SourceRect = BorderPageRect;
            PageNeighbors SourceRelationshipToCenterPage = RelationshipToCenterPage;
            if( InvertSourceRelationship )
            {
                SourceRelationshipToCenterPage = GetOppositeNeighbor( SourceRelationshipToCenterPage );
            }
            switch( SourceRelationshipToCenterPage )
            {
            case PN_TOP:
                SourceRect.top = SourceRect.bottom - BorderTexelCount;
                break;
            case PN_BOTTOM:
                SourceRect.bottom = SourceRect.top + BorderTexelCount;
                break;
            case PN_LEFT:
                SourceRect.left = SourceRect.right - BorderTexelCount;
                break;
            case PN_RIGHT:
                SourceRect.right = SourceRect.left + BorderTexelCount;
                break;
            case PN_TOPLEFT:
                SourceRect.top = SourceRect.bottom - BorderTexelCount;
                SourceRect.left = SourceRect.right - BorderTexelCount;
                break;
            case PN_TOPRIGHT:
                SourceRect.top = SourceRect.bottom - BorderTexelCount;
                SourceRect.right = SourceRect.left + BorderTexelCount;
                break;
            case PN_BOTTOMLEFT:
                SourceRect.bottom = SourceRect.top + BorderTexelCount;
                SourceRect.left = SourceRect.right - BorderTexelCount;
                break;
            case PN_BOTTOMRIGHT:
                SourceRect.bottom = SourceRect.top + BorderTexelCount;
                SourceRect.right = SourceRect.left + BorderTexelCount;
                break;
            }
        }

        // compute destination rectangle from center page and relationship of border page to center page
        const RECT CenterPageRect = pPagePool->GetPageRect( pCenterAtlasEntry );
        RECT DestRect = CenterPageRect;
        PageNeighbors DestRelationshipToCenterPage = RelationshipToCenterPage;
        switch( DestRelationshipToCenterPage )
        {
        case PN_TOP:
            DestRect.bottom = DestRect.top;
            DestRect.top -= BorderTexelCount;
            break;
        case PN_BOTTOM:
            DestRect.top = DestRect.bottom;
            DestRect.bottom += BorderTexelCount;
            break;
        case PN_LEFT:
            DestRect.right = DestRect.left;
            DestRect.left -= BorderTexelCount;
            break;
        case PN_RIGHT:
            DestRect.left = DestRect.right;
            DestRect.right += BorderTexelCount;
            break;
        case PN_TOPLEFT:
            DestRect.bottom = DestRect.top;
            DestRect.top -= BorderTexelCount;
            DestRect.right = DestRect.left;
            DestRect.left -= BorderTexelCount;
            break;
        case PN_TOPRIGHT:
            DestRect.bottom = DestRect.top;
            DestRect.top -= BorderTexelCount;
            DestRect.left = DestRect.right;
            DestRect.right += BorderTexelCount;
            break;
        case PN_BOTTOMLEFT:
            DestRect.top = DestRect.bottom;
            DestRect.bottom += BorderTexelCount;
            DestRect.right = DestRect.left;
            DestRect.left -= BorderTexelCount;
            break;
        case PN_BOTTOMRIGHT:
            DestRect.top = DestRect.bottom;
            DestRect.bottom += BorderTexelCount;
            DestRect.left = DestRect.right;
            DestRect.right += BorderTexelCount;
            break;
        }

        if( pBorderAtlasEntry != NULL )
        {
            ASSERT( ( SourceRect.right - SourceRect.left ) == ( DestRect.right - DestRect.left ) );
            ASSERT( ( SourceRect.bottom - SourceRect.top ) == ( DestRect.bottom - DestRect.top ) );
        }

        RenderOperation Op;

        Op.pResolveTexture = pPagePool->GetArrayTexture();
        Op.ResolveSliceIndex = pCenterAtlasEntry->Slice;

        Op.pIntermediateTexture = GetIntermediateTexture( pPagePool->GetFormat() );
        
        if( pBorderAtlasEntry != NULL )
        {
            Op.pSrcArrayTexture = Op.pResolveTexture;
            Op.SrcSliceIndex = pBorderAtlasEntry->Slice;
            Op.SrcRect = SourceRect;
        }
        else
        {
            Op.FillColor = XMFLOAT4( 0, 0, 0, 0 );
            Op.pSrcArrayTexture = NULL;
        }

        Op.DrawRect = DestRect;

        QueueOperation( Op );

        Trace::UpdatePageBorder( CenterPageID, BorderPageID, RelationshipToCenterPage );

        return S_OK;
    }

    //--------------------------------------------------------------------------------------
    // Name: PageRenderer::GetIntermediateTexture
    // Desc: Creates a 2D texture in the desired format, sized for a single page in that format.
    //--------------------------------------------------------------------------------------
    ID3D11Texture2D* PageRenderer::GetIntermediateTexture( DXGI_FORMAT DataFormat )
    {
        if( m_pIntermediateTextures[DataFormat] != NULL )
        {
            return m_pIntermediateTextures[DataFormat];
        }

        D3D11_TEXTURE2D_DESC TexDesc;
        ZeroMemory( &TexDesc, sizeof(D3D11_TEXTURE2D_DESC) );

        SIZE PageSize = GetPageSizeTexels( DataFormat );
        TexDesc.Width = PageSize.cx;
        TexDesc.Height = PageSize.cy;
        TexDesc.Format = GetPagePoolArrayTextureFormat( DataFormat );
        TexDesc.ArraySize = 1;
        TexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        TexDesc.MipLevels = 1;
        TexDesc.SampleDesc.Count = 1;
        TexDesc.Usage = D3D11_USAGE_DEFAULT;

        ID3D11Texture2D* pTexture2D = NULL;
        m_pd3dDevice->CreateTexture2D( &TexDesc, NULL, &pTexture2D );

        ASSERT( pTexture2D != NULL );

        m_pIntermediateTextures[DataFormat] = pTexture2D;

        return pTexture2D;
    }

    //--------------------------------------------------------------------------------------
    // Name: PageRenderer::QueueOperation
    // Desc: Adds a render operation to the queue, in a thread safe manner.
    //--------------------------------------------------------------------------------------
    VOID PageRenderer::QueueOperation( RenderOperation& Operation )
    {
        // TODO: validate op

        EnterCriticalSection( &m_QueueCritSec );
        m_PendingOperations.push_back( Operation );
        LeaveCriticalSection( &m_QueueCritSec );
    }

    //--------------------------------------------------------------------------------------
    // Name: PageRenderer::FlushPendingUpdates
    // Desc: Processes all render operations in the queue, passing each operation to its
    //       appropriate execution function.  This method is called once per render frame,
    //       and holds a lock on the queue while it is operating.
    //--------------------------------------------------------------------------------------
    VOID PageRenderer::FlushPendingUpdates()
    {
        PIXBeginNamedEvent( 0, "Page Renderer Operations" );

        // TODO: sort operations by source and destination

        BOOL CritSec = FALSE;
        if( !m_PendingOperations.empty() )
        {
            EnterCriticalSection( &m_QueueCritSec );
            CritSec = TRUE;
        }

        while( !m_PendingOperations.empty() )
        {
            RenderOperation& Op = m_PendingOperations.front();
			if( Op.pSrcBuffer != NULL )
			{
				ExecuteUpdateSubresourceOperation( Op );
				delete[] Op.pSrcBuffer;
			}
			else if( Op.pSrcArrayTexture != NULL )
			{
				ExecuteTextureCopyOperation( Op );
			}
			else
			{
				ExecuteColorFillOperation( Op );
			}
            m_PendingOperations.pop_front();
        }

        if( CritSec )
        {
            LeaveCriticalSection( &m_QueueCritSec );
        }

        PIXEndNamedEvent();
    }

    //--------------------------------------------------------------------------------------
    // Name: PageRenderer::ExecuteUpdateSubresourceOperation
    // Desc: Copies a source buffer to a region within a destination resource.  This is used
    //       to copy page data into a typed page pool array texture.
    //--------------------------------------------------------------------------------------
    VOID PageRenderer::ExecuteUpdateSubresourceOperation( const RenderOperation& Operation )
    {
        PIXBeginNamedEvent( 0, "Page Atlas Texture Update" );

		D3D11_BOX DestBox;
		DestBox.back = 1;
		DestBox.front = 0;
		DestBox.left = Operation.DrawRect.left;
		DestBox.top = Operation.DrawRect.top;
		DestBox.right = Operation.DrawRect.right;
		DestBox.bottom = Operation.DrawRect.bottom;

		m_pd3dDeviceContext->UpdateSubresource( Operation.pResolveTexture, Operation.ResolveSliceIndex, &DestBox, Operation.pSrcBuffer, Operation.SrcRowPitchBytes, 0 );

        PIXEndNamedEvent();
    }

    //--------------------------------------------------------------------------------------
    // Name: PageRenderer::ExecuteTextureCopyOperation
    // Desc: Copies texels from one region of a texture into another region of a texture.
    //       Note that when the source and destination textures are the same texture, an
    //       intermediate texture must be used, because D3D11 cannot copy from one texture
    //       directly to itself.
    //--------------------------------------------------------------------------------------
	VOID PageRenderer::ExecuteTextureCopyOperation( const RenderOperation& Operation )
	{
		PIXBeginNamedEvent( 0, "Page Atlas Texture Update" );

		D3D11_BOX SrcBox;
		SrcBox.back = 1;
		SrcBox.front = 0;
		SrcBox.left = Operation.SrcRect.left;
		SrcBox.top = Operation.SrcRect.top;
		SrcBox.right = Operation.SrcRect.right;
		SrcBox.bottom = Operation.SrcRect.bottom;

		ID3D11Resource* pSrcResource = Operation.pSrcArrayTexture;

        if( Operation.pSrcArrayTexture != Operation.pResolveTexture )
        {
            m_pd3dDeviceContext->CopySubresourceRegion( Operation.pResolveTexture, Operation.ResolveSliceIndex, Operation.DrawRect.left, Operation.DrawRect.top, 0, pSrcResource, Operation.SrcSliceIndex, &SrcBox );
        }
        else
        {
            ASSERT( Operation.pIntermediateTexture != NULL );

            // copy texels from src rect of src texture to upper left corner of intermediate texture
            m_pd3dDeviceContext->CopySubresourceRegion( Operation.pIntermediateTexture, 0, 0, 0, 0, Operation.pSrcArrayTexture, Operation.SrcSliceIndex, &SrcBox );

            D3D11_BOX IntermediateBox;
            IntermediateBox.back = 1;
            IntermediateBox.front = 0;
            IntermediateBox.left = 0;
            IntermediateBox.top = 0;
            IntermediateBox.right = Operation.SrcRect.right - Operation.SrcRect.left;
            IntermediateBox.bottom = Operation.SrcRect.bottom - Operation.SrcRect.top;

            // copy texels from intermediate texture to draw rect within destination texture
            m_pd3dDeviceContext->CopySubresourceRegion( Operation.pResolveTexture, Operation.ResolveSliceIndex, Operation.DrawRect.left, Operation.DrawRect.top, 0, Operation.pIntermediateTexture, 0, &IntermediateBox );
        }

		PIXEndNamedEvent();
	}

    //--------------------------------------------------------------------------------------
    // Name: PageRenderer::ExecuteColorFillOperation
    // Desc: Copies a single texel value into an index map texture, or alternately copies a
    //       blank region from an intermediate texture into a texture.
    //--------------------------------------------------------------------------------------
    VOID PageRenderer::ExecuteColorFillOperation( const RenderOperation& Operation )
    {
        if( Operation.IndexMapUpdate )
        {
            PIXBeginNamedEvent( 0, "Index Map Update" );

            D3D11_BOX DestBox;
            DestBox.back = 1;
            DestBox.front = 0;
            DestBox.left = Operation.DrawRect.left;
            DestBox.top = Operation.DrawRect.top;
            DestBox.right = Operation.DrawRect.right;
            DestBox.bottom = Operation.DrawRect.bottom;

            ASSERT( DestBox.right - DestBox.left == 1 );
            ASSERT( DestBox.bottom - DestBox.top == 1 );

            m_pd3dDeviceContext->UpdateSubresource( Operation.pResolveTexture, Operation.ResolveMipLevelIndex, &DestBox, &Operation.FillColor32, sizeof(UINT), 0 );

            PIXEndNamedEvent();
        }
        else
        {
            ASSERT( Operation.pIntermediateTexture != NULL );
            D3D11_TEXTURE2D_DESC TexDesc;
            Operation.pIntermediateTexture->GetDesc( &TexDesc );

            UINT Width = Operation.DrawRect.right - Operation.DrawRect.left;
            UINT Height = Operation.DrawRect.bottom - Operation.DrawRect.top;

            D3D11_BOX IntermediateBox;
            IntermediateBox.back = 1;
            IntermediateBox.front = 0;
            IntermediateBox.left = TexDesc.Width - Width;
            IntermediateBox.top = TexDesc.Height - Height;
            IntermediateBox.right = TexDesc.Width;
            IntermediateBox.bottom = TexDesc.Height;

            // copy texels from blank area of intermediate texture to draw rect within destination texture
            m_pd3dDeviceContext->CopySubresourceRegion( Operation.pResolveTexture, Operation.ResolveSliceIndex, Operation.DrawRect.left, Operation.DrawRect.top, 0, Operation.pIntermediateTexture, 0, &IntermediateBox );
        }
    }
}

