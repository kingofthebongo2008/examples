//--------------------------------------------------------------------------------------
// PageRenderer.h
//
// Queues and performs deferred page manipulation operations on the texture resources 
// that are used to emulate the tiled resource system.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#include "TiledResourceCommon.h"

namespace TiledRuntime
{
    //--------------------------------------------------------------------------------------
    // Name: struct RenderOperation
    // Desc: Describes a deferred operation on a texture resource.
    //--------------------------------------------------------------------------------------
    struct RenderOperation
    {
		VOID* pSrcBuffer;
		UINT SrcRowPitchBytes;

        // draw from the src texture, from the given rect, to the draw rect
        ID3D11Texture2D* pSrcArrayTexture;
        UINT SrcSliceIndex;
        RECT SrcRect;
        RECT DrawRect;

        // intermediate texture to be used for resource copies and rectangle clears
        ID3D11Texture2D* pIntermediateTexture;

        // resolve to the resolve texture
        ID3D11Texture2D* pResolveTexture;
        UINT ResolveSliceIndex;
        UINT ResolveMipLevelIndex;

        // fill color
        XMFLOAT4 FillColor;
        D3DCOLOR FillColor32;
        BOOL IndexMapUpdate;

        RenderOperation()
        {
            ZeroMemory( this, sizeof(RenderOperation) );
        }
    };

    //--------------------------------------------------------------------------------------
    // Name: PageRenderer
    // Desc: A singleton class that manages a queue of pending operations on texture
    //       resources, and exposes the methods that perform those operations at the desired
    //       time.
    //--------------------------------------------------------------------------------------
    class PageRenderer
    {
    protected:
        ID3D11Device* m_pd3dDevice;
        ID3D11DeviceContext* m_pd3dDeviceContext;

        std::deque<RenderOperation> m_PendingOperations;
        CRITICAL_SECTION m_QueueCritSec;

        ID3D11Texture2D* m_pIntermediateTextures[DXGI_FORMAT_MAX];

    public:
        PageRenderer( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dDeviceContext );
        ~PageRenderer(void);

        BOOL UpdatesPending() const { return !m_PendingOperations.empty(); }
        VOID FlushPendingUpdates();

        HRESULT QueuePageUpdate( TypedPagePool* pPagePool, INT PageIndex, const VOID* pPageBuffer );
        HRESULT QueueBorderUpdate( TypedPagePool* pPagePool, PhysicalPageID CenterPageID, PhysicalPageID BorderPageID, PageNeighbors RelationshipToCenterPage, BOOL InvertSourceRelationship = FALSE );

    protected:
        ID3D11Texture2D* GetIntermediateTexture( DXGI_FORMAT DataFormat );
        VOID QueueOperation( RenderOperation& Operation );
        VOID ExecuteUpdateSubresourceOperation( const RenderOperation& Operation );
		VOID ExecuteTextureCopyOperation( const RenderOperation& Operation );
		VOID ExecuteColorFillOperation( const RenderOperation& Operation );
    };
}

