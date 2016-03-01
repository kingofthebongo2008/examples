//--------------------------------------------------------------------------------------
// d3d9tiled.cpp
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "d3d11tiled.h"

#include "TiledResourceCommon.h"
#include "PhysicalPageManager.h"
#include "TypedPagePool.h"
#include "TiledResourceBase.h"

using namespace TiledRuntime;

//--------------------------------------------------------------------------------------
// Name: CTiledResource
// Desc: Internal subclass of ID3D11TiledResource.  It holds a pointer to the
//       TiledResourceBase implementation class, as well as caching the texture2D desc.
//--------------------------------------------------------------------------------------
struct CTiledResource : public ID3D11TiledResource
{
    TiledResourceBase* m_pResource;
    D3D11_TILED_TEXTURE2D_DESC m_Desc;

protected:
    VOID Terminate();
};

//--------------------------------------------------------------------------------------
// Name: CShaderResourceView
// Desc: Internal subclass of ID3D11TiledShaderResourceView.  It holds a pointer to the
//       resource that this object is a view upon.
//--------------------------------------------------------------------------------------
struct CShaderResourceView : public ID3D11TiledShaderResourceView
{
    TiledResourceBase* m_pResource;

protected:
    VOID Terminate();
};

//--------------------------------------------------------------------------------------
// Name: CTilePool
// Desc: Internal subclass of ID3D11TilePool.  It holds a pointer to a PhysicalPageManager.
//--------------------------------------------------------------------------------------
struct CTilePool : public ID3D11TilePool
{
    PhysicalPageManager* m_pPageManager;

protected:
    VOID Terminate();
};

//--------------------------------------------------------------------------------------
// Name: CTiledResourceDevice
// Desc: Internal subclass of ID3D11TiledResourceDevice.  It holds pointers to the real D3D11 device
//       and device context, and the single tile pool.  It also shadows the shader
//       constants for tiled resources, because all of them must be updated at the same
//       time via the same constant buffer.
//--------------------------------------------------------------------------------------
struct CTiledResourceDevice : public ID3D11TiledResourceDevice
{
    ID3D11Device* m_pd3dDevice;
    ID3D11DeviceContext* m_pd3dDeviceContext;
    CTilePool* m_pTilePool;

    // Shadowed shader constants for tiled resources in the pixel shader stage:
    TiledResourceBase::CB_TiledResource m_PSTiledResourceConstants[MAX_PS_TILED_SHADER_RESOURCES];
    // Constant buffer for tiled resources in the pixel shader stage:
    ID3D11Buffer* m_pPSTiledResourceCB;
    // Shadowed shader constants for the typed page pool in the pixel shader stage:
    TypedPagePool::CB_PagePool m_PSPagePoolConstants[MAX_PS_TILED_SHADER_RESOURCES];
    // Constant buffer for the typed page pool in the pixel shader stage:
    ID3D11Buffer* m_pPSPagePoolCB;

    // Shadowed shader constants for tiled resources in the vertex shader stage:
    TiledResourceBase::CB_TiledResource m_VSTiledResourceConstants[MAX_VS_TILED_SHADER_RESOURCES];
    // Constant buffer for tiled resources in the vertex shader stage:
    ID3D11Buffer* m_pVSTiledResourceCB;
    // Shadowed shader constants for the typed page pool in the vertex shader stage:
    TypedPagePool::CB_PagePool m_VSPagePoolConstants[MAX_VS_TILED_SHADER_RESOURCES];
    // Constant buffer for the typed page pool in the vertex shader stage:
    ID3D11Buffer* m_pVSPagePoolCB;

    VOID CreateConstantBuffers();
    VOID ReleaseConstantBuffers();

    VOID UpdatePSConstants();
    VOID UpdateVSConstants();

protected:
    VOID Terminate();
};

// Promotion functions that cast an external pointer to an internal pointer:
CTiledResourceDevice* Promote( ID3D11TiledResourceDevice* pDeviceEx ) { return static_cast<CTiledResourceDevice*>( pDeviceEx ); }
CTilePool* Promote( ID3D11TilePool* pTilePool ) { return static_cast<CTilePool*>( pTilePool ); }
CTiledResource* Promote( ID3D11TiledResource* pResource ) { return static_cast<CTiledResource*>( pResource ); }
CShaderResourceView* Promote( ID3D11TiledShaderResourceView* pSRView ) { return static_cast<CShaderResourceView*>( pSRView ); }

//--------------------------------------------------------------------------------------
// Name: D3D11CreateTiledResourceDevice
// Desc: Creates a new extended Direct3D device.
//--------------------------------------------------------------------------------------
HRESULT D3D11CreateTiledResourceDevice( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dDeviceContext, const D3D11_TILED_EMULATION_PARAMETERS* pEmulationParameters, ID3D11TiledResourceDevice** ppDeviceEx )
{
    // Create the new device:
    CTiledResourceDevice* pDeviceEx = new CTiledResourceDevice();

    // Fill in pointers to the real D3D11 device and device context:
    pDeviceEx->m_pd3dDevice = pd3dDevice;
    pDeviceEx->m_pd3dDevice->AddRef();
    pDeviceEx->m_pd3dDeviceContext = pd3dDeviceContext;
    pDeviceEx->m_pd3dDeviceContext->AddRef();

    // Clear the shadowed shader constants:
    ZeroMemory( &pDeviceEx->m_PSTiledResourceConstants, sizeof( pDeviceEx->m_PSTiledResourceConstants ) );
    ZeroMemory( &pDeviceEx->m_PSPagePoolConstants, sizeof( pDeviceEx->m_PSPagePoolConstants ) );
    ZeroMemory( &pDeviceEx->m_VSTiledResourceConstants, sizeof( pDeviceEx->m_VSTiledResourceConstants ) );
    ZeroMemory( &pDeviceEx->m_VSPagePoolConstants, sizeof( pDeviceEx->m_VSPagePoolConstants ) );

    // Create the constant buffers for the pixel and vertex shader emulation of tiled resources:
    pDeviceEx->CreateConstantBuffers();

    // Convert the initialization parameters to their internal format:
    PhysicalPageManagerDesc PPMDesc;
    ZeroMemory( &PPMDesc, sizeof(PhysicalPageManagerDesc) );
    if( pEmulationParameters != NULL )
    {
        PPMDesc.MaxPhysicalPages = pEmulationParameters->MaxPhysicalTileCount;
        PPMDesc.EmulationParams.DefaultResourceFormat = pEmulationParameters->DefaultPhysicalTileFormat;
    }

    // Create the single tile pool for this device:
    CTilePool* pTilePool = new CTilePool();
    pTilePool->m_pPageManager = new PhysicalPageManager( pDeviceEx->m_pd3dDevice, pDeviceEx->m_pd3dDeviceContext, &PPMDesc );

    pDeviceEx->m_pTilePool = pTilePool;
    pDeviceEx->m_pTilePool->AddRef();

    // Return the pointer to the tiled resource device:
    *ppDeviceEx = pDeviceEx;

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Name: ID3D11TiledResourceDevice::CreateTilePool
// Desc: Returns a pointer for the single tile pool for this device:
//--------------------------------------------------------------------------------------
HRESULT ID3D11TiledResourceDevice::CreateTilePool( __out ID3D11TilePool** ppTilePool )
{
    CTiledResourceDevice* p = Promote( this );

    // Returns a pointer to the tile pool:
    *ppTilePool = p->m_pTilePool;

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Name: ID3D11TiledResourceDevice::CreateTexture2D
// Desc: Creates a new tiled texture 2D resource.  Unlike a traditional texture 2D, we
//       do not support initialization data.
//--------------------------------------------------------------------------------------
HRESULT ID3D11TiledResourceDevice::CreateTexture2D( ID3D11TilePool* pTilePool, const D3D11_TILED_TEXTURE2D_DESC* pDesc, __out ID3D11TiledTexture2D** ppTexture )
{
    CTiledResourceDevice* p = Promote( this );

    // If we were not passed a tile pool pointer, use the device's default pointer:
    CTilePool* pPM = NULL;
    if( pTilePool != NULL )
    {
        pPM = Promote( pTilePool );

        // We must use the tile pool created by this device:
        ASSERT( pPM == p->m_pTilePool );
    }
    else
    {
        pPM = p->m_pTilePool;
    }

    // We only support DEFAULT usage tiled resources:
    if( pDesc->Usage != D3D11_USAGE_DEFAULT )
    {
        return E_INVALIDARG;
    }

    // The tiled texture must be bindable as a shader resource:
    if( pDesc->BindFlags != D3D11_BIND_SHADER_RESOURCE )
    {
        return E_INVALIDARG;
    }

    // The tiled texture is not accessible by the CPU:
    if( pDesc->CPUAccessFlags != 0 )
    {
        return E_INVALIDARG;
    }

    UINT ArraySize = pDesc->ArraySize;

    if( pDesc->MiscFlags & D3D11_RESOURCE_MISC_TEXTUREQUILT )
    {
        // Quilt must be at least 1x2 or 2x1:
        if( pDesc->QuiltWidth < 2 && pDesc->QuiltHeight < 2 )
        {
            return E_INVALIDARG;
        }
        // Quilt must be greater than 0 in both dimensions:
        if( pDesc->QuiltWidth == 0 || pDesc->QuiltHeight == 0 )
        {
            return E_INVALIDARG;
        }
        // Quilt width and height must match the array size:
        if( ( pDesc->QuiltHeight * pDesc->QuiltWidth ) != ArraySize )
        {
            return E_INVALIDARG;
        }
    }
    else
    {
        // We are not quilting, so the width and height must both be 0:
        if( pDesc->QuiltWidth != 0 || pDesc->QuiltHeight != 0 )
        {
            return E_INVALIDARG;
        }
    }

    // Create the tiled texture object and initialize:
    TiledTexture* pTiledTexture = new TiledTexture();
    HRESULT hr = pTiledTexture->Initialize( p->m_pd3dDevice, pPM->m_pPageManager, pDesc->Width, pDesc->Height, pDesc->MipLevels, ArraySize, pDesc->Format, pDesc->QuiltWidth, pDesc->QuiltHeight );

    if( FAILED(hr) )
    {
        delete pTiledTexture;
        return hr;
    }

    // Create a tiled resource wrapper object:
    CTiledResource* pTiledResource = new CTiledResource();
    pTiledResource->m_pResource = pTiledTexture;
    pTiledResource->m_Desc = *pDesc;
    pTiledResource->m_Desc.MipLevels = pTiledTexture->GetMipLevelCount();
    ASSERT( pTiledResource->m_Desc.MipLevels > 0 );

    // Return the wrapper object:
    *ppTexture = (ID3D11TiledTexture2D*)pTiledResource;

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Name: ID3D11TiledResourceDevice::CreateShaderResourceView
// Desc: Creates a shader resource view on a tiled resource.
//--------------------------------------------------------------------------------------
HRESULT ID3D11TiledResourceDevice::CreateShaderResourceView( __in ID3D11TiledResource* pResource, __out ID3D11TiledShaderResourceView** ppSRView )
{
    // Validate parameters:
    if( pResource == NULL )
    {
        return E_INVALIDARG;
    }
    if( ppSRView == NULL )
    {
        return E_INVALIDARG;
    }

    CTiledResource* pTiledResource = Promote( pResource );

    CShaderResourceView* pSRView = new CShaderResourceView();
    pSRView->m_pResource = pTiledResource->m_pResource;

    *ppSRView = pSRView;

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Name: ID3D11TiledResourceDevice::PreFrameRender
// Desc: Executes per-frame graphics operations that are required for the tiled resource
//       system to operate.
//--------------------------------------------------------------------------------------
HRESULT ID3D11TiledResourceDevice::PreFrameRender()
{
    CTiledResourceDevice* p = Promote( this );

    // Call into the physical tile pool to execute its page operations:
    p->m_pTilePool->m_pPageManager->ExecutePageDataOperations();

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Name: ID3D11TiledResourceDevice::PSSetShaderResources
// Desc: Sets a group of tiled shader resource views into the pixel shader stage.  Also
//       updates the shadowed constants and updates the constant buffer.
//--------------------------------------------------------------------------------------
HRESULT ID3D11TiledResourceDevice::PSSetShaderResources( UINT StartSlot, UINT NumResources, ID3D11TiledShaderResourceView** ppSRViews )
{
    CTiledResourceDevice* p = Promote( this );

    if( ( StartSlot + NumResources ) > MAX_PS_TILED_SHADER_RESOURCES )
    {
        return E_INVALIDARG;
    }

    // Loop over the resources:
    for( UINT i = 0; i < NumResources; ++i )
    {
        UINT SlotIndex = i + StartSlot;
        ASSERT( SlotIndex < MAX_PS_TILED_SHADER_RESOURCES );

        CShaderResourceView* pSRView = Promote( ppSRViews[i] );
        if( pSRView == NULL )
        {
            // Clear the shader resource views for this slot:
            ID3D11ShaderResourceView* pNullSRV = NULL;
            p->m_pd3dDeviceContext->PSSetShaderResources( SlotIndex + TiledResourceBase::GetPSBaseSlotIndex(), 1, &pNullSRV );
            p->m_pd3dDeviceContext->PSSetShaderResources( SlotIndex + TypedPagePool::GetPSBaseSlotIndex(), 1, &pNullSRV );
        }
        else
        {
            // Call into the TiledResource to set its shader resource views:
            pSRView->m_pResource->PSSetShaderResource( p->m_pd3dDeviceContext, SlotIndex );

            // Update the shadow constants:
            p->m_PSTiledResourceConstants[SlotIndex] = pSRView->m_pResource->GetShaderConstants();
            p->m_PSPagePoolConstants[SlotIndex] = pSRView->m_pResource->GetTypedPagePool()->GetShaderConstants();
        }
    }

    // Send the shadow constants to the D3D device context:
    p->UpdatePSConstants();

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Name: ID3D11TiledResourceDevice::VSSetShaderResources
// Desc: Sets a group of tiled shader resource views into the vertex shader stage.  Also
//       updates the shadowed constants and updates the constant buffer.
//--------------------------------------------------------------------------------------
HRESULT ID3D11TiledResourceDevice::VSSetShaderResources( UINT StartSlot, UINT NumResources, ID3D11TiledShaderResourceView** ppSRViews )
{
    CTiledResourceDevice* p = Promote( this );

    if( ( StartSlot + NumResources ) > MAX_VS_TILED_SHADER_RESOURCES )
    {
        return E_INVALIDARG;
    }

    // Loop over the resources:
    for( UINT i = 0; i < NumResources; ++i )
    {
        UINT SlotIndex = i + StartSlot;
        ASSERT( SlotIndex < MAX_VS_TILED_SHADER_RESOURCES );

        CShaderResourceView* pSRView = Promote( ppSRViews[i] );
        if( pSRView == NULL )
        {
            // Clear the shader resource views for this slot:
            ID3D11ShaderResourceView* pNullSRV = NULL;
            p->m_pd3dDeviceContext->VSSetShaderResources( SlotIndex + TiledResourceBase::GetVSBaseSlotIndex(), 1, &pNullSRV );
            p->m_pd3dDeviceContext->VSSetShaderResources( SlotIndex + TypedPagePool::GetVSBaseSlotIndex(), 1, &pNullSRV );
        }
        else
        {
            // Call into the TiledResource to set its shader resource views:
            pSRView->m_pResource->VSSetShaderResource( p->m_pd3dDeviceContext, SlotIndex );

            // Update the shadow constants:
            p->m_VSTiledResourceConstants[SlotIndex] = pSRView->m_pResource->GetShaderConstants();
            p->m_VSPagePoolConstants[SlotIndex] = pSRView->m_pResource->GetTypedPagePool()->GetShaderConstants();
        }
    }

    // Send the shadow constants to the D3D device context:
    p->UpdateVSConstants();

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Name: ID3D11TilePool::AllocatePhysicalTile
// Desc: Allocates a single 64KB physical tile.
//--------------------------------------------------------------------------------------
HRESULT ID3D11TilePool::AllocatePhysicalTile( __out D3D11_TILED_PHYSICAL_ADDRESS* pPhysicalAddress, __in_opt DXGI_FORMAT TileFormat )
{
    CTilePool* p = Promote( this );

    if( pPhysicalAddress == NULL )
    {
        return E_INVALIDARG;
    }

    // Call into the physical page manager to allocate the tile:
    PhysicalPageID PageID;
    HRESULT hr = p->m_pPageManager->AllocatePage( &PageID, TileFormat );

    // Convert the PhysicalPageID to the more opaque D3D11_TILED_PHYSICAL_ADDRESS:
    *pPhysicalAddress = PageID;
    return hr;
}

//--------------------------------------------------------------------------------------
// Name: ID3D11TilePool::FreePhysicalTile
// Desc: Frees a single 64KB physical tile.
//--------------------------------------------------------------------------------------
HRESULT ID3D11TilePool::FreePhysicalTile( D3D11_TILED_PHYSICAL_ADDRESS PhysicalAddress )
{
    CTilePool* p = Promote( this );
    return p->m_pPageManager->FreePage( PhysicalAddress );
}

//--------------------------------------------------------------------------------------
// Name: ID3D11TilePool::UpdateTileContents
// Desc: Copies a 64KB buffer into a physical tile that has been allocated.
//--------------------------------------------------------------------------------------
HRESULT ID3D11TilePool::UpdateTileContents( D3D11_TILED_PHYSICAL_ADDRESS PhysicalAddress, const VOID* pBuffer, DXGI_FORMAT BufferDataFormat )
{
    CTilePool* p = Promote( this );
    return p->m_pPageManager->UpdateSinglePageContents( PhysicalAddress, pBuffer, BufferDataFormat );
}

//--------------------------------------------------------------------------------------
// Name: ID3D11TilePool::MapVirtualTileToPhysicalTile
// Desc: Maps a single virtual tile address to a single physical tile address, which may
//       be invalid.
//--------------------------------------------------------------------------------------
HRESULT ID3D11TilePool::MapVirtualTileToPhysicalTile( D3D11_TILED_VIRTUAL_ADDRESS VirtualAddress, D3D11_TILED_PHYSICAL_ADDRESS PhysicalAddress )
{
    CTilePool* p = Promote( this );

    // Convert the opaque D3D11_TILED_VIRTUAL_ADDRESS into the internal VirtualPageID struct:
    VirtualPageID VPageID;
    VPageID.VirtualAddress = VirtualAddress;

    return p->m_pPageManager->MapVirtualPageToPhysicalPage( VPageID, PhysicalAddress );
}

//--------------------------------------------------------------------------------------
// Name: ID3D11TilePool::UnmapVirtualAddress
// Desc: Maps the given virtual address to an invalid physical address.
//--------------------------------------------------------------------------------------
HRESULT ID3D11TilePool::UnmapVirtualAddress( D3D11_TILED_VIRTUAL_ADDRESS VirtualAddress )
{
    return MapVirtualTileToPhysicalTile( VirtualAddress, D3D11_TILED_INVALID_PHYSICAL_ADDRESS );
}

//--------------------------------------------------------------------------------------
// Name: ID3D11TilePool::GetMemoryUsage
// Desc: Populates a struct with current memory usage statistics for the tile pool
//       and all of the tiled resources.
//--------------------------------------------------------------------------------------
VOID ID3D11TilePool::GetMemoryUsage( D3D11_TILED_MEMORY_USAGE* pMemoryUsage )
{
    CTilePool* p = Promote( this );

    p->m_pPageManager->GetMemoryUsage( pMemoryUsage );
}

//--------------------------------------------------------------------------------------
// Name: ID3D11TiledTexture2D::GetDesc
// Desc: Returns a copy of the cached texture desc.
//--------------------------------------------------------------------------------------
VOID ID3D11TiledTexture2D::GetDesc( D3D11_TILED_TEXTURE2D_DESC* pDesc )
{
    if( pDesc != NULL )
    {
        CTiledResource* p = Promote( this );
        *pDesc = p->m_Desc;
    }
}

//--------------------------------------------------------------------------------------
// Name: ID3D11TiledTexture2D::GetSubresourceDesc
// Desc: Populates a D3D11_TILED_SURFACE_DESC struct with information about a subresource.
//--------------------------------------------------------------------------------------
VOID ID3D11TiledTexture2D::GetSubresourceDesc( UINT Subresource, D3D11_TILED_SURFACE_DESC* pDesc )
{
    if( pDesc == NULL )
    {
        return;
    }

    CTiledResource* p = Promote( this );

    // Convert the subresource index into a mip level index, because the array slice index
    // doesn't matter in the surface desc:
    UINT Level = Subresource % p->m_pResource->GetMipLevelCount();

    // Get the mip level desc:
    p->m_pResource->GetLevelDesc( Level, pDesc );
}

//--------------------------------------------------------------------------------------
// Name: ID3D11TiledTexture2D::GetTileVirtualAddress
// Desc: For the given texture UV coordinates on the given subresource, return the 
//       virtual address of the page where those UV coordinates are located.
//--------------------------------------------------------------------------------------
D3D11_TILED_VIRTUAL_ADDRESS ID3D11TiledTexture2D::GetTileVirtualAddress( UINT Subresource, FLOAT TextureU, FLOAT TextureV )
{
    CTiledResource* p = Promote( this );

    // Convert the subresource index into mip level and array slice index:
    const UINT MipLevelCount = p->m_pResource->GetMipLevelCount();
    UINT Level = Subresource % MipLevelCount;
    UINT SliceIndex = Subresource / MipLevelCount;

    // Get the virtual address for the UV coords, slice index, and mip level:
    VirtualPageID VPageID = p->m_pResource->GetVirtualPageIDFloat( TextureU, TextureV, SliceIndex, Level );

    return VPageID.VirtualAddress;
}

//--------------------------------------------------------------------------------------
// Name: ID3D11TiledTexture2D::ConvertQuiltUVToArrayUVSlice
// Desc: If the texture2D is quilted, converts the given UV coordinates into normalized
//       UV coordinates plus an array slice index.
//--------------------------------------------------------------------------------------
VOID ID3D11TiledTexture2D::ConvertQuiltUVToArrayUVSlice( __inout FLOAT* pTextureU, __inout FLOAT* pTextureV, __out UINT* pSliceIndex )
{
    CTiledResource* p = Promote( this );

    if( pTextureU == NULL || pTextureV == NULL || pSliceIndex == NULL )
    {
        return;
    }

    if( !p->m_pResource->IsQuilted() )
    {
        return;
    }

    *pSliceIndex = p->m_pResource->ConvertQuiltUVToArrayUVW( pTextureU, pTextureV );
}

//--------------------------------------------------------------------------------------
// Name: CreateDynamicConstantBuffer
// Desc: Creates a DYNAMIC usage D3D11 constant buffer of the given size.
//--------------------------------------------------------------------------------------
inline HRESULT CreateDynamicConstantBuffer( ID3D11Device* pd3dDevice, UINT SizeBytes, ID3D11Buffer** ppBuffer )
{
    D3D11_BUFFER_DESC CBDesc;
    ZeroMemory( &CBDesc, sizeof(CBDesc) );
    CBDesc.Usage = D3D11_USAGE_DYNAMIC;
    CBDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    CBDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    CBDesc.ByteWidth = SizeBytes;

    return pd3dDevice->CreateBuffer( &CBDesc, NULL, ppBuffer );
}

//--------------------------------------------------------------------------------------
// Name: UpdateDynamicConstantBuffer
// Desc: Overwrites the contents of the given DYNAMIC usage constant buffer with the
//       given buffer.
//--------------------------------------------------------------------------------------
inline HRESULT UpdateDynamicConstantBuffer( ID3D11DeviceContext* pd3dContext, ID3D11Buffer* pBuffer, const VOID* pSrcData, const UINT DataSizeBytes )
{
    // Map the resource for write with discard:
    D3D11_MAPPED_SUBRESOURCE MapData;
    HRESULT hr = pd3dContext->Map( pBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &MapData );
    if( FAILED(hr) )
    {
        return hr;
    }

    // Copy the src buffer to the D3D11 buffer:
    ASSERT( MapData.RowPitch >= DataSizeBytes );
    memcpy( MapData.pData, pSrcData, DataSizeBytes );

    // Unmap the resource:
    pd3dContext->Unmap( pBuffer, 0 );

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Name: CTiledResourceDevice::CreateConstantBuffers
// Desc: Creates the D3D11 constant buffers for sending tiled resource and page pool
//       shader constants to the D3D device context.
//--------------------------------------------------------------------------------------
VOID CTiledResourceDevice::CreateConstantBuffers()
{
    ASSERT( m_pd3dDevice != NULL );

    CreateDynamicConstantBuffer( m_pd3dDevice, sizeof(m_PSTiledResourceConstants), &m_pPSTiledResourceCB );
    CreateDynamicConstantBuffer( m_pd3dDevice, sizeof(m_PSPagePoolConstants), &m_pPSPagePoolCB );
    CreateDynamicConstantBuffer( m_pd3dDevice, sizeof(m_VSTiledResourceConstants), &m_pVSTiledResourceCB );
    CreateDynamicConstantBuffer( m_pd3dDevice, sizeof(m_VSPagePoolConstants), &m_pVSPagePoolCB );
}

//--------------------------------------------------------------------------------------
// Name: CTiledResourceDevice::ReleaseConstantBuffers
// Desc: Releases the D3D11 buffer objects.
//--------------------------------------------------------------------------------------
VOID CTiledResourceDevice::ReleaseConstantBuffers()
{
    SAFE_RELEASE( m_pPSTiledResourceCB );
    SAFE_RELEASE( m_pPSPagePoolCB );
    SAFE_RELEASE( m_pVSTiledResourceCB );
    SAFE_RELEASE( m_pVSPagePoolCB );
}

//--------------------------------------------------------------------------------------
// Name: CTiledResourceDevice::UpdatePSConstants
// Desc: Copies the pixel shader shadow constants into the constant buffers and sets them
//       into the D3D device context.
//--------------------------------------------------------------------------------------
VOID CTiledResourceDevice::UpdatePSConstants()
{
    // Update the tiled resource constants:
    HRESULT hr = UpdateDynamicConstantBuffer( m_pd3dDeviceContext, m_pPSTiledResourceCB, m_PSTiledResourceConstants, sizeof(m_PSTiledResourceConstants) );
    ASSERT( SUCCEEDED(hr) );

    // Update the page pool constants:
    hr = UpdateDynamicConstantBuffer( m_pd3dDeviceContext, m_pPSPagePoolCB, m_PSPagePoolConstants, sizeof(m_PSPagePoolConstants) );
    ASSERT( SUCCEEDED(hr) );

    // Set the constant buffers to slots 12 and 13 on the D3D device context:
    ID3D11Buffer* pBuffers[] = { m_pPSTiledResourceCB, m_pPSPagePoolCB };
    m_pd3dDeviceContext->PSSetConstantBuffers( 12, 2, pBuffers );
}

//--------------------------------------------------------------------------------------
// Name: CTiledResourceDevice::UpdateVSConstants
// Desc: Copies the vertex shader shadow constants into the constant buffers and sets them
//       into the D3D device context.
//--------------------------------------------------------------------------------------
VOID CTiledResourceDevice::UpdateVSConstants()
{
    // Update the tiled resource constants:
    HRESULT hr = UpdateDynamicConstantBuffer( m_pd3dDeviceContext, m_pVSTiledResourceCB, m_VSTiledResourceConstants, sizeof(m_VSTiledResourceConstants) );
    ASSERT( SUCCEEDED(hr) );

    // Update the page pool constants:
    hr = UpdateDynamicConstantBuffer( m_pd3dDeviceContext, m_pVSPagePoolCB, m_VSPagePoolConstants, sizeof(m_VSPagePoolConstants) );
    ASSERT( SUCCEEDED(hr) );

    // Set the constant buffers to slots 12 and 13 on the D3D device context:
    ID3D11Buffer* pBuffers[] = { m_pVSTiledResourceCB, m_pVSPagePoolCB };
    m_pd3dDeviceContext->VSSetConstantBuffers( 12, 2, pBuffers );
}

//--------------------------------------------------------------------------------------
// Name: CTiledResourceDevice::Terminate
// Desc: Cleans up the extended device, releasing its D3D11 objects and its tile pool.
//--------------------------------------------------------------------------------------
VOID CTiledResourceDevice::Terminate()
{
    ReleaseConstantBuffers();

    SAFE_RELEASE( m_pTilePool );
    SAFE_RELEASE( m_pd3dDeviceContext );
    SAFE_RELEASE( m_pd3dDevice );
}

//--------------------------------------------------------------------------------------
// Name: CTilePool::Terminate
// Desc: Cleans up the tile pool by deleting the physical tile pool object.
//--------------------------------------------------------------------------------------
VOID CTilePool::Terminate()
{
    delete m_pPageManager;
    m_pPageManager = NULL;
}

//--------------------------------------------------------------------------------------
// Name: CTiledResource::Terminate
// Desc: Cleans up the tiled resource by deleting the TiledResourceBase object.
//--------------------------------------------------------------------------------------
VOID CTiledResource::Terminate()
{
    delete m_pResource;
}

//--------------------------------------------------------------------------------------
// Name: CShaderResourceView::Terminate
// Desc: Cleans up the tiled shader resource view by clearing the resource pointer.
//--------------------------------------------------------------------------------------
VOID CShaderResourceView::Terminate()
{
    m_pResource = NULL;
}
