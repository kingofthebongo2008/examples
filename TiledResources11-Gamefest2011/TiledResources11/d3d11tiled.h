//--------------------------------------------------------------------------------------
// d3d11tiled.h
//
// This is the front end of the tiled resource API, which is patterned after Direct3D 11
// semantics.  Through this API, you can create tiled resources and tile pools,
// manipulate tile memory and mappings, and render using tiled resources.
//
// Please note that this API is only a prototype for this sample only; the API does not 
// reflect current or future plans by Microsoft.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#pragma once

#ifndef _D3D11TILED_H_
#define _D3D11TILED_H_

#include <windows.h>
#include <d3d11.h>

// D3D11_TILED_PHYSICAL_ADDRESS is a handle to a physical tile address in the tiled
// resource system:
typedef UINT64 D3D11_TILED_PHYSICAL_ADDRESS;

// D3D11_TILED_VIRTUAL_ADDRESS is a handle to a virtual tile address within a tiled
// resource:
typedef UINT64 D3D11_TILED_VIRTUAL_ADDRESS;

// Invalid handles for physical and virtual addresses:
#define D3D11_TILED_INVALID_PHYSICAL_ADDRESS ((D3D11_TILED_PHYSICAL_ADDRESS)-1)
#define D3D11_TILED_INVALID_VIRTUAL_ADDRESS ((D3D11_TILED_VIRTUAL_ADDRESS)0)

// Misc flags for creating tiled textures:
#define D3D11_RESOURCE_MISC_TEXTUREQUILT (0x1000L)

//--------------------------------------------------------------------------------------
// Name: D3D11_TILED_TEXTURE2D_DESC
// Desc: The texture desc structure for initializing a tiled texture2D.  It inherits from
//       the standard D3D11 texture2D desc, and adds members that describe quilt height
//       and width.
//--------------------------------------------------------------------------------------
struct D3D11_TILED_TEXTURE2D_DESC : public D3D11_TEXTURE2D_DESC
{
    // The texture quilt width, in slices:
    UINT QuiltWidth;

    // The texture quilt height, in slices:
    UINT QuiltHeight;
};

//--------------------------------------------------------------------------------------
// Name: D3D11_TILED_SURFACE_DESC
// Desc: A description of a single mip level within a tiled texture 2D.
//--------------------------------------------------------------------------------------
struct D3D11_TILED_SURFACE_DESC
{
	// Surface format:
	DXGI_FORMAT Format;

	// Width and height of the mip level, in texels:
	UINT TexelWidth;
	UINT TexelHeight;

	// Width and height of the mip level's usable virtual address space, in tiles:
	UINT TileWidth;
	UINT TileHeight;

	// Width and height of a single tile, in texels:
	UINT TileTexelWidth;
	UINT TileTexelHeight;
};

//--------------------------------------------------------------------------------------
// Name: D3D11_TILED_MEMORY_USAGE
// Desc: A struct that is used to query the memory status of the tiled resource system.
//--------------------------------------------------------------------------------------
struct D3D11_TILED_MEMORY_USAGE
{
    // Total number of physical tiles allowed in the tiled resource system:
	UINT64 TileCapacity;

    // Total number of physical tiles currently allocated:
	UINT64 TilesAllocated;

    // Number of separate typed format pools that are currently in use:
	UINT FormatPoolsActive;

    // The amount of video memory consumed by all of the typed format pools:
	UINT64 TileTextureMemoryBytesAllocated;

    // The amount of video memory consumed by all of the resources' virtual to physical mapping textures:
	UINT64 ResourceTextureMemoryBytesAllocated;

    // The amount of system memory consumed by various tracking structures in the tiled resource system:
	UINT64 OverheadMemoryBytesAllocated;

    // The number of resources active in the tiled resource system:
	UINT ResourceCount;

    // The total amount of virtual address space spanned by all of the tiled resources:
	UINT64 ResourceVirtualBytesAllocated;
};

//--------------------------------------------------------------------------------------
// Name: IRefCount
// Desc: A refcounting base class used by the tiled resource D3D runtime interfaces.
//       This replaces refcounting behavior provided by IUnknown to the real D3D runtime
//       classes.
//--------------------------------------------------------------------------------------
class IRefCount
{
protected:
    ULONG m_RefCount;

public:
    IRefCount()
        : m_RefCount( 1 )
    {
    }

    ULONG AddRef()
    {
        return ++m_RefCount;
    }

    ULONG Release()
    {
        assert( m_RefCount > 0 );
        UINT NewRefCount = --m_RefCount;
        if( NewRefCount == 0 )
        {
            Terminate();
            delete this;
        }
        return NewRefCount;
    }

protected:
    //--------------------------------------------------------------------------------------
    // Name: IRefCount::Terminate
    // Desc: A pure virtual method that must be implemented by IRefCount subclasses.  It is
    //       called when the last reference to the object is released.
    //--------------------------------------------------------------------------------------
    virtual VOID Terminate() = NULL;
};

//--------------------------------------------------------------------------------------
// Name: ID3D11TiledResource
// Desc: A base class that represents a single tiled resource.  Currently, all tiled
//       resource methods are kept in resource type subclasses, such as ID3D11TiledTexture2D.
//--------------------------------------------------------------------------------------
struct ID3D11TiledResource : public IRefCount
{
};

//--------------------------------------------------------------------------------------
// Name: ID3D11TiledTexture2D
// Desc: Represents a tiled texture2D in the tiled resource system.  Methods are provided
//       to query the tile layout of the tiled resource, and do conversions between UV
//       coordinates and virtual addresses.
//--------------------------------------------------------------------------------------
struct ID3D11TiledTexture2D
    : public ID3D11TiledResource
{
    VOID GetDesc( D3D11_TILED_TEXTURE2D_DESC* pDesc );
    VOID GetSubresourceDesc( UINT Subresource, __out D3D11_TILED_SURFACE_DESC* pDesc );

    D3D11_TILED_VIRTUAL_ADDRESS GetTileVirtualAddress( UINT Subresource, FLOAT TextureU, FLOAT TextureV );
    VOID ConvertQuiltUVToArrayUVSlice( __inout FLOAT* pTextureU, __inout FLOAT* pTextureV, __out UINT* pSliceIndex );
};

//--------------------------------------------------------------------------------------
// Name: ID3D11TiledShaderResourceView
// Desc: Represents a shader resource view of a tiled resource.
//--------------------------------------------------------------------------------------
struct ID3D11TiledShaderResourceView : public IRefCount
{
};

//--------------------------------------------------------------------------------------
// Name: ID3D11TilePool
// Desc: The tile pool is the D3D interface to the video memory system, including
//       physical tile allocation and deallocation, physical tile data access, and
//       mappings between virtual addresses and physical addresses.
//--------------------------------------------------------------------------------------
struct ID3D11TilePool : public IRefCount
{
    HRESULT AllocatePhysicalTile( __out D3D11_TILED_PHYSICAL_ADDRESS* pPhysicalAddress, __in_opt DXGI_FORMAT TileFormat = DXGI_FORMAT_UNKNOWN );
    HRESULT FreePhysicalTile( D3D11_TILED_PHYSICAL_ADDRESS PhysicalAddress );

    HRESULT UpdateTileContents( D3D11_TILED_PHYSICAL_ADDRESS PhysicalAddress, __in const VOID* pBuffer, __in_opt DXGI_FORMAT BufferDataFormat = DXGI_FORMAT_UNKNOWN );

    HRESULT MapVirtualTileToPhysicalTile( D3D11_TILED_VIRTUAL_ADDRESS VirtualAddress, D3D11_TILED_PHYSICAL_ADDRESS PhysicalAddress );
    HRESULT UnmapVirtualAddress( D3D11_TILED_VIRTUAL_ADDRESS VirtualAddress );

    VOID GetMemoryUsage( __inout D3D11_TILED_MEMORY_USAGE* pMemoryUsage );
};

//--------------------------------------------------------------------------------------
// Name: ID3D11TiledResourceDevice
// Desc: The tiled resource device is a set of resource manipulation functions that allow the 
//       title to create tiled resources, tiled resource shader resource views, and 
//       access the tile pool.
//       It also contains device context functionality to set tiled resource shader
//       resource views into the device context.
//       It also contains an entrypoint that must be called once per frame by the title
//       app, in order to give the tiled resource system an opportunity to execute D3D
//       operations on its internal resources.
//--------------------------------------------------------------------------------------
struct ID3D11TiledResourceDevice : public IRefCount
{
    HRESULT CreateTilePool( __out ID3D11TilePool** ppTilePool );
    HRESULT CreateTexture2D( __in ID3D11TilePool* pTilePool, const D3D11_TILED_TEXTURE2D_DESC* pDesc, __out ID3D11TiledTexture2D** ppTexture );
    HRESULT CreateShaderResourceView( __in ID3D11TiledResource* pResource, __out ID3D11TiledShaderResourceView** ppSRView );

    HRESULT PreFrameRender();

    HRESULT VSSetShaderResources( UINT StartSlot, UINT NumResources, ID3D11TiledShaderResourceView** ppSRViews );
    HRESULT PSSetShaderResources( UINT StartSlot, UINT NumResources, ID3D11TiledShaderResourceView** ppSRViews );
};

//--------------------------------------------------------------------------------------
// Name: D3D11_TILED_EMULATION_PARAMETERS
// Desc: A struct that defines initialization parameters for the software implementation
//       of tiled resources.
//--------------------------------------------------------------------------------------
struct D3D11_TILED_EMULATION_PARAMETERS
{
    UINT MaxPhysicalTileCount;
    DXGI_FORMAT DefaultPhysicalTileFormat;
};

//--------------------------------------------------------------------------------------
// Name: D3D11CreateTiledResourceDevice
// Desc: Top level method to create an extended D3D device from a real D3D11 device and
//       immediate device context.
//--------------------------------------------------------------------------------------
HRESULT D3D11CreateTiledResourceDevice( __in ID3D11Device* pd3dDevice, __in ID3D11DeviceContext* pd3dDeviceContext, __in const D3D11_TILED_EMULATION_PARAMETERS* pEmulationParameters, __out ID3D11TiledResourceDevice** ppDeviceEx );

#endif
