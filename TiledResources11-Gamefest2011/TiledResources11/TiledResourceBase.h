//--------------------------------------------------------------------------------------
// TiledResourceBase.h
//
// The implementation of a tiled resource, including the concept of an index map, which
// is a texture that holds virtual-to-physical page mappings for virtual pages within
// the resource.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#include "TiledResourceCommon.h"
#include "d3d11tiled.h"

namespace TiledRuntime
{
    //--------------------------------------------------------------------------------------
    // Name: CPUTexture
    // Desc: A class that holds untyped texture data in raw buffers, without using Direct3D.  
    //       Mipmaps and array slices are supported.
    //--------------------------------------------------------------------------------------
    class CPUTexture
    {
        //--------------------------------------------------------------------------------------
        // Name: SubresourceDesc
        // Desc: Describes one subresource within the CPUTexture.
        //--------------------------------------------------------------------------------------
        struct SubresourceDesc
        {
            BYTE* pBase;
            UINT RowPitchBytes;
            UINT Height;
        };
    protected:
        // The entire texture is packed into one allocation:
        VOID* m_pAllocation;

        // An array of subresources subdivide the allocation:
        SubresourceDesc* m_pSubresources;

        // Texture array size:
        UINT m_ArraySize;

        // Texture mip level count:
        UINT m_MipLevels;

        // Bytes per pixel:
        UINT m_BytesPerPixel;

    public:
        CPUTexture();
        ~CPUTexture();

        VOID Initialize( UINT Width, UINT Height, UINT ArraySize, UINT Levels, UINT BytesPerPixel );
        VOID Terminate();
        VOID CPUMap( UINT SubresourceIndex, D3D11_MAPPED_SUBRESOURCE* pMappedSubresource, UINT* pSubresourceHeight = NULL );
        VOID CopySubresourceToTexture( ID3D11DeviceContext* pd3dDeviceContext, ID3D11Texture2D* pDestTexture, UINT SubresourceIndex );
    };

    //--------------------------------------------------------------------------------------
    // Name: ThreadSafeBitField
    // Desc: Holds a bit field of 512 bits, and uses lockless primitives to perform thread
    //       safe operations on the bits.
    //--------------------------------------------------------------------------------------
    class ThreadSafeBitField
    {
    protected:
        // 8 blocks of 64 bits each:
        volatile UINT64 m_Blocks[8];

    public:
        //--------------------------------------------------------------------------------------
        // Name: ThreadSafeBitField constructor
        // Desc: Zeros the bit field.
        //--------------------------------------------------------------------------------------
        ThreadSafeBitField()
        {
            ZeroMemory( (VOID*)m_Blocks, sizeof(m_Blocks) );
        }

        //--------------------------------------------------------------------------------------
        // Name: ThreadSafeBitField::TotalBitCount
        // Desc: Returns the total number of bits in the bitfield.
        //--------------------------------------------------------------------------------------
        inline UINT TotalBitCount() const { return ARRAYSIZE(m_Blocks) * 64; }

        //--------------------------------------------------------------------------------------
        // Name: ThreadSafeBitField::SetBit
        // Desc: Sets the indexed bit to 1 in a thread safe fashion.
        //--------------------------------------------------------------------------------------
        inline VOID SetBit( UINT Index )
        {
            // Compute the block index by dividing the index by 64:
            UINT BlockIndex = Index >> 6;
            ASSERT( BlockIndex < ARRAYSIZE(m_Blocks) );

            // Compute the bit pattern by shifting a 1 left by the lower 6 bits of the index:
            UINT64 BitPattern = (UINT64)1 << ( Index & 0x3F );

            // Safely OR the bit pattern with the selected block:
            InterlockedOr64( (volatile LONGLONG*)&m_Blocks[BlockIndex], BitPattern );
        }

        //--------------------------------------------------------------------------------------
        // Name: ThreadSafeBitField::TestBitUnsafe
        // Desc: Returns the state of an indexed bit, without thread safety.
        //--------------------------------------------------------------------------------------
        inline BOOL TestBitUnsafe( UINT Index ) const
        {
            UINT BlockIndex = Index >> 6;
            ASSERT( BlockIndex < ARRAYSIZE(m_Blocks) );
            UINT64 BitPattern = (UINT64)1 << ( Index & 0x3F );
            return ( ( m_Blocks[BlockIndex] & BitPattern ) != 0 );
        }

        //--------------------------------------------------------------------------------------
        // Name: ThreadSafeBitField::ClearBitUnsafe
        // Desc: Clears an indexed bit, without thread safety.
        //--------------------------------------------------------------------------------------
        inline VOID ClearBitUnsafe( UINT Index )
        {
            UINT BlockIndex = Index >> 6;
            ASSERT( BlockIndex < ARRAYSIZE(m_Blocks) );
            UINT64 BitPattern = (UINT64)1 << ( Index & 0x3F );
            m_Blocks[BlockIndex] &= ~BitPattern;
        }

        //--------------------------------------------------------------------------------------
        // Name: ThreadSafeBitField::CopyAndClear
        // Desc: Copies the current state of the bits to another ThreadSafeBitField, and clears
        //       the bits to 0 in the process.  This method is thread safe.
        //--------------------------------------------------------------------------------------
        inline VOID CopyAndClear( ThreadSafeBitField& Destination )
        {
            for( UINT i = 0; i < ARRAYSIZE(m_Blocks); ++i )
            {
                Destination.m_Blocks[i] = InterlockedExchange64( (volatile LONGLONG*)&m_Blocks[i], 0 );
            }
        }

        //--------------------------------------------------------------------------------------
        // Name: ThreadSafeBitField::IsEmpty
        // Desc: Returns TRUE if any bit is set, FALSE otherwise.
        //--------------------------------------------------------------------------------------
        inline BOOL IsEmpty()
        {
            UINT64 Value = 0;
            for( UINT i = 0; i < ARRAYSIZE(m_Blocks); ++i )
            {
                Value |= m_Blocks[i];
            }
            return Value == 0;
        }
    };

    //--------------------------------------------------------------------------------------
    // Name: InternalSurfaceDesc
    // Desc: Describes the layout of a tiled texture surface, where the surface dimension in
    //       pages may exceed the texel dimensions due to page alignment.
    //--------------------------------------------------------------------------------------
    struct InternalSurfaceDesc : public D3D11_TILED_SURFACE_DESC
    {
        // Width and height of the mip level's addressable virtual address space, in pages:
        UINT AddressablePageWidth;
        UINT AddressablePageHeight;
    };

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase
    // Desc: Base class for a single tiled resource.  All of the tiled resource functionality
    //       is in this class, including maintentance of CPU and GPU copies of the index map,
    //       conversion between UV/texel coordinates and virtual addresses, and setting the
    //       resource into the D3D device context.
    //--------------------------------------------------------------------------------------
    class TiledResourceBase
    {
    public:
        struct CB_TiledResource
        {
            XMFLOAT4 LODConstants[9];
            XMFLOAT4 ResourceConstant;
        };

    protected:
        friend class PhysicalPageManager;

        // The resource ID that was assigned to this resource by the physical page manager:
        UINT m_ResourceID;

        // The physical page manager that is tracking this resource:
        PhysicalPageManager* m_pPageManager;

        // The typed page pool that corresponds to this resource's data format:
        TypedPagePool* m_pTypedPagePool;

        // The size of the base level, in texels:
        SIZE m_BaseLevelSizeTexels;

        // The CPU-only respresentation of the index map texture.
        // This copy is in sync with CPU threads, for querying and updating.
        CPUTexture m_IndexMapCPU;

        // The GPU-only representation of the index map texture.
        // This copy is in sync with the GPU, and is updated at the beginning of the frame
        // following updates to the CPU index map.
        ID3D11Texture2D* m_pIndexMapGPU;

        // A bit field that indicates that a subresource of the CPU index map has been touched, and
        // needs to be copied to the GPU index map next frame:
        ThreadSafeBitField m_SubresourceNeedsGPUTextureUpdate;

        // Shader resource view for the GPU index map texture:
        ID3D11ShaderResourceView* m_pIndexMapSRV;

        // Shader resource view for the typed page pool:
        ID3D11ShaderResourceView* m_pPagePoolSRV;

        // Sampler state for sampling the index map texture:
        static ID3D11SamplerState* s_pIndexMapSamplerState;

        // Format of this resource:
        DXGI_FORMAT m_ResourceFormat;

        // Mipmap LOD bias that converts the virtual texture dimensions to the dimensions of the index map texture:
        FLOAT m_fMipLODBias;

        // Constant buffer constants for the tiled resource:
        CB_TiledResource m_CBData;

        // Resource dimension (TEXTURE2D, etc):
        D3D11_RESOURCE_DIMENSION m_ResourceDimension;

        // Number of array slices:
        UINT m_ArraySliceCount;

        // Quilting width and height:
        UINT m_QuiltWidth;
        UINT m_QuiltHeight;

        // Number of mip levels:
        UINT m_MipLevelCount;

        // Surface desc for each mip level:
        InternalSurfaceDesc m_MipLevelDesc[9];

    public:
        TiledResourceBase();
        ~TiledResourceBase();

        static UINT GetVSBaseSlotIndex() { return 7; }
        VOID VSSetShaderResource( ID3D11DeviceContext* pd3dDeviceContext, UINT SlotIndex );

        static UINT GetPSBaseSlotIndex() { return 7; }
        VOID PSSetShaderResource( ID3D11DeviceContext* pd3dDeviceContext, UINT SlotIndex );

        const CB_TiledResource& GetShaderConstants() const { return m_CBData; }

        HRESULT Initialize( ID3D11Device* pd3dDevice, PhysicalPageManager* pPageManager, UINT Width, UINT Height, UINT MipLevelCount, UINT ArraySize, DXGI_FORMAT ResourceFormat, UINT QuiltWidth, UINT QuiltHeight );

        VirtualPageID GetVirtualPageIDFloat( FLOAT U, FLOAT V, UINT SliceIndex, UINT MipLevel ) const;
        VirtualPageID GetVirtualPageIDTexel( UINT TexelX, UINT TexelY, UINT SliceIndex, UINT MipLevel ) const;
        VirtualPageID GetVirtualPageIDPage( UINT PageX, UINT PageY, UINT SliceIndex, UINT MipLevel ) const;

        HRESULT GetNeighborhood( ID3D11DeviceContext* pd3dDeviceContext, VirtualPageID CenterPage, PageNeighborhood* pNeighborhood );

        BOOL IsTexture2D() const;
        BOOL IsTexture2DArray() const;
        BOOL IsBuffer() const;
        BOOL IsQuilted() const { return m_QuiltWidth > 1 || m_QuiltHeight > 1; }
        DXGI_FORMAT GetResourceFormat() const { return m_ResourceFormat; }
        TypedPagePool* GetTypedPagePool() const { return m_pTypedPagePool; }
        ID3D11Texture2D* GetIndexMapGPUTexture() const { return m_pIndexMapGPU; }
        UINT GetResourceID() const { return m_ResourceID; }

        UINT GetMipLevelCount() const { return m_MipLevelCount; }
        UINT GetArraySliceCount() const { return m_ArraySliceCount; }
        UINT GetQuiltWidth() const { return m_QuiltWidth; }
        UINT GetQuiltHeight() const { return m_QuiltHeight; }
        VOID GetLevelDesc( UINT MipLevel, D3D11_TILED_SURFACE_DESC* pDesc ) const;

        VOID GetMemoryUsage( D3D11_TILED_MEMORY_USAGE* pMemoryUsage ) const;

        HRESULT SetIndexMapEntry( VirtualPageID VPageID, PhysicalPageID PageID, INT PagePoolIndex );
        VOID GPUTextureUpdate( ID3D11DeviceContext* pd3dDeviceContext );

        UINT ConvertQuiltUVToArrayUVW( FLOAT* pU, FLOAT* pV ) const;

    protected:
        HRESULT SetQuilted( UINT QuiltWidth, UINT QuiltHeight );
        VOID ComputeLevelDesc( UINT MipLevel );
        FLOAT ComputeIndexMapLODBias( UINT TexWidth, UINT NumPageWidth, UINT TexHeight, UINT NumPageHeight ) const;
        VOID CreateLODShaderConstants( UINT PageWidthTexels, UINT PageHeightTexels, UINT TextureWidthPixels, UINT TextureHeightPixels );
        HRESULT GetQuiltNeighborhood( VirtualPageID CenterPage, PageNeighborhood* pNeighborhood );
    };

    //--------------------------------------------------------------------------------------
    // Name: TiledTexture
    // Desc: Subclass of TiledResourceBase for texture 2D and texture 2D array
    //--------------------------------------------------------------------------------------
    class TiledTexture : public TiledResourceBase
    {

    };
}
