//--------------------------------------------------------------------------------------
// TiledResourceBase.cpp
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "TiledResourceBase.h"
#include "PhysicalPageManager.h"
#include "TypedPagePool.h"
#include "PageRenderer.h"

#include "TiledResourceRuntimeTest.h"
using namespace TiledRuntimeTest;

namespace TiledRuntime
{
    ID3D11SamplerState* TiledResourceBase::s_pIndexMapSamplerState = NULL;

    //--------------------------------------------------------------------------------------
    // Name: DumpCpuTextureLevel
    // Desc: Debug method that dumps the contents of a CPU texture subresource to debug out.
    //--------------------------------------------------------------------------------------
    VOID DumpCpuTextureLevel( CPUTexture* pTexture, UINT Subresource )
    {
        UINT Height = 0;
        D3D11_MAPPED_SUBRESOURCE MapData;
        pTexture->CPUMap( Subresource, &MapData, &Height );

        CHAR strPixel[50];

        BYTE* pRow = (BYTE*)MapData.pData;
        for( UINT y = 0; y < Height; ++y )
        {
            for ( UINT x = 0; x < MapData.RowPitch; x += 4 )
            {
                UINT Value = *(UINT*)( pRow + x );
                sprintf_s( strPixel, "<%08x> ", Value );
                OutputDebugStringA( strPixel );
            }
            pRow += MapData.RowPitch;
            OutputDebugStringA( "\n" );
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: CPUTexture constructor
    //--------------------------------------------------------------------------------------
    CPUTexture::CPUTexture()
    {
        m_pAllocation = NULL;
        m_pSubresources = NULL;
        m_ArraySize = 0;
        m_MipLevels = 0;
    }

    //--------------------------------------------------------------------------------------
    // Name: CPUTexture destructor
    //--------------------------------------------------------------------------------------
    CPUTexture::~CPUTexture()
    {
        Terminate();
    }

    //--------------------------------------------------------------------------------------
    // Name: CPUTexture::Initialize
    // Desc: Creates the memory allocation and subresource information for a CPU texture.
    //--------------------------------------------------------------------------------------
    VOID CPUTexture::Initialize( UINT Width, UINT Height, UINT ArraySize, UINT Levels, UINT BytesPerPixel )
    {
        m_BytesPerPixel = BytesPerPixel;

        // Count the mip levels if 0 is specified for levels:
        if( Levels == 0 )
        {
            UINT TestWidth = Width;
            UINT TestHeight = Height;
            while( TestWidth > 1 || TestHeight > 1 )
            {
                ++Levels;
                TestWidth = max( 1, TestWidth / 2 );
                TestHeight = max( 1, TestHeight / 2 );
            }
            ++Levels;
        }

        m_ArraySize = ArraySize;
        m_MipLevels = Levels;

        // Create subresource desc array:
        const UINT SubresourceCount = Levels * ArraySize;
        m_pSubresources = new SubresourceDesc[SubresourceCount];

        // Determine the size and offset of each subresource:
        UINT CurrentMipOffset = 0;
        for( UINT i = 0; i < Levels; ++i )
        {
            m_pSubresources[i].RowPitchBytes = Width * BytesPerPixel;
            m_pSubresources[i].Height = Height;
            m_pSubresources[i].pBase = (BYTE*)CurrentMipOffset;

            CurrentMipOffset += ( Width * Height * BytesPerPixel );

            if( Width <= 1 && Height <= 1 )
            {
                break;
            }

            Width = max( 1, Width / 2 );
            Height = max( 1, Height / 2 );
        }

        // The CurrentMipOffset variable now contains the size of one mip chain:
        const UINT MipChainSizeBytes = CurrentMipOffset;

        // Copy the first mip chain's data to the other array slices, if they exist:
        for( UINT i = 1; i < ArraySize; ++i )
        {
            for( UINT j = 0; j < Levels; ++j )
            {
                UINT SubresourceIndex = i * Levels + j;
                m_pSubresources[SubresourceIndex].pBase = m_pSubresources[j].pBase + ( MipChainSizeBytes * i );
                m_pSubresources[SubresourceIndex].RowPitchBytes = m_pSubresources[j].RowPitchBytes;
                m_pSubresources[SubresourceIndex].Height = m_pSubresources[j].Height;
            }
        }

        // Allocate and zero a single buffer for the entire texture:
        BYTE* pAllocation = new BYTE[MipChainSizeBytes * ArraySize];
        ZeroMemory( pAllocation, MipChainSizeBytes * ArraySize );

        // Offset each subresource by the base address of the allocation:
        for( UINT i = 0; i < SubresourceCount; ++i )
        {
            m_pSubresources[i].pBase += (UINT64)pAllocation;
        }

        m_pAllocation = pAllocation;
    }

    //--------------------------------------------------------------------------------------
    // Name: CPUTexture::Terminate
    // Desc: Deletes the allocation and the subresource array.
    //--------------------------------------------------------------------------------------
    VOID CPUTexture::Terminate()
    {
        if( m_pAllocation != NULL )
        {
            delete[] m_pAllocation;
            m_pAllocation = NULL;
        }
        if( m_pSubresources != NULL )
        {
            delete[] m_pSubresources;
            m_pSubresources = NULL;
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: CPUTexture::CPUMap
    // Desc: Returns a pointer to a subresource, in a D3D11_MAPPED_SUBRESOURCE struct. Note
    //       that there is no Unmap function, since we do not need to hold a lock on a CPU
    //       texture.
    //--------------------------------------------------------------------------------------
    VOID CPUTexture::CPUMap( UINT SubresourceIndex, D3D11_MAPPED_SUBRESOURCE* pMappedSubresource, UINT* pSubresourceHeight )
    {
        ASSERT( SubresourceIndex < m_ArraySize * m_MipLevels );

        // Fill in the data pointer and row pitch members:
        pMappedSubresource->pData = m_pSubresources[SubresourceIndex].pBase;
        pMappedSubresource->RowPitch = m_pSubresources[SubresourceIndex].RowPitchBytes;

        // Depth pitch is always 0 since we don't support volume textures:
        pMappedSubresource->DepthPitch = 0;

        // Return the height of the subresource in rows:
        if( pSubresourceHeight != NULL )
        {
            *pSubresourceHeight = m_pSubresources[SubresourceIndex].Height;
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: CPUTexture::CopySubresourceToTexture
    // Desc: Executes a copy from a subresource of the CPU texture to a GPU texture.
    //--------------------------------------------------------------------------------------
    VOID CPUTexture::CopySubresourceToTexture( ID3D11DeviceContext* pd3dDeviceContext, ID3D11Texture2D* pDestTexture, UINT SubresourceIndex )
    {
        D3D11_MAPPED_SUBRESOURCE MapData;
        CPUMap( SubresourceIndex, &MapData );
        pd3dDeviceContext->UpdateSubresource( pDestTexture, SubresourceIndex, NULL, MapData.pData, MapData.RowPitch, MapData.DepthPitch );
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase constructor
    //--------------------------------------------------------------------------------------
    TiledResourceBase::TiledResourceBase()
    {
        m_ResourceID = 0;
        m_pPageManager = NULL;

        m_pIndexMapGPU = NULL;
        m_pTypedPagePool = NULL;
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase destructor
    // Desc: Releases D3D11 objects.
    //--------------------------------------------------------------------------------------
    TiledResourceBase::~TiledResourceBase()
    {
        m_pPageManager = NULL;

        m_IndexMapCPU.Terminate();
        SAFE_RELEASE( m_pIndexMapGPU );
        SAFE_RELEASE( m_pPagePoolSRV );
        SAFE_RELEASE( m_pIndexMapSRV );

        SAFE_RELEASE( s_pIndexMapSamplerState );
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::VSSetShaderResource
    // Desc: Sets the tiled resource into the given "slot" on a real D3D device context.
    //       Each slot is a pair of textures - one for the index map texture, and one for
    //       the typed page pool's physical page array texture.  A pair of sampler states is
    //       also set (one for the resource, one for the typed page pool).
    //--------------------------------------------------------------------------------------
    VOID TiledResourceBase::VSSetShaderResource( ID3D11DeviceContext* pd3dDeviceContext, UINT SlotIndex )
    {
        // Set the index map texture shader resource view.
        pd3dDeviceContext->VSSetShaderResources( GetVSBaseSlotIndex() + SlotIndex, 1, &m_pIndexMapSRV );

        // Set the index map texture sampler state.
        pd3dDeviceContext->VSSetSamplers( 13, 1, &s_pIndexMapSamplerState );

        // Call the typed page pool to set itself to the D3D device context on this slot.
        m_pTypedPagePool->VSSetSRVSamplerStateAndConstants( pd3dDeviceContext, m_pPagePoolSRV, SlotIndex );
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::PSSetShaderResource
    // Desc: Sets the tiled resource into the given "slot" on a real D3D device context.
    //       Each slot is a pair of textures - one for the index map texture, and one for
    //       the typed page pool's physical page array texture.  A pair of sampler states is
    //       also set (one for the resource, one for the typed page pool).
    //--------------------------------------------------------------------------------------
    VOID TiledResourceBase::PSSetShaderResource( ID3D11DeviceContext* pd3dDeviceContext, UINT SlotIndex )
    {
        // Set the index map texture shader resource view.
        pd3dDeviceContext->PSSetShaderResources( GetPSBaseSlotIndex() + SlotIndex, 1, &m_pIndexMapSRV );

        // Set the index map texture sampler state.
        pd3dDeviceContext->PSSetSamplers( 13, 1, &s_pIndexMapSamplerState );

        // Call the typed page pool to set itself to the D3D device context on this slot.
        m_pTypedPagePool->PSSetSRVSamplerStateAndConstants( pd3dDeviceContext, m_pPagePoolSRV, SlotIndex );
    }

    //--------------------------------------------------------------------------------------
    // Name: IndexMapDimension
    // Desc: For a given texel size and page size, compute the 1D index map dimension to allow
    //       for a full mip chain.  This is not as easy as dividing the texel size by the page
    //       size; due to integer rounding, the base level might have to be expanded to ensure
    //       that higher mip levels are allocated enough pages to completely cover the required
    //       texel count for those levels.
    //       The return value is the size of the index map's base dimension, in pages.
    //--------------------------------------------------------------------------------------
    UINT IndexMapDimension( const UINT TexelSize, const UINT PageSize )
    {
        ASSERT( TexelSize <= 16384 );
        UINT MipSizes[14] = { 0 };
        UINT PageCounts[14] = { 0 };

        // Compute the lowest mip index for the given texel size.
        // The lowest mip index is the lowest mip LOD where the texel size is less than or
        // equal to the page size.  This value will be the size of the mip chain:
        INT LowestMipIndex = ARRAYSIZE(MipSizes);
        UINT Size = TexelSize;
        for( UINT i = 0; i < ARRAYSIZE(MipSizes); ++i )
        {
            // Store each mip level's size in texels:
            MipSizes[i] = Size;

            // Break out of the loop if the mip size fits within a single page:
            if( MipSizes[i] <= PageSize && (INT)i < LowestMipIndex )
            {
                LowestMipIndex = (INT)i;
                break;
            }

            Size = max( 1, Size / 2 );
        }
        ASSERT( LowestMipIndex < ARRAYSIZE(MipSizes) );

        // Initialize the lowest mip index as one page in size:
        PageCounts[LowestMipIndex] = 1;

        // If there is only one mip level, return now:
        if( LowestMipIndex == 0 )
        {
            return 1;
        }

        // Walk from the smallest mip level up to the base level, doubling
        // the page count each step.  If the doubled size isn't sufficient to
        // cover the texel size for that mip level, add one to the page count.
        // This is allowed, because odd mip dimensions will be rounded down
        // when divided by 2:
        for( INT i = LowestMipIndex - 1; i >= 0; --i )
        {
            // Double the page count:
            PageCounts[i] = PageCounts[ i + 1 ] * 2;

            // If the page count multiplied by the page size is less than the texel size for this mip level,
            // add one to the page count:
            if( ( PageCounts[i] * PageSize ) < MipSizes[i] )
            {
                PageCounts[i] += 1;
            }

            // Double check that the adjusted page count is sufficient to cover the texel size of this level:
            ASSERT( ( PageCounts[i] * PageSize ) >= MipSizes[i] );
        }

        // Return the base level's page count:
        ASSERT( PageCounts[0] >= 1 );
        return PageCounts[0];
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::Initialize
    // Desc: Initializes a texture 2D or texture 2D array tiled resource.
    //--------------------------------------------------------------------------------------
    HRESULT TiledResourceBase::Initialize( ID3D11Device* pd3dDevice, PhysicalPageManager* pPageManager, UINT Width, UINT Height, UINT MipLevelCount, UINT ArraySize, DXGI_FORMAT ResourceFormat, UINT QuiltWidth, UINT QuiltHeight )
    {
        // Create index map sampler state:
        if( s_pIndexMapSamplerState == NULL )
        {
            D3D11_SAMPLER_DESC SamplerDesc;
            ZeroMemory( &SamplerDesc, sizeof(SamplerDesc) );
            SamplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
            SamplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
            SamplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
            SamplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
            SamplerDesc.MipLODBias = 0;
            SamplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
            SamplerDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;

            pd3dDevice->CreateSamplerState( &SamplerDesc, &s_pIndexMapSamplerState );
        }

        // Validate the resoure format:
        if( ResourceFormat == DXGI_FORMAT_UNKNOWN )
        {
            return E_FAIL;
        }
        m_ResourceFormat = ResourceFormat;

        // Store the base level texel dimensions:
        m_BaseLevelSizeTexels.cx = Width;
        m_BaseLevelSizeTexels.cy = Height;

        // Get the page size in texels of this format:
        const SIZE PageSizeTexels = GetPageSizeTexels( m_ResourceFormat );

        // Compute the index map base level width and height:
        const UINT IndexMapWidth = IndexMapDimension( Width, PageSizeTexels.cx );
        const UINT IndexMapHeight = IndexMapDimension( Height, PageSizeTexels.cy );

        if( ArraySize > 1 )
        {
            // Create the GPU index map texture array:
            CreateZeroedArrayTexture( pd3dDevice, IndexMapWidth, IndexMapHeight, ArraySize, MipLevelCount, DXGI_FORMAT_INDEXMAP, FALSE, &m_pIndexMapGPU );

            // Create the CPU index map texture array:
            m_IndexMapCPU.Initialize( IndexMapWidth, IndexMapHeight, ArraySize, MipLevelCount, 4 );

            // Store the array size:
            m_ArraySliceCount = ArraySize;
        }
        else
        {
            // Create the GPU index map texture:
            CreateZeroedTexture2D( pd3dDevice, IndexMapWidth, IndexMapHeight, MipLevelCount, DXGI_FORMAT_INDEXMAP, FALSE, &m_pIndexMapGPU );

            // Create the CPU index map texture:
            m_IndexMapCPU.Initialize( IndexMapWidth, IndexMapHeight, 1, MipLevelCount, 4 );

            // Store 1 for the array size:
            m_ArraySliceCount = 1;
        }

        // Create the shader resource view for the index map texture:
        pd3dDevice->CreateShaderResourceView( m_pIndexMapGPU, NULL, &m_pIndexMapSRV );

        m_pIndexMapGPU->GetType( &m_ResourceDimension );

        // Store the mip level count:
        D3D11_TEXTURE2D_DESC TexDesc;
        m_pIndexMapGPU->GetDesc( &TexDesc );
        m_MipLevelCount = TexDesc.MipLevels;

        // Precompute the level descs (they are expensive to compute, and will be accessed frequently at runtime):
        for( UINT i = 0; i < m_MipLevelCount; ++i )
        {
            ComputeLevelDesc( i );
        }

        // Compute the mip LOD bias that causes sampling on the tiled texture texel dimensions to map to the index map dimensions:
        m_fMipLODBias = max( log2f( (UINT)PageSizeTexels.cx ), log2f( (UINT)PageSizeTexels.cy ) );

        // Create shader constants for sampling from the index map texture:
        CreateLODShaderConstants( PageSizeTexels.cx, PageSizeTexels.cy, Width, Height );

        // Initialize quilting if it is enabled:
        if( QuiltWidth > 1 || QuiltHeight > 1 )
        {
            SetQuilted( QuiltWidth, QuiltHeight );
        }
        else
        {
            m_QuiltWidth = 1;
            m_QuiltHeight = 1;
        }

        // Register this resource with the physical page manager:
        pPageManager->RegisterResource( this );

        // Create a shader resource view for the typed page pool's array texture and the resource format:
        ASSERT( m_pTypedPagePool != NULL );
        m_pPagePoolSRV = m_pTypedPagePool->CreateArrayTextureView( pd3dDevice, m_ResourceFormat );

        Trace::CreateTexture2D( m_ResourceID, Width, Height, ArraySize, ResourceFormat );

        return S_OK;
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::SetQuilted
    // Desc: Initializes quilting constants for a tiled texture2D array that is being
    //       interpreted as a rectangular texture quilt.
    //--------------------------------------------------------------------------------------
    HRESULT TiledResourceBase::SetQuilted( UINT QuiltWidth, UINT QuiltHeight )
    {
        if( !IsTexture2DArray() )
        {
            return E_FAIL;
        }

        // Compute the expected array size from the quilt width and height:
        UINT ArraySize = QuiltWidth * QuiltHeight;

        // Make sure the array size matches the quilt size:
        if( ArraySize != GetArraySliceCount() )
        {
            return E_FAIL;
        }

        // Store the quilt width and height:
        m_QuiltWidth = QuiltWidth;
        m_QuiltHeight = QuiltHeight;

        // Add the quilt width and height to the resource constants for use in shading:
        m_CBData.ResourceConstant.z = (FLOAT)m_QuiltWidth;
        m_CBData.ResourceConstant.w = (FLOAT)m_QuiltHeight;

        return S_OK;
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::ComputeIndexMapLODBias
    // Desc: Computes a mip map LOD bias from the difference between the texel dimensions
    //       and the index map dimensions.
    //--------------------------------------------------------------------------------------
    FLOAT TiledResourceBase::ComputeIndexMapLODBias( UINT TexWidth, UINT NumPageWidth, UINT TexHeight, UINT NumPageHeight ) const
    {
        // Compute the base 2 log difference between the texel width and the index map width:
        FLOAT WidthLevels = log2f( TexWidth );
        FLOAT PageWidthLevels = log2f( NumPageWidth );
        FLOAT MipBiasWidth = PageWidthLevels - WidthLevels;

        // Compute the base 2 log difference between the texel height and the index map height:
        FLOAT HeightLevels = log2f( TexHeight );
        FLOAT PageHeightLevels = log2f( NumPageHeight );
        FLOAT MipBiasHeight = PageHeightLevels - HeightLevels;

        // Return the negated average of the width and height log difference.  The value must
        // be negative because we are making the texture sample softer (instead of sampling texels,
        // we are sampling pages, which are much larger than texels):
        return ( MipBiasWidth + MipBiasHeight ) * -0.5f;
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::CreateLODShaderConstants
    // Desc: Creates an array of shader constants that are used to map UV space coordinates
    //       to page coordinates for each mip level of a texture2D.  This isn't always a
    //       trivial computation, because the dimensions of the index map may be larger than
    //       the page dimensions of the tiled resource for a given mip level.
    //--------------------------------------------------------------------------------------
    VOID TiledResourceBase::CreateLODShaderConstants( UINT PageWidthTexels, UINT PageHeightTexels, UINT TextureWidthPixels, UINT TextureHeightPixels )
    {
        const INT MaxLevels = ARRAYSIZE(m_CBData.LODConstants);

        // Determine the index of the max mip level in the texture:
        INT EndLevel = GetMipLevelCount() - 1;
        assert( EndLevel < MaxLevels );

        // Get the base dimensions of the index map texture:
        D3D11_TEXTURE2D_DESC TexDesc;
        m_pIndexMapGPU->GetDesc( &TexDesc );

        UINT LevelWidth = TexDesc.Width;
        UINT LevelHeight = TexDesc.Height;

        // Loop through the index map textures:
        for( INT i = 0; i <= MaxLevels; ++i )
        {
            // Compute the dimensions of the index map in texels, given the page size:
            UINT IndexMapWidth = LevelWidth * PageWidthTexels;
            UINT IndexMapHeight = LevelHeight * PageHeightTexels;

            // Compute the dimensions of a single page in UV space:
            FLOAT PageSizeU = (FLOAT)PageWidthTexels / (FLOAT)TextureWidthPixels;
            FLOAT PageSizeV = (FLOAT)PageHeightTexels / (FLOAT)TextureHeightPixels;

            // Store the reciprocal of the page UV dimensions:
            m_CBData.LODConstants[i].x = 1.0f / PageSizeU;
            m_CBData.LODConstants[i].y = 1.0f / PageSizeV;

            // Compute the scaling factor between the tiled texture texel dimensions and the index map texel dimensions.
            // The index map texel dimensions are sometimes larger than the tiled texture texel dimensions:
            m_CBData.LODConstants[i].z = (FLOAT)TextureWidthPixels / (FLOAT)IndexMapWidth;
            m_CBData.LODConstants[i].w = (FLOAT)TextureHeightPixels / (FLOAT)IndexMapHeight;

            // Divide the tiled texture texel dimensions by 2:
            if( i < EndLevel )
            {
                TextureWidthPixels = max( 1, TextureWidthPixels / 2 );
                TextureHeightPixels = max( 1, TextureHeightPixels / 2 );
            }

            // Divide the index map dimensions by 2:
            LevelWidth = max( 1, LevelWidth / 2 );
            LevelHeight = max( 1, LevelHeight / 2 );
        }

        // Store additional data for sampling, including the mip LOD bias and the inverse array slice count:
        m_CBData.ResourceConstant = XMFLOAT4( m_fMipLODBias, 1.0f / (FLOAT)m_ArraySliceCount, 1, 1 );
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::IsTexture2D
    // Desc: Returns TRUE if this resource is a texture 2D (not an array).
    //--------------------------------------------------------------------------------------
    BOOL TiledResourceBase::IsTexture2D() const
    {
        return m_ResourceDimension == D3D11_RESOURCE_DIMENSION_TEXTURE2D && m_ArraySliceCount == 1;
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::IsTexture2DArray
    // Desc: Returns TRUE if this resource is a texture 2D with more than one array slice.
    //--------------------------------------------------------------------------------------
    BOOL TiledResourceBase::IsTexture2DArray() const
    {
        return m_ResourceDimension == D3D11_RESOURCE_DIMENSION_TEXTURE2D && m_ArraySliceCount > 1;
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::IsBuffer
    // Desc: Returns TRUE if this resource is a buffer.
    //--------------------------------------------------------------------------------------
    BOOL TiledResourceBase::IsBuffer() const
    {
        NOTIMPL;
        return FALSE;
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::ComputeLevelDesc
    // Desc: Fills in a mip level desc struct for the given level.
    //--------------------------------------------------------------------------------------
    VOID TiledResourceBase::ComputeLevelDesc( UINT MipLevel )
    {
        ASSERT( MipLevel < GetMipLevelCount() );

        // Get a pointer to the level desc:
        InternalSurfaceDesc* pDesc = &m_MipLevelDesc[MipLevel];

        // Store the data format:
        pDesc->Format = m_ResourceFormat;

        // Store a single page's dimensions in texels:
        const SIZE PageSizeTexels = GetPageSizeTexels( m_ResourceFormat );
        pDesc->TileTexelWidth = PageSizeTexels.cx;
        pDesc->TileTexelHeight = PageSizeTexels.cy;

        // Get the dimensions of the index map GPU texture:
        D3D11_TEXTURE2D_DESC IndexMapTextureDesc;
        m_pIndexMapGPU->GetDesc( &IndexMapTextureDesc );

        // Compute the index map dimensions for this mip level:
        for( UINT i = 0; i < MipLevel; ++i )
        {
            IndexMapTextureDesc.Width = max( 1, IndexMapTextureDesc.Width / 2 );
            IndexMapTextureDesc.Height = max( 1, IndexMapTextureDesc.Height / 2 );
        }

        // The addressable dimensions are the index map dimensions at this level:
        pDesc->AddressablePageWidth = IndexMapTextureDesc.Width;
        pDesc->AddressablePageHeight = IndexMapTextureDesc.Height;

        // Compute the tiled texture texel dimensions:
        UINT Pow2 = 1 << MipLevel;
        FLOAT BaseSizeMultiple = 1.0f / (FLOAT)Pow2;
        pDesc->TexelWidth = max( 1, (UINT)( (FLOAT)m_BaseLevelSizeTexels.cx * BaseSizeMultiple ) );
        pDesc->TexelHeight = max( 1, (UINT)( (FLOAT)m_BaseLevelSizeTexels.cy * BaseSizeMultiple ) );

        // Compute the usable page dimensions at this mip level, by dividing the texel dimensions by the 
        // page dimensions:
        pDesc->TileWidth = XGNextMultiple( pDesc->TexelWidth, pDesc->TileTexelWidth ) / pDesc->TileTexelWidth;
        pDesc->TileHeight = XGNextMultiple( pDesc->TexelHeight, pDesc->TileTexelHeight ) / pDesc->TileTexelHeight;
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::GetLevelDesc
    // Desc: Returns a copy of one of the precomputed level descs for this resource.
    //--------------------------------------------------------------------------------------
    VOID TiledResourceBase::GetLevelDesc( UINT MipLevel, D3D11_TILED_SURFACE_DESC* pDesc ) const
    {
        ASSERT( pDesc != NULL );
        if( MipLevel >= GetMipLevelCount() )
        {
            ZeroMemory( pDesc, sizeof(D3D11_TILED_SURFACE_DESC) );
            return;
        }
        memcpy( pDesc, &m_MipLevelDesc[MipLevel], sizeof(D3D11_TILED_SURFACE_DESC) );
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::GetVirtualPageIDFloat
    // Desc: Converts a UV texture coordinate, array slice index, and mip level combination
    //       into a virtual page address for the page that contains those coordinates.
    //--------------------------------------------------------------------------------------
    VirtualPageID TiledResourceBase::GetVirtualPageIDFloat( FLOAT U, FLOAT V, UINT SliceIndex, UINT MipLevel ) const
    {
        if( IsTexture2D() || IsTexture2DArray() )
        {
            // Validate mip level:
            if( MipLevel >= GetMipLevelCount() )
            {
                return INVALID_VIRTUAL_PAGE_ID;
            }

            // Validate array slice index:
            if( SliceIndex >= GetArraySliceCount() )
            {
                return INVALID_VIRTUAL_PAGE_ID;
            }

            // Get the level desc for this mip level:
            const InternalSurfaceDesc& SurfDesc = m_MipLevelDesc[MipLevel];

            // Adjust U and V coordinates for resource UV transform:
            ASSERT( MipLevel < ARRAYSIZE(m_CBData.LODConstants) );
            U *= m_CBData.LODConstants[MipLevel].z;
            V *= m_CBData.LODConstants[MipLevel].w;

            // Compute the page X and Y coordinates:
            UINT PageX = (UINT)( U * (FLOAT)( SurfDesc.AddressablePageWidth ) );
            PageX = min( PageX, SurfDesc.TileWidth - 1 );
            UINT PageY = (UINT)( V * (FLOAT)( SurfDesc.AddressablePageHeight ) );
            PageY = min( PageY, SurfDesc.TileHeight - 1 );

            ASSERT( PageX < SurfDesc.TileWidth );
            ASSERT( PageY < SurfDesc.TileHeight );

            // Fill in a virtual address with the page X and Y coordinates, along
            // with the mip level and array slice index:
            VirtualPageID VPageID;
            VPageID.ResourceID = m_ResourceID;
            VPageID.PageX = PageX;
            VPageID.PageY = PageY;
            VPageID.ArraySlice = SliceIndex;
            VPageID.MipLevel = MipLevel;
            VPageID.Valid = 1;

            return VPageID;
        }
        NOTIMPL;
        return INVALID_VIRTUAL_PAGE_ID;
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::GetVirtualPageIDTexel
    // Desc: Converts a texel XY coordinate, array slice index, and mip level combination
    //       into a virtual page address for the page that contains those coordinates.
    //--------------------------------------------------------------------------------------
    VirtualPageID TiledResourceBase::GetVirtualPageIDTexel( UINT TexelX, UINT TexelY, UINT SliceIndex, UINT MipLevel ) const
    {
        if( IsTexture2D() || IsTexture2DArray() )
        {
            // Get the page size in texels for the resource format:
            const SIZE PageSizeTexels = GetPageSizeTexels( m_ResourceFormat );

            // Convert the texel coordinates into page coordinates, and return the virtual address:
            return GetVirtualPageIDPage( TexelX / PageSizeTexels.cx, TexelY / PageSizeTexels.cy, SliceIndex, MipLevel );
        }
        NOTIMPL;
        return INVALID_VIRTUAL_PAGE_ID;
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::GetVirtualPageIDPage
    // Desc: Converts a page XY coordinate, array slice index, and mip level combination
    //       into a virtual page address.
    //--------------------------------------------------------------------------------------
    VirtualPageID TiledResourceBase::GetVirtualPageIDPage( UINT PageX, UINT PageY, UINT SliceIndex, UINT MipLevel ) const
    {
        if( IsTexture2D() || IsTexture2DArray() )
        {
            // Validate mip level:
            if( MipLevel >= GetMipLevelCount() )
            {
                return INVALID_VIRTUAL_PAGE_ID;
            }

            // Validate array slice index:
            if( SliceIndex >= GetArraySliceCount() )
            {
                return INVALID_VIRTUAL_PAGE_ID;
            }

            // Validate page X and Y coordinates:
            const InternalSurfaceDesc& MipDesc = m_MipLevelDesc[MipLevel];
            if( PageX >= MipDesc.AddressablePageWidth || PageY >= MipDesc.AddressablePageHeight )
            {
                return INVALID_VIRTUAL_PAGE_ID;
            }

            // Fill in a virtual address with the page X and Y coordinates, along
            // with the mip level and array slice index:
            VirtualPageID VPageID;
            VPageID.ResourceID = m_ResourceID;
            VPageID.PageX = PageX;
            VPageID.PageY = PageY;
            VPageID.ArraySlice = SliceIndex;
            VPageID.MipLevel = MipLevel;
            VPageID.Valid = 1;
            return VPageID;
        }
        NOTIMPL;
        return INVALID_VIRTUAL_PAGE_ID;
    }

    //--------------------------------------------------------------------------------------
    // Name: FetchFromIndexMap
    // Desc: Helper function that decodes an index map entry from a locked CPU index map
    //       texture.
    //--------------------------------------------------------------------------------------
    inline VOID FetchFromIndexMap( const D3D11_MAPPED_SUBRESOURCE& LockRect, const SIZE& SurfSize, INT TexelX, INT TexelY, UINT* pAtlasX, UINT* pAtlasY, UINT* pAtlasSlice, BOOL* pValid )
    {
        // Return an invalid sample for texel addresses that are out of range:
        if( TexelX < 0 || TexelY < 0 || TexelX >= (INT)SurfSize.cx || TexelY >= (INT)SurfSize.cy )
        {
            *pValid = FALSE;
            return;
        }

        // Select the proper index map texel:
        const BYTE* pBits = (const BYTE*)LockRect.pData;
        pBits += TexelY * LockRect.RowPitch;
        pBits += TexelX * sizeof(UINT);

        // Decode the valid flag:
        BYTE Valid = pBits[3];
        *pValid = Valid > 0;

        // Decode the atlas location:
        *pAtlasX = (UINT)pBits[0];
        *pAtlasY = (UINT)pBits[1];
        *pAtlasSlice = (UINT)pBits[2];
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::GetNeighborhood
    // Desc: For a given virtual page address, this method finds that page and its immediate
    //       8 neighbors in the index map, and returns the neighbor pages' physical addresses.
    //--------------------------------------------------------------------------------------
    HRESULT TiledResourceBase::GetNeighborhood( ID3D11DeviceContext* pd3dDeviceContext, VirtualPageID CenterPage, PageNeighborhood* pNeighborhood )
    {
        if( IsTexture2D() || IsTexture2DArray() )
        {
            // For quilting, we go through a different codepath:
            if( IsQuilted() )
            {
                return GetQuiltNeighborhood( CenterPage, pNeighborhood );
            }

            // Decode the center page location:
            INT CenterX = (INT)CenterPage.PageX;
            INT CenterY = (INT)CenterPage.PageY;

            BOOL ValidEntry[9] = { FALSE };
            UINT AtlasX[9];
            UINT AtlasY[9];
            UINT AtlasSlice[9];

            // Get the index map mip level size for the given mip level:
            const SIZE SurfSize = GetMipLevelSize( m_pIndexMapGPU, CenterPage.MipLevel );

            // Compute the subresource index for the given mip level and array slice:
            UINT SubresourceIndex = (UINT)CenterPage.ArraySlice * GetMipLevelCount() + (UINT)CenterPage.MipLevel;

            // "Map" the CPU texture for reading:
            D3D11_MAPPED_SUBRESOURCE MapData;
            m_IndexMapCPU.CPUMap( SubresourceIndex, &MapData );

            // Using the index map texels, convert the 8 neighbors and center page into 9 physical page array locations:
            FetchFromIndexMap( MapData, SurfSize, CenterX - 1, CenterY - 1, &AtlasX[0], &AtlasY[0], &AtlasSlice[0], &ValidEntry[0] );
            FetchFromIndexMap( MapData, SurfSize, CenterX + 0, CenterY - 1, &AtlasX[1], &AtlasY[1], &AtlasSlice[1], &ValidEntry[1] );
            FetchFromIndexMap( MapData, SurfSize, CenterX + 1, CenterY - 1, &AtlasX[2], &AtlasY[2], &AtlasSlice[2], &ValidEntry[2] );

            FetchFromIndexMap( MapData, SurfSize, CenterX - 1, CenterY + 0, &AtlasX[3], &AtlasY[3], &AtlasSlice[3], &ValidEntry[3] );
            FetchFromIndexMap( MapData, SurfSize, CenterX + 0, CenterY + 0, &AtlasX[4], &AtlasY[4], &AtlasSlice[4], &ValidEntry[4] );
            FetchFromIndexMap( MapData, SurfSize, CenterX + 1, CenterY + 0, &AtlasX[5], &AtlasY[5], &AtlasSlice[5], &ValidEntry[5] );

            FetchFromIndexMap( MapData, SurfSize, CenterX - 1, CenterY + 1, &AtlasX[6], &AtlasY[6], &AtlasSlice[6], &ValidEntry[6] );
            FetchFromIndexMap( MapData, SurfSize, CenterX + 0, CenterY + 1, &AtlasX[7], &AtlasY[7], &AtlasSlice[7], &ValidEntry[7] );
            FetchFromIndexMap( MapData, SurfSize, CenterX + 1, CenterY + 1, &AtlasX[8], &AtlasY[8], &AtlasSlice[8], &ValidEntry[8] );

            const PageNeighbors OutputLocations[] = { PN_TOPLEFT, PN_TOP, PN_TOPRIGHT, PN_LEFT, PN_COUNT, PN_RIGHT, PN_BOTTOMLEFT, PN_BOTTOM, PN_BOTTOMRIGHT };
            C_ASSERT( ARRAYSIZE(ValidEntry) == ARRAYSIZE(OutputLocations) );

            // Loop over the 9 physical page array locations:
            for( UINT i = 0; i < ARRAYSIZE(ValidEntry); ++i )
            {
                // Convert the array location into a physical page address using the typed page pool:
                PhysicalPageID PageID = INVALID_PHYSICAL_PAGE_ID;
                if( ValidEntry[i] )
                {
                    PageID = m_pTypedPagePool->GetPageByAtlasLocation( AtlasSlice[i], AtlasX[i], AtlasY[i] );
                }

                // Place the physical page ID into the proper slot in the neighborhood structure:
                if( OutputLocations[i] == PN_COUNT )
                {
                    pNeighborhood->m_CenterPage = PageID;
                }
                else
                {
                    pNeighborhood->m_Neighbors[OutputLocations[i]] = PageID;
                }
            }

            return S_OK;
        }
        return E_NOTIMPL;
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::GetQuiltNeighborhood
    // Desc: For a given virtual page address, this method finds that page and its immediate
    //       8 neighbors in the index map, and returns the neighbor pages' physical addresses.
    //       This method has additional logic to deal with quilt boundaries.
    //--------------------------------------------------------------------------------------
    HRESULT TiledResourceBase::GetQuiltNeighborhood( VirtualPageID CenterPage, PageNeighborhood* pNeighborhood )
    {
        ASSERT( IsTexture2DArray() && IsQuilted() );

        // Get the dimensions of the mip level:
        const SIZE SurfSize = GetMipLevelSize( m_pIndexMapGPU, CenterPage.MipLevel );

        // Get the level desc of the mip level:
        const InternalSurfaceDesc& MipSurfDesc = m_MipLevelDesc[CenterPage.MipLevel];

        // Determine the quilt location from the array slice index:
        INT QuiltX = (UINT)CenterPage.ArraySlice % GetQuiltWidth();
        INT QuiltY = (UINT)CenterPage.ArraySlice / GetQuiltWidth();

        // Determine the page location:
        INT CenterX = (INT)CenterPage.PageX;
        INT CenterY = (INT)CenterPage.PageY;

        // Build a static array of X and Y offsets for the given neighbor directions:
        static const INT XOffset[PN_COUNT] = { 0, 0, -1, 1, -1, 1, 1, -1 };
        static const INT YOffset[PN_COUNT] = { -1, 1, 0, 0, -1, 1, -1, 1 };

        // Loop over the neighbors:
        for( UINT Neighbor = 0; Neighbor < PN_COUNT; ++Neighbor )
        {
            // Get the page location of the neighbor:
            INT PageX = CenterX + XOffset[Neighbor];
            INT PageY = CenterY + YOffset[Neighbor];

            INT CurrentQuiltX = QuiltX;
            INT CurrentQuiltY = QuiltY;

            // If the page location is outside the mip level bounds, then go to the quilt neighbor:
            if( PageX < 0 )
            {
                CurrentQuiltX--;
                PageX = MipSurfDesc.TileWidth - 1;
            }
            else if( PageX >= (INT)MipSurfDesc.TileWidth )
            {
                CurrentQuiltX++;
                PageX = 0;
            }

            if( PageY < 0 )
            {
                CurrentQuiltY--;
                PageY = MipSurfDesc.TileHeight - 1;
            }
            else if( PageY >= (INT)MipSurfDesc.TileHeight )
            {
                CurrentQuiltY++;
                PageY = 0;
            }

            // Find the neighboring page if the quilt location is still valid:
            if( CurrentQuiltX >= 0 && CurrentQuiltX < (INT)GetQuiltWidth() && CurrentQuiltY >= 0 && CurrentQuiltY < (INT)GetQuiltHeight() )
            {
                // Compute a new array slice index for the neighbor page:
                UINT SliceIndex = CurrentQuiltY * GetQuiltWidth() + CurrentQuiltX;

                // Compute the subresource index:
                UINT SubresourceIndex = SliceIndex * GetMipLevelCount() + (UINT)CenterPage.MipLevel;

                // Map the CPU texture for reading:
                D3D11_MAPPED_SUBRESOURCE MapData;
                m_IndexMapCPU.CPUMap( SubresourceIndex, &MapData );

                BOOL ValidEntry;
                UINT AtlasX;
                UINT AtlasY;
                UINT AtlasSlice;

                // Get the atlas location of the physical page:
                FetchFromIndexMap( MapData, SurfSize, PageX, PageY, &AtlasX, &AtlasY, &AtlasSlice, &ValidEntry );

                // Fill in the appropriate slot on the neighborhood struct:
                if( ValidEntry )
                {
                    PhysicalPageID PageID = m_pTypedPagePool->GetPageByAtlasLocation( AtlasSlice, AtlasX, AtlasY );
                    pNeighborhood->m_Neighbors[Neighbor] = PageID;
                }
                else
                {
                    pNeighborhood->m_Neighbors[Neighbor] = INVALID_PHYSICAL_PAGE_ID;
                }
            }
            else
            {
                pNeighborhood->m_Neighbors[Neighbor] = INVALID_PHYSICAL_PAGE_ID;
            }
        }

        return S_OK;
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::GetMemoryUsage
    // Desc: Returns memory usage statistics for this resource.
    //--------------------------------------------------------------------------------------
    VOID TiledResourceBase::GetMemoryUsage( D3D11_TILED_MEMORY_USAGE* pMemoryUsage ) const
    {
        ASSERT( pMemoryUsage != NULL );

        // Increment resource count:
        pMemoryUsage->ResourceCount++;

        // Accumulate the GPU index map texture size in bytes:
        pMemoryUsage->ResourceTextureMemoryBytesAllocated += GetTextureSizeBytes( m_pIndexMapGPU );

        // Accumulate the resource's total virtual memory size in bytes:
        UINT64 ResourceVMBytes = 0;
        for( UINT i = 0; i < m_MipLevelCount; ++i )
        {
            UINT PageCount = m_MipLevelDesc[i].AddressablePageWidth * m_MipLevelDesc[i].AddressablePageHeight;
            ResourceVMBytes += (UINT64)PageCount * (UINT64)PAGE_SIZE_BYTES;
        }
        ResourceVMBytes *= m_ArraySliceCount;

        pMemoryUsage->ResourceVirtualBytesAllocated += ResourceVMBytes;
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::SetIndexMapEntry
    // Desc: Updates the index map CPU texture with a new virtual to physical page mapping.
    //--------------------------------------------------------------------------------------
    HRESULT TiledResourceBase::SetIndexMapEntry( VirtualPageID VPageID, PhysicalPageID PageID, INT PagePoolIndex )
    {
        ASSERT( VPageID.Valid );
        ASSERT( VPageID.ResourceID == m_ResourceID );

        if( IsTexture2D() || IsTexture2DArray() )
        {
            // Look up the atlas entry by index in the typed page pool.
            // The atlas entry contains the data that will need to be written to the index map texel.
            AtlasEntry* pEntry = NULL;
            if( PagePoolIndex != -1 )
            {
                pEntry = m_pTypedPagePool->GetAtlasEntry( PagePoolIndex );
                ASSERT( pEntry->PageID == PageID );
            }

            // Compute the subresource index from the virtual address:
            UINT SubresourceIndex = (UINT)VPageID.ArraySlice * GetMipLevelCount() + (UINT)VPageID.MipLevel;

            // Map the index map CPU texture for writing:
            D3D11_MAPPED_SUBRESOURCE MapData;
            m_IndexMapCPU.CPUMap( SubresourceIndex, &MapData );

            // Offset to the correct texel in the index map given the page X and Y location:
            BYTE* pBits = (BYTE*)MapData.pData;
            pBits += VPageID.PageY * MapData.RowPitch;
            pBits += VPageID.PageX * sizeof(UINT);

            if( PageID != INVALID_PHYSICAL_PAGE_ID )
            {
                // Write the atlas location to the index map texel:
                ASSERT( pEntry != NULL );
                pBits[3] = 255;
                pBits[0] = (BYTE)pEntry->X;
                pBits[1] = (BYTE)pEntry->Y;
                pBits[2] = (BYTE)pEntry->Slice;
            }
            else
            {
                // Invalidate the texel for a null physical address:
                pBits[3] = 0;
            }

            // Ensure that the texel write is flushed:
            MemoryBarrier();

            // Set a bit that indicates that this subresource needs to be updated in
            // the index map GPU texture:
            m_SubresourceNeedsGPUTextureUpdate.SetBit( SubresourceIndex );

            return S_OK;
        }
        NOTIMPL;
        return E_NOTIMPL;
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::GPUTextureUpdate
    // Desc: Based on the state of the m_SubresourceNeedsGPUTextureUpdate bitfield, this
    //       method will copy subresources from the index map CPU texture to the index map
    //       GPU texture.
    //--------------------------------------------------------------------------------------
    VOID TiledResourceBase::GPUTextureUpdate( ID3D11DeviceContext* pd3dDeviceContext )
    {
        // In a thread safe manner, retrieve a snap of the subresource GPU update flags and
        // clear the flags at the same time:
        ThreadSafeBitField SubresourceFlags;
        m_SubresourceNeedsGPUTextureUpdate.CopyAndClear( SubresourceFlags );

        // If none of the bits are set, then we can exit now:
        if( SubresourceFlags.IsEmpty() )
        {
            return;
        }

        // Loop over the bits.  We can use unsafe bit access because we are working with a local
        // copy of the bits:
        for( UINT Subresource = 0; Subresource < SubresourceFlags.TotalBitCount(); ++Subresource )
        {
            BOOL NeedsUpdate = SubresourceFlags.TestBitUnsafe( Subresource );
            if( NeedsUpdate )
            {
                // Copy a subresource from the CPU texture to the GPU texture:
                m_IndexMapCPU.CopySubresourceToTexture( pd3dDeviceContext, m_pIndexMapGPU, Subresource );

                // Clear the bit:
                SubresourceFlags.ClearBitUnsafe( Subresource );
            }

            // Break early if we have cleared all of the bits:
            if( SubresourceFlags.IsEmpty() )
            {
                break;
            }
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: TiledResourceBase::ConvertQuiltUVToArrayUVW
    // Desc: Converts an extended UV coordinate (0..M, 0..N) to normalized UV coordinates
    //       (0..1) plus the array slice index returned as a result.
    //--------------------------------------------------------------------------------------
    UINT TiledResourceBase::ConvertQuiltUVToArrayUVW( FLOAT* pU, FLOAT* pV ) const
    {
        FLOAT U = *pU;
        FLOAT V = *pV;

        const INT SliceCount = (INT)GetArraySliceCount();
        const INT QuiltWidth = (INT)GetQuiltWidth();
        const INT QuiltHeight = (INT)GetQuiltHeight();

        // Compute integer quilt location from the UV coordinates:
        INT QuiltU = min( QuiltWidth - 1, max( 0, (INT)U ) );
        INT QuiltV = min( QuiltHeight - 1, max( 0, (INT)V ) );

        // Compute a slice index from the quilt location:
        INT SliceIndex = QuiltV * m_QuiltWidth + QuiltU;
        ASSERT( SliceIndex < SliceCount );

        // Return the fractional component of U and V:
        *pU = U - floorf( U );
        *pV = V - floorf( V );

        // Return the slice index:
        return (UINT)SliceIndex;
    }
}
