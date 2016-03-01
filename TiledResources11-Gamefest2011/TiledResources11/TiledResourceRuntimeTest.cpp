#include "DXUT.h"
#include "TiledResourceRuntimeTest.h"

#ifdef _DEBUG
#define TRACE_SPEW 0
#else
#define TRACE_SPEW 0
#endif

namespace TiledRuntimeTest
{
    // String names for the neighbor directions:
    static const CHAR* g_strNeighborNames[] =
    {
        "Top",
        "Bottom",
        "Left",
        "Right",
        "Top Left",
        "Bottom Right",
        "Top Right",
        "Bottom Left"
    };
    C_ASSERT( ARRAYSIZE(g_strNeighborNames) == PN_COUNT );

    //--------------------------------------------------------------------------------------
    // Name: TestTileData::FillRect32Bit
    // Desc: Fills a rectangular block of texels with 32bpp ARGB colors.
    //--------------------------------------------------------------------------------------
    VOID TestTileData::FillRect32Bit( BYTE* pDestBits, DWORD Width, DWORD Height, DWORD PitchBytes, DWORD ColorARGB, BOOL Checker )
    {
        for( DWORD y = 0; y < Height; ++y )
        {
            DWORD* pRow = (DWORD*)pDestBits;
            for( DWORD x = 0; x < Width; ++x )
            {
                DWORD PixelColor = ColorARGB;
                if( Checker && ( x + y ) % 2 == 1 )
                {
                    DWORD Red = ( ( PixelColor >> 16 ) & 0xFF ) * 2 / 4;
                    DWORD Green = ( ( PixelColor >> 8 ) & 0xFF ) * 2 / 4;
                    DWORD Blue = ( ( PixelColor >> 0 ) & 0xFF ) * 2 / 4;
                    PixelColor = D3DCOLOR_ARGB( 0xFF, Red, Green, Blue );
                }
                pRow[x] = PixelColor;
            }
            pDestBits += PitchBytes;
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: TestTileData::FillRect16Bit
    // Desc: Fills a rectangular block of texels with 16bpp 565 BGR colors.
    //--------------------------------------------------------------------------------------
    VOID TestTileData::FillRect16Bit( BYTE* pDestBits, DWORD Width, DWORD Height, DWORD PitchBytes, DWORD ColorARGB, BOOL Checker )
    {
        for( DWORD y = 0; y < Height; ++y )
        {
            WORD* pRow = (WORD*)pDestBits;
            for( DWORD x = 0; x < Width; ++x )
            {
                DWORD PixelColor = ColorARGB;
                if( Checker && ( x + y ) % 2 == 1 )
                {
                    DWORD Red = ( ( PixelColor >> 16 ) & 0xFF ) * 2 / 4;
                    DWORD Green = ( ( PixelColor >> 8 ) & 0xFF ) * 2 / 4;
                    DWORD Blue = ( ( PixelColor >> 0 ) & 0xFF ) * 2 / 4;
                    PixelColor = D3DCOLOR_ARGB( 0xFF, Red, Green, Blue );
                }
                pRow[x] = Color32To565( PixelColor );
            }
            pDestBits += PitchBytes;
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: GetFormatName
    // Desc: Returns a string name for each DXGI format.
    //--------------------------------------------------------------------------------------
    const CHAR* GetFormatName( DXGI_FORMAT Format )
    {
        static const CHAR* s_FormatNames[] =
        {
            "UNKNOWN",
            "R32G32B32A32_TYPELESS",
            "R32G32B32A32_FLOAT",
            "R32G32B32A32_UINT",
            "R32G32B32A32_SINT",
            "R32G32B32_TYPELESS",
            "R32G32B32_FLOAT",
            "R32G32B32_UINT",
            "R32G32B32_SINT",
            "R16G16B16A16_TYPELESS",
            "R16G16B16A16_FLOAT",
            "R16G16B16A16_UNORM",
            "R16G16B16A16_UINT",
            "R16G16B16A16_SNORM",
            "R16G16B16A16_SINT",
            "R32G32_TYPELESS",
            "R32G32_FLOAT",
            "R32G32_UINT",
            "R32G32_SINT",
            "R32G8X24_TYPELESS",
            "D32_FLOAT_S8X24_UINT",
            "R32_FLOAT_X8X24_TYPELESS",
            "X32_TYPELESS_G8X24_UINT",
            "R10G10B10A2_TYPELESS",
            "R10G10B10A2_UNORM",
            "R10G10B10A2_UINT",
            "R11G11B10_FLOAT",
            "R8G8B8A8_TYPELESS",
            "R8G8B8A8_UNORM",
            "R8G8B8A8_UNORM_SRGB",
            "R8G8B8A8_UINT",
            "R8G8B8A8_SNORM",
            "R8G8B8A8_SINT",
            "R16G16_TYPELESS",
            "R16G16_FLOAT",
            "R16G16_UNORM",
            "R16G16_UINT",
            "R16G16_SNORM",
            "R16G16_SINT",
            "R32_TYPELESS",
            "D32_FLOAT",
            "R32_FLOAT",
            "R32_UINT",
            "R32_SINT",
            "R24G8_TYPELESS",
            "D24_UNORM_S8_UINT",
            "R24_UNORM_X8_TYPELESS",
            "X24_TYPELESS_G8_UINT",
            "R8G8_TYPELESS",
            "R8G8_UNORM",
            "R8G8_UINT",
            "R8G8_SNORM",
            "R8G8_SINT",
            "R16_TYPELESS",
            "R16_FLOAT",
            "D16_UNORM",
            "R16_UNORM",
            "R16_UINT",
            "R16_SNORM",
            "R16_SINT",
            "R8_TYPELESS",
            "R8_UNORM",
            "R8_UINT",
            "R8_SNORM",
            "R8_SINT",
            "A8_UNORM",
            "R1_UNORM",
            "R9G9B9E5_SHAREDEXP",
            "R8G8_B8G8_UNORM",
            "G8R8_G8B8_UNORM",
            "BC1_TYPELESS",
            "BC1_UNORM",
            "BC1_UNORM_SRGB",
            "BC2_TYPELESS",
            "BC2_UNORM",
            "BC2_UNORM_SRGB",
            "BC3_TYPELESS",
            "BC3_UNORM",
            "BC3_UNORM_SRGB",
            "BC4_TYPELESS",
            "BC4_UNORM",
            "BC4_SNORM",
            "BC5_TYPELESS",
            "BC5_UNORM",
            "BC5_SNORM",
            "B5G6R5_UNORM",
            "B5G5R5A1_UNORM",
            "B8G8R8A8_UNORM",
            "B8G8R8X8_UNORM",
            "R10G10B10_XR_BIAS_A2_UNORM",
            "B8G8R8A8_TYPELESS",
            "B8G8R8A8_UNORM_SRGB",
            "B8G8R8X8_TYPELESS",
            "B8G8R8X8_UNORM_SRGB",
            "BC6H_TYPELESS",
            "BC6H_UF16",
            "BC6H_SF16",
            "BC7_TYPELESS",
            "BC7_UNORM",
            "BC7_UNORM_SRGB",
        };

        ASSERT( Format >= DXGI_FORMAT_UNKNOWN && Format < DXGI_FORMAT_MAX );
        return s_FormatNames[Format];
    }

    //--------------------------------------------------------------------------------------
    // Name: SpewThreadID
    // Desc: Prints a trace header to the debug output, including the current thread ID.
    //--------------------------------------------------------------------------------------
    VOID SpewThreadID()
    {
        if( TRACE_SPEW )
        {
            DebugSpew( "(%08x)TRACE: ", GetCurrentThreadId() );
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: Trace::CreatePage
    // Desc: Trace event for the creation of a physical page.
    //--------------------------------------------------------------------------------------
    VOID Trace::CreatePage( PhysicalPageID PageID, DXGI_FORMAT Format )
    {
        if( TRACE_SPEW > 1 )
        {
            SpewThreadID();
            DebugSpew( "Creating page %I64d in format %s\n", PageID, GetFormatName( Format ) );
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: Trace::PageCreateFailure
    // Desc: Trace event for when physical page creation fails.
    //--------------------------------------------------------------------------------------
    VOID Trace::PageCreateFailure( PhysicalPageID PageID, DXGI_FORMAT Format )
    {
        if( TRACE_SPEW )
        {
            SpewThreadID();
            DebugSpew( "Could not create page %I64d in format %s\n", PageID, GetFormatName( Format ) );
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: Trace::CreateTexture2D
    // Desc: Trace event for the creation of a tiled texture 2D.
    //--------------------------------------------------------------------------------------
    VOID Trace::CreateTexture2D( INT ResourceID, INT Width, INT Height, INT ArraySize, DXGI_FORMAT Format )
    {
        if( TRACE_SPEW )
        {
            SpewThreadID();
            DebugSpew( "Creating resource ID %d: tiled texture2D size (%d, %d) array size %d format %s\n", ResourceID, Width, Height, ArraySize, GetFormatName( Format ) );
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: Trace::MovePage
    // Desc: Trace event for when a physical page is moved from one typed page pool to another.
    //--------------------------------------------------------------------------------------
    VOID Trace::MovePage( PhysicalPageID PageID, DXGI_FORMAT SrcFormat, DXGI_FORMAT DestFormat )
    {
        if( TRACE_SPEW )
        {
            SpewThreadID();
            DebugSpew( "Moving page %I64d from format %s to format %s\n", PageID, GetFormatName( SrcFormat ), GetFormatName( DestFormat ) );
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: Trace::FillPage
    // Desc: Trace event for when a physical page is filled with contents. 
    //--------------------------------------------------------------------------------------
    VOID Trace::FillPage( PhysicalPageID PageID, DXGI_FORMAT Format )
    {
        if( TRACE_SPEW )
        {
            SpewThreadID();
            DebugSpew( "Filling page %I64d in format %s\n", PageID, GetFormatName( Format ) );
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: Trace::MapPage
    // Desc: Trace event for when a virtual page is mapped to a physical page.
    //--------------------------------------------------------------------------------------
    VOID Trace::MapPage( VirtualPageID VPageID, PhysicalPageID PageID )
    {
        if( TRACE_SPEW )
        {
            SpewThreadID();
            DebugSpew( "Mapping page %I64d to resource ID %I64u, slice %I64u, mip %I64u, X %I64u, Y %I64u\n", PageID, VPageID.ResourceID, VPageID.ArraySlice, VPageID.MipLevel, VPageID.PageX, VPageID.PageY );
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: Trace::QueueMapUpdate
    // Desc: Trace event for when a request is queued to map a virtual page to a physical
    //       page (the GPU index map update operation).
    //--------------------------------------------------------------------------------------
    VOID Trace::QueueMapPageUpdate( VirtualPageID VPageID, PhysicalPageID PageID )
    {
        if( TRACE_SPEW )
        {
            SpewThreadID();
            DebugSpew( "Queueing request to map page %I64d to resource ID %I64u, slice %I64u, mip %I64u, X %I64u, Y %I64u\n", PageID, VPageID.ResourceID, VPageID.ArraySlice, VPageID.MipLevel, VPageID.PageX, VPageID.PageY );
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: Trace::RetireMapPageUpdate
    // Desc: Trace event for when a queued virtual to physical map request is executed.
    //--------------------------------------------------------------------------------------
    VOID Trace::RetireMapPageUpdate( VirtualPageID VPageID, PhysicalPageID PageID )
    {
        if( TRACE_SPEW )
        {
            SpewThreadID();
            DebugSpew( "Retiring request to map page %I64d to resource ID %I64u, slice %I64u, mip %I64u, X %I64u, Y %I64u\n", PageID, VPageID.ResourceID, VPageID.ArraySlice, VPageID.MipLevel, VPageID.PageX, VPageID.PageY );
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: Trace::UpdatePageBorder
    // Desc: Trace event for when a page border texel update request is queued.
    //--------------------------------------------------------------------------------------
    VOID Trace::UpdatePageBorder( PhysicalPageID CenterPage, PhysicalPageID BorderPage, PageNeighbors BorderLocation )
    {
        if( TRACE_SPEW > 1 )
        {
            if( BorderPage != INVALID_PHYSICAL_PAGE_ID || TRACE_SPEW > 2 )
            {
                SpewThreadID();
                DebugSpew( "Setting %s border of page %I64d with contents from page %I64d\n", g_strNeighborNames[BorderLocation], CenterPage, BorderPage );
            }
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: Trace::AddPageToPool
    // Desc: Trace event for when a physical page is added to a typed page pool.
    //--------------------------------------------------------------------------------------
    VOID Trace::AddPageToPool( PhysicalPageID PageID, INT PoolIndex, DXGI_FORMAT PoolFormat )
    {
        if( TRACE_SPEW )
        {
            SpewThreadID();
            DebugSpew( "Adding page %I64d to %s pool at index %d\n", PageID, GetFormatName( PoolFormat ), PoolIndex );
        }
    }

    //--------------------------------------------------------------------------------------
    // Name: Trace::RemovePageFromPool
    // Desc: Trace event for when a physical page is removed from a typed page pool.
    //--------------------------------------------------------------------------------------
    VOID Trace::RemovePageFromPool( PhysicalPageID PageID, INT PoolIndex, DXGI_FORMAT PoolFormat )
    {
        if( TRACE_SPEW )
        {
            SpewThreadID();
            DebugSpew( "Removing page %I64d from %s pool at index %d\n", PageID, GetFormatName( PoolFormat ), PoolIndex );
        }
    }
}
