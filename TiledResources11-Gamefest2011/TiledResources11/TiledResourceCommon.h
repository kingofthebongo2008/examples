//--------------------------------------------------------------------------------------
// TiledResourceCommon.h
//
// Typedefs, structs, and functions that are the core of the software emulation of tiled
// resources.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#ifdef _XBOX

#include <xtl.h>
#include <xgraphics.h>

#else

#include <windows.h>
#include <d3d11.h>
#include <d3d9.h>

#endif

#include <xnamath.h>
#include <assert.h>

#include <hash_map>
#include <hash_set>
#include <vector>
#include <deque>
#include <stack>
#include <algorithm>

#pragma warning( disable: 4201 ) // nameless union/struct
#pragma warning( disable: 4127 ) // conditional expression is constant
#pragma warning( disable: 4189 ) // unused variable

//----------------------------------------------------------------------------------------
// Debug functions
//----------------------------------------------------------------------------------------

#ifdef _DEBUG
#undef PARAMETER_CHECK
#define PARAMETER_CHECK 1
#else
#undef PARAMETER_CHECK
#define PARAMETER_CHECK 0
#endif

#ifndef ASSERT
#define ASSERT(x) assert(x)
#endif

#define RIP ASSERT(FALSE)

#define NOTIMPL ASSERT(!"Not implemented")

namespace TiledRuntime
{
    //--------------------------------------------------------------------------------------
    // Name: DebugSpewV()
    // Desc: Internal helper function
    //--------------------------------------------------------------------------------------
    static inline VOID DebugSpewV( const CHAR* strFormat, const va_list pArgList )
    {
        CHAR str[2048];
        // Use the secure CRT to avoid buffer overruns. Specify a count of
        // _TRUNCATE so that too long strings will be silently truncated
        // rather than triggering an error.
        _vsnprintf_s( str, _TRUNCATE, strFormat, pArgList );
        OutputDebugStringA( str );
    }


    //--------------------------------------------------------------------------------------
    // Name: DebugSpew()
    // Desc: Prints formatted debug spew
    //--------------------------------------------------------------------------------------
#ifdef  _Printf_format_string_  // VC++ 2008 and later support this annotation
    inline VOID CDECL DebugSpew( _In_z_ _Printf_format_string_ const CHAR* strFormat, ... )
#else
    inline VOID CDECL DebugSpew( const CHAR* strFormat, ... )
#endif
    {
        va_list pArgList;
        va_start( pArgList, strFormat );
        DebugSpewV( strFormat, pArgList );
        va_end( pArgList );
    }
}

//----------------------------------------------------------------------------------------
// Tiled runtime declarations, enums and functions
//----------------------------------------------------------------------------------------

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(Resource) { if( (Resource) != NULL ) { (Resource)->Release(); (Resource) = NULL; } }
#endif

namespace TiledRuntime
{
    struct PhysicalPageEntry;
    class PageRenderer;
    class PhysicalPageManager;
    class TypedPagePool;
    class TiledResourceBase;

    //--------------------------------------------------------------------------------------
    // Name: PageNeighbors
    // Desc: An enum that describes 2D neighbor relations to a center page, including the
    //       corners.  The enum is structured so that the opposite neighbor is always the
    //       opposite least significant bit.
    //--------------------------------------------------------------------------------------
    enum PageNeighbors
    {
        PN_TOP = 0,
        PN_BOTTOM,
        PN_LEFT,
        PN_RIGHT,
        PN_TOPLEFT,
        PN_BOTTOMRIGHT,
        PN_TOPRIGHT,
        PN_BOTTOMLEFT,
        PN_COUNT
    };

    //--------------------------------------------------------------------------------------
    // Name: GetOppositeNeighbor
    // Desc: Returns the opposite to the given neighbor, by inverting the least significant
    //       bit.
    //--------------------------------------------------------------------------------------
    inline PageNeighbors GetOppositeNeighbor( const PageNeighbors Neighbor )
    {
        return (PageNeighbors)( Neighbor ^ 0x1 );
    }

    //--------------------------------------------------------------------------------------
    // Name: PhysicalPageID
    // Desc: PhysicalPageID represents a 64-bit address to a piece of physical memory within
    //       the tiled resource system.
    //--------------------------------------------------------------------------------------
    typedef UINT64 PhysicalPageID;

    //--------------------------------------------------------------------------------------
    // Name: VirtualPageID
    // Desc: VirtualPageID represents a 64-bit address to a location within the virtual
    //       address space of a tiled resource.  This struct breaks down the 64 bits into a
    //       variety of fields that allow the address to be easily decomposed into a location
    //       within a resource.
    //--------------------------------------------------------------------------------------
    typedef union _VirtualPageID
    {
        struct
        {
            // The Valid bit distinguishes a virtual address from NULL virtual address:
            UINT64 Valid : 1;

            // The ResourceID field supports up to 32768 individual resources:
            UINT64 ResourceID : 15;

            // X location of page within a mip level:
            UINT64 PageX : 16;

            // Y location of page within a mip level:
            UINT64 PageY : 16;

            // The MipLevel field supports up to 64 mip levels (probably excessive):
            UINT64 MipLevel : 6;

            // The ArraySlice field supports up to 1024 array slices:
            UINT64 ArraySlice: 10;
        };

        // The fields together as a 64-bit uint value:
        UINT64 VirtualAddress;

        // Comparison operator overloads:
        bool operator==( const _VirtualPageID& RHS ) const { return VirtualAddress == RHS.VirtualAddress; }
        bool operator!=( const _VirtualPageID& RHS ) const { return VirtualAddress != RHS.VirtualAddress; }

        bool operator<( const _VirtualPageID& RHS ) const { return VirtualAddress < RHS.VirtualAddress; }

        // size_t conversion operator to support VirtualPageID used as a sort predicate:
        operator size_t() const { return (size_t)VirtualAddress; }

    } VirtualPageID;

    //--------------------------------------------------------------------------------------
    // Name: INVALID_PHYSICAL_PAGE_ID
    // Desc: Represents an invalid physical address.
    //--------------------------------------------------------------------------------------
    static const PhysicalPageID INVALID_PHYSICAL_PAGE_ID = (PhysicalPageID)-1;

    //--------------------------------------------------------------------------------------
    // Name: CreateInvalidVirtualPageID
    // Desc: Creates a NULL virtual address.
    //--------------------------------------------------------------------------------------
    inline VirtualPageID CreateInvalidVirtualPageID()
    {
        VirtualPageID VPageID;
        VPageID.VirtualAddress = 0;
        return VPageID;
    }

    //--------------------------------------------------------------------------------------
    // Name: INVALID_VIRTUAL_PAGE_ID
    // Desc: Represents a NULL virtual address.
    //--------------------------------------------------------------------------------------
    static const VirtualPageID INVALID_VIRTUAL_PAGE_ID = CreateInvalidVirtualPageID();

    //--------------------------------------------------------------------------------------
    // Name: PageNeighborhood
    // Desc: Represents a physical page and its 8 immediate neighbors in 2D space.
    //--------------------------------------------------------------------------------------
    struct PageNeighborhood
    {
        PhysicalPageID m_CenterPage;
        PhysicalPageID m_Neighbors[PN_COUNT];
    };

    // A single physical page's size in bytes:
    static const UINT PAGE_SIZE_BYTES = 64 * 1024;

    // The dimensions of the page atlas used in the typed page pool:
    static const UINT ATLAS_ROWS = 6;
    static const UINT ATLAS_COLUMNS = 6;
    static const UINT ATLAS_PAGES_PER_SLICE = ATLAS_ROWS * ATLAS_COLUMNS;

    // An invalid page pool result:
    static const INT INVALID_PAGE_POOL_INDEX = -1;

    // Set this variable to TRUE to automatically mirror page edges to the borders when neighboring pages are not present.
    // This has the effect of making bilinear fetches always return valid texels:
    static const BOOL FETCH_ONLY_VALID_TEXELS_OPTION = TRUE;

    // The max amount of simultaneous shader resources allowed in PS and VS stages:
    static const UINT MAX_PS_TILED_SHADER_RESOURCES = 4;
    static const UINT MAX_VS_TILED_SHADER_RESOURCES = 4;

    // Constants and functions for computing the base 2 logarithm of float and unsigned ints:
    static const FLOAT INVLOG2 = 1.0f / logf( 2.0f );

    inline FLOAT log2f( FLOAT Value )
    {
        return logf( Value ) * INVLOG2;
    }

    inline FLOAT log2f( UINT Value )
    {
        return log2f( (FLOAT)Value );
    }
}

#ifdef _XBOX
#include "TiledResourceXbox360.h"
#else
#include "TiledResourceD3D11.h"
#endif
