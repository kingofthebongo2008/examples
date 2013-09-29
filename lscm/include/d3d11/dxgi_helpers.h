#ifndef __dxgi_helpers_h__
#define __dxgi_helpers_h__

#include <cstdint>

#include <DXGI.h>
#include <DXGI1_2.h>


#include <d3d11/d3d11_pointers.h>
#include <d3d11/d3d11_error.h>

namespace dxgi
{
    //return the back buffer
    inline d3d11::itexture2d_ptr get_buffer(IDXGISwapChain* swap_chain)
    {
        using namespace os::windows;
        d3d11::itexture2d_ptr	result;
        throw_if_failed< d3d11::exception> ( swap_chain->GetBuffer(0, __uuidof( ID3D11Texture2D ), (void**) &result ) );
        return result;
    }

    inline DXGI_SWAP_CHAIN_DESC get_desc(IDXGISwapChain* swap_chain)
    {
        DXGI_SWAP_CHAIN_DESC desc = {};
        
        using namespace os::windows;
		throw_if_failed<d3d11::exception>( swap_chain->GetDesc(&desc) );
        return desc;
    }

	
	inline DXGI_FORMAT format_2_srgb_format( DXGI_FORMAT format)
	{
		switch (format)
		{
			case DXGI_FORMAT_R8G8B8A8_TYPELESS:
			case DXGI_FORMAT_BC1_TYPELESS:
			case DXGI_FORMAT_BC2_TYPELESS:
			case DXGI_FORMAT_BC3_TYPELESS:
			case DXGI_FORMAT_BC7_TYPELESS:
				{
					return static_cast<DXGI_FORMAT> ( static_cast<uint32_t> ( format ) + 2 );
				}

			default:
					return format;
		}
	}
}


#endif

