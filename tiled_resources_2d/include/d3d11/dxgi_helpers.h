#pragma once

#include <cstdint>

#include <d3d11/d3d11_pointers.h>
#include <d3d11/d3d11_error.h>

namespace dxgi
{
    inline d3d11::resource get_buffer(IDXGISwapChain* swap_chain, uint32_t buffer)
    {
        using namespace os::windows;
        d3d11::resource result;
        throw_if_failed< d3d11::exception> ( swap_chain->GetBuffer(buffer, __uuidof( ID3D11Resource ), (void**) &result ) );
        return result;
    }

    inline d3d11::texture2d get_buffer_as_texture(IDXGISwapChain* swap_chain, uint32_t buffer)
    {
        using namespace os::windows;
        d3d11::texture2d result;
        throw_if_failed< d3d11::exception>(swap_chain->GetBuffer(buffer, __uuidof(ID3D11Texture2D), (void**)&result));
        return result;
    }

    inline DXGI_SWAP_CHAIN_DESC get_desc(IDXGISwapChain* swap_chain)
    {
        DXGI_SWAP_CHAIN_DESC desc = {};
        
        using namespace os::windows;
		throw_if_failed<d3d11::exception>( swap_chain->GetDesc(&desc) );
        return desc;
    }
	
}




