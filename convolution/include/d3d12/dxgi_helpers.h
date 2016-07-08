#pragma once

#include <cstdint>

#include <dxgi1_4.h>

#include <d3d12/d3d12_pointers.h>
#include <d3d12/d3d12_error.h>

namespace dxgi
{
    inline d3d12::resource get_buffer(IDXGISwapChain* swap_chain, uint32_t buffer)
    {
        using namespace os::windows;
        d3d12::resource result;
        throw_if_failed< d3d12::exception> ( swap_chain->GetBuffer(buffer, __uuidof( ID3D12Resource ), (void**) &result ) );
        return std::move(result);
    }

    inline DXGI_SWAP_CHAIN_DESC get_desc(IDXGISwapChain* swap_chain)
    {
        DXGI_SWAP_CHAIN_DESC desc = {};
        
        using namespace os::windows;
		throw_if_failed<d3d12::exception>( swap_chain->GetDesc(&desc) );
        return desc;
    }
	
}




