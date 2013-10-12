#ifndef __d3d11_HELPERS_H__
#define __d3d11_HELPERS_H__

#include <cstdint>

#include <d3d11/dxgi_helpers.h>
#include <d3d11/d3d11_pointers.h>
#include <d3d11/d3d11_error.h>

namespace d3d11
{
    inline iblendstate_ptr          create_blend_state(ID3D11Device* device,   const D3D11_BLEND_DESC* description )
    {
        d3d11::iblendstate_ptr result;
        os::windows::throw_if_failed< d3d11::create_blend_state_exception> ( device->CreateBlendState(description, &result ) );
        return result;
    }
    
    inline idepthstencilstate_ptr    create_depth_stencil_state(ID3D11Device* device,  const D3D11_DEPTH_STENCIL_DESC* description )
    {
        idepthstencilstate_ptr result;
        os::windows::throw_if_failed< d3d11::create_depth_stencil_state_exception> ( device->CreateDepthStencilState( description, &result ) );
        return result;
    }

    inline idepthstencilview_ptr    create_depth_stencil_view(ID3D11Device* device, ID3D11Resource* resource,  const D3D11_DEPTH_STENCIL_VIEW_DESC* description )
    {
        idepthstencilview_ptr result;
        os::windows::throw_if_failed< d3d11::create_depth_stencil_view_exception> ( device->CreateDepthStencilView(resource, description, &result ) );
        return result;
    }

    inline irasterizerstate_ptr      create_raster_state(ID3D11Device* device, const D3D11_RASTERIZER_DESC* description )
    {
        irasterizerstate_ptr result;
        os::windows::throw_if_failed< d3d11::create_rasterizer_state_exception> (device->CreateRasterizerState(description, &result ) );
        return result;
    }
    
    inline id3d11rendertargetview_ptr    create_render_target_view(ID3D11Device* device, ID3D11Resource* resource, const D3D11_RENDER_TARGET_VIEW_DESC* description )
    {
        id3d11rendertargetview_ptr result;
        os::windows::throw_if_failed< d3d11::create_render_target_view_exception> (device->CreateRenderTargetView(resource, description, &result ) );
        return result;
    }

    inline id3d11rendertargetview_ptr    create_render_target_view(ID3D11Device* device, ID3D11Resource* resource )
    {
        return create_render_target_view(device, resource, nullptr);
    }

    inline isamplerstate_ptr            create_sampler_state(ID3D11Device* device, const D3D11_SAMPLER_DESC* description )
    {
        isamplerstate_ptr result;
        os::windows::throw_if_failed< d3d11::create_sampler_state_exception> (device->CreateSamplerState(description, &result ) );
        return result;
    }

    inline ishaderresourceview_ptr create_shader_resource_view(ID3D11Device* device, ID3D11Resource* resource , const D3D11_SHADER_RESOURCE_VIEW_DESC* description)
    {
        ishaderresourceview_ptr result;
        os::windows::throw_if_failed< d3d11::create_shader_resource_view_exception> ( device->CreateShaderResourceView( resource, description, &result )  );
        return result;
    }

    inline ishaderresourceview_ptr create_shader_resource_view(ID3D11Device* device, ID3D11Resource* resource)
    {
        return create_shader_resource_view(device, resource, nullptr);
    }

    inline itexture2d_ptr           create_texture_2d(ID3D11Device* device,  const D3D11_TEXTURE2D_DESC* description, const D3D11_SUBRESOURCE_DATA* initial_data )
    {
        itexture2d_ptr result;
        os::windows::throw_if_failed< d3d11::create_texture2d_exception> (device->CreateTexture2D(description, initial_data, &result ) );
        return result;
    }

    inline itexture2d_ptr           create_texture_2d(ID3D11Device* device,  const D3D11_TEXTURE2D_DESC* description)
    {
        return create_texture_2d(device, description, nullptr);
    }

    namespace
    {
        inline static d3d11::ibuffer_ptr create_vertex_buffer(ID3D11Device* device, const void* initial_data, size_t size, D3D11_USAGE d3d11_bind_flags, uint32_t  cpu_access_flags )
        {
            D3D11_BUFFER_DESC desc = {};
            d3d11::ibuffer_ptr result;
            desc.ByteWidth = static_cast<UINT> (size);
            desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
            desc.CPUAccessFlags = cpu_access_flags;
            desc.Usage = d3d11_bind_flags;
            D3D11_SUBRESOURCE_DATA initial_data_dx = { initial_data, 0, 0};

            os::windows::throw_if_failed<d3d11::create_buffer_exception> (device->CreateBuffer(&desc, &initial_data_dx, &result));
            return result;
        }

        inline static d3d11::ibuffer_ptr create_index_buffer(ID3D11Device* device, const void* initial_data, size_t size, D3D11_USAGE d3d11_bind_flags, uint32_t cpu_access_flags  )
        {
            D3D11_BUFFER_DESC desc = {};
            d3d11::ibuffer_ptr result;
            desc.ByteWidth = static_cast<UINT> (size);
            desc.BindFlags = D3D11_BIND_INDEX_BUFFER ;
            desc.CPUAccessFlags = cpu_access_flags;
            desc.Usage = d3d11_bind_flags;
            D3D11_SUBRESOURCE_DATA initial_data_dx = { initial_data, 0, 0};
            os::windows::throw_if_failed<d3d11::create_buffer_exception> (device->CreateBuffer(&desc, &initial_data_dx, &result));
            return result;
        }
    }

    inline d3d11::ibuffer_ptr create_constant_buffer(ID3D11Device* device, size_t size)
    {
        D3D11_BUFFER_DESC desc = {};
        d3d11::ibuffer_ptr result;

        desc.ByteWidth = std::max< uint32_t> ( 16, static_cast<uint32_t> ( size ) );
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        os::windows::throw_if_failed<d3d11::create_buffer_exception> (device->CreateBuffer(&desc, nullptr, &result ));
        return result;
    }

    inline d3d11::ibuffer_ptr create_default_vertex_buffer(ID3D11Device* device, const void* initial_data, size_t size )
    {
        return create_vertex_buffer(device, initial_data, size, D3D11_USAGE_DEFAULT, 0);
    }

    inline d3d11::ibuffer_ptr create_default_index_buffer(ID3D11Device* device, const void* initial_data, size_t size )
    {
        return create_index_buffer(device, initial_data, size, D3D11_USAGE_DEFAULT, 0);
    }

    inline d3d11::ibuffer_ptr create_dynamic_vertex_buffer(ID3D11Device* device, const void* initial_data, size_t size )
    {
        return create_vertex_buffer(device, initial_data, size, D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE );
    }

    inline d3d11::ibuffer_ptr create_dynamic_index_buffer(ID3D11Device* device, const void* initial_data, size_t size )
    {
        return create_index_buffer(device, initial_data, size, D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE );
    }

    inline d3d11::ibuffer_ptr create_immutable_vertex_buffer(ID3D11Device* device, const void* initial_data, size_t size )
    {
        return create_vertex_buffer(device, initial_data, size, D3D11_USAGE_IMMUTABLE, 0 );
    }

    inline d3d11::ibuffer_ptr create_immutable_index_buffer(ID3D11Device* device, const void* initial_data, size_t size )
    {
        return create_index_buffer(device, initial_data, size, D3D11_USAGE_IMMUTABLE, 0);
    }

    inline d3d11::ibuffer_ptr create_stream_out_vertex_buffer(ID3D11Device* device, const void* initial_data, size_t size )
    {
        D3D11_BUFFER_DESC desc = {};
        d3d11::ibuffer_ptr result;
        desc.ByteWidth = static_cast<UINT> (size);
        desc.BindFlags = D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_STREAM_OUTPUT ;
        desc.CPUAccessFlags = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;
        D3D11_SUBRESOURCE_DATA initial_data_dx = { initial_data, 0, 0};
        os::windows::throw_if_failed<d3d11::create_buffer_exception> (device->CreateBuffer(&desc, &initial_data_dx, &result));
        return result;
    }

    inline  d3d11::iunordered_access_view_ptr create_unordered_access_view_structured( ID3D11Device* device, ID3D11Texture2D* const texture, DXGI_FORMAT format )
    {
        D3D11_UNORDERED_ACCESS_VIEW_DESC desc = {};

        d3d11::iunordered_access_view_ptr r;
        desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
        desc.Texture2D.MipSlice = 0;
        desc.Format = format;
        os::windows::throw_if_failed<d3d11::create_unordered_access_view_exception> (device->CreateUnorderedAccessView( texture, &desc, &r));
        return r;
    }

    struct d3d11_buffer_scope_lock
    {
        d3d11_buffer_scope_lock( ID3D11DeviceContext* context, ID3D11Buffer* buffer) : m_context(context), m_buffer(buffer)
        {
            os::windows::throw_if_failed<d3d11::exception>(context->Map( buffer, 0,  D3D11_MAP_WRITE_DISCARD, 0, &m_mapped_resource) ) ;
        }

        ~d3d11_buffer_scope_lock()
        {
            m_context->Unmap(m_buffer, 0);
        }

        D3D11_MAPPED_SUBRESOURCE    m_mapped_resource;
        ID3D11DeviceContext*        m_context;
        ID3D11Buffer*               m_buffer;
    };

    struct resource_slot
    {
        resource_slot(uint32_t value ) : m_value(value)
        {

        }

        operator uint32_t () const
        {

            return m_value;
        }

        uint32_t m_value;
    };

    struct view_count
    {
        view_count(uint32_t value ) : m_value(value)
        {

        }

        operator uint32_t () const
        {

            return m_value;
        }

        uint32_t m_value;
    };

    
    //----------------------------------------------------------------------------------------------------------
    inline void gs_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, view_count num_views, ID3D11ShaderResourceView * const * shader_resource_view)
    {
        device_context->GSSetShaderResources( slot, num_views, shader_resource_view);
    }

    inline void gs_set_shader_resources(ID3D11DeviceContext* device_context, view_count num_views, ID3D11ShaderResourceView * const * shader_resource_view)
    {
        gs_set_shader_resources( device_context, 0, num_views, shader_resource_view );
    }

    inline void gs_set_shader_resources(ID3D11DeviceContext* device_context, ID3D11ShaderResourceView * const shader_resource_view [] )
    {
        gs_set_shader_resources( device_context, 0, sizeof( shader_resource_view ) / sizeof (shader_resource_view[0] ), &shader_resource_view[0] );
    }

    inline void gs_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, ID3D11ShaderResourceView * const shader_resource_view [] )
    {
        gs_set_shader_resources( device_context, slot, sizeof( shader_resource_view ) / sizeof (shader_resource_view[0] ), &shader_resource_view[0] );
    }

    template <typename t> inline void gs_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, t begin, t end )
    {
        //array and vector will work here, not general
        gs_set_shader_resources( device_context, slot, static_cast< uint32_t > ( end - begin ), & ( static_cast< ID3D11ShaderResourceView * const> ( *begin ) ) );
    }

    template <typename t> inline void gs_set_shader_resources(ID3D11DeviceContext* device_context, t begin, t end )
    {
        gs_set_shader_resources( device_context, 0, begin, end );
    }

    inline void gs_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, ID3D11ShaderResourceView * const resource_0, ID3D11ShaderResourceView * const resource_1, ID3D11ShaderResourceView * const resource_2 )
    {
        ID3D11ShaderResourceView * const resources[3] = { resource_0 , resource_1, resource_2};
        gs_set_shader_resources( device_context, slot,  resources);
    }

    inline void gs_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, ID3D11ShaderResourceView * const resource_0, ID3D11ShaderResourceView * const resource_1)
    {
        ID3D11ShaderResourceView * const resources[2] = { resource_0 , resource_1};
        gs_set_shader_resources( device_context, slot,  resources);
    }

    inline void gs_set_shader_resources(ID3D11DeviceContext* device_context, ID3D11ShaderResourceView * const resource_0, ID3D11ShaderResourceView * const resource_1, ID3D11ShaderResourceView * const resource_2 )
    {
        gs_set_shader_resources(device_context, 0, resource_0, resource_1, resource_2 );
    }

    inline void gs_set_shader_resources(ID3D11DeviceContext* device_context, ID3D11ShaderResourceView * const resource_0, ID3D11ShaderResourceView * const resource_1)
    {
        gs_set_shader_resources(device_context, 0, resource_0, resource_1 );
    }

    inline void gs_set_shader_resource(ID3D11DeviceContext* device_context, ID3D11ShaderResourceView * const shader_resource_view)
    {
        gs_set_shader_resources( device_context, 0, 1, &shader_resource_view );
    }
    
    inline void gs_set_shader_resource(ID3D11DeviceContext* device_context, resource_slot slot, ID3D11ShaderResourceView * const shader_resource_view)
    {
        gs_set_shader_resources( device_context, slot, 1, &shader_resource_view );
    }

    //----------------------------------------------------------------------------------------------------------
    inline void gs_set_shader(ID3D11DeviceContext* device_context, ID3D11GeometryShader * vertex_shader )
    {
        device_context->GSSetShader( vertex_shader, nullptr, 0) ;
    }

    //----------------------------------------------------------------------------------------------------------
    inline void cs_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, view_count num_views, ID3D11ShaderResourceView * const * shader_resource_view)
    {
        device_context->CSSetShaderResources( slot, num_views, shader_resource_view);
    }

    inline void cs_set_shader_resources(ID3D11DeviceContext* device_context, view_count num_views, ID3D11ShaderResourceView * const * shader_resource_view)
    {
        cs_set_shader_resources( device_context, 0, num_views, shader_resource_view );
    }

    inline void cs_set_shader_resources(ID3D11DeviceContext* device_context, ID3D11ShaderResourceView * const shader_resource_view [] )
    {
        cs_set_shader_resources( device_context, 0, sizeof( shader_resource_view ) / sizeof (shader_resource_view[0] ), &shader_resource_view[0] );
    }

    inline void cs_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, ID3D11ShaderResourceView * const shader_resource_view [] )
    {
        cs_set_shader_resources( device_context, slot, sizeof( shader_resource_view ) / sizeof (shader_resource_view[0] ), &shader_resource_view[0] );
    }

    template <typename t> inline void cs_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, t begin, t end )
    {
        //array and vector will work here, not general
        cs_set_shader_resources( device_context, slot, static_cast< uint32_t > ( end - begin ), & ( static_cast< ID3D11ShaderResourceView * const> ( *begin ) ) );
    }

    template <typename t> inline void cs_set_shader_resources(ID3D11DeviceContext* device_context, t begin, t end )
    {
        cs_set_shader_resources( device_context, 0, begin, end );
    }

    inline void cs_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, ID3D11ShaderResourceView * const resource_0, ID3D11ShaderResourceView * const resource_1, ID3D11ShaderResourceView * const resource_2 )
    {
        ID3D11ShaderResourceView * const resources[3] = { resource_0 , resource_1, resource_2};
        cs_set_shader_resources( device_context, slot,  resources);
    }

    inline void cs_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, ID3D11ShaderResourceView * const resource_0, ID3D11ShaderResourceView * const resource_1)
    {
        ID3D11ShaderResourceView * const resources[2] = { resource_0 , resource_1};
        cs_set_shader_resources( device_context, slot,  resources);
    }

    inline void cs_set_shader_resources(ID3D11DeviceContext* device_context, ID3D11ShaderResourceView * const resource_0, ID3D11ShaderResourceView * const resource_1, ID3D11ShaderResourceView * const resource_2 )
    {
        cs_set_shader_resources(device_context, 0, resource_0, resource_1, resource_2 );
    }

    inline void cs_set_shader_resources(ID3D11DeviceContext* device_context, ID3D11ShaderResourceView * const resource_0, ID3D11ShaderResourceView * const resource_1)
    {
        cs_set_shader_resources(device_context, 0, resource_0, resource_1 );
    }

    inline void cs_set_shader_resource(ID3D11DeviceContext* device_context, ID3D11ShaderResourceView * const shader_resource_view)
    {
        cs_set_shader_resources( device_context, 0, 1, &shader_resource_view );
    }
    
    inline void cs_set_shader_resource(ID3D11DeviceContext* device_context, resource_slot slot, ID3D11ShaderResourceView * const shader_resource_view)
    {
        cs_set_shader_resources( device_context, slot, 1, &shader_resource_view );
    }
    //----------------------------------------------------------------------------------------------------------
    inline void cs_set_shader(ID3D11DeviceContext* device_context, ID3D11ComputeShader * const compute_shader )
    {
        device_context->CSSetShader( const_cast< ID3D11ComputeShader* > (compute_shader), nullptr, 0) ;
    }

    inline void cs_set_shader_unordered_access_view(ID3D11DeviceContext* device_context, ID3D11UnorderedAccessView * const shader_resource_view)
    {
        device_context->CSSetUnorderedAccessViews( 0, 1, const_cast< ID3D11UnorderedAccessView ** > ( &shader_resource_view ), 0 );
    }

    inline void cs_set_sampler_state(ID3D11DeviceContext* device_context, ID3D11SamplerState* const sampler)
    {
        device_context->CSSetSamplers( 0, 1, const_cast<ID3D11SamplerState** > (&sampler) );
    }
    //----------------------------------------------------------------------------------------------------------

    inline void vs_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, view_count num_views, ID3D11ShaderResourceView * const * shader_resource_view)
    {
        device_context->VSSetShaderResources( slot, num_views, shader_resource_view);
    }

    inline void vs_set_shader_resources(ID3D11DeviceContext* device_context, view_count num_views, ID3D11ShaderResourceView * const * shader_resource_view)
    {
        vs_set_shader_resources( device_context, 0, num_views, shader_resource_view );
    }

    inline void vs_set_shader_resources(ID3D11DeviceContext* device_context, ID3D11ShaderResourceView * const shader_resource_view [] )
    {
        vs_set_shader_resources( device_context, 0, sizeof( shader_resource_view ) / sizeof (shader_resource_view[0] ), &shader_resource_view[0] );
    }

    inline void vs_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, ID3D11ShaderResourceView * const shader_resource_view [] )
    {
        vs_set_shader_resources( device_context, slot, sizeof( shader_resource_view ) / sizeof (shader_resource_view[0] ), &shader_resource_view[0] );
    }

    template <typename t> inline void vs_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, t begin, t end )
    {
        //array and vector will work here, not general
        vs_set_shader_resources( device_context, slot, static_cast< uint32_t > ( end - begin ), & ( static_cast< ID3D11ShaderResourceView * const> ( *begin ) ) );
    }

    template <typename t> inline void vs_set_shader_resources(ID3D11DeviceContext* device_context, t begin, t end )
    {
        vs_set_shader_resources( device_context, 0, begin, end );
    }

    inline void vs_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, ID3D11ShaderResourceView * const resource_0, ID3D11ShaderResourceView * const resource_1, ID3D11ShaderResourceView * const resource_2 )
    {
        ID3D11ShaderResourceView * const resources[3] = { resource_0 , resource_1, resource_2};
        vs_set_shader_resources( device_context, slot,  resources);
    }

    inline void vs_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, ID3D11ShaderResourceView * const resource_0, ID3D11ShaderResourceView * const resource_1)
    {
        ID3D11ShaderResourceView * const resources[2] = { resource_0 , resource_1};
        vs_set_shader_resources( device_context, slot,  resources);
    }

    inline void vs_set_shader_resources(ID3D11DeviceContext* device_context, ID3D11ShaderResourceView * const resource_0, ID3D11ShaderResourceView * const resource_1, ID3D11ShaderResourceView * const resource_2 )
    {
        vs_set_shader_resources(device_context, 0, resource_0, resource_1, resource_2 );
    }

    inline void vs_set_shader_resources(ID3D11DeviceContext* device_context, ID3D11ShaderResourceView * const resource_0, ID3D11ShaderResourceView * const resource_1)
    {
        vs_set_shader_resources(device_context, 0, resource_0, resource_1 );
    }

    inline void vs_set_shader_resource(ID3D11DeviceContext* device_context, ID3D11ShaderResourceView * const shader_resource_view)
    {
        vs_set_shader_resources( device_context, 0, 1, &shader_resource_view );
    }
    
    inline void vs_set_shader_resource(ID3D11DeviceContext* device_context, resource_slot slot, ID3D11ShaderResourceView * const shader_resource_view)
    {
        vs_set_shader_resources( device_context, slot, 1, &shader_resource_view );
    }

    inline void vs_set_shader(ID3D11DeviceContext* device_context, ID3D11VertexShader * const vertex_shader )
    {
        device_context->VSSetShader( vertex_shader, nullptr, 0) ;
    }

    //----------------------------------------------------------------------------------------------------------
    inline void ps_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, view_count num_views, ID3D11ShaderResourceView * const * shader_resource_view)
    {
        device_context->PSSetShaderResources( slot, num_views, shader_resource_view);
    }

    inline void ps_set_shader_resources(ID3D11DeviceContext* device_context, view_count num_views, ID3D11ShaderResourceView * const * shader_resource_view)
    {
        ps_set_shader_resources( device_context, 0, num_views, shader_resource_view );
    }

    inline void ps_set_shader_resources(ID3D11DeviceContext* device_context, ID3D11ShaderResourceView * const shader_resource_view [] )
    {
        ps_set_shader_resources( device_context, 0, sizeof( shader_resource_view ) / sizeof (shader_resource_view[0] ), &shader_resource_view[0] );
    }

    inline void ps_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, ID3D11ShaderResourceView * const shader_resource_view [] )
    {
        ps_set_shader_resources( device_context, slot, sizeof( shader_resource_view ) / sizeof (shader_resource_view[0] ), &shader_resource_view[0] );
    }

    template <typename t> inline void ps_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, t begin, t end )
    {
        //array and vector will work here, not general
        ps_set_shader_resources( device_context, slot, static_cast< uint32_t > ( end - begin ), & ( static_cast< ID3D11ShaderResourceView * const> ( *begin ) ) );
    }

    template <typename t> inline void ps_set_shader_resources(ID3D11DeviceContext* device_context, t begin, t end )
    {
        ps_set_shader_resources( device_context, 0, begin, end );
    }

    inline void ps_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, ID3D11ShaderResourceView * const resource_0, ID3D11ShaderResourceView * const resource_1, ID3D11ShaderResourceView * const resource_2 )
    {
        ID3D11ShaderResourceView * const resources[3] = { resource_0 , resource_1, resource_2};
        ps_set_shader_resources( device_context, slot,  resources);
    }

    inline void ps_set_shader_resources(ID3D11DeviceContext* device_context, resource_slot slot, ID3D11ShaderResourceView * const resource_0, ID3D11ShaderResourceView * const resource_1)
    {
        ID3D11ShaderResourceView * const resources[2] = { resource_0 , resource_1};
        ps_set_shader_resources( device_context, slot,  resources);
    }

    inline void ps_set_shader_resources(ID3D11DeviceContext* device_context, ID3D11ShaderResourceView * const resource_0, ID3D11ShaderResourceView * const resource_1, ID3D11ShaderResourceView * const resource_2 )
    {
        ps_set_shader_resources(device_context, 0, resource_0, resource_1, resource_2 );
    }

    inline void ps_set_shader_resources(ID3D11DeviceContext* device_context, ID3D11ShaderResourceView * const resource_0, ID3D11ShaderResourceView * const resource_1)
    {
        ps_set_shader_resources(device_context, 0, resource_0, resource_1 );
    }

    inline void ps_set_shader_resource(ID3D11DeviceContext* device_context, ID3D11ShaderResourceView * const shader_resource_view)
    {
        ps_set_shader_resources( device_context, 0, 1, &shader_resource_view );
    }
    
    inline void ps_set_shader_resource(ID3D11DeviceContext* device_context, resource_slot slot, ID3D11ShaderResourceView * const shader_resource_view)
    {
        ps_set_shader_resources( device_context, slot, 1, &shader_resource_view );
    }
    //----------------------------------------------------------------------------------------------------------

    inline void ps_set_shader(ID3D11DeviceContext* device_context, ID3D11PixelShader * const pixel_shader )
    {
        device_context->PSSetShader( pixel_shader, nullptr, 0) ;
    }

    inline void ps_set_sampler_state(ID3D11DeviceContext* device_context, ID3D11SamplerState* const sampler)
    {
        device_context->PSSetSamplers(0, 1, &sampler );
    }

    inline void ps_set_sampler_state(ID3D11DeviceContext* device_context, ID3D11SamplerState* const sampler, uint32_t slot)
    {
        device_context->PSSetSamplers(slot, 1, &sampler);
    }

    inline void ia_set_input_layout(ID3D11DeviceContext* device_context, ID3D11InputLayout * const layout)
    {
        device_context->IASetInputLayout( layout );
    } 

    inline void ia_set_vertex_buffer( ID3D11DeviceContext* device_context, ID3D11Buffer* const buffer, uint32_t stride, uint32_t offset )
    {
        device_context->IASetVertexBuffers ( 0, 1, &buffer , &stride, &offset );
    }

    inline void ia_set_vertex_buffer( ID3D11DeviceContext* device_context, ID3D11Buffer* const buffer, uint32_t stride )
    {
        ia_set_vertex_buffer(device_context, buffer, stride, 0 );
    }

    inline void ia_set_index_buffer(ID3D11DeviceContext* device_context, ID3D11Buffer* const buffer)
    {
        device_context->IASetIndexBuffer( buffer, DXGI_FORMAT_R16_UINT, 0);
    }

    inline void ia_set_index_buffer(ID3D11DeviceContext* device_context, ID3D11Buffer* const buffer, DXGI_FORMAT format)
    {
        device_context->IASetIndexBuffer( buffer , format, 0);
    }

    inline void ia_set_primitive_topology(ID3D11DeviceContext* device_context, D3D11_PRIMITIVE_TOPOLOGY topology)
    {
        device_context->IASetPrimitiveTopology(topology);
    }

    inline void rs_set_state(ID3D11DeviceContext* device_context, ID3D11RasterizerState * const state)
    {
        device_context->RSSetState(state);
    }

    inline void rs_set_view_port(ID3D11DeviceContext* device_context, const D3D11_VIEWPORT* const view_port)
    {
        device_context->RSSetViewports(1, const_cast<const D3D11_VIEWPORT*> (view_port) );
    }

    inline void om_set_depth_state(ID3D11DeviceContext* device_context, ID3D11DepthStencilState * const state)
    {
        device_context->OMSetDepthStencilState( state, 0);
    }

    inline void om_set_blend_state(ID3D11DeviceContext* device_context, ID3D11BlendState * const state)
    {
        device_context->OMSetBlendState( state, nullptr, 0xFFFFFFFF);
    }

    inline void clear_render_target_view(ID3D11DeviceContext* device_context, ID3D11RenderTargetView* const view, math::float4 value)
    {
        //allocate on 16 byte boundary
        void* v = _alloca ( 4 * sizeof(float) );
        math::store4( v, value);
        device_context->ClearRenderTargetView( view, reinterpret_cast<float*> (v) );
    }

    inline void clear_depth_stencil_view(ID3D11DeviceContext* device_context, ID3D11DepthStencilView* const view, float depth, uint8_t stencil)
    {
        device_context->ClearDepthStencilView( view, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, depth, stencil );
    }

    inline void clear_depth_stencil_view(ID3D11DeviceContext* device_context, ID3D11DepthStencilView* const view)
    {
        device_context->ClearDepthStencilView(view, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0xFF);
    }

    inline void om_set_render_target( ID3D11DeviceContext* device_context, ID3D11RenderTargetView* const render_target_view, ID3D11DepthStencilView* const depth_view )
    {
        ID3D11RenderTargetView* const views[1] =
        {
            render_target_view
        };

        device_context->OMSetRenderTargets( 1, &views[0], depth_view );
    }

    inline void om_set_render_target( ID3D11DeviceContext* device_context, ID3D11RenderTargetView* const render_target_view )
    {
        ID3D11RenderTargetView* const views[1] =
        {
            render_target_view
        };

        device_context->OMSetRenderTargets(1, &views[0], nullptr );
    }
}



#endif

