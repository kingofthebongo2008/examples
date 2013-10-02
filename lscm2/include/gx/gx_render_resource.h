 #ifndef __GX_RENDER_RESOURCE_H__
#define __GX_RENDER_RESOURCE_H__

#include <cstdint>
#include <limits>

#include <d3d11/d3d11_helpers.h>
#include <d3d11/dxgi_helpers.h>

namespace gx
{
    class render_target_resource
    {
        public:

        render_target_resource (    
                                    d3d11::itexture2d_ptr               resource,
                                    d3d11::id3d11rendertargetview_ptr   resource_rtv,
                                    d3d11::ishaderresourceview_ptr	    resource_srv
                                ) : m_resource(resource) , m_resource_rtv(resource_rtv), m_resource_srv(resource_srv)
        {

        }

        operator ID3D11Texture2D* ()
        {
            return m_resource.get();
        }

        operator const ID3D11Texture2D* () const
        {
            return m_resource.get();
        }

        operator ID3D11RenderTargetView* ()
        {
            return m_resource_rtv.get();
        }

        operator const ID3D11RenderTargetView* () const
        {
            return m_resource_rtv.get();
        }

        operator ID3D11ShaderResourceView* ()
        {
            return m_resource_srv.get();
        }

        operator const ID3D11ShaderResourceView* () const
        {
            return m_resource_srv.get();
        }

        d3d11::itexture2d_ptr               m_resource;
        d3d11::id3d11rendertargetview_ptr   m_resource_rtv;
        d3d11::ishaderresourceview_ptr      m_resource_srv;

    };

    class depth_resource
    {
        public:

        depth_resource (    
                                    d3d11::itexture2d_ptr           resource,
                                    d3d11::idepthstencilview_ptr    resource_dsv,
                                    d3d11::ishaderresourceview_ptr  resource_srv
                                ) : m_resource(resource) , m_resource_dsv(resource_dsv), m_resource_srv(resource_srv)
        {

        }

        operator ID3D11Texture2D* ()
        {
            return m_resource.get();
        }

        operator const ID3D11Texture2D* () const
        {
            return m_resource.get();
        }

        operator ID3D11DepthStencilView* ()
        {
            return m_resource_dsv.get();
        }

        operator const ID3D11DepthStencilView* () const
        {
            return m_resource_dsv.get();
        }

        operator ID3D11ShaderResourceView* ()
        {
            return m_resource_srv.get();
        }

        operator const ID3D11ShaderResourceView* () const
        {
            return m_resource_srv.get();
        }

        d3d11::itexture2d_ptr           m_resource;
        d3d11::idepthstencilview_ptr    m_resource_dsv;
        d3d11::ishaderresourceview_ptr  m_resource_srv;

    };

    inline render_target_resource create_render_target_resource(ID3D11Device* device, uint32_t width, uint32_t height, DXGI_FORMAT format)
    {
        D3D11_TEXTURE2D_DESC description = {};

        description.ArraySize = 1;
        description.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        description.CPUAccessFlags = 0;
        description.Format = format;    
        description.Height = height;
        description.MipLevels = 1;
        description.MiscFlags = 0;
        description.SampleDesc.Count = 1;
        description.SampleDesc.Quality = 0;

        description.Usage = D3D11_USAGE_DEFAULT;
        description.Width = width;

        d3d11::itexture2d_ptr texture = d3d11::create_texture_2d( device, &description);

        return render_target_resource( texture , d3d11::create_render_target_view( device, texture.get() ),  d3d11::create_shader_resource_view( device, texture.get() ) );
    }

    inline render_target_resource create_normal_resource(ID3D11Device* device, uint32_t width, uint32_t height)
    {
        return create_render_target_resource ( device, width, height, DXGI_FORMAT_R16G16B16A16_FLOAT );
    }

    inline render_target_resource create_diffuse_resource(ID3D11Device* device, uint32_t width, uint32_t height)
    {
        return create_render_target_resource ( device, width, height, DXGI_FORMAT_R8G8B8A8_TYPELESS );
    }

    inline render_target_resource create_specular_resource(ID3D11Device* device, uint32_t width, uint32_t height)
    {
        return create_render_target_resource ( device, width, height, DXGI_FORMAT_R8G8B8A8_TYPELESS );
    }

    inline render_target_resource create_light_buffer_resource(ID3D11Device* device, uint32_t width, uint32_t height)
    {
        return create_render_target_resource ( device, width, height, DXGI_FORMAT_R16G16B16A16_FLOAT );
    }

    inline d3d11::ishaderresourceview_ptr  create_depth_resource_view( ID3D11Device* device, ID3D11Resource* resource )
    {
        D3D11_SHADER_RESOURCE_VIEW_DESC srv = {};
        srv.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
        srv.Texture2D.MostDetailedMip = 0;
        srv.Texture2D.MipLevels = 1;
        srv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;

        return d3d11::create_shader_resource_view( device, resource, &srv );
    }

    inline d3d11::idepthstencilview_ptr    create_read_depth_stencil_view( ID3D11Device* device, ID3D11Resource* resource )
    {
        D3D11_DEPTH_STENCIL_VIEW_DESC dsv = {};
        dsv.Flags = D3D11_DSV_READ_ONLY_DEPTH;
        dsv.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        dsv.Texture2D.MipSlice = 0;
        dsv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;

        return d3d11::create_depth_stencil_view( device, resource, &dsv );
    }

    inline d3d11::idepthstencilview_ptr    create_write_depth_stencil_view( ID3D11Device* device, ID3D11Resource* resource )
    {
        D3D11_DEPTH_STENCIL_VIEW_DESC dsv = {};

        dsv.Flags = 0;
        dsv.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        dsv.Texture2D.MipSlice = 0;
        dsv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;

        return d3d11::create_depth_stencil_view( device, resource, &dsv ); 
    }

    inline depth_resource  create_depth_resource(ID3D11Device* device, uint32_t width, uint32_t height)
    {
        D3D11_TEXTURE2D_DESC description = {};

        description.ArraySize = 1;
        description.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE ;
        description.CPUAccessFlags = 0;
        description.Format = DXGI_FORMAT_R24G8_TYPELESS;
        description.Height = height;
        description.MipLevels = 1;
        description.MiscFlags = 0;
        description.SampleDesc.Count = 1;
        description.SampleDesc.Quality = 0;
        description.Usage = D3D11_USAGE_DEFAULT;
        description.Width = width;

        d3d11::itexture2d_ptr texture = d3d11::create_texture_2d( device, &description);

        return depth_resource( texture, create_write_depth_stencil_view( device, texture.get() ),  create_depth_resource_view( device, texture.get() ) );
    }

    inline d3d11::idepthstencilstate_ptr   create_depth_test_less_state( ID3D11Device* device )
    {
        D3D11_DEPTH_STENCIL_DESC dss = {};

        dss.DepthEnable = true;
        dss.DepthFunc = D3D11_COMPARISON_LESS;
        dss.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;

        return d3d11::create_depth_stencil_state( device, &dss );
    }

    inline d3d11::idepthstencilstate_ptr   create_depth_test_disable_state( ID3D11Device* device)
    {
        D3D11_DEPTH_STENCIL_DESC dss = {};
        return d3d11::create_depth_stencil_state( device, &dss );
    }

    inline d3d11::iblendstate_ptr  create_gbuffer_opaque_blend_state( ID3D11Device* device )
    {
        D3D11_BLEND_DESC blend = {};

        blend.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        blend.RenderTarget[1].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        blend.RenderTarget[2].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        blend.RenderTarget[3].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

        return d3d11::create_blend_state( device, &blend );
    }

    inline d3d11::iblendstate_ptr  create_opaque_blend_state( ID3D11Device* device )
    {
        D3D11_BLEND_DESC blend = {};

        blend.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

        return d3d11::create_blend_state( device, &blend );
    }

    inline d3d11::iblendstate_ptr          create_additive_blend_state( ID3D11Device* device )
    {
        D3D11_BLEND_DESC blend = {};

        blend.RenderTarget[0].BlendEnable = true;
        blend.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
        blend.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
        blend.RenderTarget[0].DestBlend = D3D11_BLEND_ONE;
        blend.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        blend.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
        blend.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ZERO;
        blend.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;

        return d3d11::create_blend_state( device, &blend );
    }

    inline d3d11::isamplerstate_ptr              create_default_sampler_state( ID3D11Device* device )
    {
        D3D11_SAMPLER_DESC sampler = {};

        sampler.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
        sampler.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
        sampler.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
        sampler.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
        sampler.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sampler.MaxAnisotropy = 1;
        sampler.MaxLOD = std::numeric_limits<float>::max();
        sampler.MinLOD = std::numeric_limits<float>::min();
        sampler.MipLODBias = 0;

        return d3d11::create_sampler_state(device, &sampler );
    }

    inline d3d11::iblendstate_ptr          create_premultiplied_alpha_blend_state(ID3D11Device* device)
    {
        D3D11_BLEND_DESC blend = {};

        blend.RenderTarget[0].BlendEnable = true;
        blend.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
        blend.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
        blend.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
        blend.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        blend.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
        blend.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
        blend.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;

        return d3d11::create_blend_state(device, &blend);
    }

    inline d3d11::isamplerstate_ptr         create_point_sampler_state( ID3D11Device* device )
    {
        D3D11_SAMPLER_DESC sampler = {};

        sampler.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
        sampler.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
        sampler.MaxAnisotropy = 1;
        sampler.MaxLOD = std::numeric_limits<float>::max();
        sampler.MinLOD = std::numeric_limits<float>::min();
        sampler.MipLODBias = 0;

        return d3d11::create_sampler_state( device, &sampler );
    }

    inline d3d11::irasterizerstate_ptr      create_cull_back_rasterizer_state( ID3D11Device* device )
    {
        D3D11_RASTERIZER_DESC rasterizer = {};
        rasterizer.FillMode = D3D11_FILL_SOLID;
        rasterizer.CullMode = D3D11_CULL_BACK;
        rasterizer.DepthClipEnable = 1;

        return d3d11::create_raster_state( device, &rasterizer);
    }
}



#endif

