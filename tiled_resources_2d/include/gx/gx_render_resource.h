#pragma once

#include <cstdint>
#include <limits>

#include <d3d11/d3d11_helpers.h>
#include <d3d11/dxgi_helpers.h>

namespace gx
{
    enum msaa_samples : uint32_t
    {
        msaa_2x = 2,
        msaa_4x = 4,
        msaa_8x = 8,
        msaa_16x = 16
    };

    class render_target_resource
    {
        public:

        render_target_resource()
        {

        }

        render_target_resource (    
                                    d3d11::texture2d               resource,
                                    d3d11::render_target_view      resource_rtv,
                                    d3d11::shader_resource_view    resource_srv
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

        d3d11::texture2d                    m_resource;
        d3d11::render_target_view           m_resource_rtv;
        d3d11::shader_resource_view         m_resource_srv;

    };

    class depth_resource
    {
        public:

        depth_resource()
        {

        }

        depth_resource (    
                                    d3d11::texture2d                resource,
                                    d3d11::depth_stencil_view       resource_dsv,
                                    d3d11::shader_resource_view     resource_srv
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

        d3d11::texture2d            m_resource;
        d3d11::depth_stencil_view   m_resource_dsv;
        d3d11::shader_resource_view m_resource_srv;
    };

    inline render_target_resource create_render_target_resource(ID3D11Device* device, uint32_t width, uint32_t height, DXGI_FORMAT format, msaa_samples msaa_samples)
    {
        D3D11_TEXTURE2D_DESC description = {};

        description.ArraySize = 1;
        description.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        description.CPUAccessFlags = 0;
        description.Format = format;    
        description.Height = height;
        description.MipLevels = 1;
        description.MiscFlags = 0;
        description.SampleDesc.Count = msaa_samples;
        description.SampleDesc.Quality = 0;

        description.Usage = D3D11_USAGE_DEFAULT;
        description.Width = width;

        d3d11::texture2d texture = d3d11::helpers::create_texture_2d( device, &description);

        return render_target_resource( texture , d3d11::helpers::create_render_target_view( device, texture.get() ),  d3d11::helpers::create_shader_resource_view( device, texture.get()) );
    }

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

        d3d11::texture2d texture = d3d11::helpers::create_texture_2d(device, &description);

        return render_target_resource(texture, d3d11::helpers::create_render_target_view(device, texture.get() ), d3d11::helpers::create_shader_resource_view(device, texture.get()));
    }

    inline render_target_resource create_render_target_resource(ID3D11Device* device, uint32_t width, uint32_t height, DXGI_FORMAT format, DXGI_FORMAT target, DXGI_FORMAT view, msaa_samples msaa_samples)
    {
        D3D11_TEXTURE2D_DESC description = {};

        description.ArraySize = 1;
        description.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        description.CPUAccessFlags = 0;
        description.Format = format;
        description.Height = height;
        description.MipLevels = 1;
        description.MiscFlags = 0;
        description.SampleDesc.Count = msaa_samples;
        description.SampleDesc.Quality = 0;

        description.Usage = D3D11_USAGE_DEFAULT;
        description.Width = width;

        D3D11_SHADER_RESOURCE_VIEW_DESC view_desc = {};
        D3D11_RENDER_TARGET_VIEW_DESC   render_desc = {};

        render_desc.Format = target;
        render_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMS;
        
        view_desc.Format = view;
        view_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMS;
        

        d3d11::texture2d texture = d3d11::helpers::create_texture_2d(device, &description);

        return render_target_resource(texture, d3d11::helpers::create_render_target_view(device, texture.get(), &render_desc), d3d11::helpers::create_shader_resource_view(device, texture.get(), &view_desc));
    }

    inline render_target_resource create_render_target_resource(ID3D11Device* device, uint32_t width, uint32_t height, DXGI_FORMAT format, DXGI_FORMAT target, DXGI_FORMAT view)
    {
        D3D11_TEXTURE2D_DESC description = {};

        description.ArraySize = 1;
        description.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        description.CPUAccessFlags = 0;
        description.Format = format;
        description.Height = height;
        description.MipLevels = 1;
        description.MiscFlags = 0;
        description.SampleDesc.Count = 1;
        description.SampleDesc.Quality = 0;

        description.Usage = D3D11_USAGE_DEFAULT;
        description.Width = width;

        D3D11_SHADER_RESOURCE_VIEW_DESC view_desc = {};
        D3D11_RENDER_TARGET_VIEW_DESC   render_desc = {};

        render_desc.Format = target;
        render_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        render_desc.Texture2D.MipSlice = 0;

        view_desc.Format = view;
        view_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        view_desc.Texture2D.MostDetailedMip = 0;
        view_desc.Texture2D.MipLevels = 1;

        d3d11::texture2d texture = d3d11::helpers::create_texture_2d(device, &description);

        return render_target_resource(texture, d3d11::helpers::create_render_target_view(device, texture.get(), &render_desc), d3d11::helpers::create_shader_resource_view(device, texture.get(), &view_desc));
    }

    inline d3d11::shader_resource_view  create_depth_resource_view( ID3D11Device* device, ID3D11Resource* resource )
    {
        D3D11_SHADER_RESOURCE_VIEW_DESC srv = {};
        srv.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
        srv.Texture2D.MostDetailedMip = 0;
        srv.Texture2D.MipLevels = 1;
        srv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;

        return d3d11::helpers::create_shader_resource_view( device, resource, &srv );
    }

    inline d3d11::shader_resource_view  create_depth_resource_view(ID3D11Device* device, ID3D11Resource* resource, msaa_samples )
    {
        D3D11_SHADER_RESOURCE_VIEW_DESC srv = {};
        srv.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
        srv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMS;

        return d3d11::helpers::create_shader_resource_view(device, resource, &srv);
    }

    inline d3d11::depth_stencil_view    create_read_depth_stencil_view( ID3D11Device* device, ID3D11Resource* resource )
    {
        D3D11_DEPTH_STENCIL_VIEW_DESC dsv = {};
        dsv.Flags = D3D11_DSV_READ_ONLY_DEPTH;
        dsv.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        dsv.Texture2D.MipSlice = 0;
        dsv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;

        return d3d11::helpers::create_depth_stencil_view( device, resource, &dsv );
    }

    inline d3d11::depth_stencil_view    create_read_depth_stencil_view(ID3D11Device* device, ID3D11Resource* resource, msaa_samples)
    {
        D3D11_DEPTH_STENCIL_VIEW_DESC dsv = {};
        dsv.Flags = D3D11_DSV_READ_ONLY_DEPTH;
        dsv.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        dsv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;

        return d3d11::helpers::create_depth_stencil_view(device, resource, &dsv);
    }


    inline d3d11::depth_stencil_view    create_write_depth_stencil_view( ID3D11Device* device, ID3D11Resource* resource )
    {
        D3D11_DEPTH_STENCIL_VIEW_DESC dsv = {};

        dsv.Flags = 0;
        dsv.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        dsv.Texture2D.MipSlice = 0;
        dsv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;

        return d3d11::helpers::create_depth_stencil_view( device, resource, &dsv );
    }

    inline d3d11::depth_stencil_view    create_write_depth_stencil_view(ID3D11Device* device, ID3D11Resource* resource, msaa_samples samples)
    {
        D3D11_DEPTH_STENCIL_VIEW_DESC dsv = {};

        dsv.Flags = 0;
        dsv.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        dsv.Texture2D.MipSlice = 0;
        dsv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;

        return d3d11::helpers::create_depth_stencil_view(device, resource, &dsv);
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

        d3d11::texture2d texture = d3d11::helpers::create_texture_2d( device, &description);

        return depth_resource( texture, create_write_depth_stencil_view( device, texture ),  create_depth_resource_view( device, texture ) );
    }

    inline depth_resource  create_depth_resource(ID3D11Device* device, uint32_t width, uint32_t height, msaa_samples samples)
    {
        D3D11_TEXTURE2D_DESC description = {};

        description.ArraySize = 1;
        description.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
        description.CPUAccessFlags = 0;
        description.Format = DXGI_FORMAT_R24G8_TYPELESS;
        description.Height = height;
        description.MipLevels = 1;
        description.MiscFlags = 0;
        description.SampleDesc.Count = samples;
        description.SampleDesc.Quality = 0;
        description.Usage = D3D11_USAGE_DEFAULT;
        description.Width = width;

        d3d11::texture2d texture = d3d11::helpers::create_texture_2d(device, &description);

        return depth_resource(texture, create_write_depth_stencil_view(device, texture, samples), create_depth_resource_view(device, texture, samples));
    }

    inline d3d11::depth_stencil_state   make_depth_test_less( ID3D11Device* device )
    {
        D3D11_DEPTH_STENCIL_DESC dss = {};

        dss.DepthEnable = true;
        dss.DepthFunc = D3D11_COMPARISON_LESS;
        dss.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;

        return d3d11::helpers::create_depth_stencil_state( device, &dss );
    }

    inline d3d11::depth_stencil_state   make_depth_test_disable( ID3D11Device* device)
    {
        D3D11_DEPTH_STENCIL_DESC dss = {};
        return d3d11::helpers::create_depth_stencil_state( device, &dss );
    }

    inline d3d11::blend_state  make_gbuffer_opaque( ID3D11Device* device )
    {
        D3D11_BLEND_DESC blend = {};

        blend.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        blend.RenderTarget[1].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        blend.RenderTarget[2].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        blend.RenderTarget[3].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

        return d3d11::helpers::create_blend_state( device, &blend );
    }

    inline d3d11::blend_state  make_opaque( ID3D11Device* device )
    {
        D3D11_BLEND_DESC blend = {};

        blend.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

        return d3d11::helpers::create_blend_state( device, &blend );
    }

    inline d3d11::blend_state   make_additive( ID3D11Device* device )
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

        return d3d11::helpers::create_blend_state( device, &blend );
    }

    inline d3d11::sampler_state make_default_sampler( ID3D11Device* device )
    {
        D3D11_SAMPLER_DESC sampler = {};

        sampler.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
        sampler.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
        sampler.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
        sampler.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
        sampler.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
        sampler.MaxAnisotropy = 1;
        sampler.MaxLOD = std::numeric_limits<float>::max();
        sampler.MinLOD = std::numeric_limits<float>::min();
        sampler.MipLODBias = 0;

        return d3d11::helpers::create_sampler_state(device, &sampler );
    }

    inline d3d11::blend_state   make_premultiplied_alpha(ID3D11Device* device)
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

        return d3d11::helpers::create_blend_state(device, &blend);
    }

    inline d3d11::sampler_state make_point_sampler( ID3D11Device* device )
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

        return d3d11::helpers::create_sampler_state( device, &sampler );
    }

    inline d3d11::rasterizer_state  make_cull_back( ID3D11Device* device )
    {
        D3D11_RASTERIZER_DESC rasterizer = {};
        rasterizer.FillMode = D3D11_FILL_SOLID;
        rasterizer.CullMode = D3D11_CULL_BACK;
        rasterizer.FrontCounterClockwise = 1;
        rasterizer.DepthClipEnable = 1;

        return d3d11::helpers::create_rasterizer_state( device, &rasterizer);
    }

    inline d3d11::rasterizer_state  make_cull_front(ID3D11Device* device)
    {
        D3D11_RASTERIZER_DESC rasterizer = {};
        rasterizer.FillMode = D3D11_FILL_SOLID;
        rasterizer.CullMode = D3D11_CULL_FRONT;
        rasterizer.FrontCounterClockwise = 1;
        rasterizer.DepthClipEnable = 1;

        return d3d11::helpers::create_rasterizer_state(device, &rasterizer);
    }

    inline d3d11::rasterizer_state  make_cull_back_wireframe(ID3D11Device* device)
    {
        D3D11_RASTERIZER_DESC rasterizer = {};
        rasterizer.FillMode = D3D11_FILL_WIREFRAME;
        rasterizer.CullMode = D3D11_CULL_BACK;
        rasterizer.FrontCounterClockwise = 1;
        rasterizer.DepthClipEnable = 1;

        return d3d11::helpers::create_rasterizer_state(device, &rasterizer);
    }

    inline d3d11::rasterizer_state  make_cull_none(ID3D11Device* device)
    {
        D3D11_RASTERIZER_DESC rasterizer = {};
        rasterizer.FillMode = D3D11_FILL_SOLID;
        rasterizer.CullMode = D3D11_CULL_NONE;
        rasterizer.FrontCounterClockwise = 1;
        rasterizer.DepthClipEnable = 1;

        return d3d11::helpers::create_rasterizer_state(device, &rasterizer);
    }
}




