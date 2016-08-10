#pragma once
#include <d3d11/d3d11_helpers.h>

namespace d3d11
{
    namespace helpers
    {
        struct mapped_constant
        {
            mapped_constant(const D3D11_MAPPED_SUBRESOURCE& mr, ID3D11Resource* r, ID3D11DeviceContext2* ctx, UINT subresource = 0) :
                m_mapped_resource(mr)
                , m_resource(r)
                , m_context(ctx)
                , m_subresource(subresource)
            {

            }

            mapped_constant(mapped_constant&& o) :
                m_mapped_resource(std::move(o.m_mapped_resource))
                , m_resource(std::move(o.m_resource))
                , m_context(std::move(o.m_context))
            {
                o.m_resource = nullptr;
                o.m_context = nullptr;
            }

            mapped_constant& operator=(mapped_constant&& o)
            {
                m_mapped_resource = std::move(o.m_mapped_resource);
                m_resource = std::move(o.m_resource);
                m_context = std::move(o.m_context);

                o.m_resource = nullptr;
                o.m_context = nullptr;
            }

            ~mapped_constant()
            {
                if (m_context && m_resource)
                {
                    m_context->Unmap(m_resource, m_subresource);
                }
            }

            template <typename t> t* data() const
            {
                return reinterpret_cast<t*> (m_mapped_resource.pData);
            }

            D3D11_MAPPED_SUBRESOURCE m_mapped_resource;
            ID3D11Resource*          m_resource;
            ID3D11DeviceContext2*    m_context;
            UINT                     m_subresource;

            mapped_constant(const mapped_constant&) = delete;
            mapped_constant& operator=(const mapped_constant&) = delete;
        };

        inline mapped_constant map_constant_buffer(ID3D11DeviceContext2* ctx, ID3D11Resource* r)
        {
            D3D11_MAPPED_SUBRESOURCE resource;
            d3d11::throw_if_failed(ctx->Map(r, 0, D3D11_MAP_WRITE_DISCARD, 0, &resource));

            return mapped_constant(resource, r, ctx);
        }

        template <typename t> void update_constant_buffer(ID3D11DeviceContext2* ctx, ID3D11Resource*r, const t* d)
        {
            auto&& m    = map_constant_buffer(ctx, r);
            t* aliased  = m.data<t>();
            *aliased    = *d;
        }

        template <typename t> struct typed_constant_buffer
        {
            d3d11::buffer    m_resource;

            void update(ID3D11DeviceContext2* ctx, const t* s)
            {
                update_constant_buffer<t>(ctx, m_resource, s);
            }

            void update(ID3D11DeviceContext2* ctx, const t& s)
            {
                update_constant_buffer<t>(ctx, m_resource, &s);
            }

            ID3D11Buffer* to_constant_buffer() const
            {
                return m_resource.get();
            }
        };

        template <typename t> inline typed_constant_buffer<t> make_constant_buffer(ID3D11Device* d)
        {
            typed_constant_buffer<t> r;
            r.m_resource = create_constant_buffer(d, sizeof(t));
            return r;
        }

    }
}
