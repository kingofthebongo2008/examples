#ifndef __GX_SHADERS_DEPTH_PREPASS_VS_H__
#define __GX_SHADERS_DEPTH_PREPASS_VS_H__

#include <cstdint>
#include <future>
#include <tuple>

#include <d3d11/d3d11_error.h>
#include <d3d11/d3d11_pointers.h>
#include <d3d11/d3d11_helpers.h>
#include <gx/gx_constant_buffer_helper.h>

#include <math/math_matrix.h>

namespace lscm
{
    class __declspec(align(16)) shader_depth_prepass_vs_buffer final
    {
    public:

        explicit shader_depth_prepass_vs_buffer(ID3D11Device* device) :
            m_buffer( d3d11::create_constant_buffer(device, size() ) )
        {


        }

        void set_w(math::float4x4 value)
        {
            m_w = value;
        }

        void update(ID3D11DeviceContext* context, math::float4x4* value)
        {
            gx::constant_buffer_update(context, m_buffer, value);
        }

        void flush(ID3D11DeviceContext* context)
        {
            update(context, &m_w);
        }

        void bind_as_vertex(ID3D11DeviceContext* context)
        {
            context->VSSetConstantBuffers(gx::slot_per_draw_call, 1, &m_buffer);
        }

        void bind_as_vertex(ID3D11DeviceContext* context, uint32_t slot)
        {
            context->VSSetConstantBuffers(slot, 1, &m_buffer);
        }

        operator ID3D11Buffer*()
        {
            return m_buffer.get();
        }

        operator const ID3D11Buffer*() const
        {
            return m_buffer.get();
        }

        size_t size() const
        {
            return sizeof(m_w);
        }

    private:

        d3d11::ibuffer_ptr	m_buffer;
        math::float4x4		m_w;
    };

    typedef std::tuple< d3d11::ivertexshader_ptr, const void*, uint32_t> vertex_shader_create_info;

    namespace details
    {
        inline vertex_shader_create_info  create_shader_depth_prepass_vs(ID3D11Device* device)
        {
            d3d11::ivertexshader_ptr   shader;

            using namespace os::windows;

            static
            #include "gx_shader_depth_prepass_vs_compiled.hlsl"

            throw_if_failed<d3d11::create_vertex_shader>(device->CreateVertexShader(gx_shader_depth_prepass_vs, sizeof(gx_shader_depth_prepass_vs), nullptr, &shader));

            return std::make_tuple(shader, &gx_shader_depth_prepass_vs[0], static_cast<uint32_t> (sizeof(gx_shader_depth_prepass_vs)));
        }
    }

    class shader_depth_prepass_vs final
    {
    public:

        shader_depth_prepass_vs()
        {

        }

        explicit shader_depth_prepass_vs( vertex_shader_create_info info ) :
            m_shader(std::get<0>(info) )
            , m_code(std::get<1>(info) )
            , m_code_size( std::get<2> ( info ) )
        {
        }

        operator ID3D11VertexShader* const () const
        {
            return m_shader.get();
        }

        d3d11::ivertexshader_ptr    m_shader;
        const void*                 m_code;
        uint32_t                    m_code_size;
    };

    shader_depth_prepass_vs create_shader_depth_prepass_vs(ID3D11Device* device)
    {
        return shader_depth_prepass_vs(details::create_shader_depth_prepass_vs(device));

    }

    std::future< shader_depth_prepass_vs > create_shader_depth_prepass_vs_async(ID3D11Device* device)
    {
        return std::async(std::launch::async, create_shader_depth_prepass_vs, device);
    }

    class shader_depth_prepass_layout final
    {
        public:

        shader_depth_prepass_layout(ID3D11Device* device, const shader_depth_prepass_vs& shader)
        {
            const D3D11_INPUT_ELEMENT_DESC desc[] = 
            {
                { "position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }
            };

            os::windows::throw_if_failed<d3d11::create_input_layout> ( device->CreateInputLayout(&desc[0], sizeof(desc) / sizeof(desc[0]), shader.m_code, shader.m_code_size, &m_input_layout ) );
        }

        operator ID3D11InputLayout*()
        {
            return m_input_layout.get();
        }

        operator const ID3D11InputLayout*() const
        {
            return m_input_layout.get();
        }

        d3d11::iinputlayout_ptr	m_input_layout;
    };
}

#endif