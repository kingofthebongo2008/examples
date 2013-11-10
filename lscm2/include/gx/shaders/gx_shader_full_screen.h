#ifndef __GX_SHADERS_FULL_SCREEN_H__
#define __GX_SHADERS_FULL_SCREEN_H__

#include <cstdint>
#include <future>

#include <d3d11/d3d11_error.h>
#include <d3d11/d3d11_pointers.h>

namespace gx
{
    typedef std::tuple< d3d11::ivertexshader_ptr, const void*, uint32_t> vertex_shader_create_info;

    namespace details
    {
        inline vertex_shader_create_info  create_shader_depth_prepass_vs(ID3D11Device* device)
        {
            d3d11::ivertexshader_ptr   shader;

            using namespace os::windows;

            //strange? see in the hlsl file
            static
            #include "gx_shader_full_screen_vs_compiled.hlsl"
            throw_if_failed<d3d11::create_vertex_shader>(device->CreateVertexShader(gx_shader_full_screen_vs, sizeof(gx_shader_full_screen_vs), nullptr, &shader));

            return std::make_tuple(shader, &gx_shader_full_screen_vs[0], static_cast<uint32_t> (sizeof(gx_shader_full_screen_vs)));
        }
    }

    class shader_full_screen final
    {
        public:
        shader_full_screen() {}

        explicit shader_full_screen( vertex_shader_create_info info ) :
            m_shader(std::get<0>(info))
            , m_code(std::get<1>(info))
            , m_code_size(std::get<2>(info))
        {
        }

        operator ID3D11VertexShader*() const
        {
            return m_shader.get();
        }

        d3d11::ivertexshader_ptr    m_shader;
        const void*                 m_code;
        uint32_t                    m_code_size;
    };

    shader_full_screen create_shader_depth_prepass_vs( ID3D11Device* device )
    {
        return shader_full_screen( details::create_shader_depth_prepass_vs(device) );
    }

    std::future< shader_full_screen > create_screate_shader_depth_prepass_vs_async( ID3D11Device* device )
    {
        return  std::async(std::launch::async, create_shader_depth_prepass_vs, device);
    }

    class shader_full_screen_layout final
    {
        public:

        shader_full_screen_layout( ID3D11Device* device, const shader_full_screen& shader)
        {
            const D3D11_INPUT_ELEMENT_DESC desc[2] = 
            {
                { "position",   0,  DXGI_FORMAT_R16G16B16A16_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
                { "texcoord",   0,  DXGI_FORMAT_R16G16_FLOAT,       0, 8, D3D11_INPUT_PER_VERTEX_DATA, 0 }
            };

            //create description of the vertices that will go into the vertex shader
            //example here: half4, half2
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