#ifndef __GX_SHADERS_FULL_SCREEN_H__
#define __GX_SHADERS_FULL_SCREEN_H__

#include <cstdint>
#include <d3d11/d3d11_error.h>
#include <d3d11/d3d11_pointers.h>

namespace gx
{
    class shader_full_screen final
    {
        public:

        explicit shader_full_screen ( ID3D11Device* device )
        {
            using namespace os::windows;

            //strange? see in the hlsl file
            static 
            #include "gx_shader_full_screen_vs_compiled.hlsl"

            //load, compile and create a pixel shader with the code in the hlsl file, might get slow (this is a compilation), consider offloading to another thread
            throw_if_failed<d3d11::create_vertex_shader> (device->CreateVertexShader( gx_shader_full_screen_vs, sizeof(gx_shader_full_screen_vs), nullptr, &m_shader));
            m_code = &gx_shader_full_screen_vs[0];
            m_code_size = sizeof(gx_shader_full_screen_vs);
        }

        operator ID3D11VertexShader*() const
        {
            return m_shader.get();
        }

        d3d11::ivertexshader_ptr    m_shader;
        const void*                 m_code;
        uint32_t                    m_code_size;
    };

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