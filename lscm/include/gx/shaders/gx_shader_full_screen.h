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

            #include "gx_shader_full_screen_vs_compiled.hlsl"

            throw_if_failed<d3d11::create_vertex_shader> (device->CreateVertexShader( gx_shader_full_screen_vs, sizeof(gx_shader_full_screen_vs), nullptr, &m_shader));
            m_code = &gx_shader_full_screen_vs[0];
            m_code_size = sizeof(gx_shader_full_screen_vs);
        }

        operator const ID3D11VertexShader*() const
        {
            return m_shader.get();
        }

        d3d11::ivertexshader_ptr    m_shader;
        const void*                 m_code;
        uint32_t                    m_code_size;
    };
}

#endif