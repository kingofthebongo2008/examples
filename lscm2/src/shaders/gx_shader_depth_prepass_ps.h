#ifndef __GX_SHADERS_DEPTH_PREPASS_PS_H__
#define __GX_SHADERS_DEPTH_PREPASS_PS_H__

#include <cstdint>
#include <d3d11/d3d11_error.h>
#include <d3d11/d3d11_pointers.h>

namespace lscm
{
    class shader_depth_prepass_ps final
    {
        public:
        explicit shader_depth_prepass_ps(ID3D11Device* device)
        {
            using namespace os::windows;

            static
            #include "gx_shader_depth_prepass_ps_compiled.hlsl"

            throw_if_failed<d3d11::create_pixel_shader>(device->CreatePixelShader(gx_shader_depth_prepass_ps, sizeof(gx_shader_depth_prepass_ps), nullptr, &m_shader));
            m_code = &gx_shader_depth_prepass_ps[0];
            m_code_size = sizeof(gx_shader_depth_prepass_ps);
        }

        operator const ID3D11PixelShader*() const
        {
            return m_shader.get();
        }

        d3d11::ipixelshader_ptr     m_shader;
        const void*                 m_code;
        uint32_t                    m_code_size;
    };
}

#endif