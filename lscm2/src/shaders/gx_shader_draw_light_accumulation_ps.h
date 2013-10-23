#ifndef __GX_SHADERS_DRAW_LIGHT_ACCUMULATION_PS_H__
#define __GX_SHADERS_DRAW_LIGHT_ACCUMULATION_PS_H__

#include <cstdint>
#include <d3d11/d3d11_error.h>
#include <d3d11/d3d11_pointers.h>

namespace lscm
{
    class shader_draw_light_accumulation_ps final
    {
        public:
        explicit shader_draw_light_accumulation_ps(ID3D11Device* device)
        {
            using namespace os::windows;

            //strange? see in the hlsl file
            static 
            #include "gx_shader_draw_light_accumulation_ps_compiled.hlsl"

            //load, compile and create a pixel shader with the code in the hlsl file, might get slow (this is a compilation), consider offloading to another thread
            throw_if_failed<d3d11::create_pixel_shader>(device->CreatePixelShader(gx_shader_draw_light_accumulation_ps, sizeof(gx_shader_draw_light_accumulation_ps), nullptr, &m_shader));
            m_code = &gx_shader_draw_light_accumulation_ps[0];
            m_code_size = sizeof(gx_shader_draw_light_accumulation_ps);
        }

        operator ID3D11PixelShader* () const
        {
            return m_shader.get();
        }

        d3d11::ipixelshader_ptr     m_shader;
        const void*                 m_code;
        uint32_t                    m_code_size;
    };
}

#endif