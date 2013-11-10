#ifndef __GX_SHADERS_DRAW_LIGHT_ACCUMULATION_PS_H__
#define __GX_SHADERS_DRAW_LIGHT_ACCUMULATION_PS_H__

#include <cstdint>
#include <future>

#include <d3d11/d3d11_error.h>
#include <d3d11/d3d11_pointers.h>

namespace lscm
{
    inline d3d11::ipixelshader_ptr   create_shader_draw_light_accumulation_ps(ID3D11Device* device)
    {
        d3d11::ipixelshader_ptr   shader;

        using namespace os::windows;

        //strange? see in the hlsl file
        static
        #include "gx_shader_draw_light_accumulation_ps_compiled.hlsl"

        //load, compile and create a pixel shader with the code in the hlsl file, might get slow (this is a compilation), consider offloading to another thread
        throw_if_failed<d3d11::create_pixel_shader>(device->CreatePixelShader( gx_shader_draw_light_accumulation_ps, sizeof(gx_shader_draw_light_accumulation_ps), nullptr, &shader) );

        return shader;
    }

    std::future< d3d11::ipixelshader_ptr> create_shader_draw_light_accumulation_ps_async(ID3D11Device* device)
    {
        return std::async(std::launch::async, create_shader_draw_light_accumulation_ps, device);
    }

    class shader_draw_light_accumulation_ps final
    {
        public:

        explicit shader_draw_light_accumulation_ps(d3d11::ipixelshader_ptr shader) : m_shader(shader)
        {

        }

        operator ID3D11PixelShader* () const
        {
            return m_shader.get();
        }

        d3d11::ipixelshader_ptr     m_shader;
    };
}

#endif