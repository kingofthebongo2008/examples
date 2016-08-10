#pragma once

#include <d3d11/d3d11.h>

namespace app
{
    namespace clear_color_shader
    {
        struct shader
        {
            shader(ID3D11Device* d)
            {
                static 
                #include <clear_color.h>

                m_shader = d3d11::helpers::create_pixel_shader(d, g_clear_color, sizeof(g_clear_color));
            }

            d3d11::pixel_shader m_shader;

            inline ID3D11PixelShader* to_shader()
            {
                return m_shader.get();
            }
        };

        shader* create_shader(ID3D11Device* d)
        {
            static shader s(d);

            return &s;
        }

    }

}
