#pragma once

#include <d3d11/d3d11.h>

namespace app
{
    namespace display_texture_shader
    {
        struct shader
        {
            shader(ID3D11Device* d)
            {
                static 
                #include <display_texture.h>

                m_shader = d3d11::helpers::create_pixel_shader(d, g_display_texture, sizeof(g_display_texture));
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
