#pragma once

#include <d3d11/d3d11.h>

namespace app
{
    namespace sampling_pixel_shader
    {
        struct shader
        {
            shader(ID3D11Device* d)
            {
                static 
                #include <sampling_pixel.h>

                m_shader = d3d11::helpers::create_pixel_shader(d, g_sampling_pixel, sizeof(g_sampling_pixel));
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
