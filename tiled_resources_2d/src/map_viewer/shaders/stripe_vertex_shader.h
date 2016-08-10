#pragma once

#include <d3d11/d3d11.h>

namespace app
{
    namespace stripe_vertex_shader
    {
        struct shader
        {
            shader(ID3D11Device* d)
            {
                static
                #include <stripe_vertex.h>

                m_shader = d3d11::helpers::create_vertex_shader(d, g_stripe_vertex, sizeof(g_stripe_vertex));
            }

            d3d11::vertex_shader m_shader;

            inline ID3D11VertexShader* to_shader()
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
