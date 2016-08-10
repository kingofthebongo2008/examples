#pragma once

#include <d3d11/d3d11.h>

namespace app
{
    namespace sampling_vertex_shader
    {
        struct shader
        {
            shader(ID3D11Device* d)
            {
                static
                #include <sampling_vertex.h>

                m_shader = d3d11::helpers::create_vertex_shader(d, g_sampling_vertex, sizeof(g_sampling_vertex));
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
