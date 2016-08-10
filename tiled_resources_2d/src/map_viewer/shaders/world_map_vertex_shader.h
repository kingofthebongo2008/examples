#pragma once

#include <d3d11/d3d11.h>

namespace app
{
    namespace world_map_vertex_shader
    {
        static
        #include <world_map_vertex.h>

        struct shader
        {
            shader(ID3D11Device* d)
            {
                m_shader = d3d11::helpers::create_vertex_shader(d, g_world_map_vertex, sizeof(g_world_map_vertex));
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

        d3d11::input_layout create_input_layout(ID3D11Device* d, const D3D11_INPUT_ELEMENT_DESC *pInputElementDescs, UINT NumElements)
        {
            return d3d11::helpers::create_input_layout(d, pInputElementDescs, NumElements, g_world_map_vertex, sizeof(g_world_map_vertex));
        }
    }

}
