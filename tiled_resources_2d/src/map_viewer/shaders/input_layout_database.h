#pragma once

#include <d3d11/d3d11.h>

#include <unordered_map>

namespace app
{
    struct input_layout_database
    {
        enum format
        {
            pos3_uv2
        };

        std::unordered_map< format, d3d11::input_layout> m_formats;

        static input_layout_database* database()
        {
            static input_layout_database db;
            return &db;
        }

        void initialize_format_0( ID3D11Device* d )
        {
            static
            #include <prototype_vertex.h>

            /*
            LPCSTR SemanticName;
            UINT SemanticIndex;
            DXGI_FORMAT Format;
            UINT InputSlot;
            UINT AlignedByteOffset;
            D3D11_INPUT_CLASSIFICATION InputSlotClass;
            UINT InstanceDataStepRate;
            */
        

            D3D11_INPUT_ELEMENT_DESC v[] =
            {
                { "position",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
                { "texcoord",  0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            };

            auto r = d3d11::helpers::create_input_layout(d, v, 2, g_prototype_vertex, sizeof(g_prototype_vertex));

            m_formats.insert(std::make_pair( format::pos3_uv2, r ));
        }

        void initialize( ID3D11Device* d )
        {
            initialize_format_0(d);
        }

        d3d11::input_layout get_layout(input_layout_database::format f)
        {
            d3d11::input_layout r;

            auto it = m_formats.find(f);
            if (it != m_formats.end())
            {
                r = it->second;
            }

            return r;
        }

        static d3d11::input_layout layout(input_layout_database::format f )
        {
            return database()->get_layout(f);
        }
    };


}
