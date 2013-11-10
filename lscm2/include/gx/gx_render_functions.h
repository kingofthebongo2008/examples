#ifndef __GX_RENDER_FUNCTIONS_H__
#define __GX_RENDER_FUNCTIONS_H__

#include <array>
#include <cstdint>
#include <limits>

#include <math/math_half.h>
#include <math/math_matrix.h>

#include <d3d11/d3d11_helpers.h>

#include <gx/shaders/gx_shader_full_screen.h>

namespace gx
{
    class full_screen_draw
    {
        public:

        full_screen_draw ( ID3D11Device* device ) : 
            m_shader( create_shader_depth_prepass_vs(device) )
            , m_input_layout (device, m_shader)
        {
            using namespace math;

            struct vertex
            {
                half v[4];
                half uv[2];
            };

            struct vertex_float
            {
                float v[4];
                float uv[2];
            };

            const vertex_float v_1[ 6 + 2 ] =
            { 
                 { {-1.0f,	-1.0f,	0.0f, 1.0f},  {0.0f, 1.0f}},
                 { {-1.0f,	 1.0f,	0.0f, 1.0f},  {0.0f, 0.0f}},
                 { {1.0f,	 1.0f,	0.0f, 1.0f},  {1.0f, 0.0f}},

                 { {1.0f,	 1.0f,	0.0f, 1.0f} , {1.0f, 0.0f}},
                 { {1.0f,	-1.0f,	0.0f, 1.0f} , {1.0f, 1.0f}},
                 { {-1.0f,	-1.0f,	0.0f, 1.0f} , {0.0f, 1.0f}},

                 { {0.0f,	0.0f,	0.0f, 0.0f} , {0.0f,0.0f}}, //padding
                 { {0.0f,	0.0f,	0.0f, 0.0f} , {0.0f,0.0f}}, //padding
            };

            __declspec( align(16) ) math::half h_1 [ 40 ];

            math::convert_f32_f16_stream(reinterpret_cast<const float*> (&v_1[0]), static_cast<uint32_t>(40), &h_1[0] );
            m_geometry = d3d11::create_immutable_vertex_buffer( device, &h_1[0],  6 * sizeof(vertex) );
        }
            

        void    draw ( ID3D11DeviceContext* device_context )
        {

            d3d11::ia_set_input_layout( device_context, m_input_layout );
            d3d11::vs_set_shader( device_context, m_shader );
            d3d11::ia_set_primitive_topology(device_context, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            d3d11::ia_set_vertex_buffer ( device_context, m_geometry, 12 );

            device_context->Draw( 6, 0 );
        }

        shader_full_screen          m_shader;

        shader_full_screen_layout   m_input_layout;
        d3d11::ibuffer_ptr          m_geometry;
    };

    void reset_shader_resources(ID3D11DeviceContext* device_context)
    {
        ID3D11ShaderResourceView* resources[ D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT ];

        for (auto i = 0; i < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT; ++i )
        {
            resources[i] = nullptr;
        }

        d3d11::cs_set_shader_resources ( device_context, resources );
        d3d11::gs_set_shader_resources ( device_context, resources );
        d3d11::ps_set_shader_resources ( device_context, resources );
        d3d11::vs_set_shader_resources ( device_context, resources );
    }

    void reset_constant_buffers(ID3D11DeviceContext* device_context)
    {
        ID3D11Buffer * buffers[D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT];

        for (auto i = 0; i <D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT;++i)
        {
            buffers[i] = nullptr;
        }

        device_context->VSSetConstantBuffers(0, sizeof(buffers)/ sizeof(buffers[0]), &buffers[0]); 
        device_context->PSSetConstantBuffers(0, sizeof(buffers)/ sizeof(buffers[0]), &buffers[0]); 
    }

    void reset_render_targets(ID3D11DeviceContext* device_context)
    {
        std::array<ID3D11RenderTargetView* const , D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT> views = 
        { 
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,

                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr
        };

        d3d11::om_set_render_targets ( device_context, std::cbegin(views), std::cend(views) );

    }


}



#endif

