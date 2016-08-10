#pragma once

#include <d3d11/d3d11_helpers_constants.h>
#include <math/math.h>

#include <shaders/input_layout_database.h>
#include <array>

namespace app
{
    struct world_map_model
    {
        struct aabb
        {
            math::float4 m_min;
            math::float4 m_max;
        };

        struct vertex
        {
            float x;
            float y;
            float z;

            float u;
            float v;
        };

        d3d11::buffer       m_vertices;
        d3d11::buffer       m_indices;
        d3d11::input_layout m_input_layout;
        aabb                m_aabb;

        d3d11::vertex_buffer_view to_vertex_buffer_view() const
        {
            d3d11::vertex_buffer_view r = { m_vertices.get(), sizeof(vertex), 0 };
            return r;
        }

        d3d11::index_buffer_view to_index_buffer_view() const
        {
            d3d11::index_buffer_view r = { m_indices.get(), DXGI_FORMAT_R16_UINT, 0 };
            return r;
        }
    };

    world_map_model make_world_map_model(ID3D11Device* d)
    {
        world_map_model m;

        using index = uint16_t;
        using vertex = world_map_model::vertex;

        float s = 16384.0f; // meters

        //big quad in meters to simulate the world map
        std::array<vertex,4> vertices 
        {
            vertex{ 0.0f, 0.0f,   0.0f, 0.0f, 0.0f },
            vertex{ s,    0.0f,   0.0f, 1.0f, 0.0f },
            vertex{ s,    s,      0.0f, 1.0f, 1.0f },
            vertex{ 0.0f, s,      0.0f, 0.0f, 1.0f }
        };

        //counter clock wise
        std::array<index, 6> indices  =
        {
            0,1,2, 0, 2, 3
        };

        m.m_vertices     = d3d11::helpers::create_vertex_buffer(d, sizeof(vertex), vertices.size(), &vertices[0]);
        m.m_indices      = d3d11::helpers::create_index_buffer(d, sizeof(index), indices.size(), &indices[0]);
        m.m_input_layout = input_layout_database::layout(input_layout_database::pos3_uv2);

        math::float4 min = math::splat(std::numeric_limits<float>::max());
        math::float4 max = math::splat(std::numeric_limits<float>::lowest());

        for ( auto i = 0; i < sizeof(vertices) / sizeof(vertices[0]); ++i )
        {
            math::float4 v0 = math::load3u_point(&vertices[i]);
            min = math::min(v0, min);
            max = math::max(v0, max);
        }

        m.m_aabb.m_min = min;
        m.m_aabb.m_max = max;

        return m;
    }

    struct world_map
    {
        world_map_model m_model;
        math::float4x4  m_world;
    };

    world_map make_world_map(ID3D11Device* d)
    {
        world_map r;

        r.m_model = make_world_map_model(d);
        r.m_world = math::identity_matrix();

        auto s      = math::sub(r.m_model.m_aabb.m_max, r.m_model.m_aabb.m_min);
        auto center = math::div(s, math::set(2.0f, 2.0f, 2.0f, 1.0f));

        r.m_world   = math::translation( math::negate(center) );
        

        return r;
    }

    struct world_map_render
    {
        d3d11::buffer   m_per_draw_call;
        world_map       m_world_map;

        d3d11::vertex_buffer_view to_vertex_buffer_view() const
        {
            return m_world_map.m_model.to_vertex_buffer_view();
        }

        d3d11::index_buffer_view to_index_buffer_view() const
        {
            return m_world_map.m_model.to_index_buffer_view();
        }

        void set_world_matrix(ID3D11DeviceContext2*ctx, math::float4x4 m)
        {
            auto&& mapped = d3d11::helpers::map_constant_buffer(ctx, m_per_draw_call.get());
            math::float4x4* mat = mapped.data<math::float4x4>();
            *mat = m;
        }

        void update_constants(ID3D11DeviceContext2*ctx)
        {
            set_world_matrix(ctx, math::transpose(m_world_map.m_world));
        }

        ID3D11Buffer* to_constant_buffer() const 
        {
            return m_per_draw_call.get();
        }

        ID3D11InputLayout* to_input_layout() const 
        {
            return m_world_map.m_model.m_input_layout.get();
        }
    };

    world_map_render make_world_map_render( ID3D11Device* d )
    {
        world_map_render r;

        r.m_world_map = make_world_map(d);
        r.m_per_draw_call = d3d11::helpers::create_constant_buffer(d, sizeof(r.m_world_map.m_world));

        return r;
    }
}
