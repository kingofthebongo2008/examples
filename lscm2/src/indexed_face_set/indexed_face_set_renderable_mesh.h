#pragma once

namespace lscm
{
    namespace indexed_face_set
    {

        class renderable_mesh
        {

        public:

            renderable_mesh(d3d11::ibuffer_ptr vertices, d3d11::ibuffer_ptr triangles, uint32_t vertex_stride, uint32_t vertex_count, uint32_t index_count) :
                m_positions(vertices)
                , m_triangles(triangles)
                , m_vertex_stride( vertex_stride )
                , m_vertex_count(vertex_count)
                , m_index_count(index_count)
            {

            }

            void draw(ID3D11DeviceContext* context)
            {
                d3d11::ia_set_primitive_topology(context, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                d3d11::ia_set_vertex_buffer(context, m_positions, m_vertex_stride);

                d3d11::ia_set_index_buffer(context, m_triangles, DXGI_FORMAT_R32_UINT);
                context->DrawIndexed(m_index_count, 0, 0);
            }

        private:

            d3d11::ibuffer_ptr  m_positions;
            d3d11::ibuffer_ptr  m_triangles;
            uint32_t            m_vertex_count;
            uint32_t            m_index_count;
            uint32_t            m_vertex_stride;
        };

        inline std::shared_ptr< renderable_mesh > create_renderable_mesh(ID3D11Device* device, const std::shared_ptr< mesh >& mesh)
        {
            auto positions = gx::create_positions_x_y_z((const float*)&mesh->m_vertices[0], static_cast<uint32_t> (mesh->m_vertices.size()));
            auto vertex_count = static_cast<uint32_t> (mesh->m_vertices.size());

            //triangle list
            auto index_count = 3 * static_cast<uint32_t> (mesh->m_faces.size());

            return std::make_shared<renderable_mesh>(
                d3d11::create_immutable_vertex_buffer(device, &positions[0], positions.size() * sizeof(math::half)),
                d3d11::create_immutable_index_buffer(device, &mesh->m_faces[0], mesh->m_faces.size() * sizeof(mesh::face)),
                4 * sizeof(math::half),
                vertex_count,
                index_count
                );
        }
    }
}