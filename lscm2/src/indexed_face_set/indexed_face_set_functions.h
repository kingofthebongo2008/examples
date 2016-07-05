#pragma once

#include <numeric>

#include "indexed_face_set_mesh.h"

namespace lscm
{
    namespace indexed_face_set
    {
        using vector = math::float4;
                
        inline vector make_vector( vertex a, vertex b )
        {
            return math::sub( b, a );
        }

        inline float area( const vertex a, const vertex b, const vertex c)
        {
            auto v0 = make_vector(a, c);
            auto v1 = make_vector(b, c);
            auto v2 = math::splat(0.5f);

            auto v3 = math::cross3( v0, v1 );
            auto v4 = math::norm3(v3);
            auto v5 = math::mul(v2, v4);

            alignas(16) float f;
            math::store1(&f, v5);
            return f;
        }

        inline float area(const mesh* m)
        {
            return std::accumulate(std::begin(m->m_faces), std::end(m->m_faces), 0.0f, [m]
            ( float s, const mesh::face& b )
            {
                vertex v0 = *m->get_vertex(b.v0);
                vertex v1 = *m->get_vertex(b.v1);
                vertex v2 = *m->get_vertex(b.v2);
                auto a = area(v0, v1, v2);

                return s + a;
            });
        }
    };
}







