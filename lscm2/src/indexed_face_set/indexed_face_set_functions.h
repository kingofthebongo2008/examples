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

        inline float area( vertex a, vertex b, vertex c)
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
            //todo: accelerate
            return std::accumulate(std::begin(m->m_faces), std::end(m->m_faces), 0.0f, [m]
            ( float s, const mesh::face& b )
            {
                vertex v0 = math::load3u_point(m->get_vertex(b.v0));
                vertex v1 = math::load3u_point(m->get_vertex(b.v1));
                vertex v2 = math::load3u_point(m->get_vertex(b.v2));
                auto a = area(v0, v1, v2);

                return s + a;
            });
        }

        inline vector distance(vertex a, const mesh* m)
        {
            auto init = math::splat(std::numeric_limits<float>::max());

            //todo: accelerate
            return std::accumulate(std::begin(m->m_vertices), std::end(m->m_vertices), init, [ m, a ]
            (math::float4 distance, const mesh::vertex& b)
            {
                vertex v0 = math::load3u_point(&b);
                auto d1 = math::norm2( math::sub(v0, a) );
                return math::min(d1, distance);
            });
        }

        inline vector hausdorff_distance( const mesh* m1, const mesh* m2 )
        {
            auto init = math::splat(std::numeric_limits<float>::max());

            //todo: accelerate
            return std::accumulate(std::begin(m1->m_vertices), std::end(m1->m_vertices), init, [m2]
            (math::float4 d, const mesh::vertex& b)
            {
                vertex v0 = math::load3u_point(&b);
                math::float4 d1 = math::min(d, distance(v0, m2));
                return d1;
            });
        }

        inline vector symmetric_hausdorff_distance(const mesh* m1, const mesh* m2)
        {
            return math::max( hausdorff_distance(m1, m2), hausdorff_distance(m2, m1));
        }

    };
}







