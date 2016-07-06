#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>

#include <ppl.h>
#include <ppltasks.h>
#include <concrt.h>

namespace lscm
{
    namespace indexed_face_set
    {
        using vertex    = math::float4;
        using normal    = math::float4;
        using pointer   = uint32_t;

        static const uint32_t null_pointer = 0xFFFFFFFF;

        struct storage_vertex
        {
            float x;
            float y;
            float z;
        };

        struct storage_normal
        {
            float x;
            float y;
            float z;
        };

        struct face
        {
            pointer v0 = null_pointer;
            pointer v1 = null_pointer;
            pointer v2 = null_pointer;
        };

        struct edge
        {
            pointer v0 = null_pointer;
            pointer v1 = null_pointer;
        };

        struct winged_edge
        {
            pointer v0 = null_pointer; //start vertex
            pointer v1 = null_pointer; //end   vertex

            pointer f0 = null_pointer; // left   face
            pointer f1 = null_pointer; // right  face

            pointer l_p = null_pointer; // left  predecessor
            pointer r_p = null_pointer; // right predecessor

            pointer l_s = null_pointer; // left  successor
            pointer r_s = null_pointer; // right successor
        };

        struct half_edge
        {
            pointer o = null_pointer;  //opposite half edge
            pointer n = null_pointer;  //next half edge

            pointer v = null_pointer;  //vertex
            pointer f = null_pointer;  //face
        };
    }
}

namespace std
{
    // hash functor for bool
    template<> struct hash<lscm::indexed_face_set::edge>
    {
        using this_type = lscm::indexed_face_set::edge;

        size_t operator()(const this_type val) const
        {
            uint64_t op = ( (uint64_t ) val.v0 << 32ULL ) | val.v1;
            hash<uint64_t> v;
            return v(op);
        }
    };
}

namespace lscm
{
    namespace indexed_face_set
    {

        struct mesh
        {

        public:

            using vertex = storage_vertex;
            using normal = storage_normal;
            using face = ::lscm::indexed_face_set::face;

            struct progress_notifier
            {

            };

            mesh(
                const std::vector< vertex >&   vertices,
                const std::vector< normal >&   normals,
                const std::vector< face >&     faces,
                const progress_notifier&       notifier
            ) : m_vertices(vertices)
                , m_normals(normals)
                , m_faces(faces)
                , m_notifier(notifier)
            {
                initialize();
            }

            mesh(
                std::vector< vertex > &&     vertices,
                std::vector< normal > &&     normals,
                std::vector< face >   &&     faces,
                progress_notifier     &&    notifier
            ) :
                m_vertices(std::move(vertices))
                , m_normals(std::move(normals))
                , m_faces(std::move(faces))
                , m_notifier(std::move(notifier))
            {
                initialize();
            }

            vertex* get_vertex(pointer p)
            {
                return &m_vertices[static_cast<uint32_t> (p)];
            }

            const vertex* get_vertex(pointer p) const
            {
                return &m_vertices[static_cast<uint32_t> (p)];
            }

            face*   get_face(pointer p)
            {
                return &m_faces[static_cast<uint32_t> (p)];
            }

            const face*   get_face(pointer p) const
            {
                return &m_faces[static_cast<uint32_t> (p)];
            }

            half_edge*   get_edge(pointer p)
            {
                return &m_edges_storage[static_cast<uint32_t> (p)];
            }

            const half_edge*   get_edge(pointer p) const
            {
                return &m_edges_storage[static_cast<uint32_t> (p)];
            }

            bool is_half_edge( pointer p ) const
            {
                return static_cast<uint32_t>(p) < m_edges_storage.size();
            }

            std::vector< vertex >                m_vertices;
            std::vector< normal >                m_normals;
            std::vector< face >                  m_faces;
            std::vector< half_edge >             m_edges_storage;
            std::unordered_map < edge, pointer > m_edges;
            progress_notifier                    m_notifier;

            void initialize()
            {
                clean_duplicate_faces();
                clear_vertices_not_referenced_by_faces();
                calculate_pivot();

                concurrency::task_group r;

                r.run([this]
                {
                    build_normals();
                });

                r.run([this]
                {
                    build_edges();
                });

                r.wait();
            }

            void build_edges()
            {


            }

            std::vector< normal > build_face_normals() const
            {
                std::vector< normal > face_normals(m_faces.size());

                for (uint32_t i = 0; i < face_normals.size(); ++i)
                {
                    math::float4 v0 = math::load3u_point(&m_vertices[m_faces[i].v0]);
                    math::float4 v1 = math::load3u_point(&m_vertices[m_faces[i].v1]);
                    math::float4 v2 = math::load3u_point(&m_vertices[m_faces[i].v2]);

                    math::float4 n      = math::cross3(math::sub(v0, v1), math::sub(v1, v2));
                    math::float4 normal = math::normalize3(n);

                    math::store3u_vector(&face_normals[i], normal);
                }

                return face_normals;
            }

            void calculate_pivot()
            {
                std::vector< mesh::vertex > vertices(m_vertices.size());
                

                math::float4 sum = math::zero();

                //todo: parallel
                std::for_each(m_vertices.begin(), m_vertices.end(), [&](const mesh::vertex& v)
                {
                    math::float4 v0 = math::load3u_point(&v);
                    sum = math::add(sum, v0);
                }
                );

                auto vertex_size = static_cast<float>(m_vertices.size());
                sum = math::div(sum, math::set(vertex_size, vertex_size, vertex_size, vertex_size ));
                sum = math::mul(sum, math::set(1.0f, 1.0f, 1.0f, 0.0f));

                std::transform(m_vertices.begin(), m_vertices.end(), vertices.begin(), [&](const mesh::vertex& v)
                {
                    math::float4 v0 = math::load3u_point(&v);

                    v0 = math::sub(v0, sum);
                    
                    mesh::vertex r;

                    math::store3u_point(&r, v0);

                    return r;
                }
                );

                m_vertices = std::move(vertices);
            }

            void clean_degenerate_faces()
            {
                std::vector< face > faces(m_faces.size());

                auto last = std::copy_if(m_faces.begin(), m_faces.end(), faces.begin(), [=](const face& f)
                {
                    return (f.v0 != f.v1 && f.v0 != f.v2 && f.v1 != f.v2);
                });

                faces.resize(std::distance(faces.begin(), last));
                m_faces = std::move(faces);
            }

            void build_normals()
            {
                std::vector< normal > normals(m_vertices.size());

                auto face_size = m_faces.size();

                for (uint32_t i = 0; i < face_size; ++i)
                {
                    auto i0 = m_faces[i].v0;
                    auto i1 = m_faces[i].v1;
                    auto i2 = m_faces[i].v2;

                    math::float4 v0     = math::load3u_point(&m_vertices[i0]);
                    math::float4 v1     = math::load3u_point(&m_vertices[i1]);
                    math::float4 v2     = math::load3u_point(&m_vertices[i2]);

                    math::float4 n      = math::cross3(math::sub(v0, v1), math::sub(v1, v2));

                    math::float4 n0     = math::load3u_vector(&normals[i0]);
                    math::float4 n1     = math::load3u_vector(&normals[i1]);
                    math::float4 n2     = math::load3u_vector(&normals[i2]);

                    n0 = math::add(n0, n);
                    n1 = math::add(n1, n);
                    n2 = math::add(n2, n);
                    
                    math::store3u_point(&normals[i0], n0);
                    math::store3u_point(&normals[i1], n1);
                    math::store3u_point(&normals[i2], n2);

                }

                std::transform(normals.begin(), normals.end(), normals.begin(), [=](normal& n0)
                {
                    math::float4 n  = math::load3u_vector(&n0);
                    math::float4 n1 = math::normalize3(n);

                    normal result;

                    math::store3u_vector(&result, n1);
                    return result;
                });

                m_normals = std::move(normals);
            }

            void clear_vertices_not_referenced_by_faces()
            {
                std::vector< mesh::vertex > vertices(m_vertices.size());

                uint32_t j = 0;

                for (uint32_t i = 0; i < m_vertices.size(); ++i)
                {
                    for (uint32_t k = 0; k < m_faces.size(); ++k)
                    {
                        const face& f = m_faces[k];

                        //face references the i-th vertex, then it is used
                        if (f.v0 == i || f.v1 == i || f.v2 == i)
                        {
                            vertices[j] = m_vertices[i];
                            ++j;
                            break;
                        }
                    }
                }

                vertices.resize(j);
                m_vertices = std::move(vertices);
            }

            void clean_duplicate_faces()
            {
                struct equal_faces
                {
                    struct hash_function
                    {
                        size_t operator() (const face& f) const
                        {
                            return (((size_t)f.v0) << 42UL) | (((size_t)f.v1) << 21UL) | f.v2;
                        }
                    };

                    static void sort(uint32_t* f)
                    {
                        uint32_t n = 3;

                        do
                        {
                            uint32_t new_n = 0;

                            for (uint32_t i = 1; i <= n - 1; ++i)
                            {
                                if (f[i - 1] > f[i])
                                {
                                    std::swap(f[i - 1], f[i]);
                                    new_n = i;
                                }
                            }

                            n = new_n;
                        } while (n > 0);
                    }

                    bool operator()(const face& f0, const face& f1) const
                    {
                        uint32_t f_0[3] = { f0.v0, f0.v1, f0.v2 };
                        uint32_t f_1[3] = { f1.v0, f1.v1, f1.v2 };

                        sort(&f_0[0]);
                        sort(&f_1[0]);

                        uint32_t difference[3] = { f_0[0] - f_1[0] , f_0[1] - f_1[1] , f_0[2] - f_1[2] };
                        return difference[0] == 0 && difference[1] == 0 && difference[2] == 0;
                    }
                };

                std::unordered_set< face, equal_faces::hash_function, equal_faces > unique_faces;

                std::for_each(m_faces.begin(), m_faces.end(), [&](const face& f)
                {
                    face f0 = f;

                    equal_faces::sort(&f0.v0);

                    if (unique_faces.find(f0) == unique_faces.end())
                    {
                        unique_faces.insert(f0);
                    }
                });

                std::vector< mesh::face> faces;
                faces.resize(unique_faces.size());

                std::copy(unique_faces.begin(), unique_faces.end(), faces.begin());

                m_faces = std::move(faces);
            }
        };

        std::shared_ptr<mesh> create_from_noff_file(const std::wstring& filename)
        {
            std::vector< mesh::vertex >   vertices;
            std::vector< mesh::normal >   normals;
            std::vector< mesh::face >     faces;

            mesh::progress_notifier       notifier;

            std::ifstream file(filename, std::ifstream::in);

            if (file.good())
            {
                std::string type;
                file >> type;

                uint32_t vertex_count = 0;
                uint32_t face_count = 0;
                uint32_t edge_count = 0;

                file >> vertex_count;
                file >> face_count;
                file >> edge_count;

                vertices.reserve(vertex_count);
                faces.reserve(face_count);

                for (uint32_t i = 0; i < vertex_count && file.good(); ++i)
                {
                    {
                        mesh::vertex v;
                        file >> v.x >> v.y >> v.z;
                        vertices.push_back(v);
                    }

                    {
                        mesh::normal n = {};
                        file >> n.x >> n.y >> n.z; //skip normals, we calculate them
                    }
                }

                for (uint32_t i = 0; i < face_count && file.good(); ++i)
                {
                    mesh::face  face;
                    uint32_t    face_size;

                    file >> face_size;
                    file >> face.v0 >> face.v1 >> face.v2;
                    faces.push_back(face);
                }
            }

            return std::shared_ptr<mesh>(new mesh(std::move(vertices), std::move(normals), std::move(faces), std::move(notifier)));
        };

        bool mesh_is_manifold(std::shared_ptr<mesh> mesh)
        {
            return false;
        }
    }


    class half_vertex
    {

    };

    class half_edge
    {

    };

    class half_face
    {

    };

    
}







