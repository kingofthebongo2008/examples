#pragma once

#include <cstdint>
#include <vector>

namespace lscm
{
    namespace indexed_face_set
    {
        struct mesh
        {
        public:

            typedef uint32_t pointer;

            struct vertex
            {
                float x;
                float y;
                float z;
                float w;
            };

            struct normal
            {
                float nx;
                float ny;
                float nz;
                float nw;
            };

            struct face
            {
                pointer v0;
                pointer v1;
                pointer v2;
            };

            struct winged_edge
            {
                pointer v0; //start vertex
                pointer v1; //end   vertex

                pointer f0; // left face
                pointer f1; // right face

                pointer l_p; // left predeccesor
                pointer r_p; // right predeccesor

                pointer l_s; // left successor
                pointer r_s; // right successor
            };

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

            winged_edge*   get_edge(pointer p)
            {
                return &m_edges[static_cast<uint32_t> (p)];
            }

            const winged_edge*   get_edge(pointer p) const
            {
                return &m_edges[static_cast<uint32_t> (p)];
            }

            std::vector< vertex >          m_vertices;
            std::vector< normal >          m_normals;
            std::vector< face >            m_faces;
            std::vector< normal >          m_face_normals;
            std::vector< winged_edge >     m_edges;
            progress_notifier              m_notifier;

            void initialize()
            {

                auto f1 = std::async([=]
                {
                    clean_degenerate_faces();
                });

                auto f2 = std::async([&]
                {
                    f1.wait();
                    clear_vertices_not_referenced_by_faces();
                });

                auto f3 = std::async([&]
                {
                    f1.wait();
                    f2.wait();

                    calculate_pivot();

                    auto f6 = std::async([=]
                    {
                        build_face_normals();
                    });

                    f6.wait();
                });

                auto f4 = std::async([=]
                {
                    normalize_normals();
                });

                auto f5 = std::async([&]
                {
                    f2.wait();
                    build_edges();
                });

                f3.wait();
                f4.wait();
                f5.wait();

            }

            void build_edges()
            {

            }

            void build_face_normals()
            {
                std::vector< normal > face_normals(m_faces.size());

                for (uint32_t i = 0; i < face_normals.size(); ++i)
                {
                    math::float4 v0 = math::load4(&m_vertices[m_faces[i].v0]);
                    math::float4 v1 = math::load4(&m_vertices[m_faces[i].v1]);
                    math::float4 v2 = math::load4(&m_vertices[m_faces[i].v2]);

                    math::float4 n = math::cross3(math::sub(v0, v1), math::sub(v1, v2));
                    math::float4 normal = math::normalize3(n);

                    math::store4(&face_normals[i], normal);
                }
            }

            void calculate_pivot()
            {
                std::vector< mesh::vertex > vertices(m_vertices.size());
                auto vertex_size = m_vertices.size();

                double sum0 = 0.0;
                double sum1 = 0.0;
                double sum2 = 0.0;
                double sum3 = 0.0;

                std::for_each(m_vertices.begin(), m_vertices.end(), [&](const vertex& v)
                {
                    sum0 += v.x;
                    sum1 += v.y;
                    sum2 += v.z;
                    sum3 += v.w;
                }
                );

                sum0 /= vertex_size;
                sum1 /= vertex_size;
                sum2 /= vertex_size;
                sum3 = 0.0;


                std::transform(m_vertices.begin(), m_vertices.end(), vertices.begin(), [&](const vertex& v)
                {
                    vertex v_new = { static_cast<float> (v.x - sum0) , static_cast<float> (v.y - sum1) , static_cast<float> (v.z - sum2), static_cast<float> (v.w - sum3) };
                    return std::move(v_new);
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

            void normalize_normals()
            {
                std::vector< normal > normals(m_normals.size());

                std::transform(m_normals.begin(), m_normals.end(), normals.begin(), [=](normal& n0)
                {
                    math::float4 n = math::load3(&n0);
                    math::float4 n1 = math::normalize3(n);

                    normal result;

                    math::store3(&result, n1);

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

                        //face references the ith vertex, then it is used
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
                    mesh::vertex v = { 0.0f, 0.0f, 0.0f, 1.0f };
                    mesh::normal n = {};

                    file >> v.x >> v.y >> v.z;
                    file >> n.nx >> n.ny >> n.nz;

                    vertices.push_back(v);
                    normals.push_back(n);
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







