// geodesics.cpp : Defines the entry point for the console application.
//
#include "precompiled.h"

#include <algorithm>
#include <cstdint>
#include <fstream>


#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

#include <utility>

#include <mem/mem_alloc_aligned.h>
#include <mem/mem_alloc_std.h>
#include <math/math_vector.h>
#include <math/math_graphics.h>

namespace geodesics
{
    class renderable_mesh;

    namespace indexed_face_set
    {
        class mesh
        {
            public:

            typedef uint32_t pointer;

            struct vertex : public mem::alloc_aligned< vertex >
            {
				public:

				static size_t alignment()
				{
					return 16;
				}

				public:

                float x;
                float y;
                float z;
                float w;

				vertex( float _x, float _y, float _z, float _w ) :
				x(_x)
				, y(_y)
				, z(_z)
				, w(_w)
				{}

				vertex():
				x(0.0f)
				, y(0.0f)
				, z(0.0f)
				, w(1.0f)
				{}
			};

			struct normal : public mem::alloc_aligned< normal >
            {
                float nx;
                float ny;
                float nz;
                float nw;

				normal( float _nx, float _ny, float _nz, float _nw ) :
				nx(_nx)
				, ny(_ny)
				, nz(_nz)
				, nw(_nw)
				{}

				normal() :
				nx(0.0f)
				, ny(0.0f)
				, nz(0.0f)
				, nw(0.0f)
				{}
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

			typedef std::vector< vertex, mem::allocator<vertex> > vertex_container;
			typedef std::vector< normal, mem::allocator<normal> > normal_container;

            mesh (
                    const vertex_container&			vertices,
                    const normal_container&			normals,
                    const std::vector< face >&		faces,
                    const progress_notifier&		notifier
                 )  : m_vertices(vertices)
                    , m_normals(normals)
                    , m_faces(faces)
                    , m_notifier(notifier)
            {
                initialize();
            }

            mesh (
                    vertex_container &&     vertices,
                    normal_container &&     normals,
                    std::vector< face >   &&     faces,
                    progress_notifier     &&    notifier
                 ) : 
                      m_vertices(std::move( vertices ) ) 
                    , m_normals(std::move (normals ) )
                    , m_faces( std::move(faces) )
                    , m_notifier( std::move(notifier) )
            {
                initialize();
            }

            vertex* get_vertex( pointer p )
            {
                return &m_vertices[ static_cast<uint32_t> ( p ) ];
            }

            const vertex* get_vertex( pointer p ) const
            {
                return &m_vertices[ static_cast<uint32_t> ( p ) ];
            }

            face*   get_face( pointer p )
            {
                return &m_faces[ static_cast<uint32_t> ( p ) ];
            }

            const face*   get_face( pointer p ) const
            {
                return &m_faces[ static_cast<uint32_t> ( p ) ];
            }

            winged_edge*   get_edge( pointer p )
            {
                return &m_edges[ static_cast<uint32_t> ( p ) ];
            }

            const winged_edge*   get_edge( pointer p ) const
            {
                return &m_edges[ static_cast<uint32_t> ( p                     ) ];
            }

            private:

            vertex_container			m_vertices;
            normal_container			m_normals;
            std::vector< face >			m_faces;
            normal_container			m_face_normals;
            std::vector< winged_edge >	m_edges;
            progress_notifier           m_notifier;

            void initialize()
            {
				calculate_pivot();
				build_face_normals();
				normalize_normals();
				build_edges();
            }

            void build_edges()
            {

            }

            void build_face_normals()
            {
                normal_container face_normals( m_faces.size() );

                for (uint32_t i = 0; i < face_normals.size(); ++i)
                {
					auto address = &m_vertices[ m_faces[i].v0 ] ;
					auto address1 = &m_vertices[ 0 ] ;

                    math::float4 v0 = math::load4( address );
                    math::float4 v1 = math::load4(&m_vertices[ m_faces[i].v1 ] );
                    math::float4 v2 = math::load4(&m_vertices[ m_faces[i].v2 ] );

                    math::float4 n = math::cross3 ( math::sub ( v0, v1 ), math::sub ( v1, v2 ) );
                    math::float4 normal = math::normalize3 ( n );

                    math::store4( &face_normals[i], normal );
                }
            }

            void calculate_pivot()
            {
                auto vertices = vertex_container( m_vertices.size() ) ;
                auto vertex_size = m_vertices.size();

                double sum0 = 0.0;
                double sum1 = 0.0;
                double sum2 = 0.0;
                double sum3 = 0.0;

                std::for_each ( m_vertices.begin(), m_vertices.end(), [&] ( const vertex& v )
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
                sum3  = 0.0;


                std::transform ( m_vertices.begin(), m_vertices.end(), vertices.begin(),    [&] ( const vertex& v ) -> vertex
                {
                    auto v_new = mesh::vertex ( static_cast<float> ( v.x - sum0 ) , static_cast<float> ( v.y - sum1 ) , static_cast<float> ( v.z - sum2 ), static_cast<float> ( v.w - sum3 ) );
                    return v_new;
                }
                );

                m_vertices = std::move(vertices);
            }

            void normalize_normals()
            {
                auto normals = normal_container( m_normals.size() ) ;

                std::transform( m_normals.begin(), m_normals.end(), normals.begin(),  [=]( const normal& n0 ) -> normal
                {
                    math::float4 n = math::load3(&n0);
                    math::float4 n1 = math::normalize3(n);

                    normal result;

                    math::store3( &result, n1 );

                    return result;
                });

                m_normals = std::move(normals);
            }
        };

        std::shared_ptr<mesh> create_from_noff_file( const std::wstring& filename )
        {
            mesh::vertex_container   vertices;
            mesh::normal_container   normals;
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

                for ( uint32_t i = 0; i < vertex_count && file.good(); ++i )
                {
                    auto v = mesh::vertex ( 0.0f, 0.0f, 0.0f, 1.0f );
                    mesh::normal n;

                    file >> v.x >> v.y >> v.z;
                    file >> n.nx >> n.ny >> n.nz;

                    vertices.push_back ( v );
                    normals.push_back ( n );
                }

                for ( uint32_t i = 0; i < face_count && file.good(); ++i )
                {
                    mesh::face  face;
                    uint32_t    face_size;

                    file >> face_size;
                    file >> face.v0 >> face.v1 >> face.v2;
                    faces.push_back ( face );
                }
            }

            return std::shared_ptr<mesh> ( new mesh( std::move(vertices), std::move(normals), std::move( faces ), std::move(notifier) ) );
        };


        bool mesh_is_manifold( std::shared_ptr<mesh> mesh )
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

int wmain(int argc, wchar_t* argv[])
{
	argc;
	argv;
	using namespace geodesics::indexed_face_set;

	auto m = create_from_noff_file(L"../media/meshes/bunny_nf4000.noff");

	
    
	return 0;
}

