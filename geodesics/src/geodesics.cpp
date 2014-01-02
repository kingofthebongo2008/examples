// geodesics.cpp : Defines the entry point for the console application.
//
#include "precompiled.h"

#include <algorithm>
#include <cstdint>
#include <fstream>


#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>



#include <utility>

#include <sys/sys_profile_timer.h>

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

            struct face : public mem::alloc_aligned< face >
            {
                pointer v0;
                pointer v1;
                pointer v2;

				face() :
				v0(0)
				, v1(0)
				, v2(0)
				{}

				face( pointer _v0, pointer _v1, pointer _v2 ) :
				v0(_v0)
				, v1(_v1)
				, v2(_v2)
				{}
            };

			typedef std::vector< vertex, mem::allocator<vertex> > vertex_container;
			typedef std::vector< normal, mem::allocator<normal> > normal_container;
			typedef std::vector< face, mem::allocator<face> >	  face_container;

            mesh (
                    const vertex_container&			vertices,
                    const normal_container&			normals,
                    const face_container&			faces
                 )  : m_vertices(vertices)
                    , m_normals(normals)
                    , m_faces(faces)
            {
                initialize();
            }

            mesh (
                    vertex_container &&       vertices,
                    normal_container &&       normals,
                    face_container   &&		  faces
                 ) : 
                      m_vertices(std::move( vertices ) ) 
                    , m_normals(std::move (normals ) )
                    , m_faces( std::move(faces) )
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

			vertex_container::const_iterator vertices_begin() const
			{
				return m_vertices.begin();
			}

			vertex_container::const_iterator vertices_end() const
			{
				return m_vertices.end();
			}

			face_container::const_iterator faces_begin() const
			{
				return m_faces.begin();
			}

			face_container::const_iterator faces_end() const
			{
				return m_faces.end();
			}

            private:

            vertex_container			m_vertices;
            normal_container			m_normals;
            face_container				m_faces;
            normal_container			m_face_normals;

            void initialize()
            {
				//calculate_pivot();
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
                    math::float4 v0 = math::load4(&m_vertices[ m_faces[i].v0 ] );
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

                std::for_each ( m_vertices.begin(), m_vertices.end(), [&] ( const vertex& v ) -> void
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
            mesh::face_container     faces;

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

            return std::shared_ptr<mesh> ( new mesh( std::move(vertices), std::move(normals), std::move( faces ) ) );
        };

        bool mesh_is_manifold( std::shared_ptr<mesh> mesh )
        {
            return false;
        }
    }
	
	class half_vertex;
	class half_edge;
	class half_face;

	class half_vertex : public mem::alloc_aligned< half_vertex >
    {
		public:

		half_vertex( ) :
		m_x(0.0f)
		, m_y(0.0f)
		, m_z(0.0f)
		, m_w(1.0f)
		{}

		half_vertex( float x, float y, float z, float w ) :
		 m_x(x)
		,m_y(y)
		,m_z(z)
		,m_w(w)
		{ }
		
		float	   m_x;
		float	   m_y;
		float	   m_z;
		float	   m_w;

		std::shared_ptr<half_edge> m_edge;
    };

    class half_face
    {
		public:

		half_face( ) :
		m_edge(nullptr)
		{}

		std::shared_ptr<half_edge> m_edge;
    };

	class half_edge
    {
		public:

		half_edge()
		{}

		std::shared_ptr<half_edge>		m_next;
		std::shared_ptr<half_edge>		m_previous;

		std::shared_ptr<half_edge>		m_opposite;	

		std::shared_ptr<half_face>		m_face;
		std::shared_ptr<half_vertex>	m_vertex;

		bool is_boundary() const
		{
			return !m_face;
		}
    };

	class half_mesh
	{
		public:

		typedef std::vector< std::shared_ptr<half_edge> >	edges_container;
		typedef std::vector< std::shared_ptr<half_vertex> >	vertex_container;
		typedef std::vector< std::shared_ptr<half_face> >	faces_container;

		typedef vertex_container::iterator			vertex_iterator;
		typedef faces_container::iterator			face_iterator;
		typedef edges_container::iterator			half_edge_iterator;

		typedef vertex_container::const_iterator	const_vertex_iterator;
		typedef faces_container::const_iterator		const_face_iterator;
		typedef edges_container::const_iterator		const_half_edge_iterator;

		half_mesh ( 
					const edges_container& edges,
					const vertex_container& vertices,
					const faces_container& faces
				) :
		m_edges(edges)
		, m_vertices(vertices)
		, m_faces(faces)
		{}

		half_mesh ( 
					edges_container&& edges,
					vertex_container&& vertices,
					faces_container&& faces
				) :
		m_edges( std::move(edges) )
		, m_vertices(std::move ( vertices ))
		, m_faces(std::move( faces ) )
		{}


		vertex_iterator vertices_begin()
		{
			return m_vertices.begin();
		}

		vertex_iterator vertices_end()
		{
			return m_vertices.end();
		}

		const_vertex_iterator vertices_begin() const
		{
			return m_vertices.begin();
		}

		const_vertex_iterator vertices_end() const
		{
			return m_vertices.end();
		}

		face_iterator faces_begin()
		{
			return m_faces.begin();
		}

		face_iterator faces_end()
		{
			return m_faces.end();
		}

		const_face_iterator faces_begin() const
		{
			return m_faces.begin();
		}

		const_face_iterator faces_end() const
		{
			return m_faces.end();
		}

		half_edge_iterator edges_begin()
		{
			return m_edges.begin();
		}

		half_edge_iterator edges_end()
		{
			return m_edges.end();
		}

		const_half_edge_iterator edges_begin() const
		{
			return m_edges.begin();
		}

		const_half_edge_iterator edges_end() const
		{
			return m_edges.end();
		}

		void check_invariants() const
		{
			std::for_each( edges_begin(), edges_end(), [] ( const std::shared_ptr<half_edge>& he ) -> void
			{
				if (!he->m_previous)
				{
					throw std::exception("validation check");
				}

				if (!he->m_opposite)
				{
					throw std::exception("validation check");
				}

				if (!he->m_next)
				{
					throw std::exception("validation check");
				}

				if (he->m_opposite == he )
				{
					throw std::exception("validation check");
				}

				

				if (he->m_opposite->m_opposite != he)
				{
					throw std::exception("validation check");
				}

				if (he->m_next->m_previous != he )
				{
					throw std::exception("validation check");
				}

				if (he->m_previous->m_next != he )
				{
					throw std::exception("validation check");
				}

				if (he->m_face != he->m_next->m_face)
				{
					throw std::exception("validation check");
				}

				if (!he->m_vertex)
				{
					throw std::exception("validation check");
				}
			});

			std::for_each( vertices_begin(), vertices_end(), [] ( const std::shared_ptr<half_vertex>& vertex ) -> void
			{
				if (vertex->m_edge->m_vertex != vertex)
				{
					throw std::exception("validation check");
				}
			});

			std::for_each( faces_begin(), faces_end(), [] ( const std::shared_ptr<half_face>& face ) -> void
			{
				if (face->m_edge->m_face != face)
				{
					throw std::exception("validation check");
				}
			});
		}


		class vertex_half_edge_iterator : public std::iterator<
			std::input_iterator_tag, std::shared_ptr<half_edge> , std::ptrdiff_t,
			const  std::shared_ptr<half_edge>* , const std::shared_ptr<half_edge>& >
		{
			public:
			typedef vertex_half_edge_iterator this_type;

			explicit vertex_half_edge_iterator ( std::shared_ptr<half_vertex> vertex ) :
			m_vertex( vertex )
			, m_current( vertex->m_edge )
			, m_loop_counter(0)
			{
				
			}

			reference operator*() const
			{	
				// return designated value
				return (m_current);
			}

			pointer operator->() const
			{	
				// return pointer to class object
				return (&m_current);
			}

			this_type& operator++()
			{
				// preincrement
				m_current = m_current->m_opposite->m_next;

				if ( m_current == m_vertex->m_edge )
				{
					++m_loop_counter;
				}

				return (*this);
			}

			this_type operator++(int)
			{
				// postincrement
				this_type tmp = *this;
				++*this;
				return (tmp);
			}

			bool is_valid() const
			{
				return  (m_loop_counter == 0 || m_current != m_vertex->m_edge );
			}

		private:
			std::shared_ptr<half_vertex>	m_vertex;
			std::shared_ptr<half_edge>		m_current;
			uint32_t						m_loop_counter;
		};

		vertex_half_edge_iterator vertex_half_edge ( uint32_t vertex_index )
		{
			return vertex_half_edge_iterator( m_vertices[vertex_index] );
		}

		public:
		edges_container		m_edges;
		vertex_container	m_vertices;
		faces_container		m_faces;
	};

	std::shared_ptr< half_mesh > create_half_mesh ( std::shared_ptr<indexed_face_set::mesh> mesh )
	{
		using namespace geodesics::indexed_face_set;	

		half_mesh::edges_container		edges;
		half_mesh::vertex_container		vertices;
		half_mesh::faces_container		faces;

		auto vertex_count = mesh->vertices_end() - mesh->vertices_begin();
		
		edges.reserve ( vertex_count * 3 *2);
		vertices.reserve( vertex_count  );
		faces.reserve( vertex_count * 2  );
		
		std::map < std::pair < mesh::pointer , mesh::pointer >, std::shared_ptr<half_edge> > half_edges;

		std::unordered_map < mesh::pointer, std::shared_ptr<half_vertex> > half_vertices_set;

		std::for_each ( mesh->faces_begin(), mesh->faces_end(), [&] ( const indexed_face_set::mesh::face& face ) -> void
		{
			auto h_face = std::make_shared<half_face>();

			faces.push_back( h_face );

			auto edge01 = std::make_pair( face.v0, face.v1 );
			auto edge12 = std::make_pair( face.v1, face.v2 );
			auto edge20 = std::make_pair( face.v2, face.v0 );

			auto edge10 = std::make_pair( face.v1, face.v0 );
			auto edge21 = std::make_pair( face.v2, face.v1 );
			auto edge02 = std::make_pair( face.v0, face.v2 );

			auto half_edge_01 = std::make_shared<half_edge>();
			auto half_edge_12 = std::make_shared<half_edge>();
			auto half_edge_20 = std::make_shared<half_edge>();
			
			half_edges[ edge01 ] = half_edge_01;
			half_edge_01-> m_face = h_face;

			half_edges[ edge12 ] = half_edge_12;
			half_edge_12-> m_face = h_face;

			half_edges[ edge20 ] = half_edge_20;
			half_edge_20-> m_face = h_face;

			//next edges
			half_edge_01->m_previous = half_edges[ edge12 ];
			half_edge_12->m_previous = half_edges[ edge20 ];
			half_edge_20->m_previous = half_edges[ edge01 ];

			//previous edges
			half_edge_01->m_next = half_edges[ edge20 ];
			half_edge_12->m_next = half_edges[ edge01 ];
			half_edge_20->m_next = half_edges[ edge12 ];

			auto half_edge_10 = half_edges.find( edge10 );
			auto half_edge_21 = half_edges.find( edge21 );
			auto half_edge_02 = half_edges.find( edge02 );

			//opposite edges
			if ( half_edge_10  != half_edges.end() )
			{
				half_edge_10->second->m_opposite = half_edge_01;
				half_edge_01->m_opposite = half_edge_10->second;
			}

			//opposite edges
			if ( half_edge_21 != half_edges.end() )
			{
				half_edge_21->second->m_opposite = half_edge_12;
				half_edge_12->m_opposite = half_edge_21->second;
			}

			//opposite edges
			if ( half_edge_02 != half_edges.end() )
			{
				half_edge_02->second->m_opposite = half_edge_20;
				half_edge_20->m_opposite = half_edge_02->second;
			}

			h_face->m_edge = half_edge_01;

			edges.push_back(half_edge_01);
			edges.push_back(half_edge_12);
			edges.push_back(half_edge_20);

			std::shared_ptr< half_vertex > half_vertex_v1;
			std::shared_ptr< half_vertex > half_vertex_v2;
			std::shared_ptr< half_vertex > half_vertex_v0;

			//connect to the vertex the half edge with the second vertex, also called incident vertex
			if ( half_vertices_set.find( face.v0 ) == half_vertices_set.end() )
			{
				auto vertex = mesh->get_vertex( face.v0 );
				half_vertex_v0 = std::make_shared<half_vertex>( vertex->x, vertex->y, vertex->z, vertex->w );
				half_vertices_set[face.v0] = half_vertex_v0;
				half_vertex_v0->m_edge = half_edge_01;
			}
			else
			{
				half_vertex_v0 = half_vertices_set[face.v0];
			}

			half_edge_01->m_vertex = half_vertex_v0;

			if ( half_vertices_set.find( face.v1 ) == half_vertices_set.end() )
			{
				auto vertex = mesh->get_vertex( face.v1 );
				half_vertex_v1 = std::make_shared<half_vertex>( vertex->x, vertex->y, vertex->z, vertex->w );
				half_vertices_set[face.v1] = half_vertex_v1;
				half_vertex_v1->m_edge = half_edge_12;
			}
			else
			{
				half_vertex_v1 = half_vertices_set[face.v1];
			}
			half_edge_12->m_vertex = half_vertex_v1;

			if ( half_vertices_set.find( face.v2 ) == half_vertices_set.end() )
			{
				auto vertex = mesh->get_vertex( face.v2 );
				half_vertex_v2 = std::make_shared<half_vertex>( vertex->x, vertex->y, vertex->z, vertex->w );
				half_vertices_set[face.v2] = half_vertex_v2;
				half_vertex_v2->m_edge = half_edge_20;
			}
			else
			{
				half_vertex_v2 = half_vertices_set[face.v2];
			}
			
			half_edge_20->m_vertex = half_vertex_v2;

		});

		std::for_each( half_vertices_set.begin(), half_vertices_set.end(), [&] ( const std::pair< mesh::pointer, std::shared_ptr<half_vertex> > & pair ) -> void
		{
			vertices.push_back( pair.second );
		});

		std::unordered_map < mesh::pointer , std::shared_ptr<half_edge> > boundary_start_edges;
		std::unordered_map < mesh::pointer , std::shared_ptr<half_edge> >	boundary_end_edges;

		//create boundary edges with empty faces and mark the boundary vertices to point to them
		//see dennis zorin presentations
		
		std::for_each ( mesh->faces_begin(), mesh->faces_end(), [&] ( const indexed_face_set::mesh::face& face ) -> void
		{
			auto edge01 = std::make_pair( face.v0, face.v1 );
			auto edge12 = std::make_pair( face.v1, face.v2 );
			auto edge20 = std::make_pair( face.v2, face.v0 );

			auto edge10 = std::make_pair( face.v1, face.v0 );
			auto edge21 = std::make_pair( face.v2, face.v1 );
			auto edge02 = std::make_pair( face.v0, face.v2 );

			if ( half_edges.find( edge10 ) == half_edges.end() )
			{
				auto half_edge_10 = std::make_shared<half_edge>();
				auto half_edge_01 = half_edges[edge01];

				half_edge_10->m_opposite = half_edge_01;
				half_edge_01->m_opposite = half_edge_10;

				boundary_start_edges[ face.v1 ] = half_edge_10;
				boundary_end_edges  [ face.v0 ] = half_edge_10;
			}

			if ( half_edges.find( edge21 ) == half_edges.end() )
			{
				auto half_edge_21 = std::make_shared<half_edge>();
				auto half_edge_12 = half_edges[edge12];
				half_edge_21->m_opposite = half_edge_12;
				half_edge_12->m_opposite = half_edge_21;
				boundary_start_edges[ face.v2 ] = half_edge_21;
				boundary_end_edges	[ face.v1 ] = half_edge_21;
			}

			if ( half_edges.find( edge02 ) == half_edges.end() )
			{
				auto half_edge_02 = std::make_shared<half_edge>();
				auto half_edge_20 = half_edges[edge20];
				half_edge_02->m_opposite = half_edge_20;
				half_edge_20->m_opposite = half_edge_02;
				boundary_start_edges[ face.v0 ]   = half_edge_02;
				boundary_end_edges	[ face.v2   ] = half_edge_02;
			}
		});
		

		std::for_each ( boundary_end_edges.begin(), boundary_end_edges.end(), [&] ( const std::pair< mesh::pointer, std::shared_ptr<half_edge> > & he ) -> void
		{
			he.second->m_next = boundary_start_edges[ he.first ];
			boundary_start_edges[ he.first ] -> m_previous = he.second;
		});

		std::for_each ( boundary_start_edges.begin(), boundary_start_edges.end(), [&] ( const std::pair< mesh::pointer, std::shared_ptr<half_edge> > & he ) -> void
		{
			auto half_vertex = half_vertices_set[ he.first ];
			he.second->m_vertex = half_vertex;
			
			//point to the internal of the half edges
			he.second->m_opposite->m_vertex = half_vertex;
			half_vertex->m_edge = he.second->m_opposite;
		});

		std::for_each ( boundary_end_edges.begin(), boundary_end_edges.end(), [&] ( const std::pair< mesh::pointer, std::shared_ptr<half_edge> > & he ) -> void
		{
			edges.push_back(he.second);	
		});

		return std::make_shared<half_mesh> (std::move( edges), std::move(vertices), std::move(faces) );
	}

	class dart
	{
		std::shared_ptr<half_edge> m_edge;
		bool					   m_dir;

		public:

		dart( std::shared_ptr<half_edge> edge, bool dir ) :
		m_edge(edge)
		, m_dir(dir)
		{

		}

		dart( dart&& o ) : 
		m_edge( std::move(o.m_edge) )
		, m_dir ( std::move(o.m_dir) )
		{

		}

		dart& operator=(dart&& o )
		{
			m_edge = std::move(o.m_edge);
			m_dir = std::move(o.m_dir);
			return *this;
		}

		bool operator==(const dart& o ) const
		{
			return (m_edge == o.m_edge && m_dir == o.m_dir );
		}

		bool operator!=(const dart&o ) const
		{
			return !this->operator==(o);
		}

		dart& alpha0()
		{
			m_dir = !m_dir;
			return *this;
		}

		dart& alpha1()
		{
			if (m_dir)
			{
				m_edge = m_edge->m_previous;
			}
			else
			{
				m_edge = m_edge->m_next;
			}

			m_dir = !m_dir;
			return *this;
		}

		dart& alpha2()
		{
			m_edge = m_edge->m_opposite;
			m_dir = !m_dir;
			return *this;
		}
	};
}

int wmain(int argc, wchar_t* argv[])
{
	argc;
	argv;
	using namespace geodesics::indexed_face_set;

	sys::profile_timer timer;
	auto m = create_from_noff_file(L"../media/meshes/bunny_nf4000.noff");
	
	auto seconds_loaded_elapsed = timer.milliseconds();
	timer.reset();

	mesh::vertex_container   vertices;
	mesh::normal_container   normals;
    mesh::face_container     faces;

	
	
	vertices.push_back ( mesh::vertex( 0.0f, 0.0f, 0.0f, 1.0f ) );
	vertices.push_back ( mesh::vertex( 1.0f, 0.0f, 0.0f, 1.0f ) );
	vertices.push_back ( mesh::vertex( 1.0f, 1.0f, 0.0f, 1.0f ) );
	vertices.push_back ( mesh::vertex( 0.0f, 1.0f, 0.0f, 1.0f ) );
	vertices.push_back ( mesh::vertex( 0.75f, 0.25f, 0.0f, 1.0f ) );
	vertices.push_back ( mesh::vertex( 0.25f, 0.75f, 0.0f, 1.0f ) );

	normals.push_back( mesh::normal( 0.0f, -1.0f, 0.0f, 0.0f ) );
	normals.push_back( mesh::normal( 0.0f, -1.0f, 0.0f, 0.0f ) );
	normals.push_back( mesh::normal( 0.0f, -1.0f, 0.0f, 0.0f ) );
	normals.push_back( mesh::normal( 0.0f, -1.0f, 0.0f, 0.0f ) );
	normals.push_back( mesh::normal( 0.0f, -1.0f, 0.0f, 0.0f ) );
	normals.push_back( mesh::normal( 0.0f, -1.0f, 0.0f, 0.0f ) );

	faces.push_back ( mesh::face( 0, 1, 4 ) );
	faces.push_back ( mesh::face( 1, 2, 4 ) );
	faces.push_back ( mesh::face( 0, 4, 2 ) );

	faces.push_back ( mesh::face( 0, 2, 5 ) );
	//faces.push_back ( mesh::face( 0, 5, 3 ) );
	faces.push_back ( mesh::face( 5, 2, 3 ) );

	
	/*
	vertices.push_back ( mesh::vertex( 0.0f, 0.0f, 0.0f, 1.0f ) );
	vertices.push_back ( mesh::vertex( 1.0f, 0.0f, 0.0f, 1.0f ) );
	vertices.push_back ( mesh::vertex( 0.0f, 1.0f, 0.0f, 1.0f ) );

	normals.push_back( mesh::normal( 0.0f, -1.0f, 0.0f, 0.0f ) );
	normals.push_back( mesh::normal( 0.0f, -1.0f, 0.0f, 0.0f ) );
	normals.push_back( mesh::normal( 0.0f, -1.0f, 0.0f, 0.0f ) );
	
	faces.push_back ( mesh::face( 0, 1, 2 ) );
	
	*/
	auto m1 = std::make_shared<mesh> ( vertices, normals, faces );


	auto h = geodesics::create_half_mesh( m1 );

	h->check_invariants();

	auto end   = h->m_vertices[2]->m_edge;
	auto iter  = end;
	
	do
	{
		auto edge = iter;

		std::cout<< edge->m_vertex->m_x << ", " << edge->m_vertex->m_y << ", " << edge->m_vertex->m_z << std::endl;

		iter = iter->m_opposite->m_next;
	}
	while ( iter != end );

	/*
	for ( ; iter.is_valid(); ++iter )
	{
		auto edge = *iter;

		std::cout<< edge->m_vertex->m_x << ", " << edge->m_vertex->m_y << ", " << edge->m_vertex->m_z << std::endl;
	}
	*/

	auto seconds_created_elapsed = timer.milliseconds();

	std::cout<<"mesh loaded for "<< seconds_loaded_elapsed <<" milliseconds" << std::endl;
	std::cout<<"half_mesh created for "<< seconds_created_elapsed <<" milliseconds" << std::endl;
	
   
	return 0;
}

