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
#include <mem/mem_intrusive_ptr.h>
#include <mem/mem_ref_counter.h>

#include <math/math_vector.h>
#include <math/math_graphics.h>

#define USE_SHARED_PTR

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
	
	namespace hds
	{
		class vertex;
		class half_edge;
		class face;

#if defined(USE_SHARED_PTR)		
		typedef std::shared_ptr< half_edge > half_edge_pointer;
		typedef std::shared_ptr< vertex >	 vertex_pointer;
		typedef std::shared_ptr < face >	 face_pointer;
#else
		typedef mem::intrusive_ptr< half_edge > half_edge_pointer;
		typedef mem::intrusive_ptr< vertex >	vertex_pointer;
		typedef mem::intrusive_ptr< face >		face_pointer;
#endif

		class vertex : public mem::alloc_aligned< vertex >, public mem::ref_counter<vertex>
		{
			public:

			vertex( ) :
			m_x(0.0f)
			, m_y(0.0f)
			, m_z(0.0f)
			, m_w(1.0f)
			, m_index(0)
			{}

			vertex( float x, float y, float z, float w, uint32_t index) :
			 m_x(x)
			,m_y(y)
			,m_z(z)
			,m_w(w)
			, m_index(index)
			{ }

			static size_t alignment()
			{
				return 16;
			}
		
			float	   m_x;
			float	   m_y;
			float	   m_z;
			float	   m_w;
			uint32_t m_index;

			half_edge_pointer m_incident_edge;
		};

		class face : public mem::ref_counter<face>
		{
			public:

			face( ) :
			m_incident_edge(nullptr)
			, m_index(0)
			{}

			half_edge_pointer m_incident_edge;

			uint32_t m_index;
		};

		class half_edge : public mem::ref_counter<half_edge>
		{
			public:

			half_edge()
			{}

			half_edge_pointer		m_next;
			half_edge_pointer		m_twin;	

			face_pointer	m_incident_face;
			vertex_pointer	m_incident_vertex;

			bool is_boundary() const
			{
				return false;//!m_incident_face;
			}
		};

		class mesh
		{
			public:

			typedef std::vector< half_edge_pointer >			edges_container;
			typedef std::vector< vertex_pointer >		vertex_container;
			typedef std::vector< face_pointer >		faces_container;

			typedef vertex_container::iterator					vertex_iterator;
			typedef faces_container::iterator					face_iterator;
			typedef edges_container::iterator					half_edge_iterator;

			typedef vertex_container::const_iterator			const_vertex_iterator;
			typedef faces_container::const_iterator				const_face_iterator;
			typedef edges_container::const_iterator				const_half_edge_iterator;

			mesh ( 
						const edges_container& edges,
						const vertex_container& vertices,
						const faces_container& faces
					) :
			m_incident_edges(edges)
			, m_vertices(vertices)
			, m_faces(faces)
			{}

			mesh ( 
						edges_container&& edges,
						vertex_container&& vertices,
						faces_container&& faces
					) :
			m_incident_edges( std::move(edges) )
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
				return m_incident_edges.begin();
			}

			half_edge_iterator edges_end()
			{
				return m_incident_edges.end();
			}

			const_half_edge_iterator edges_begin() const
			{
				return m_incident_edges.begin();
			}

			const_half_edge_iterator edges_end() const
			{
				return m_incident_edges.end();
			}

			class vertex_vertex_iterator : public std::iterator<
				std::input_iterator_tag, half_edge_pointer , std::ptrdiff_t,
				const  half_edge_pointer* , const half_edge_pointer& >
			{
				public:
				typedef vertex_vertex_iterator this_type;

				explicit vertex_vertex_iterator ( vertex_pointer vertex ) :
				m_incident_vertex( vertex )
				, m_current( vertex->m_incident_edge )
				, m_loop_counter(0)
				{
				
				}

				vertex_vertex_iterator( vertex_vertex_iterator&& o ) :
				m_incident_vertex(std::move(o.m_incident_vertex))
				, m_current(std::move(o.m_current))
				, m_loop_counter(std::move(o.m_loop_counter))
				{

				}

				vertex_vertex_iterator& operator=(vertex_vertex_iterator&& o)
				{
					m_incident_vertex = std::move(o.m_incident_vertex);
					m_current = std::move(o.m_current);
					m_loop_counter = std::move(o.m_loop_counter);
					return *this;
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
					m_current = m_current->m_twin->m_next;

					if ( m_current == m_incident_vertex->m_incident_edge )
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
					return  (m_loop_counter == 0 || m_current != m_incident_vertex->m_incident_edge );
				}

				bool operator==(const vertex_vertex_iterator& o ) const
				{
					return (m_current == o.m_current && m_incident_vertex == o.m_incident_vertex && m_loop_counter == o.m_loop_counter);
				}

				bool operator !=(const vertex_vertex_iterator& o ) const
				{
					return ! this->operator==(o);
				}

			private:
				vertex_pointer			m_incident_vertex;
				half_edge_pointer		m_current;
				uint32_t						m_loop_counter;
			};

			class vertex_face_iterator : public std::iterator<
				std::input_iterator_tag, half_edge_pointer , std::ptrdiff_t,
				const  half_edge_pointer* , const half_edge_pointer& >
			{
				public:
				typedef vertex_face_iterator this_type;

				explicit vertex_face_iterator ( vertex_pointer vertex ) :
				m_incident_vertex( vertex )
				, m_current( vertex->m_incident_edge )
				, m_loop_counter(0)
				{
					//if we are the boundary
					if (!m_current->m_incident_face)
					{
						m_current = m_current->m_twin->m_next;

						if ( m_current == m_incident_vertex->m_incident_edge )
						{
							++m_loop_counter;
						}
					}
				}

				vertex_face_iterator( vertex_face_iterator&& o ) :
				m_incident_vertex(std::move(o.m_incident_vertex))
				, m_current(std::move(o.m_current))
				, m_loop_counter(std::move(o.m_loop_counter))
				{

				}

				vertex_face_iterator& operator=(vertex_face_iterator&& o)
				{
					m_incident_vertex = std::move(o.m_incident_vertex);
					m_current = std::move(o.m_current);
					m_loop_counter = std::move(o.m_loop_counter);
					return *this;
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
					m_current = m_current->m_twin->m_next;
				
					if ( m_current == m_incident_vertex->m_incident_edge )
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
					return  (m_loop_counter == 0 || m_current != m_incident_vertex->m_incident_edge );
				}

				bool operator==(const vertex_face_iterator& o ) const
				{
					return (m_current == o.m_current && m_incident_vertex == o.m_incident_vertex && m_loop_counter == o.m_loop_counter);
				}

				bool operator !=(const vertex_face_iterator& o ) const
				{
					return ! this->operator==(o);
				}

			private:
				vertex_pointer			m_incident_vertex;
				half_edge_pointer		m_current;
				uint32_t				m_loop_counter;
			};

			class face_face_iterator : public std::iterator<
				std::input_iterator_tag, half_edge_pointer , std::ptrdiff_t,
				const  half_edge_pointer* , const half_edge_pointer& >
			{
				public:

				typedef face_face_iterator this_type;

				explicit face_face_iterator ( face_pointer face ) :
				m_face( face )
				, m_current( face->m_incident_edge )
				, m_loop_counter(0)
				{
					//if we are on a boundary, do one iteration
					if (! m_current->m_twin->m_incident_face )
					{
						this->operator++();
					}
				}

				face_face_iterator( face_face_iterator&& o ) :
				m_face(std::move(o.m_face))
				, m_current(std::move(o.m_current))
				, m_loop_counter(std::move(o.m_loop_counter))
			
				{

				}

				face_face_iterator& operator=(face_face_iterator&& o)
				{
					m_face = std::move(o.m_face);
					m_current = std::move(o.m_current);
					m_loop_counter = std::move(o.m_loop_counter);
					return *this;
				}

				reference operator*() const
				{	
					// return designated value
					return (m_current->m_twin);
				}

				pointer operator->() const
				{	
					// return pointer to class object
					return (&m_current->m_twin);
				}

				this_type& operator++()
				{
					// preincrement
					m_current = m_current->m_next;
				
					if ( m_current == m_face->m_incident_edge )
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
					return  (m_loop_counter == 0 || m_current != m_face->m_incident_edge );
				}

				bool operator==(const face_face_iterator& o ) const
				{
					return (m_current == o.m_current && m_face == o.m_face && m_loop_counter == o.m_loop_counter);
				}

				bool operator !=(const face_face_iterator& o ) const
				{
					return ! this->operator==(o);
				}

			private:
				face_pointer			m_face;
				half_edge_pointer		m_current;
				uint32_t						m_loop_counter;
			};

			class face_vertex_iterator : public std::iterator<
				std::input_iterator_tag, half_edge_pointer , std::ptrdiff_t,
				const  half_edge_pointer* , const half_edge_pointer& >
			{
				public:

				typedef face_vertex_iterator this_type;

				explicit face_vertex_iterator ( face_pointer face ) :
				m_face( face )
				, m_current( face->m_incident_edge )
				, m_loop_counter(0)
				{

				}

				face_vertex_iterator( face_vertex_iterator&& o ) :
				m_face(std::move(o.m_face))
				, m_current(std::move(o.m_current))
				, m_loop_counter(std::move(o.m_loop_counter))
			
				{

				}

				face_vertex_iterator& operator=(face_vertex_iterator&& o)
				{
					m_face = std::move(o.m_face);
					m_current = std::move(o.m_current);
					m_loop_counter = std::move(o.m_loop_counter);
					return *this;
				}

				reference operator*() const
				{	
					// return designated value
					return (m_current->m_twin);
				}

				pointer operator->() const
				{	
					// return pointer to class object
					return (&m_current->m_twin);
				}

				this_type& operator++()
				{
					// preincrement
					m_current = m_current->m_next;
				
					if ( m_current == m_face->m_incident_edge )
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
					return  (m_loop_counter == 0 || m_current != m_face->m_incident_edge );
				}

				bool operator==(const face_vertex_iterator& o ) const
				{
					return (m_current == o.m_current && m_face == o.m_face && m_loop_counter == o.m_loop_counter);
				}

				bool operator !=(const face_vertex_iterator& o ) const
				{
					return ! this->operator==(o);
				}

			private:
				face_pointer			m_face;
				half_edge_pointer		m_current;
				uint32_t						m_loop_counter;
			};

			vertex_vertex_iterator vertex_vertex ( uint32_t vertex_index )
			{
				return vertex_vertex_iterator( m_vertices[vertex_index] );
			}

			vertex_face_iterator vertex_face ( uint32_t vertex_index )
			{
				return vertex_face_iterator( m_vertices[vertex_index] );
			}

			face_face_iterator face_face ( uint32_t face_index )
			{
				return face_face_iterator( m_faces[face_index] );
			}

			face_vertex_iterator face_vertex ( uint32_t face_index )
			{
				return face_vertex_iterator( m_faces[face_index] );
			}

			private:
			edges_container		m_incident_edges;
			vertex_container	m_vertices;
			faces_container		m_faces;

			void throw_invariant_error() const
			{
				throw std::exception("validation check");
			}

			public:

			void check_invariants() const
			{
				std::for_each( edges_begin(), edges_end(), [&] ( const half_edge_pointer& he ) -> void
				{
					if (!he->m_twin)
					{
						throw_invariant_error();
					}

					if (!he->m_next)
					{
						throw_invariant_error();
					}

					if (he->m_twin == he )
					{
						throw_invariant_error();
					}

					if (he->m_twin->m_twin != he)
					{
						throw_invariant_error();
					}

					if (!he->is_boundary() && he->m_next->m_next->m_next != he )
					{
						throw_invariant_error();
					}

					if (!he->is_boundary() && he->m_incident_face != he->m_next->m_incident_face)
					{
						throw_invariant_error();
					}

					if (!he->m_incident_vertex)
					{
						throw_invariant_error();
					}
				});

				std::for_each( faces_begin(), faces_end(), [&] ( const face_pointer& face ) -> void
				{
					if (face->m_incident_edge->m_incident_face != face)
					{
						throw_invariant_error();
					}
				});
			}

		};

	}

	class mesh_pointer_hash : public std::unary_function< std::pair < indexed_face_set::mesh::pointer, indexed_face_set::mesh::pointer >, size_t>
	{
		public:
			result_type operator()(const argument_type& v) const
			{
				// hash _Keyval to size_t value by pseudorandomizing transform
				uint64_t v0 = std::get<0>(v);
				uint64_t v1 = std::get<1>(v);
				uint64_t value = v1 << 32 | v0;
				return (std::hash<uint64_t>()((value & (_ULLONG_MAX >> 1)) == 0 ? 0 : value));
			}
	};

	inline hds::half_edge_pointer make_half_edge()
	{
#if defined(USE_SHARED_PTR)	
		return std::make_shared<hds::half_edge> ( );
#else
		return mem::intrusive_ptr<hds::half_edge>( new hds::half_edge() );
#endif
	}

	inline hds::face_pointer	make_face()
	{
#if defined(USE_SHARED_PTR)	
		return std::make_shared<hds::face> ( );
#else
		return mem::intrusive_ptr<hds::face>( new hds::face() );
#endif
	}

	inline hds::vertex_pointer	make_vertex()
	{
#if defined(USE_SHARED_PTR)	
		return std::make_shared<hds::vertex> ( );
#else
		return mem::intrusive_ptr<hds::vertex>( new hds::vertex() );
#endif
	}

	template<typename t1> inline hds::vertex_pointer make_vertex(t1&& arg1)
	{
#if defined(USE_SHARED_PTR)	
		return std::make_shared<hds::vertex>( std::forward<t1>(arg1) );
#else
		return mem::intrusive_ptr<hds::vertex>( new hds::vertex( std::forward<t1>(arg1) ) );
#endif
	}

	template<typename t1, typename t2> inline hds::vertex_pointer make_vertex(t1&& arg1, t2&& arg2)
	{
#if defined(USE_SHARED_PTR)	
		return std::make_shared<hds::vertex>( std::forward<t1>(arg1), std::forward<t2>(arg2) );
#else
		return mem::intrusive_ptr<hds::vertex>( new hds::vertex( std::forward<t1>(arg1), std::forward<t2>(arg2) ) );
#endif
	}

	template<typename t1, typename t2, typename t3> inline hds::vertex_pointer make_vertex(t1&& arg1, t2&& arg2, t3&& arg3)
	{
#if defined(USE_SHARED_PTR)	
		return std::make_shared<hds::vertex>( std::forward<t1>(arg1), std::forward<t2>(arg2), std::forward<t3>(arg3) );
#else
		return mem::intrusive_ptr<hds::vertex>( new hds::vertex( std::forward<t1>(arg1), std::forward<t2>(arg2), std::forward<t3>(arg3) ) );
#endif
	}

	template<typename t1, typename t2, typename t3, typename t4> inline hds::vertex_pointer make_vertex(t1&& arg1, t2&& arg2, t3&& arg3, t4&& arg4)
	{
#if defined(USE_SHARED_PTR)	
		return std::make_shared<hds::vertex>( std::forward<t1>(arg1), std::forward<t2>(arg2), std::forward<t3>(arg3), std::forward<t4>(arg4)   );
#else
		return mem::intrusive_ptr<hds::vertex>( new hds::vertex( std::forward<t1>(arg1), std::forward<t2>(arg2), std::forward<t3>(arg3), std::forward<t4>(arg4) ) );
#endif
	}

	template<typename t1, typename t2, typename t3, typename t4, typename t5 > inline hds::vertex_pointer make_vertex(t1&& arg1, t2&& arg2, t3&& arg3, t4&& arg4, t5&& arg5)
	{
#if defined(USE_SHARED_PTR)	
		return std::make_shared<hds::vertex>( std::forward<t1>(arg1), std::forward<t2>(arg2), std::forward<t3>(arg3), std::forward<t4>(arg4), std::forward<t5>(arg5) );
#else
		return mem::intrusive_ptr<hds::vertex>( new hds::vertex( std::forward<t1>(arg1), std::forward<t2>(arg2), std::forward<t3>(arg3), std::forward<t4>(arg4), std::forward<t5>(arg5) ) );
#endif
	}


	std::shared_ptr< hds::mesh > create_half_mesh ( std::shared_ptr<indexed_face_set::mesh> mesh )
	{
		hds::mesh::edges_container			edges;
		hds::mesh::vertex_container		vertices;
		hds::mesh::faces_container			faces;

		typedef indexed_face_set::mesh::pointer pointer;

		auto vertex_count = mesh->vertices_end() - mesh->vertices_begin();
		
		edges.reserve ( vertex_count * 3 *3 );
		vertices.reserve( vertex_count  );
		faces.reserve( vertex_count * 2  );

		uint32_t vertex_indexer = 0;
		std::for_each(mesh->vertices_begin(), mesh->vertices_end(), [&](const indexed_face_set::mesh::vertex& vertex ) -> void 
		{
			vertices.push_back( make_vertex( vertex.x, vertex.y, vertex.z, vertex.w, vertex_indexer++ ) );
		});

		std::unordered_map < std::pair < pointer, pointer >, hds::half_edge_pointer, mesh_pointer_hash > half_edges;
		//std::map < std::pair < pointer, pointer >, hds::half_edge_pointer> half_edges;

		std::for_each ( mesh->faces_begin(), mesh->faces_end(), [&] ( const indexed_face_set::mesh::face& face ) -> void
		{
			auto edge01 = std::make_pair( face.v0, face.v1 );
			auto edge12 = std::make_pair( face.v1, face.v2 );
			auto edge20 = std::make_pair( face.v2, face.v0 );

			auto half_edge_01 = make_half_edge();
			auto half_edge_12 = make_half_edge();
			auto half_edge_20 = make_half_edge();

			half_edges.insert ( std::make_pair( std::move( edge01 ), half_edge_01 ) );
			half_edges.insert ( std::make_pair( std::move( edge12 ), half_edge_12 ) );
			half_edges.insert ( std::make_pair( std::move( edge20 ), half_edge_20 ) );

			edges.push_back(std::move( half_edge_01) );
			edges.push_back(std::move( half_edge_12) );
			edges.push_back(std::move( half_edge_20) );
		});

		uint32_t face_count = 0;
		std::for_each ( mesh->faces_begin(), mesh->faces_end(), [&] ( const indexed_face_set::mesh::face& face ) -> void
		{
			auto edge01 = std::make_pair( face.v0, face.v1 );
			auto edge12 = std::make_pair( face.v1, face.v2 );
			auto edge20 = std::make_pair( face.v2, face.v0 );

			auto half_edge_01 = half_edges[ edge01 ];
			auto half_edge_12 = half_edges[ edge12 ];
			auto half_edge_20 = half_edges[ edge20 ];

			auto h_face = make_face();

			h_face->m_incident_edge = half_edge_01;
			h_face->m_index = face_count++;

			half_edge_01->m_incident_face = h_face;
			half_edge_12->m_incident_face = h_face;
			half_edge_20->m_incident_face = h_face;

			faces.push_back(std::move( h_face ) );

			auto vertex_0 = vertices[face.v0];
			auto vertex_1 = vertices[face.v1];
			auto vertex_2 = vertices[face.v2];

			if ( !vertex_0->m_incident_edge)
			{
				vertex_0->m_incident_edge = half_edge_01;
			}

			if ( !vertex_1->m_incident_edge)
			{
				vertex_1->m_incident_edge = half_edge_12;
			}

			if ( !vertex_2->m_incident_edge)
			{
				vertex_2->m_incident_edge = half_edge_20;
			}


			half_edge_01->m_next = half_edge_12;
			half_edge_12->m_next = half_edge_20;
			half_edge_20->m_next = half_edge_01;

			half_edge_01->m_incident_vertex = vertex_1;
			half_edge_12->m_incident_vertex = vertex_2;
			half_edge_20->m_incident_vertex = vertex_0;

		});

		std::unordered_map < pointer, hds::half_edge_pointer >	boundary_start_edges;
		std::unordered_map < pointer, hds::half_edge_pointer >	boundary_end_edges;

		std::for_each( half_edges.begin(), half_edges.end(), [&] ( const std::pair < std::pair< pointer, pointer >, hds::half_edge_pointer > & he ) -> void
		{
			auto edge_pair_0 = he.first;

			auto i = std::get<0>( edge_pair_0 );
			auto j = std::get<1>( edge_pair_0 );

			auto edge_pair_1 = std::make_pair ( j, i );

			auto edge_0 = he;
			auto edge_1 = half_edges.find(edge_pair_1);

			// edge_0 is a boundary edge?
			if ( edge_1 == half_edges.end() )
			{
				auto hedge = make_half_edge();

				hedge->m_twin = edge_0.second;
				edge_0.second->m_twin = hedge;

				hedge->m_incident_vertex = vertices[i];

				//make the boundary edge to be the edge in the face and the vertex, this allows easier circulation
				vertices[j]->m_incident_edge = hedge;
				edge_0.second->m_incident_face->m_incident_edge = edge_0.second;

				boundary_start_edges[ j ] = hedge;
				boundary_end_edges[ i ] = hedge;
			}
			else
			{
				edge_0.second->m_twin = edge_1->second;
				edge_1->second->m_twin = edge_0.second;
			}
		});

		std::for_each ( boundary_end_edges.begin(), boundary_end_edges.end(), [&] ( const std::pair< pointer, hds::half_edge_pointer > & he ) -> void
		{
			he.second->m_next = boundary_start_edges[ he.first ];
		});

		std::for_each ( boundary_end_edges.begin(), boundary_end_edges.end(), [&] ( const std::pair< pointer, hds::half_edge_pointer > & he ) -> void
		{
			edges.push_back(he.second);	
		});
		
		return std::make_shared<hds::mesh> (std::move( edges), std::move(vertices), std::move(faces) );
	}

	class dart
	{
		hds::half_edge_pointer	m_incident_edge;
		bool					m_dir;

		public:

		dart( hds::half_edge_pointer edge, bool dir ) :
		m_incident_edge(edge)
		, m_dir(dir)
		{

		}

		dart( dart&& o ) : 
		m_incident_edge( std::move(o.m_incident_edge) )
		, m_dir ( std::move(o.m_dir) )
		{

		}

		dart& operator=(dart&& o )
		{
			m_incident_edge = std::move(o.m_incident_edge);
			m_dir = std::move(o.m_dir);
			return *this;
		}

		bool operator==(const dart& o ) const
		{
			return (m_incident_edge == o.m_incident_edge && m_dir == o.m_dir );
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
				m_incident_edge = m_incident_edge->m_next->m_next;
			}
			else
			{
				m_incident_edge = m_incident_edge->m_next;
			}

			m_dir = !m_dir;
			return *this;
		}

		dart& alpha2()
		{
			m_incident_edge = m_incident_edge->m_twin;
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


	auto h = geodesics::create_half_mesh( m );

	auto seconds_created_elapsed = timer.milliseconds();

	h->check_invariants();
	
	std::cout<<"first"<<std::endl;

	auto face_index = 1;
	auto vertex_index = 4;
	auto iter1 = h->face_vertex( face_index );

	for ( ; iter1.is_valid(); ++iter1 )
	{
		auto edge = *iter1;

		if (edge->m_incident_face)
		{
			//std::cout<< edge->m_incident_face->m_index << std::endl;
		}
		else
		{
			//std::cout<< "boundary " << std::endl;
		}

		std::cout<<edge->m_incident_vertex->m_index << std::endl;
	}
	

	std::cout<<"mesh loaded for "<< seconds_loaded_elapsed <<" milliseconds" << std::endl;
	std::cout<<"half_mesh created for "<< seconds_created_elapsed <<" milliseconds" << std::endl;
	
	return 0;
}

