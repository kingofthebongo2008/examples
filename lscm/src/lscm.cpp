// lscm.cpp : Defines the entry point for the console application.
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


#include <math/math_vector.h>


#include <gx/gx_application.h>

#include <io/io_keyboard.h>
#include <io/io_mouse.h>

#include <d3d11/dxgi_helpers.h>
#include <d2d/d2d_helpers.h>
#include <d2d/dwrite_helpers.h>

#include <gx/shaders/gx_shader_copy_texture.h>
#include <gx/shaders/gx_shader_full_screen.h>

namespace lscm
{
    namespace indexed_face_set
    {
        class mesh
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


            mesh (
                    const std::vector< vertex >&   vertices,
                    const std::vector< normal >&   normals,
                    const std::vector< face >&     faces,
                    const progress_notifier&       notifier
                 )  : m_vertices(vertices)
                    , m_normals(normals)
                    , m_faces(faces)
                    , m_notifier(notifier)
            {
                clean_degenerate_faces();
                clean_duplicate_faces();
                clear_vertices_not_referenced_by_faces();
                normalize_normals();
                build_edges();
                build_face_normals();
            }

            mesh (
                    std::vector< vertex > &&     vertices,
                    std::vector< normal > &&     normals,
                    std::vector< face >   &&     faces,
                    progress_notifier     &&    notifier
                 ) : 
                      m_vertices(std::move( vertices ) ) 
                    , m_normals(std::move (normals ) )
                    , m_faces( std::move(faces) )
                    , m_notifier( std::move(notifier) )
            {
                clean_degenerate_faces();
                clean_duplicate_faces();
                clear_vertices_not_referenced_by_faces();
                normalize_normals();
                build_edges();
                build_face_normals();
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
                return &m_edges[ static_cast<uint32_t> ( p ) ];
            }

            private:

            std::vector< vertex >          m_vertices;
            std::vector< normal >          m_normals;
            std::vector< face >            m_faces;
            std::vector< normal >          m_face_normals;
            std::vector< winged_edge >     m_edges;
            progress_notifier              m_notifier;

            void build_edges()
            {

            }

            void build_face_normals()
            {
                std::vector< normal > face_normals( m_faces.size() );

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

            void clean_degenerate_faces()
            {
                std::vector< face > faces( m_faces.size() ) ;
                
                auto last = std::copy_if ( m_faces.begin(), m_faces.end(), faces.begin(),  [ =  ] ( const face& f )
                {
                    return ( f.v0 != f.v1 && f.v0 != f.v2 && f.v1 != f.v2 ) ;
                });

                faces.resize ( std::distance( faces.begin(), last ) );
                
                m_faces = std::move( faces );
            }

            void normalize_normals()
            {
                std::vector< normal > normals( m_normals.size() ) ;

                std::transform( m_normals.begin(), m_normals.end(), normals.begin(), [=]( normal& n0 ) 
                {
                    math::float4 n = math::load3(&n0);
                    math::float4 n1 = math::normalize3(n);

                    normal result;

                    math::store3( &result, n1 );

                    return result;
                });

                m_normals = std::move(normals);
            }

            void clear_vertices_not_referenced_by_faces()
            {
                std::vector< mesh::vertex > vertices( m_vertices.size() ) ;

                uint32_t j = 0;

                for( uint32_t i = 0; i < m_vertices.size(); ++i)
                {
                    for (uint32_t k = 0; k < m_faces.size(); ++k )
                    {
                        const face& f = m_faces[k];

                        //face references the ith vertex, then it is used
                        if (  f.v0 == i || f.v1 == i || f.v2 == i )
                        {
                            vertices[j] = m_vertices[i];
                            ++j;
                            break;
                        }
                    }
                }
                
                vertices.resize ( j );
                m_vertices = std::move( vertices );
            }

            void clean_duplicate_faces()
            {
                struct equal_faces
                {
                    struct hash_function
                    {
                        size_t operator() ( const face& f ) const 
                        {
                            return ( ( ( size_t ) f.v0 ) << 42UL ) | ( ( ( size_t ) f.v1 ) << 21UL ) | f.v2 ;
                        }
                    };

                    static void sort( uint32_t* f )
                    {
                        uint32_t n = 3;

                        do
                        {
                            uint32_t new_n = 0;

                            for (uint32_t i = 1;  i <= n-1; ++i )
                            {
                                if ( f[i-1] > f[i] )
                                {
                                    std::swap ( f[i-1], f[i] );
                                    new_n = i;
                                }
                            }

                            n = new_n;
                        }
                        while ( n > 0 );
                    }

                    bool operator()( const face& f0, const face& f1 ) const
                    {
                        uint32_t f_0[3] = { f0.v0, f0.v1, f0.v2 };
                        uint32_t f_1[3] = { f1.v0, f1.v1, f1.v2 };

                        sort (&f_0[0]);
                        sort (&f_1[0]);

                        uint32_t difference [3] = { f_0[0] - f_1[0] , f_0[1] - f_1[1] , f_0[2] - f_1[2]  };
                        return difference[0] == 0 && difference[1] == 0 && difference[2] == 0;
                    }
                };

                std::unordered_set< face, equal_faces::hash_function, equal_faces > unique_faces;

                std::for_each ( m_faces.begin(), m_faces.end(), [&] ( const face& f )
                {
                    face f0 = f;

                    equal_faces::sort(&f0.v0);

                    if ( unique_faces.find( f0 ) == unique_faces.end() )
                    {
                        unique_faces.insert(f0);
                    }
                });

                std::vector< mesh::face> faces;
                faces.resize ( unique_faces.size() );

                std::copy( unique_faces.begin(), unique_faces.end(), faces.begin() );

                m_faces = std::move( faces );
            }
        };

        std::shared_ptr<mesh> create_from_noff_file( const std::wstring& filename )
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

                for ( uint32_t i = 0; i < vertex_count && file.good(); ++i )
                {
                    mesh::vertex v = { 0.0f, 0.0f, 0.0f, 1.0f};
                    mesh::normal n = {};

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

class windowed_with_input_application : public gx::application
{
    typedef gx::application base;

    public:

    windowed_with_input_application( const wchar_t* window_title  ) : base ( window_title)
        , m_mouse_state(window_width( get_window() ), window_height( get_window() ) )
    {
        using namespace os::windows;    
  
        throw_if_failed< com_exception > ( DirectInput8Create ( GetModuleHandle(nullptr), DIRECTINPUT_VERSION, IID_IDirectInput8, reinterpret_cast<void**> (&m_direct_input), nullptr ) );

        throw_if_failed< com_exception > ( m_direct_input->CreateDevice( GUID_SysKeyboard, &m_keyboard, nullptr) );
        throw_if_failed< com_exception > ( m_keyboard->SetDataFormat( &c_dfDIKeyboard ) );
        throw_if_failed< com_exception > ( m_keyboard->SetCooperativeLevel( this->get_window(), DISCL_BACKGROUND | DISCL_NONEXCLUSIVE )  );

        throw_if_failed< com_exception > ( m_direct_input->CreateDevice( GUID_SysMouse, &m_mouse, nullptr) );
        throw_if_failed< com_exception > ( m_mouse->SetDataFormat( &c_dfDIMouse2  ) );
        throw_if_failed< com_exception > ( m_mouse->SetCooperativeLevel( this->get_window(), DISCL_BACKGROUND | DISCL_NONEXCLUSIVE )  );


        ShowCursor(false);
    }

    protected:

    io::keyboard_state get_keyboard_state() const
    {
        return m_keyboard_state;        
    }

    io::mouse_state get_mouse_state() const
    {
            return m_mouse_state;
    }

    private:

    uint16_t window_width( HWND window) const
    {
        RECT r;
        ::GetClientRect ( window, &r);
        return static_cast<uint16_t> ( r.right - r.left );
    }

    uint16_t window_height(HWND window) const
    {
        RECT r;
        ::GetClientRect ( window, &r);
        return static_cast<uint16_t> ( r.bottom - r.top );
    }

    protected:

    void on_update()
    {
        using namespace os::windows;

        uint8_t data[256] = {};
        auto hr = m_keyboard->GetDeviceState( sizeof(data), &data[0] );

        //swap the shadow states
        m_keyboard_state.swap();

        if (hr == S_OK )
        {
            m_keyboard_state.set_state( data );
        }

        DIMOUSESTATE2 mouse_state = {};

        m_mouse_state.swap();

        hr = m_mouse->GetDeviceState( sizeof(mouse_state), &mouse_state );

        if (hr == S_OK )
        {
            m_mouse_state.set_state ( mouse_state );

            std::cout << "x:" << m_mouse_state.get_x() << "y:"<<m_mouse_state.get_y() << std::endl; 
        }
    }

    void on_render_frame()
    {

    }

    void on_activate()
    {
        using namespace os::windows;

        auto hr = m_keyboard->Acquire();

        if ( ! (hr == S_OK || hr == S_FALSE) )
        {
            throw_if_failed< com_exception > (  hr );
        }

        hr = m_mouse->Acquire();
        if ( ! (hr == S_OK || hr == S_FALSE) )
        {
            throw_if_failed< com_exception > (  hr );
        }
    }

    void on_deactivate() override
    {
        using namespace os::windows;
        throw_if_failed< com_exception > ( m_keyboard->Unacquire() );

        using namespace os::windows;
        throw_if_failed< com_exception > ( m_mouse->Unacquire() );
    }

    void on_resize(uint32_t width, uint32_t height) override
    {
        base::on_resize(width, height);

        m_mouse_state.set_width(width);
        m_mouse_state.set_height(height);
    }

    private:

    os::windows::com_ptr<IDirectInput8>         m_direct_input;
    os::windows::com_ptr<IDirectInputDevice8>   m_keyboard;
    os::windows::com_ptr<IDirectInputDevice8>   m_mouse;

    io::keyboard_state                          m_keyboard_state;
    io::mouse_state                             m_mouse_state;
};


class cursor
{

};

inline cursor create_cursor( d2d::irendertarget_ptr render_target)
{

    //static uint8_t data = {};
    D2D1_SIZE_U size = {};
    size.width = 64;
    size.height = 64;

    return cursor();
}


class sample_application : public windowed_with_input_application
{
    typedef windowed_with_input_application base;

    public:

    sample_application( const wchar_t* window_title  ) : base ( window_title)
        , m_d2d_factory( d2d::create_d2d_factory_single_threaded() )
        , m_dwrite_factory( dwrite::create_dwrite_factory() )
        , m_text_format ( dwrite::create_text_format(m_dwrite_factory) )
    {

    }

    protected:

    void on_update()
    {

    }

    void on_render_frame()
    {
        return;
        //render text
        m_window_render_target->BeginDraw();
        m_window_render_target->Clear();

        RECT r;
        ::GetClientRect(get_window(), &r);
        ::InflateRect(&r, -10, -10 );
    
        D2D1_RECT_F rf = {static_cast<float> (r.left), static_cast<float>(r.top), static_cast<float>(r.right), static_cast<float>(r.bottom)};

        m_window_render_target->DrawTextW(L"Testo", 4, m_text_format.get(), &rf, m_brush.get());

        m_window_render_target->EndDraw();

    }

    void on_resize (uint32_t width, uint32_t height)
    {
        base::on_resize(width, height);


        using namespace os::windows;
        d3d11::itexture2d_ptr   texture = dxgi::get_buffer( m_context.m_swap_chain.get() );
        dxgi::isurface_ptr      dxgi;

        throw_if_failed<d3d11::exception> ( texture->QueryInterface(__uuidof(IDXGISurface), reinterpret_cast<void**> (&dxgi) ) );

        m_window_render_target = d2d::create_render_target( m_d2d_factory, dxgi );
        m_brush = d2d::create_solid_color_brush(m_window_render_target);
    }

    private:
    d2d::ifactory_ptr           m_d2d_factory;
    d2d::irendertarget_ptr      m_window_render_target;
    d2d::isolid_color_brush_ptr m_brush;
    dwrite::ifactory_ptr        m_dwrite_factory;
    dwrite::itextformat_ptr     m_text_format;
            
            
};

int _tmain(int argc, _TCHAR* argv[])
{
    using namespace lscm::indexed_face_set;

    auto app = sample_application (  L"Least Squares Conformal Maps" );

    //auto mesh = create_from_noff_file( L"../media/meshes/bunny_nf4000.noff" ) ;

    return app.run();
}

