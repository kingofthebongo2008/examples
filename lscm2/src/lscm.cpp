// lscm.cpp : Defines the entry point for the console application.
//

#include "precompiled.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <future>

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

#include <utility>

#include <DXGIDebug.h>

#include <math/math_vector.h>
#include <math/math_graphics.h>

#include <sys/sys_profile_timer.h>

#include <gx/gx_default_application.h>
#include <gx/gx_view_port.h>

#include <d3d11/dxgi_helpers.h>
#include <d2d/d2d_helpers.h>
#include <d2d/dwrite_helpers.h>

#include <gx/gx_compute_resource.h>
#include <gx/gx_geometry_pass_common.h>
#include <gx/gx_geometry_helpers.h>
#include <gx/gx_pinhole_camera.h>
#include <gx/gx_render_resource.h>
#include <gx/gx_render_functions.h>

#include <gx/shaders/gx_shader_copy_texture.h>
#include <gx/shaders/gx_shader_full_screen.h>

#include "shaders/gx_global_buffers.h"

#include "shaders/gx_shader_depth_prepass_ps.h"
#include "shaders/gx_shader_depth_prepass_vs.h"
#include "shaders/gx_shader_draw_light_accumulation_ps.h"
#include "shaders/gx_shader_light_accumulation_cs.h"


namespace lscm
{
    class renderable_mesh;

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
                initialize();
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
                return &m_edges[ static_cast<uint32_t> ( p ) ];
            }

            private:

            std::vector< vertex >          m_vertices;
            std::vector< normal >          m_normals;
            std::vector< face >            m_faces;
            std::vector< normal >          m_face_normals;
            std::vector< winged_edge >     m_edges;
            progress_notifier              m_notifier;

            friend std::shared_ptr< renderable_mesh > create_renderable_mesh(ID3D11Device* device, std::shared_ptr< lscm::indexed_face_set::mesh > mesh);

            void initialize()
            {

                auto f1 = std::async ( [=]
                {
                    clean_degenerate_faces();
                });

                auto f2 = std::async ( [&]
                {
                    f1.wait();
                    clear_vertices_not_referenced_by_faces();
                });

                auto f3 = std::async ( [&]
                {
                    f1.wait();
                    f2.wait();

                    calculate_pivot();

                    auto f6 = std::async ( [=]
                    {
                        build_face_normals();
                    });

                    f6.wait();
                });

                auto f4 = std::async ( [=]
                {
                    normalize_normals();
                });

                auto f5 = std::async ( [&]
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

            void calculate_pivot()
            {
                std::vector< mesh::vertex > vertices( m_vertices.size() ) ;
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


                std::transform ( m_vertices.begin(), m_vertices.end(), vertices.begin(),  [&] ( const vertex& v )
                {
                    vertex v_new = { static_cast<float> ( v.x - sum0 ) , static_cast<float> ( v.y - sum1 ) , static_cast<float> ( v.z - sum2 ), static_cast<float> ( v.w - sum3 ) };
                    return std::move( v_new );
                }
                );

                m_vertices = std::move(vertices);
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

class sample_application : public gx::default_application
{
    typedef gx::default_application base;

    public:

    sample_application( const wchar_t* window_title  ) : base ( window_title)
        , m_d2d_factory( d2d::create_d2d_factory_single_threaded() )
        , m_dwrite_factory( dwrite::create_dwrite_factory() )
        , m_text_format ( dwrite::create_text_format(m_dwrite_factory) )
        , m_full_screen_draw( m_context.m_device )
        , m_copy_texture_ps(m_context.m_device )
        , m_d2d_resource ( gx::create_render_target_resource( m_context.m_device, 8, 8, DXGI_FORMAT_R8G8B8A8_UNORM ) )
        , m_opaque_state ( gx::create_opaque_blend_state( m_context.m_device ) )
        , m_premultiplied_alpha_state(gx::create_premultiplied_alpha_blend_state(m_context.m_device))
        , m_cull_back_raster_state ( gx::create_cull_back_rasterizer_state( m_context.m_device ) )
        , m_cull_none_raster_state(gx::create_cull_none_rasterizer_state(m_context.m_device))
        , m_depth_disable_state( gx::create_depth_test_disable_state( m_context.m_device ) )
        , m_point_sampler(gx::create_point_sampler_state(m_context.m_device ))
        , m_elapsed_update_time(0.0)
    {

    }

    protected:

    virtual void on_render_scene()
    {

    }

    void render_scene()
    {
        on_render_scene();
    }

    virtual void on_update_scene()
    {

    }

    void update_scene()
    {
        on_update_scene();
    }

    void on_update()
    {
        sys::profile_timer timer;

        update_scene();

        //Measure the update time and pass it to the render function
        m_elapsed_update_time = timer.milliseconds();
    }

    void on_render_frame()
    {
        sys::profile_timer timer;

        //get immediate context to submit commands to the gpu
        auto device_context= m_context.m_immediate_context.get();


        //set render target as the back buffer, goes to the operating system
        d3d11::om_set_render_target ( device_context, m_back_buffer_render_target );
        d3d11::clear_render_target_view ( device_context, m_back_buffer_render_target, math::zero() );


        on_render_scene();

        /*
        //Draw the gui and the texts
        m_d2d_render_target->BeginDraw();
        m_d2d_render_target->Clear();

        RECT r;
        ::GetClientRect(get_window(), &r);

        //Get a description of the GPU or another simulator device
        DXGI_ADAPTER_DESC d;
        m_context.m_adapter->GetDesc(&d);
            
        D2D1_RECT_F rf = {static_cast<float> (r.left), static_cast<float>(r.top), static_cast<float>(r.right), static_cast<float>(r.bottom)};

        const std::wstring w = L"Update time: " + std::to_wstring(m_elapsed_update_time) + L"ms Render time: " + std::to_wstring(timer.milliseconds()) + L" ms\n";
        const std::wstring w2 = w + d.Description + L" Video Memory(MB): " + std::to_wstring(d.DedicatedVideoMemory / (1024 * 1024)) + L" System Memory(MB): " + std::to_wstring(d.DedicatedSystemMemory / (1024 * 1024)) + L" Shared Memory(MB): " + std::to_wstring(d.SharedSystemMemory / (1024 * 1024));
      
        m_d2d_render_target->SetTransform(D2D1::Matrix3x2F::Identity());
        m_d2d_render_target->FillRectangle(rf, m_brush2);
        m_d2d_render_target->DrawTextW(w2.c_str(),  static_cast<uint32_t> ( w2.length() ) , m_text_format, &rf, m_brush);
        m_d2d_render_target->EndDraw();

        //set a view port for rendering
        D3D11_VIEWPORT v = m_view_port;
        device_context->RSSetViewports(1, &v);

        //clear the back buffer
        const float fraction = 1.0f / 128.0f;
        d3d11::clear_render_target_view(device_context, m_back_buffer_render_target, math::set(fraction, fraction, fraction, 1.0f));

        //compose direct2d render target over the back buffer by rendering full screen quad that copies one texture onto another with alpha blending
        d3d11::ps_set_shader( device_context, m_copy_texture_ps );
        d3d11::ps_set_shader_resources( device_context,  m_d2d_resource );
        d3d11::ps_set_sampler_state(device_context, m_point_sampler);
        
        //cull all back facing triangles
        d3d11::rs_set_state(device_context, m_cull_back_raster_state);

        d3d11::om_set_blend_state(device_context, m_premultiplied_alpha_state);
        
        //disable depth culling
        d3d11::om_set_depth_state(device_context, m_depth_disable_state);
        m_full_screen_draw.draw(device_context);
        */
    }

    void on_resize (uint32_t width, uint32_t height)
    {
        //Reset back buffer render targets
        m_back_buffer_render_target.reset();

        base::on_resize( width, height );

        //Recreate the render target to the back buffer again
        m_back_buffer_render_target =  d3d11::create_render_target_view ( m_context.m_device, dxgi::get_buffer( m_context.m_swap_chain ) ) ;

        /*
        using namespace os::windows;
     
        //Direct 2D resources
        m_d2d_resource = gx::create_render_target_resource( m_context.m_device, width, height, DXGI_FORMAT_R8G8B8A8_UNORM );
        m_d2d_render_target = d2d::create_render_target( m_d2d_factory, m_d2d_resource );
        m_brush = d2d::create_solid_color_brush( m_d2d_render_target );
        m_brush2 = d2d::create_solid_color_brush2(m_d2d_render_target);
        */
        //Reset view port dimensions
        m_view_port.set_dimensions(width, height);

    }


    protected:

    gx::render_target_resource              m_d2d_resource;

    d2d::ifactory_ptr                       m_d2d_factory;
    dwrite::ifactory_ptr                    m_dwrite_factory;

    d2d::irendertarget_ptr		            m_d2d_render_target;
    d2d::isolid_color_brush_ptr             m_brush;
    d2d::isolid_color_brush_ptr             m_brush2;
    dwrite::itextformat_ptr                 m_text_format;
    
    gx::full_screen_draw                    m_full_screen_draw;
    gx::shader_copy_texture                 m_copy_texture_ps;
    d3d11::id3d11rendertargetview_ptr       m_back_buffer_render_target;

    d3d11::iblendstate_ptr                  m_opaque_state;
    d3d11::iblendstate_ptr                  m_premultiplied_alpha_state;
    
    d3d11::iblendstate_ptr                  m_alpha_blend_state;
    d3d11::irasterizerstate_ptr             m_cull_back_raster_state;
    d3d11::irasterizerstate_ptr             m_cull_none_raster_state;

    d3d11::idepthstencilstate_ptr           m_depth_disable_state;
    d3d11::isamplerstate_ptr                m_point_sampler;

    gx::view_port                           m_view_port;

    double                                  m_elapsed_update_time;
};


namespace lscm
{
    class renderable_mesh
    {
    public:

        renderable_mesh( d3d11::ibuffer_ptr vertices, d3d11::ibuffer_ptr triangles, uint32_t vertex_count, uint32_t index_count) :
              m_positions ( vertices )
            , m_triangles ( triangles )
            , m_vertex_count( vertex_count )
            , m_index_count( index_count )
        {

        }

        void draw(ID3D11DeviceContext* context)
        {
            d3d11::ia_set_primitive_topology(context, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST );
            d3d11::ia_set_vertex_buffer(context, m_positions, 16);


            d3d11::ia_set_index_buffer(context, m_triangles, DXGI_FORMAT_R32_UINT);
            context->DrawIndexed( m_index_count, 0, 0);
        }

    private:

        d3d11::ibuffer_ptr  m_positions;
        d3d11::ibuffer_ptr  m_triangles;
        uint32_t            m_vertex_count;
        uint32_t            m_index_count;
    };

    namespace indexed_face_set
    {
        std::shared_ptr< renderable_mesh > create_renderable_mesh(ID3D11Device* device, std::shared_ptr< lscm::indexed_face_set::mesh > mesh)
        {
            auto positions = gx::create_positions_x_y_z_w( (const float*)&mesh->m_vertices[0], static_cast<uint32_t> ( mesh->m_vertices.size() ) );
            auto vertex_count = static_cast<uint32_t> (mesh->m_vertices.size());
            
            //triangle list
            auto index_count = 3 * static_cast<uint32_t> (mesh->m_faces.size());

            return std::make_shared<renderable_mesh>(
                d3d11::create_immutable_vertex_buffer(device, &mesh->m_vertices[0], mesh->m_vertices.size() * sizeof( mesh::vertex ) ),
                d3d11::create_immutable_index_buffer(device, &mesh->m_faces[0], mesh->m_faces.size() * sizeof(  mesh::face ) ), 
                vertex_count,
                index_count
                );
        }
    }
}

class sample_application2 : public sample_application
{
    typedef sample_application base;

    public:

    sample_application2( const wchar_t* window_title ) : base(window_title)
    , m_visibility_buffer  ( gx::create_render_target_resource(m_context.m_device, 8, 8, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_UINT, gx::msaa_8x ) )
    , m_depth_buffer ( gx::create_depth_resource(m_context.m_device, 8, 8, gx::msaa_8x ) )
    , m_light_buffer( gx::create_structured_compute_resource(m_context.m_device, 8 * 8, 2 * sizeof(uint32_t) ) )
    , m_depth_less( gx::create_depth_test_less_state(m_context.m_device ) )
    , m_depth_disabled( gx::create_depth_test_disable_state(m_context.m_device))
    , m_depth_prepass_ps_buffer( m_context.m_device )
    , m_depth_prepass_vs_buffer(m_context.m_device)
    , m_depth_prepass_ps(m_context.m_device)
    , m_depth_prepass_vs(m_context.m_device)
    , m_depth_prepass_layout( m_context.m_device, m_depth_prepass_vs )
    , m_depth_prepass_buffer( m_context.m_device )
    , m_global_per_frame_buffer( m_context.m_device )
    , m_lighting_cs( m_context.m_device )
    , m_draw_lighting_ps(m_context.m_device)
    {
        m_camera.set_view_position( math::set(0.0, 0.0f, -15.0f, 0.0f) );
    }

    protected:

    void     on_update_scene() override
    {

    }

    void     on_render_scene() override
    {
        auto device_context = this->m_context.m_immediate_context.get();

        device_context->ClearState();

        gx::reset_render_targets( device_context );
        gx::reset_shader_resources( device_context );
        gx::reset_constant_buffers( device_context );
        

        //visibility pass
        D3D11_VIEWPORT v = m_view_port;
        d3d11::rs_set_view_port(device_context, &v);

        //cull all back facing triangles
        d3d11::rs_set_state(device_context, m_cull_back_raster_state);

        d3d11::om_set_render_target( device_context, m_visibility_buffer, m_depth_buffer );
        d3d11::om_set_blend_state( device_context, m_opaque_state );
        d3d11::om_set_depth_state( device_context, m_depth_less );

        d3d11::clear_render_target_view(device_context, m_visibility_buffer, math::one());
        d3d11::clear_depth_stencil_view(device_context, m_depth_buffer);

        d3d11::ia_set_input_layout(device_context, m_depth_prepass_layout);

        d3d11::vs_set_shader( device_context, m_depth_prepass_vs );
        d3d11::ps_set_shader( device_context, m_depth_prepass_ps );

        auto scale = math::scaling( math::set( 20.0f, 20.0f, 20.0f, 1.0f) );
        auto w = math::mul(scale, math::identity_matrix());

        m_depth_prepass_vs_buffer.set_w( w );
        m_depth_prepass_ps_buffer.set_instance_id( 255 );

        m_depth_prepass_buffer.set_view(gx::create_view_matrix( m_camera ) );
        m_depth_prepass_buffer.set_projection(gx::create_perspective_matrix(m_camera));
        m_depth_prepass_buffer.set_reverse_projection( static_cast<math::float4> ( gx::create_perspective_reprojection_params( m_camera ) ) );

        m_depth_prepass_buffer.flush(device_context);
        m_depth_prepass_vs_buffer.flush(device_context);
        m_depth_prepass_ps_buffer.flush(device_context);

        m_depth_prepass_buffer.bind_as_vertex(device_context);
        m_depth_prepass_buffer.bind_as_pixel(device_context);
        
        m_depth_prepass_vs_buffer.bind_as_vertex(device_context);
        m_depth_prepass_ps_buffer.bind_as_pixel(device_context);

        //m_full_screen_draw.draw(device_context);
        m_mesh->draw(device_context);

        d3d11::clear_state( device_context );

        auto dimensions = m_global_per_frame_buffer.get_light_accumulation_buffer_dimensions();

        //light calculation in the light_buffer_1
        d3d11::cs_set_shader( device_context, m_lighting_cs );
        d3d11::cs_set_shader_resource( device_context, 0, m_visibility_buffer );
        d3d11::cs_set_unordered_access_view(device_context, 0, m_light_buffer );
        
        m_global_per_frame_buffer.flush(device_context);
        m_global_per_frame_buffer.bind_as_compute(device_context);

        d3d11::cs_dispatch(device_context, std::get<0>(dimensions), std::get<1>(dimensions) );
        d3d11::clear_state( device_context );

        d3d11::om_set_render_target (   device_context, m_back_buffer_render_target );
        d3d11::om_set_blend_state(  device_context,     m_opaque_state );
        d3d11::om_set_depth_state(  device_context,     m_depth_disabled );

        //draw on the screen what we have calculated
        d3d11::ps_set_shader_resource(  device_context, m_light_buffer  );
        d3d11::ps_set_shader(   device_context, m_draw_lighting_ps  );
        d3d11::ps_set_sampler_state(    device_context, m_point_sampler );
        d3d11::rs_set_state(    device_context, m_cull_none_raster_state );
        d3d11::rs_set_view_port(device_context, &v);
        
        m_global_per_frame_buffer.bind_as_pixel(device_context);

        m_full_screen_draw.draw(device_context);

    }

    void    on_resize(uint32_t width, uint32_t height) override
    {
        base::on_resize(width, height);

        m_visibility_buffer         =   gx::create_render_target_resource(m_context.m_device, width, height, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_UINT, gx::msaa_8x);
        m_depth_buffer              =   gx::create_depth_resource( m_context.m_device, width, height, gx::msaa_8x );

        m_light_buffer              =   gx::create_structured_compute_resource(m_context.m_device, width * height, 2 * sizeof(uint32_t));

        m_global_per_frame_buffer.set_light_accumulation_buffer_dimensions( width, height );
    }

    public:

    void set_mesh( std::shared_ptr< lscm::indexed_face_set::mesh > m )
    {

        m_mesh = lscm::indexed_face_set::create_renderable_mesh( m_context.m_device, m );

    }

    private:

    gx::pinhole_camera                      m_camera;
    gx::depth_resource                      m_depth_buffer;
    gx::render_target_resource              m_visibility_buffer;
    gx::compute_resource                    m_light_buffer;
    d3d11::idepthstencilstate_ptr           m_depth_less;
    d3d11::idepthstencilstate_ptr           m_depth_disabled;

    //visibility pass buffers
    lscm::shader_depth_prepass_ps_buffer    m_depth_prepass_ps_buffer;
    lscm::shader_depth_prepass_vs_buffer    m_depth_prepass_vs_buffer;

    lscm::shader_depth_prepass_ps           m_depth_prepass_ps;
    lscm::shader_depth_prepass_vs           m_depth_prepass_vs;
    lscm::shader_depth_prepass_layout       m_depth_prepass_layout;
    lscm::visibility_per_pass_buffer        m_depth_prepass_buffer;

    lscm::global_per_frame_buffer           m_global_per_frame_buffer;

    //lighting
    lscm::shader_light_accumulation_cs      m_lighting_cs;
    lscm::shader_draw_light_accumulation_ps m_draw_lighting_ps;


    //debug output
    d3d11::iunordered_access_view_ptr       m_visibility_buffer_view;
    d3d11::iunordered_access_view_ptr       m_back_buffer_view;

    //scene
    std::shared_ptr<lscm::renderable_mesh>  m_mesh;
};

#define DEFINE_GUID(name, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8) \
    EXTERN_C const GUID DECLSPEC_SELECTANY name \
    = { l, w1, w2, { b1, b2, b3, b4, b5, b6, b7, b8 } }

DEFINE_GUID(DXGI_DEBUG_ALL, 0xe48ae283, 0xda80, 0x490b, 0x87, 0xe6, 0x43, 0xe9, 0xa9, 0xcf, 0xda, 0x8);

int _tmain(int argc, _TCHAR* argv[])
{
    using namespace lscm::indexed_face_set;

    std::shared_ptr<mesh> m;

    auto loading = std::async( [&]
    {

        m = create_from_noff_file(L"../media/meshes/bunny_nf4000.noff");

    });

    auto app = new sample_application2 (  L"Sample Application" );

    loading.wait();

    app->set_mesh(m);

    auto result = app->run();

    delete app;

    HMODULE h = LoadLibrary(L"DXGIDebug.dll");

    if (h)
    {
        os::windows::com_ptr < IDXGIDebug > debug;
        HRESULT WINAPI DXGIGetDebugInterface(REFIID riid, void **ppDebug);

        typedef HRESULT(*DXGIDebug) (REFIID riid, void **ppDebug);

        DXGIDebug g = (DXGIDebug)GetProcAddress(h, "DXGIGetDebugInterface");
        g(__uuidof(IDXGIDebug), (void**)&debug);
        debug->ReportLiveObjects(DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_ALL);
    }

    return result;
}

