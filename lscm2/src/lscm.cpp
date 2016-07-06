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

#include "shaders/gx_shader_clear_light_accumulation_cs.h"
#include "shaders/gx_shader_depth_prepass_ps.h"
#include "shaders/gx_shader_depth_prepass_vs.h"
#include "shaders/gx_shader_draw_light_accumulation_ps.h"
#include "shaders/gx_shader_light_accumulation_cs.h"
#include "shaders/gx_shader_work_list_build_cs.h"
#include "shaders/gx_shader_work_list_sort_cs.h"

#include "indexed_face_set/indexed_face_set_mesh.h"
#include "indexed_face_set/indexed_face_set_renderable_mesh.h"
#include "indexed_face_set/indexed_face_set_functions.h"


class sample_application : public gx::default_application
{
    typedef gx::default_application base;

    public:

    sample_application( const wchar_t* window_title  ) : base ( window_title)
        , m_d2d_factory( d2d::create_d2d_factory_single_threaded() )
        , m_dwrite_factory( dwrite::create_dwrite_factory() )
        , m_text_format ( dwrite::create_text_format(m_dwrite_factory) )
        , m_full_screen_draw( m_context.m_device )
        , m_copy_texture_ps( gx::create_shader_copy_texture_ps ( m_context.m_device ) )
        , m_d2d_resource ( gx::create_render_target_resource( m_context.m_device, 8, 8, DXGI_FORMAT_R8G8B8A8_UNORM ) )
        , m_opaque_state ( gx::create_opaque_blend_state( m_context.m_device ) )
        , m_premultiplied_alpha_state(gx::create_premultiplied_alpha_blend_state(m_context.m_device))
        , m_cull_back_raster_state ( gx::create_cull_back_rasterizer_state_wireframe( m_context.m_device ) )
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
    gx::shader_copy_texture_ps              m_copy_texture_ps;
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


typedef std::future < d3d11::icomputeshader_ptr >   compute_shader;
typedef std::future < d3d11::ipixelshader_ptr >     pixel_shader;

typedef std::tuple < compute_shader, compute_shader, compute_shader, compute_shader, pixel_shader, pixel_shader, std::future< lscm::shader_depth_prepass_vs > > shader_database_create_info;

shader_database_create_info create_shaders(ID3D11Device* device)
{
    return std::make_tuple(
                            lscm::create_shader_work_list_build_cs_async(device) 
                            , lscm::create_shader_work_list_sort_cs_async(device)
                            , lscm::create_shader_light_accumulation_cs_async(device)
                            , lscm::create_shader_clear_light_accumulation_cs_async(device)
                            , lscm::create_shader_draw_light_accumulation_ps_async(device)
                            , lscm::create_shader_depth_prepass_ps_async(device)
                            , lscm::create_shader_depth_prepass_vs_async(device)
                            );
}

class shader_database
{
    public:
        shader_database(shader_database_create_info& shaders) :
          m_work_list_build(std::get<0>(shaders).get() )
          , m_work_list_sort(std::get<1>(shaders).get() )
          , m_lighting_cs(std::get<2>(shaders).get() )
          , m_clear_lighting_cs( std::get<3>(shaders).get() )
          , m_draw_lighting_ps( std::get<4>(shaders).get() )
          , m_depth_prepass_ps( std::get<5>(shaders).get() )
          , m_depth_prepass_vs(std::get<6>(shaders).get())
    {

    }

    public:

    lscm::shader_work_list_build_cs             m_work_list_build;
    lscm::shader_work_list_sort_cs              m_work_list_sort;

    //lighting
    lscm::shader_light_accumulation_cs          m_lighting_cs;
    lscm::shader_clear_light_accumulation_cs    m_clear_lighting_cs;

    lscm::shader_draw_light_accumulation_ps     m_draw_lighting_ps;


    //visibility buffer generation
    lscm::shader_depth_prepass_ps               m_depth_prepass_ps;
    lscm::shader_depth_prepass_vs               m_depth_prepass_vs;
};

class draw_instance_info
{
    public:
    draw_instance_info( uint32_t closure_id, uint32_t vertex_offset, uint32_t index_offset ) : 
        m_closure_id(0)
        , m_vertex_offset( vertex_offset )
        , m_index_offset( index_offset )
    {}

    explicit draw_instance_info(uint32_t closure_id) : draw_instance_info( closure_id, 0, 0)
    {}

    uint32_t m_closure_id;      //shader 
    uint32_t m_vertex_offset;   //offset into a big vertex buffer with geometry
    uint32_t m_index_offset;    //offset into a big index buffer with geometry
};


template <typename const_iterator>
gx::compute_resource create_draw_instance_info_compute_resource(ID3D11Device* device, const_iterator begin, const_iterator end )
{
    return gx::create_structured_compute_resource(device, static_cast<uint32_t> (end - begin), static_cast<uint32_t> (sizeof(draw_instance_info)), begin);
}

inline float rotation_radian(uint32_t step, uint32_t max_steps)
{
    static const float PI = std::atanf(1.0f) * 4;
    return  ( (step %  max_steps) * 2 * PI / max_steps ) ;
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
    , m_depth_prepass_buffer( m_context.m_device )
    , m_global_per_frame_buffer( m_context.m_device )
    , m_shader_database(create_shaders( m_context.m_device ) )
    , m_depth_prepass_layout(m_context.m_device, m_shader_database.m_depth_prepass_vs)
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

        d3d11::clear_state(device_context);

        gx::reset_render_targets( device_context );
        gx::reset_shader_resources( device_context );
        gx::reset_constant_buffers( device_context );
        

        //visibility pass
        D3D11_VIEWPORT v = m_view_port;
        d3d11::rs_set_view_port(device_context, &v);

        d3d11::rs_set_state(device_context, m_cull_back_raster_state);

        d3d11::om_set_render_target( device_context, m_visibility_buffer, m_depth_buffer );
        d3d11::om_set_blend_state( device_context, m_opaque_state );
        d3d11::om_set_depth_state( device_context, m_depth_less );

        d3d11::clear_render_target_view( device_context, m_visibility_buffer, math::one() );
        d3d11::clear_depth_stencil_view( device_context, m_depth_buffer );

        d3d11::ia_set_input_layout( device_context, m_depth_prepass_layout );

        d3d11::vs_set_shader( device_context, m_shader_database.m_depth_prepass_vs );
        d3d11::ps_set_shader( device_context, m_shader_database.m_depth_prepass_ps );

        auto scale      = math::scaling( math::set( 40.0f, 40.0f, 40.0f, 1.0f) );

        //do simple update
        static uint32_t step = 0;
        auto rotation        = math::rotation_y(rotation_radian(step++, 360));
        auto w               = math::mul(rotation, math::mul(scale, math::identity_matrix()) ) ;

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
        m_mesh->draw(device_context);

        //light pass
        d3d11::clear_state( device_context );

        auto dimensions = m_global_per_frame_buffer.get_light_accumulation_buffer_dimensions();

        m_global_per_frame_buffer.flush(device_context);
        m_global_per_frame_buffer.bind_as_compute(device_context);

        //clear the light buffer
        d3d11::cs_set_shader(device_context, m_shader_database.m_clear_lighting_cs);
        d3d11::cs_set_unordered_access_view(device_context, 0, m_light_buffer);
        d3d11::cs_dispatch(device_context, std::get<0>(dimensions), std::get<1>(dimensions));

        draw_instance_info instance_info( 1, 0, 0 );
        auto info = create_draw_instance_info_compute_resource<draw_instance_info*>( this->m_context.m_device, &instance_info, &instance_info + 1);

        //do light accumulation    
        d3d11::cs_set_shader( device_context, m_shader_database.m_lighting_cs );
        d3d11::cs_set_shader_resource( device_context, 0, m_visibility_buffer );
        d3d11::cs_dispatch(device_context, std::get<0>(dimensions), std::get<1>(dimensions) );

        d3d11::clear_state( device_context );

        d3d11::om_set_render_target (   device_context, m_back_buffer_render_target );
        d3d11::om_set_blend_state(  device_context,     m_opaque_state );
        d3d11::om_set_depth_state(  device_context,     m_depth_disabled );

        //draw on the screen what we have calculated
        d3d11::ps_set_shader_resource(  device_context, m_light_buffer  );
        d3d11::ps_set_shader(   device_context, m_shader_database.m_draw_lighting_ps  );
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

    shader_database                         m_shader_database;

    lscm::shader_depth_prepass_layout       m_depth_prepass_layout;
    lscm::visibility_per_pass_buffer        m_depth_prepass_buffer;

    lscm::global_per_frame_buffer           m_global_per_frame_buffer;

    //debug output
    d3d11::iunordered_access_view_ptr       m_visibility_buffer_view;
    d3d11::iunordered_access_view_ptr       m_back_buffer_view;

    //scene
    std::shared_ptr<lscm::indexed_face_set::renderable_mesh>  m_mesh;
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

    //std::cout << "Area of mesh: " << area(m.get()) << std::endl;
    //std::cout << "Symmetric hausdorff distance: " << math::get_x(symmetric_hausdorff_distance(m.get(), m.get())) << std::endl;

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

