#ifndef __composer_application_h__
#define __composer_application_h__

#include <cstdint>

#include <gx/gx_default_application.h>
#include <gx/gx_render_resource.h>
#include <d2d/d2d_helpers.h>

#include <d2d/dwrite_helpers.h>
#include <gx/gx_geometry_helpers.h>
#include <gx/gx_view_port.h>

#include <gx/gx_render_functions.h>
#include <gx/shaders/gx_shader_copy_texture.h>
#include <gx/gx_view_port.h>


namespace coloryourway
{
    namespace composer
    {
        class renderable;

        class sample_application : public gx::default_application
        {
            typedef gx::default_application base;

        public:
            sample_application(const wchar_t* window_title);

            void register_renderable( std::shared_ptr<renderable> r );
            void unregister_renderable(std::shared_ptr<renderable> r);

            d3d11::idevice_ptr get_device()
            {
                return m_context.m_device;
            }

            d3d11::idevicecontext_ptr get_immediate_context()
            {
                return m_context.m_immediate_context;
            }

        protected:

            virtual void on_render_scene();
            void render_scene();
            virtual void on_update_scene(float dt);
            void update_scene(float dt);
            void on_update();
            void on_render_frame();
            void on_resize(uint32_t width, uint32_t height);

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

            std::vector< std::shared_ptr< renderable> > m_renderables;
        };
    }
}



#endif
