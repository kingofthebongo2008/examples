#include "precompiled.h"
#include "composer_application.h"

#include <string>
#include <sys/sys_profile_timer.h>

#include "composer_renderable.h"
#include "composer_render_context.h"

namespace coloryourway
{
    namespace composer
    {

            sample_application::sample_application(const wchar_t* window_title) : base(window_title)
                , m_d2d_factory(d2d::create_d2d_factory_single_threaded())
                , m_dwrite_factory(dwrite::create_dwrite_factory())
                , m_text_format(dwrite::create_text_format(m_dwrite_factory))
                , m_full_screen_draw(m_context.m_device)
                , m_copy_texture_ps(gx::create_shader_copy_texture_ps(m_context.m_device))
                , m_d2d_resource(gx::create_render_target_resource(m_context.m_device, 8, 8, DXGI_FORMAT_R8G8B8A8_UNORM))
                , m_opaque_state(gx::create_opaque_blend_state(m_context.m_device))
                , m_premultiplied_alpha_state(gx::create_premultiplied_alpha_blend_state(m_context.m_device))
                , m_cull_back_raster_state(gx::create_cull_back_rasterizer_state(m_context.m_device))
                , m_cull_none_raster_state(gx::create_cull_none_rasterizer_state(m_context.m_device))
                , m_depth_disable_state(gx::create_depth_test_disable_state(m_context.m_device))
                , m_point_sampler(gx::create_point_sampler_state(m_context.m_device))
                , m_elapsed_update_time(0.0)
            {
                m_renderables.reserve(1000);
            }

            void sample_application::register_renderable(std::shared_ptr<renderable> r)
            {
                m_renderables.push_back(r);
            }

            void sample_application::unregister_renderable(std::shared_ptr<renderable> r)
            {
                std::remove( std::begin(m_renderables), std::end(m_renderables), r);
            }

            void sample_application::on_render_scene()
            {
                //get immediate context to submit commands to the gpu
                auto device_context = m_context.m_immediate_context;

                render_context context
                    (
                        device_context,
                        m_opaque_state,
                        m_premultiplied_alpha_state,

                        m_alpha_blend_state,
                        m_cull_back_raster_state,
                        m_cull_none_raster_state,

                        m_depth_disable_state,
                        m_point_sampler,

                        m_view_port
                    );

                for (auto i = 0U; i < m_renderables.size(); ++i)
                {
                    const auto& r = m_renderables[i];

                    r->draw( context );
                }
            }

            void sample_application::render_scene()
            {
                on_render_scene();
            }

            void sample_application::on_update_scene( float dt)
            {

            }

            void sample_application::update_scene( float dt)
            {
                on_update_scene(dt);
            }

            void sample_application::on_update()
            {
                sys::profile_timer timer;

                //Measure the update time and pass it to the render function
                m_elapsed_update_time = timer.milliseconds();

                update_scene( static_cast<float>( m_elapsed_update_time ) );
            }

            void sample_application::on_render_frame()
            {
                sys::profile_timer timer;

                //get immediate context to submit commands to the gpu
                auto device_context = m_context.m_immediate_context.get();

                //set render target as the back buffer, goes to the operating system
                d3d11::om_set_render_target(device_context, m_back_buffer_render_target);
                d3d11::clear_render_target_view(device_context, m_back_buffer_render_target, math::zero());

                on_render_scene( );

                //Draw the gui and the texts
                m_d2d_render_target->BeginDraw();


                RECT r;
                ::GetClientRect(get_window(), &r);

                //Get a description of the GPU or another simulator device
                DXGI_ADAPTER_DESC d;
                m_context.m_adapter->GetDesc(&d);

                D2D1_RECT_F rf = { static_cast<float> (r.left), static_cast<float>(r.top), static_cast<float>(r.right), static_cast<float>(r.bottom) };

                const std::wstring w = L"Update time: " + std::to_wstring(m_elapsed_update_time) + L"ms Render time: " + std::to_wstring(timer.milliseconds()) + L" ms\n";
                const std::wstring w2 = w + d.Description + L" Video Memory(MB): " + std::to_wstring(d.DedicatedVideoMemory / (1024 * 1024)) + L" System Memory(MB): " + std::to_wstring(d.DedicatedSystemMemory / (1024 * 1024)) + L" Shared Memory(MB): " + std::to_wstring(d.SharedSystemMemory / (1024 * 1024));

                m_d2d_render_target->SetTransform(D2D1::Matrix3x2F::Identity());
                m_d2d_render_target->FillRectangle(rf, m_brush2);
                m_d2d_render_target->DrawTextW(w2.c_str(), static_cast<uint32_t> (w2.length()), m_text_format, &rf, m_brush);
                m_d2d_render_target->EndDraw();
            }

            void sample_application::on_resize(uint32_t width, uint32_t height)
            {
                //Reset back buffer render targets
                m_back_buffer_render_target.reset();

                base::on_resize(width, height);

                //Recreate the render target to the back buffer again
                m_back_buffer_render_target = d3d11::create_render_target_view(m_context.m_device, dxgi::get_buffer(m_context.m_swap_chain));


                using namespace os::windows;

                //Direct 2D resources
                m_d2d_resource = gx::create_render_target_resource(m_context.m_device, width, height, DXGI_FORMAT_B8G8R8A8_UNORM);// DXGI_FORMAT_R8G8B8A8_UNORM);
                m_d2d_render_target = d2d::create_render_target(m_d2d_factory, dxgi::get_buffer(m_context.m_swap_chain).get());
                m_brush = d2d::create_solid_color_brush(m_d2d_render_target);
                m_brush2 = d2d::create_solid_color_brush2(m_d2d_render_target);

                //Reset view port dimensions
                m_view_port.set_dimensions(width, height);
            }
    }
}
