#pragma once

#include <cstdint>
#include <ppl.h>

#include <sys/sys_profile_timer.h>

#include <d3d11/d3d11_helpers_constants.h>

#include <gx/gx_default_application.h>
#include <gx/gx_view_port.h>
#include <gx/gx_render_resource.h>
#include <gx/gx_pinhole_camera.h>

#include <gxu/gxu_pinhole_camera_dispatcher.h>

#include <img/targa_image.h>
#include <io/io_pad.h>

#include <sys/sys_profile_timer.h>

#include <shaders/full_screen_quad_shader.h>
#include <shaders/clear_color_shader.h>
#include <shaders/display_texture_shader.h>
#include <shaders/world_map_vertex_shader.h>
#include <shaders/world_map_pixel_shader.h>
#include <shaders/input_layout_database.h>
#include <shaders/sampling_pixel_shader.h>
#include <shaders/sampling_vertex_shader.h>

#include "map_utils.h"
#include "world_map.h"

#include <shaders/stripe_vertex_shader.h>
#include <shaders/stripe_pixel_shader.h>

#include "tiled_texture.h"
#include "world_map_sampling_renderer.h"
#include "world_map_residency_manager.h"

#include <d2d/d2d.h>
#include <d2d/dwrite.h>


namespace app
{
    struct tiled_texture_resource
    {
        tiled_texture               m_texture;
        d3d11::shader_resource_view m_view;

        ID3D11ShaderResourceView* to_view() const
        {
            return m_view.get();
        }
    };

    struct pass_constants
    {
        math::float4x4 m_view;
        math::float4x4 m_perspective;
    };


    void map_to_null_tile(ID3D11DeviceContext2* ctx, tiled_texture* t, uint32_t total_tile_count)
    {
        //region
        D3D11_TILED_RESOURCE_COORDINATE  region = {};
        D3D11_TILE_REGION_SIZE           region_size = {};
        uint32_t                         range_flag = D3D11_TILE_RANGE_REUSE_SINGLE_TILE;

        region_size.NumTiles = 1;
        uint32_t tile_pool_start_offsets = 0;
        uint32_t range_tile_counts = 1;

        //map region 0 to the null tile
        d3d11::helpers::update_tile_mappings
        (
            ctx,
            t->m_resource.get(),
            1,  //1 region
            &region,
            &region_size,
            t->m_tile_pool.get(),
            1,
            &range_flag,
            &tile_pool_start_offsets,
            &range_tile_counts,
            D3D11_TILE_MAPPING_NO_OVERWRITE
        );

        ctx->TiledResourceBarrier(NULL, t->m_resource.get());

        auto size = 128 * 128 * sizeof(uint32_t);
        std::unique_ptr< uint8_t[] > tile_data(new uint8_t[size]);

        std::memset(&tile_data[0], 0x00, size);

        ctx->UpdateTiles(
            t->m_resource.get(),
            &region,
            &region_size,
            &tile_data[0],
            D3D11_TILE_COPY_NO_OVERWRITE);

        ctx->TiledResourceBarrier(NULL, t->m_resource.get());

        //now map everything to this tile
        region_size.NumTiles = total_tile_count;
        tile_pool_start_offsets = 0;
        range_tile_counts = total_tile_count;

        D3D11_TILED_RESOURCE_COORDINATE  regions[8]      = {};
        D3D11_TILE_REGION_SIZE           region_sizes[8] = {};


        for (auto mip = 0; mip < 8; ++mip)
        {
            regions[mip].Subresource = mip;

            auto w = 128 / (1 << mip);
            auto h = w;
            auto d = 1;

            region_sizes[mip].bUseBox = TRUE;
            region_sizes[mip].Depth = d;
            region_sizes[mip].Width = w;
            region_sizes[mip].Height = h;
            region_sizes[mip].NumTiles = w * h * d;
        }

        //map region 0 to the null tile
        d3d11::helpers::update_tile_mappings
        (
            ctx,
            t->m_resource.get(),
            8,  //8 regions
            &regions[0],
            &region_sizes[0],
            t->m_tile_pool.get(),
            1,
            &range_flag,
            &tile_pool_start_offsets,
            &range_tile_counts,
            D3D11_TILE_MAPPING_NO_OVERWRITE
        );

        ctx->TiledResourceBarrier(NULL, t->m_resource.get());
    }

    void update_tile(ID3D11DeviceContext2* ctx, const void* tile_data, tiled_texture* t, uint32_t tile_x, uint32_t tile_y, uint32_t mip = 0 )
    {
        //region
        D3D11_TILED_RESOURCE_COORDINATE region = {};
        D3D11_TILE_REGION_SIZE          region_size = {};

        region.X = tile_x;
        region.Y = tile_y;
        region.Subresource = mip;

        region_size.NumTiles = 1;

        ctx->UpdateTiles(
            t->m_resource.get(),
            &region,
            &region_size,
            tile_data,
            D3D11_TILE_COPY_NO_OVERWRITE);
    }

    tiled_texture make_tiled_texture( ID3D11Device* device, ID3D11DeviceContext2* ctx, uint32_t width_texels, uint32_t height_texels)
    {
        tiled_texture r;
        
        r.m_width_texels                = 16384;  //width_texels;
        r.m_height_texels               = 16384;  //height_texels;

        r.m_width_tiles                 = 128;
        r.m_height_tiles                = 128;

        auto total_tiles                = tile_residency::pool_size_in_tiles;

        r.m_resource                    = d3d11::helpers::create_tiled_texture_2d(device, 16384, 16384, 8 );
        r.m_tile_pool                   = d3d11::helpers::create_tiled_pool(device, total_tiles );   //null tile + buffered titles

        
        //todo: calculate total tiles
        map_to_null_tile(ctx, &r, 21845 );

        return r;
    }

    tiled_texture_resource make_tiled_render_resource(ID3D11Device* device, const tiled_texture& resource)
    {
        tiled_texture_resource r;

        r.m_texture = resource;
        r.m_view = d3d11::helpers::create_shader_resource_view(device, resource.m_resource.get() );

        return r;
    }

    template <typename t> inline void vs_setshader(ID3D11DeviceContext2* ctx, t* s)
    {
        ctx->VSSetShader(s->to_shader(), nullptr, 0);
    }

    template <typename t> inline void vs_set_vertex_buffer(ID3D11DeviceContext2* ctx, uint32_t slot, t* s)
    {
        d3d11::vertex_buffer_view view = s->to_vertex_buffer_view();
        ID3D11Buffer* v[]       = { view.buffer };
        uint32_t      offset[]  = { view.offset};
        uint32_t      stride[]  = { view.stride };
        ctx->IASetVertexBuffers(slot, 1, v, stride, offset);
    }

    inline void vs_set_vertex_buffer(ID3D11DeviceContext2* ctx, uint32_t slot, nullptr_t s)
    {
        ID3D11Buffer* v[] = { nullptr };
        uint32_t      offset[] = { 0 };
        uint32_t      stride[] = { 0 };
        ctx->IASetVertexBuffers(slot, 1, v, stride, offset);
    }

    template <typename t> inline void vs_set_index_buffer(ID3D11DeviceContext2* ctx, t* s)
    {
        d3d11::index_buffer_view view = s->to_index_buffer_view();
        ctx->IASetIndexBuffer( view.buffer, view.format, view.offset);
    }

    inline void vs_set_index_buffer(ID3D11DeviceContext2* ctx, nullptr_t* s)
    {
        ctx->IASetIndexBuffer(nullptr, DXGI_FORMAT_UNKNOWN, 0);
    }

    template <typename t> inline void ps_setshader(ID3D11DeviceContext2* ctx, t* s)
    {
        ctx->PSSetShader(s->to_shader(), nullptr, 0);
    }

    template <typename t> inline void ps_setshader_resource(ID3D11DeviceContext2* ctx, uint32_t slot, t* s)
    {
        ID3D11ShaderResourceView* v[] = { s->to_view() };
        ctx->PSSetShaderResources( slot, 1, v);
    }

    template <typename t> inline void ps_setshader_sampler(ID3D11DeviceContext2* ctx, uint32_t slot, t& st)
    {
        ID3D11SamplerState* v[] = { st.get() };
        ctx->PSSetSamplers(slot, 1, v);
    }

    template <typename t0, typename t1> void vs_set_constant_buffers(ID3D11DeviceContext2* ctx, t0* s0, t1* s1)
    {
        ID3D11Buffer* v[] = { s0->to_constant_buffer(), s1->to_constant_buffer() };
        ctx->VSSetConstantBuffers(0, 2, v);
    }

    template <typename t0, typename t1> void vs_set_constant_buffers(ID3D11DeviceContext2* ctx, const t0& s0, const t1& s1)
    {
        ID3D11Buffer* v[] = { s0.to_constant_buffer(), s1.to_constant_buffer() };
        ctx->VSSetConstantBuffers(0, 2, v);
    }

    template <typename t0> void ia_set_input_layput(ID3D11DeviceContext2* ctx, const t0& s0)
    {
        ctx->IASetInputLayout(s0.to_input_layout());
    }

    class base_application : public gx::default_application
    {
        using base = gx::default_application;

    public:

        struct create_parameters : public base::create_parameters
        {

        };

        base_application(const create_parameters& p) : base(p)
        {
            input_layout_database::database()->initialize(device());

            m_texture          = make_tiled_texture(device(), immediate_context(), 128, 128);
            m_tiled_resource   = make_tiled_render_resource(device(), m_texture);

            m_full_screen      = full_screen_quad_shader::create_shader(device());
            m_clear_color      = clear_color_shader::create_shader(device());
            m_display_texture  = display_texture_shader::create_shader(device());
            m_world_map_pixel  = world_map_pixel_shader::create_shader(device());
            m_world_map_vertex = world_map_vertex_shader::create_shader(device());

            m_stripe_vertex     = stripe_vertex_shader::create_shader(device());
            m_stripe_pixel      = stripe_pixel_shader::create_shader(device());

            m_sampling_vertex   = sampling_vertex_shader::create_shader(device());
            m_sampling_pixel    = sampling_pixel_shader::create_shader(device());

            m_back_buffer_view = d3d11::helpers::create_render_target_view(device(), dxgi::get_buffer(m_context.m_swap_chain, 0));

            m_cull_none        = gx::make_cull_none(device());
            m_cull_back        = gx::make_cull_back(device());
            m_cull_front       = gx::make_cull_front(device());
            m_default_sampler  = gx::make_default_sampler(device());

            auto tga           = img::targa_load(make_tile_file_name(0, 0, 0));

            update_tile(immediate_context(), tga->data(), &m_texture, 0, 0);

            m_world_map_render = make_world_map_render( device() );
            m_pass_constants   = d3d11::helpers::make_constant_buffer<pass_constants>(device());


            m_camera.set_far(50000.0f);
            m_camera.set_view_position(math::set(0.0, 0.0f, -5911.0f, 0.0f));

            m_sampling_renderer = make_world_sampling_renderer(device());
            m_residency_manager = make_residency_manager(device(), &m_texture);

            m_depth_buffer      = gx::create_depth_resource( device(), 1600, 900) ;

            /*
            m_d2d_factory = d2d::helpers::create_d2d_factory_single_threaded();
            m_dwrite_factory    = dwrite::helpers::create_dwrite_factory();
            m_text_format       = dwrite::helpers::create_text_format(m_dwrite_factory);
            */
        }

        ~base_application()
        {

        }

    protected:

        void render_scene()
        {
            on_render_scene();
        }

        virtual void on_update_scene()
        {
            process_user_input();
        }

        void update_scene()
        {
            on_update_scene();
        }

        void on_update() override
        {
            sys::profile_timer timer;

            base::on_update();
            update_scene();

            //Measure the update time and pass it to the render function
            m_elapsed_update_time = timer.milliseconds();
        }

        void on_render_frame() override
        {
            sys::profile_timer timer;
            on_render_scene();
        }

        void process_user_input()
        {
            const float movement_camera_speed = 100.0f;
            const float rotation_camera_speed = 0.002f * 3.1415f;

            auto pad = pad_state();

            if (pad.m_state.m_thumb_left_x != 0.0f || pad.m_state.m_thumb_left_y != 0.0f)
            {
                auto move_x = std::copysign(movement_camera_speed, pad.m_state.m_thumb_left_x);
                auto move_y = std::copysign(movement_camera_speed, pad.m_state.m_thumb_left_y);

                auto a = fabsf(pad.m_state.m_thumb_left_x);
                auto b = fabsf(pad.m_state.m_thumb_left_y);

                gxu::camera_command command;

                if ( a > b )
                {
                     command = gxu::create_move_camera_xy_command(move_x, 0);
                }
                else
                {
                    command = gxu::create_move_camera_xy_command(0, move_y);
                }

                gxu::pinhole_camera_command_dispatcher procesor(&m_camera);
                procesor.process(&command);
            }

            if ( pad.m_state.m_thumb_right_x != 0.0f || pad.m_state.m_thumb_right_y !=  0.0f )
            {
                auto move_x = movement_camera_speed;
                auto move_y = movement_camera_speed;

                auto v = math::set(move_x, move_y, 0.0f, 0.0f);

                float magnitude = math::get_x(math::length2(v));

                if (pad.m_state.m_thumb_right_y < 0.0f)
                {
                    magnitude = -magnitude;
                }

                gxu::camera_command command = gxu::create_move_camera_z_command(magnitude);

                gxu::pinhole_camera_command_dispatcher procesor(&m_camera);
                procesor.process(&command);
            }

            pad.swap();

            //clamp the camera
            auto camera_position = m_camera.position();
            auto camera_min = math::set(-15000.0f, -15000.0f, -35666.0f, 1.0f);
            auto camera_max = math::set(+15000.0f, +15000.0f, -127.0f,  1.0f); // -666.0f
            camera_position = math::clamp(camera_position, camera_min, camera_max);
            m_camera.set_view_position(camera_position);
        }

        virtual void on_render_scene()
        {
            sys::profile_timer timer;

            auto ctx = immediate_context();

            draw_function s = [this]( ID3D11DeviceContext2* ctx)
            {
                ctx->RSSetState(m_cull_back.get());
                ctx->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                vs_setshader(ctx, m_sampling_vertex);
                ps_setshader(ctx, m_sampling_pixel);

                vs_set_vertex_buffer(ctx, 0, &m_world_map_render);
                vs_set_index_buffer(ctx, &m_world_map_render);
                m_world_map_render.update_constants(ctx);

                ps_setshader_resource(ctx, 0, &m_tiled_resource);
                ps_setshader_sampler(ctx, 0, m_default_sampler);

                pass_constants pass;
                pass.m_view = math::transpose(gx::view_matrix(m_camera));
                pass.m_perspective = math::transpose(gx::perspective_matrix(m_camera));
                m_pass_constants.update(ctx, pass);

                vs_set_constant_buffers(ctx, m_pass_constants, m_world_map_render);
                ia_set_input_layput(ctx, m_world_map_render);

                ctx->DrawIndexed(6, 0, 0);
            };

            static bool save_sampling = false;

            m_sampling_renderer.render( ctx, m_frame_count, s );
            auto samples = m_sampling_renderer.sample(ctx, m_frame_count - 2, save_sampling );

            m_residency_manager->process_samples(samples, m_frame_count);
            m_residency_manager->update(ctx);

            ID3D11RenderTargetView* rtv[] = { m_back_buffer_view.get() };

            ctx->OMSetRenderTargets(1, rtv, m_depth_buffer.m_resource_dsv);
            float clear_color[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            ctx->ClearRenderTargetView(m_back_buffer_view.get(), &clear_color[0]);
            ctx->ClearDepthStencilView(m_depth_buffer, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

            D3D11_VIEWPORT v = m_view_port;
            ctx->RSSetViewports(1, &v);

            ctx->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

            vs_setshader(ctx, m_world_map_vertex);
            ps_setshader(ctx, m_world_map_pixel);

            ps_setshader_resource(ctx, 0, &m_tiled_resource);
            ps_setshader_sampler(ctx, 0, m_default_sampler);

            ctx->RSSetState(m_cull_back.get());

            vs_set_vertex_buffer(ctx, 0, &m_world_map_render);
            vs_set_index_buffer(ctx, &m_world_map_render);
            m_world_map_render.update_constants( ctx );

            pass_constants pass;
            pass.m_view        = math::transpose(gx::view_matrix(m_camera));
            pass.m_perspective = math::transpose(gx::perspective_matrix(m_camera));
            m_pass_constants.update(ctx, pass);

            vs_set_constant_buffers(ctx, m_pass_constants, m_world_map_render);
            ia_set_input_layput(ctx, m_world_map_render);

            ctx->DrawIndexed(6, 0, 0);

            //draw stripe
            ctx->RSSetState(m_cull_none.get());
            vs_set_vertex_buffer(ctx, 0, nullptr);
            vs_set_index_buffer(ctx, nullptr);

            vs_setshader(ctx, m_stripe_vertex);
            ps_setshader(ctx, m_stripe_pixel);

            ctx->Draw(12, 0);

            /*

            //Draw the gui and the texts
            m_d2d_render_target->BeginDraw();
            //m_d2d_render_target->Clear();

            RECT r;
            ::GetClientRect(get_window(), &r);

            //Get a description of the GPU or another simulator device
            DXGI_ADAPTER_DESC d;
            m_context.m_adapter->GetDesc(&d);

            D2D1_RECT_F rf = { static_cast<float> (r.left), static_cast<float>(r.top), static_cast<float>(r.right), static_cast<float>(r.bottom) };

            const std::wstring w = L"Update time: " + std::to_wstring(m_elapsed_update_time) + L"ms Render time: " + std::to_wstring(timer.milliseconds()) + L" ms\n";
            const std::wstring w2 = w + d.Description + L" Video Memory(MB): " + std::to_wstring(d.DedicatedVideoMemory / (1024 * 1024)) + L" System Memory(MB): " + std::to_wstring(d.DedicatedSystemMemory / (1024 * 1024)) + L" Shared Memory(MB): " + std::to_wstring(d.SharedSystemMemory / (1024 * 1024));

            m_d2d_render_target->SetTransform(D2D1::Matrix3x2F::Identity());
            m_d2d_render_target->FillRectangle(rf, m_brush2.get());
            m_d2d_render_target->DrawTextW(w2.c_str(), static_cast<uint32_t> (w2.length()), m_text_format.get(), &rf, m_brush.get());
            m_d2d_render_target->EndDraw();
            */

            m_frame_count++;
        }

        void on_resize(uint32_t w, uint32_t h) override
        {
            m_back_buffer_view.reset();

            base::on_resize(w, h);

            auto r = dxgi::get_buffer_as_texture(m_context.m_swap_chain, 0);
            
            D3D11_TEXTURE2D_DESC d;
            r->GetDesc(&d);

            m_back_buffer_view = d3d11::helpers::create_render_target_view( device(), r.get() );

            m_view_port.set_dimensions(d.Width, d.Height);
            m_depth_buffer = gx::create_depth_resource(device(), d.Width, d.Height);

            m_sampling_renderer.create_render_targets( device(), d.Width, d.Height );

            /*
            //Direct 2D resources
            m_d2d_render_target = d2d::helpers::create_render_target(m_d2d_factory, r.get() );
            m_brush             = d2d::helpers::create_solid_color_brush(m_d2d_render_target);
            m_brush2            = d2d::helpers::create_solid_color_brush2(m_d2d_render_target);
            */
        }

    protected:

        double                           m_elapsed_update_time;

        tiled_texture                    m_texture;
        tiled_texture_resource           m_tiled_resource;

        world_map_render                 m_world_map_render;

        full_screen_quad_shader::shader* m_full_screen;
        clear_color_shader::shader*      m_clear_color;
        display_texture_shader::shader*  m_display_texture;

        sampling_vertex_shader::shader*  m_sampling_vertex;
        sampling_pixel_shader::shader*   m_sampling_pixel;

        world_map_vertex_shader::shader* m_world_map_vertex;
        world_map_pixel_shader::shader*  m_world_map_pixel;

        stripe_vertex_shader::shader*    m_stripe_vertex;
        stripe_pixel_shader::shader*     m_stripe_pixel;

        d3d11::render_target_view        m_back_buffer_view;
        gx::depth_resource               m_depth_buffer;

        d3d11::rasterizer_state          m_cull_none;
        d3d11::rasterizer_state          m_cull_back;
        d3d11::rasterizer_state          m_cull_front;
        d3d11::sampler_state             m_default_sampler;

        d3d11::helpers::typed_constant_buffer<pass_constants>   m_pass_constants;

        gx::view_port                    m_view_port;
        gx::pinhole_camera               m_camera;

        world_map_sampling_renderer      m_sampling_renderer;
        size_t                           m_frame_count = 0;

        std::unique_ptr<world_map_residency_manager> m_residency_manager;

        /*
        d2d::factory                      m_d2d_factory;
        dwrite::factory                   m_dwrite_factory;
        d2d::rendertarget		          m_d2d_render_target;
        d2d::solid_color_brush            m_brush;
        d2d::solid_color_brush            m_brush2;
        dwrite::textformat                m_text_format;
        */

    private:

        ID3D11Device2* device() const
        {
            return m_context.m_device.get();
        }

        ID3D11DeviceContext2* immediate_context() const
        {
            return m_context.m_immediate_context.get();
        }
    };
}

