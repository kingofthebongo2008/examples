#pragma once

#include <unordered_set>

#include <d3d11/d3d11_helpers.h>
#include <gx/gx_render_resource.h>
#include <img/targa_image.h>

#include "tile.h"

namespace app
{
    using draw_function = std::function< void(ID3D11DeviceContext2*) >;

    struct world_map_sampling_renderer
    {
        gx::render_target_resource  m_render_target;
        gx::depth_resource          m_depth_taget;
        d3d11::texture2d            m_staging_sampling[3];
        std::unique_ptr<uint8_t[]>  m_cpu_data[3];
        gx::view_port               m_view_port;

        uint32_t width() const
        {
            return m_view_port.get_width();
        }

        uint32_t height() const
        {
            return m_view_port.get_height();
        }

        void create_render_targets(ID3D11Device*device, uint32_t width, uint32_t height)
        {
            auto w = (width + 7) / 8;
            auto h = (height + 7) / 8;

            auto f = DXGI_FORMAT_R8G8B8A8_UINT;

            m_render_target = gx::create_render_target_resource(device, w, h, f);
            m_depth_taget = gx::create_depth_resource(device, w, h);

            m_staging_sampling[0] = d3d11::helpers::create_staging_texture_2d(device, w, h, f);
            m_staging_sampling[1] = d3d11::helpers::create_staging_texture_2d(device, w, h, f);
            m_staging_sampling[2] = d3d11::helpers::create_staging_texture_2d(device, w, h, f);

            m_view_port.set_dimensions( w, h );

            m_cpu_data[0] = std::unique_ptr<uint8_t[]>(new uint8_t[w * h * 4]);
            m_cpu_data[1] = std::unique_ptr<uint8_t[]>(new uint8_t[w * h * 4]);
            m_cpu_data[2] = std::unique_ptr<uint8_t[]>(new uint8_t[w * h * 4]);
        }

        void render( ID3D11DeviceContext2* ctx, size_t frame_count, draw_function f  )
        {
            ID3D11RenderTargetView* rtv[] = { m_render_target };

            ctx->OMSetRenderTargets(1, rtv, m_depth_taget);
            float clear_color[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            ctx->ClearRenderTargetView(m_render_target, &clear_color[0]);
            ctx->ClearDepthStencilView(m_depth_taget, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

            D3D11_VIEWPORT v = m_view_port;
            ctx->RSSetViewports(1, &v);

            f(ctx);

            auto buffer = frame_count % 3;
            ctx->CopyResource( m_staging_sampling[buffer], m_render_target);
        }

        void* row_start(void* start, size_t row, size_t row_pitch)
        {
            uintptr_t p = reinterpret_cast<uintptr_t>(start);
            return reinterpret_cast<void*>(p + row * row_pitch);
        }

        std::vector<tile_coordinates> sample( ID3D11DeviceContext2* ctx, size_t frame_count, bool save_image)
        {
            auto buffer     = frame_count % 3;
            auto& staging   = m_staging_sampling[buffer];
            auto& cpu_data  = m_cpu_data[buffer];

            D3D11_MAPPED_SUBRESOURCE r;
            d3d11::throw_if_failed(ctx->Map(staging.get(), 0, D3D11_MAP_READ, 0, &r ));

            auto w = width();
            auto h = height();

            for (auto i = 0U; i < h; ++i)
            {
                auto src = row_start( r.pData, i, r.RowPitch );
                auto tgt = row_start( cpu_data.get(), i, w * 4 );

                std::memcpy( tgt, src, w * 4);
            }

            ctx->Unmap( staging.get(), 0 );
            

            //todo: move this to the gpu
            std::unordered_set< tile_coordinates > tiles;

            for (auto i = 0U; i < h; ++i)
            {
                auto row = row_start(cpu_data.get(), i, w * 4);

                auto tile_row = reinterpret_cast<tile_coordinates*>(row);

                for ( auto j = 0U; j < w; ++j )
                {
                    if (tile_row[j].m_padding > 0)
                    {
                        if (tiles.find( tile_row[j] ) == tiles.end())
                        {
                            tiles.insert(tile_row[j]);
                        }
                    }
                }
            }

            if ( save_image )
            {
                img::targa_write_rgb_rle("sampling.tga", cpu_data.get(), w, h);
            }

            std::vector<tile_coordinates> result;

            result.resize(tiles.size());

            std::copy(tiles.begin(), tiles.end(), result.begin());

            return result;
        }
    };

    inline world_map_sampling_renderer make_world_sampling_renderer( ID3D11Device* device )
    {
        world_map_sampling_renderer r;
        
        r.create_render_targets(device, 1600, 900);

        return r;
    }
}
