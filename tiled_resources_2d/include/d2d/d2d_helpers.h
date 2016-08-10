#pragma once

#include <cstdint>

#include <DXGI.h>
#include <DXGI1_2.h>

#include <d2d1.h>
#include <d2d1_2.h>
#include <d2d1_3.h>

#include <d2d/d2d_pointers.h>
#include <d2d/d2d_error.h>

#include <d3d11.h>

#include <os/windows/dxgi_pointers.h>

namespace d2d
{
    namespace helpers
    {
        inline factory create_d2d_factory_single_threaded()
        {
            factory result;

            throw_if_failed(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &result));

            return result;
        }

        inline factory create_d2d_factory_multithreaded()
        {
            factory result;

            throw_if_failed(D2D1CreateFactory(D2D1_FACTORY_TYPE_MULTI_THREADED, &result));

            return result;
        }

        inline D2D1_RENDER_TARGET_PROPERTIES create_default_render_target_properties(factory factory)
        {
            float	dpi_x = 0.0f;
            float	dpi_y = 0.0f;

            factory->GetDesktopDpi(&dpi_x, &dpi_y);

            D2D1_PIXEL_FORMAT pixel_format =
            {
                    DXGI_FORMAT_UNKNOWN,     
                    D2D1_ALPHA_MODE_PREMULTIPLIED
            };

            D2D1_RENDER_TARGET_PROPERTIES properties =
            {
                D2D1_RENDER_TARGET_TYPE_DEFAULT,
                pixel_format,
                dpi_x,
                dpi_y,
                D2D1_RENDER_TARGET_USAGE_NONE,
                D2D1_FEATURE_LEVEL_10
            };

            return properties;
        }

        inline rendertarget create_render_target(factory factory, dxgi::surface surface)
        {
            rendertarget result;

            D2D1_RENDER_TARGET_PROPERTIES properties = create_default_render_target_properties(factory);
            throw_if_failed(factory->CreateDxgiSurfaceRenderTarget(surface.get(), &properties, &result));
            return result;
        }

        inline rendertarget create_render_target(factory factory, ID3D11Texture2D* const texture)
        {
            dxgi::surface surface;
            throw_if_failed(texture->QueryInterface(IID_IDXGISurface, reinterpret_cast<void**> (&surface)));
            return create_render_target(factory, surface);
        }

        inline solid_color_brush create_solid_color_brush(rendertarget render_target)
        {
            solid_color_brush result;
            throw_if_failed(render_target->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::White, 1.0f), &result));
            return result;
        }

        inline solid_color_brush create_solid_color_brush2(rendertarget render_target)
        {
            solid_color_brush result;

            const float fraction = 1.0f / 32.0f;
            throw_if_failed(render_target->CreateSolidColorBrush(D2D1::ColorF(fraction, fraction, fraction, fraction), &result));
            return result;
        }
    }
}




