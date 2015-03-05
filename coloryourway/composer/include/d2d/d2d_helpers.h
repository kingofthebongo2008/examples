#ifndef __d2d_helpers_h__
#define __d2d_helpers_h__

#include <cstdint>

#include <DXGI.h>
#include <DXGI1_2.h>

#include <d2d1.h>

#include <d2d/d2d_pointers.h>
#include <d2d/d2d_error.h>

#include <d3d11.h>

#include <os/windows/dxgi_pointers.h>

namespace d2d
{
    inline ifactory_ptr create_d2d_factory_single_threaded()
    {
        using namespace os::windows;

        ifactory_ptr result;

        throw_if_failed<exception> ( D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &result)) ;

        return result;
    }

    inline ifactory_ptr create_d2d_factory_multithreaded()
    {
        using namespace os::windows;

        ifactory_ptr result;

        throw_if_failed<exception> ( D2D1CreateFactory(D2D1_FACTORY_TYPE_MULTI_THREADED, &result)) ;

        return result;
    }

    inline D2D1_RENDER_TARGET_PROPERTIES create_default_render_target_properties( ifactory_ptr factory )
    {

        float	dpi_x = 0.0f;
        float	dpi_y = 0.0f;

        factory->GetDesktopDpi(&dpi_x, &dpi_y);

        D2D1_PIXEL_FORMAT pixel_format = 
        {
                DXGI_FORMAT_R8G8B8A8_UNORM,
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

    inline irendertarget_ptr create_render_target(ifactory_ptr factory, dxgi::isurface_ptr surface)
    {
        using namespace os::windows;

        irendertarget_ptr result;
        
        D2D1_RENDER_TARGET_PROPERTIES properties = create_default_render_target_properties(factory);

        throw_if_failed<exception> ( factory->CreateDxgiSurfaceRenderTarget( surface, &properties, &result) );

        return result;
    }

    inline irendertarget_ptr create_render_target( ifactory_ptr factory,  ID3D11Texture2D* const texture )
    {
        using namespace os::windows;
        dxgi::isurface_ptr surface;
        throw_if_failed<exception>(texture->QueryInterface(IID_IDXGISurface, reinterpret_cast<void**> (&surface)));

        return create_render_target(factory, surface);
    }

    inline isolid_color_brush_ptr create_solid_color_brush(irendertarget_ptr render_target)
    {
        using namespace os::windows;
        isolid_color_brush_ptr result;
        throw_if_failed<exception>(render_target->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::White, 1.0f), &result));
        return result;
    }

    inline isolid_color_brush_ptr create_solid_color_brush2(irendertarget_ptr render_target)
    {
        using namespace os::windows;
        isolid_color_brush_ptr result;

        const float fraction = 1.0f / 32.0f;
        throw_if_failed<exception>(render_target->CreateSolidColorBrush(D2D1::ColorF( fraction, fraction, fraction, fraction ), &result));
        return result;
    }
}


#endif

