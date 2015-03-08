#ifndef __composer_render_context_h__
#define __composer_render_context_h__

#include <d3d11/d3d11_pointers.h>

#include <gx/gx_view_port.h>

namespace coloryourway
{
    namespace composer
    {
        class render_context
        {
            public:

            render_context
                ( 
                d3d11::idevicecontext_ptr                device_context
                , d3d11::iblendstate_ptr                  opaque_state
                , d3d11::iblendstate_ptr                  premultiplied_alpha_state

                , d3d11::iblendstate_ptr                  alpha_blend_state
                , d3d11::irasterizerstate_ptr             cull_back_raster_state
                , d3d11::irasterizerstate_ptr             cull_none_raster_state

                , d3d11::idepthstencilstate_ptr           depth_disable_state
                , d3d11::isamplerstate_ptr                point_sampler

                , gx::view_port                           view_port
                ) :
                m_device_context(device_context)
                , m_opaque_state( opaque_state )
                , m_premultiplied_alpha_state(premultiplied_alpha_state)
                , m_alpha_blend_state(alpha_blend_state)
                , m_cull_back_raster_state(cull_back_raster_state)
                , m_cull_none_raster_state(cull_none_raster_state)
                , m_depth_disable_state(depth_disable_state)
                , m_point_sampler(point_sampler)
                , m_view_port(view_port)
            {

            }

            ID3D11BlendState* get_opaque_state() const
            {
                return m_opaque_state.get();
            }

            private:

            d3d11::idevicecontext_ptr               m_device_context;

            d3d11::iblendstate_ptr                  m_opaque_state;
            d3d11::iblendstate_ptr                  m_premultiplied_alpha_state;

            d3d11::iblendstate_ptr                  m_alpha_blend_state;
            d3d11::irasterizerstate_ptr             m_cull_back_raster_state;
            d3d11::irasterizerstate_ptr             m_cull_none_raster_state;

            d3d11::idepthstencilstate_ptr           m_depth_disable_state;
            d3d11::isamplerstate_ptr                m_point_sampler;

            gx::view_port                           m_view_port;

        };
    }
}


#endif

