#ifndef __d3d11_EXCEPTION_H__
#define __d3d11_EXCEPTION_H__

#include <os/windows/com_error.h>

namespace d3d11
{
    class exception : public os::windows::com_exception
    {
        public:

        exception ( const HRESULT hr ) : os::windows::com_exception(hr)
        {

        }

    };

    class create_resource_exception : public exception
    {
        public:
        create_resource_exception ( const HRESULT hr ) : exception(hr)
        {

        }
    };

    class create_device_exception : public exception
    {
        public:
        create_device_exception ( const HRESULT hr ) : exception(hr)
        {

        }

    };

    class create_dxgi_factory_exception : public exception
    {
        public:
        create_dxgi_factory_exception ( const HRESULT hr ) : exception(hr)
        {

        }
    };

    class create_swap_chain_exception : public exception
    {
        public:
        create_swap_chain_exception ( const HRESULT hr ) : exception(hr)
        {

        }
    };

    class create_texture_exception : public create_resource_exception
    {
        public:
        create_texture_exception ( const HRESULT hr ) : create_resource_exception(hr)
        {

        }
    };

    class create_texture2d_exception : public create_texture_exception
    {
        public:
        create_texture2d_exception ( const HRESULT hr ) : create_texture_exception(hr)
        {

        }
    };

    class create_buffer_exception : public create_resource_exception
    {
        public:
        create_buffer_exception ( const HRESULT hr ) : create_resource_exception(hr)
        {

        }
    };

    class create_view_exception : public create_resource_exception
    {
        public:
        create_view_exception ( const HRESULT hr ) : create_resource_exception(hr)
        {

        }
    };

    class create_shader_exception : public create_resource_exception
    {
        public:
        create_shader_exception ( const HRESULT hr ) : create_resource_exception(hr)
        {

        }
    };

    class create_state_exception : public create_resource_exception
    {
        public:
        create_state_exception ( const HRESULT hr ) : create_resource_exception(hr)
        {

        }
    };

    class create_blend_state_exception : public create_state_exception
    {
        public:
        create_blend_state_exception ( const HRESULT hr ) : create_state_exception(hr)
        {

        }
    };

    class create_depth_stencil_state_exception : public create_state_exception
    {
        public:
        create_depth_stencil_state_exception ( const HRESULT hr ) : create_state_exception(hr)
        {

        }
    };

    class create_rasterizer_state_exception : public create_state_exception
    {
        public:
        create_rasterizer_state_exception ( const HRESULT hr ) : create_state_exception(hr)
        {

        }
    };

    class create_sampler_state_exception : public create_state_exception
    {
        public:
        create_sampler_state_exception ( const HRESULT hr ) : create_state_exception(hr)
        {

        }
    };

	class create_deferred_context_exception : public exception
    {
        public:
        create_deferred_context_exception ( const HRESULT hr ) : exception(hr)
        {

        }
    };

	class create_depth_stencil_view_exception : public create_resource_exception
    {
        public:
        create_depth_stencil_view_exception ( const HRESULT hr ) : create_resource_exception(hr)
        {

        }
    };

	class create_render_target_view_exception : public create_resource_exception
    {
        public:
        create_render_target_view_exception ( const HRESULT hr ) : create_resource_exception(hr)
        {

        }
    };

	class create_resource_view_exception : public create_resource_exception
    {
        public:
        create_resource_view_exception ( const HRESULT hr ) : create_resource_exception(hr)
        {

        }
    };

    class create_shader_resource_view_exception : public create_resource_exception
    {
        public:
        create_shader_resource_view_exception ( const HRESULT hr ) : create_resource_exception(hr)
        {

        }
    };

	class create_vertex_shader : public create_resource_exception
    {
        public:
        create_vertex_shader ( const HRESULT hr ) : create_resource_exception(hr)
        {

        }
    };

	class create_pixel_shader : public create_resource_exception
    {
        public:
        create_pixel_shader ( const HRESULT hr ) : create_resource_exception(hr)
        {

        }
    };

    class create_geometry_shader : public create_resource_exception
    {
        public:
        create_geometry_shader ( const HRESULT hr ) : create_resource_exception(hr)
        {

        }
    };

    class create_input_layout : public create_resource_exception
    {
        public:
        create_input_layout ( const HRESULT hr ) : create_resource_exception(hr)
        {

        }
    };

    class create_unordered_access_view_exception : public create_resource_exception
    {
        public:
        create_unordered_access_view_exception ( const HRESULT hr ) : create_resource_exception(hr)
        {

        }
    };
}


#endif