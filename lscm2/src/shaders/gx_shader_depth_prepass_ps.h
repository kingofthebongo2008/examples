#ifndef __GX_SHADERS_DEPTH_PREPASS_PS_H__
#define __GX_SHADERS_DEPTH_PREPASS_PS_H__

#include <cstdint>
#include <future>

#include <d3d11/d3d11_error.h>
#include <d3d11/d3d11_pointers.h>
#include <gx/gx_constant_buffer_helper.h>

namespace lscm
{
    class __declspec(align(16)) shader_depth_prepass_ps_buffer final
    {
    public:

        explicit shader_depth_prepass_ps_buffer(ID3D11Device* device) :
            m_buffer ( d3d11::create_constant_buffer(device, size() ) )
        {

        }

        void set_instance_id(uint32_t value)
        {
            m_instance_id = value;
        }

        void update(ID3D11DeviceContext* context, uint32_t instance_id)
        {
            gx::constant_buffer_update(context, m_buffer.get(), instance_id);
        }

        void flush(ID3D11DeviceContext* context)
        {
            update(context, m_instance_id);
        }

        void bind_as_pixel(ID3D11DeviceContext* context)
        {
            context->PSSetConstantBuffers(gx::slot_per_draw_call, 1, &m_buffer);
        }

        void bind_as_pixel(ID3D11DeviceContext* context, uint32_t slot)
        {
            context->PSSetConstantBuffers(slot, 1, &m_buffer);
        }

        operator ID3D11Buffer*()
        {
            return m_buffer.get();
        }

        operator const ID3D11Buffer*() const
        {
            return m_buffer.get();
        }

        size_t size() const
        {
            return sizeof(m_instance_id);
        }

    private:

        d3d11::ibuffer_ptr	m_buffer;
        uint32_t            m_instance_id;
    };

    inline d3d11::ipixelshader_ptr   create_shader_depth_prepass_ps(ID3D11Device* device)
    {
        d3d11::ipixelshader_ptr   shader;

        using namespace os::windows;

        static
        #include "gx_shader_depth_prepass_ps_compiled.hlsl"

        throw_if_failed<d3d11::create_pixel_shader>(device->CreatePixelShader(gx_shader_depth_prepass_ps, sizeof(gx_shader_depth_prepass_ps), nullptr, &shader));

        return shader;
    }

    std::future< d3d11::ipixelshader_ptr> create_shader_depth_prepass_ps_async(ID3D11Device* device)
    {
        return std::async(std::launch::async, create_shader_depth_prepass_ps, device);
    }

    class shader_depth_prepass_ps final
    {
        public:
        explicit shader_depth_prepass_ps(d3d11::ipixelshader_ptr shader) : m_shader(shader)
        {
        
        }

        operator ID3D11PixelShader* const() const
        {
            return m_shader.get();
        }

        d3d11::ipixelshader_ptr     m_shader;
    };
}

#endif