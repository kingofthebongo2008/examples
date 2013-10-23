#ifndef __GX_GLOBAL_BUFFERS_H__
#define __GX_GLOBAL_BUFFERS_H__

#include <cstdint>

#include <d3d11/d3d11_error.h>
#include <d3d11/d3d11_pointers.h>

#include <gx/gx_constant_buffer_helper.h>

namespace lscm
{
    class __declspec(align(16)) visibility_per_pass_buffer final
    {

    public:

        explicit visibility_per_pass_buffer(ID3D11Device* device) :
            m_buffer ( d3d11::create_constant_buffer(device, size() ) )
        {

        }

        void set_view( math::float4x4 value)
        {
            m_view = value;
        }

        void set_projection( math::float4x4 value )
        {
            m_projection = value;
        }

        void set_reverse_projection(math::float4 value)
        {
            m_reverse_projection = value;
        }

        void flush(ID3D11DeviceContext* context)
        {
            gx::constant_buffer_update(context, m_buffer, &m_view, size());
        }

        void bind_as_vertex(ID3D11DeviceContext* context)
        {
            context->VSSetConstantBuffers(gx::slot_per_pass, 1, &m_buffer);
        }

        void bind_as_pixel(ID3D11DeviceContext* context)
        {
            context->PSSetConstantBuffers(gx::slot_per_pass, 1, &m_buffer);
        }

        void bind_as_vertex(ID3D11DeviceContext* context, uint32_t slot)
        {
            context->VSSetConstantBuffers(slot, 1, &m_buffer);
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
            return sizeof( m_view ) + sizeof ( m_projection ) + sizeof( m_reverse_projection );
        }

    private:

        math::float4x4      m_view;
        math::float4x4      m_projection;
        math::float4        m_reverse_projection;

        d3d11::ibuffer_ptr	m_buffer;
    };


    class __declspec(align(16)) global_per_frame_buffer final
    {

    public:

        explicit global_per_frame_buffer(ID3D11Device* device) :
            m_buffer(d3d11::create_constant_buffer(device, size()))
        {

        }

        void set_light_accumulation_buffer_dimensions(uint32_t width, uint32_t height)
        {
            m_light_accumulation_buffer_width = width;
            m_light_accumulation_buffer_height = height;

        }

        std::tuple< uint32_t, uint32_t > get_light_accumulation_buffer_dimensions() const
        {
            return std::make_tuple(m_light_accumulation_buffer_width, m_light_accumulation_buffer_height);
        }

        void flush(ID3D11DeviceContext* context)
        {
            gx::constant_buffer_update(context, m_buffer, &m_light_accumulation_buffer_width, size());
        }

        void bind_as_compute(ID3D11DeviceContext* context, uint32_t slot)
        {
            context->CSSetConstantBuffers(slot, 1, &m_buffer);
        }

        void bind_as_compute(ID3D11DeviceContext* context)
        {
            context->CSSetConstantBuffers(gx::slot_per_frame, 1, &m_buffer);
        }

        void bind_as_vertex(ID3D11DeviceContext* context)
        {
            context->VSSetConstantBuffers(gx::slot_per_frame, 1, &m_buffer);
        }

        void bind_as_pixel(ID3D11DeviceContext* context)
        {
            context->PSSetConstantBuffers(gx::slot_per_frame, 1, &m_buffer);
        }

        void bind_as_vertex(ID3D11DeviceContext* context, uint32_t slot)
        {
            context->VSSetConstantBuffers(slot, 1, &m_buffer);
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
            return sizeof(m_light_accumulation_buffer_width) + sizeof(m_light_accumulation_buffer_height) + sizeof(m_padding);
        }

    private:

        uint32_t            m_light_accumulation_buffer_width;
        uint32_t            m_light_accumulation_buffer_height;

        uint32_t            m_padding[2];

        d3d11::ibuffer_ptr	m_buffer;

    };
}

#endif