#ifndef __COMPOSER_SAMPLES_VS_H__
#define __COMPOSER_SAMPLES_VS_H__

#include <cstdint>
#include <future>

#include <d3d11/d3d11_error.h>
#include <d3d11/d3d11_pointers.h>
#include <d3d11/d3d11_helpers.h>

#include <gx/gx_constant_buffer_helper.h>

namespace coloryourway
{
    namespace composer
    {
        namespace details
        {
            inline d3d11::ivertexshader_ptr   create_shader_samples_vs(ID3D11Device* device)
            {
                d3d11::ivertexshader_ptr   shader;

                using namespace os::windows;

                //strange? see in the hlsl file
                static
                #include "samples_vs_compiled.hlsl"

                    //load, compile and create a pixel shader with the code in the hlsl file, might get slow (this is a compilation), consider offloading to another thread
                    throw_if_failed<d3d11::create_vertex_shader>(device->CreateVertexShader(g_vertex_main, sizeof(g_vertex_main), nullptr, &shader));
                return shader;
            }
        }

        class shader_samples_vs final
        {
        public:
            shader_samples_vs()
            {

            }

            explicit shader_samples_vs(d3d11::ivertexshader_ptr shader) : m_shader(shader)
            {

            }

            shader_samples_vs( shader_samples_vs&& shader ) : m_shader( std::move(shader))
            {

            }

            operator ID3D11VertexShader* () const
            {
                return m_shader.get();
            }

            shader_samples_vs& operator = (shader_samples_vs&& o)
            {
                m_shader = std::move(o.m_shader);
                return *this;
            }

            d3d11::ivertexshader_ptr     m_shader;
        };

        inline shader_samples_vs   create_shader_samples_vs(ID3D11Device* device)
        {
            return shader_samples_vs(std::move(details::create_shader_samples_vs(device)));
        }

        inline std::future< shader_samples_vs > create_shader_samples_vs_async(ID3D11Device* device)
        {
            return std::async(std::launch::async, create_shader_samples_vs, device);
        }

        class shader_samples_vs_layout
        {
            public:

                shader_samples_vs_layout(ID3D11Device* device)
                {
                    D3D11_INPUT_ELEMENT_DESC desc[] =
                    {
                        { "position", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }

                    };

                    //strange? see in the hlsl file
                    static
                    #include "samples_vs_compiled.hlsl"

                    using namespace os::windows;

                    throw_if_failed<d3d11::create_input_layout_exception>(device->CreateInputLayout(&desc[0], sizeof(desc) / sizeof(desc[0]), g_vertex_main, sizeof(g_vertex_main), &m_input_layout));
                }

                shader_samples_vs_layout(shader_samples_vs_layout&& o) : m_input_layout(std::move(o.m_input_layout))
                {

                }

                operator d3d11::iinputlayout_ptr()
                {
                    return m_input_layout;
                }

                operator ID3D11InputLayout*()
                {
                    return m_input_layout.get();
                }

                operator const ID3D11InputLayout*() const
                {
                    return m_input_layout.get();
                }

                shader_samples_vs_layout& operator=(shader_samples_vs_layout&& o)
                {
                    m_input_layout = std::move(o.m_input_layout);
                }


                d3d11::iinputlayout_ptr	m_input_layout;
        };

        class constant_buffer
        {

            public:

                explicit constant_buffer(d3d11::ibuffer_ptr	buffer) : m_buffer(buffer)
            {
                D3D11_BUFFER_DESC d = {};
                buffer->GetDesc(&d);
                m_shadow.resize(d.ByteWidth);
            }

            constant_buffer(const constant_buffer& o) : m_buffer(o.m_buffer), m_shadow(o.m_shadow)
            {

            }

            constant_buffer(constant_buffer&& o) : m_buffer( std::move(o.m_buffer)) , m_shadow(std::move(o.m_shadow))
            {

            }

            constant_buffer& operator=(const constant_buffer& o)
            {
                m_buffer = o.m_buffer;
                m_shadow = o.m_shadow;
                return *this;
            }

            constant_buffer& operator=(const constant_buffer&& o)
            {
                m_buffer = std::move(o.m_buffer);
                m_shadow = std::move(o.m_shadow);
                return *this;
            }

            operator ID3D11Buffer*()
            {
                return m_buffer.get();
            }

            operator const ID3D11Buffer*() const
            {
                return m_buffer.get();
            }

            template <typename t> t* map()
            {
                return reinterpret_cast<t*> (&m_shadow[0]);
            }

            void unmap()
            {

            }

            void flush( ID3D11DeviceContext* context )
            {
                gx::constant_buffer_update( context, m_buffer, &m_shadow[0], m_shadow.size() );
            }

            protected:

            d3d11::ibuffer_ptr      m_buffer;
            std::vector<uint8_t>    m_shadow;
        };

        class shader_samples_vs_constant_buffer final : public constant_buffer
        {
            typedef constant_buffer base;
            
            public:
            
            enum { size = sizeof(uint32_t) };


            explicit shader_samples_vs_constant_buffer( d3d11::ibuffer_ptr	buffer ) : base( buffer)
            {

            }


            shader_samples_vs_constant_buffer(const shader_samples_vs_constant_buffer& o) : base(o)
            {

            }

            shader_samples_vs_constant_buffer(shader_samples_vs_constant_buffer&& o) : base(std::move(o))
            {

            }

            shader_samples_vs_constant_buffer& operator=(const shader_samples_vs_constant_buffer& o)
            {
                base::operator=(o);
                return *this;
            }

            constant_buffer& operator=(const constant_buffer&& o)
            {
                base::operator=(std::move(o));
                return *this;
            }

            void set_instance_offset(uint32_t offset)
            {
                auto i = map<uint32_t>();

                *i = offset;

                unmap();
            }
        };

        inline shader_samples_vs_constant_buffer create_samples_vs_constant_buffer(ID3D11Device* device)
        {
            return shader_samples_vs_constant_buffer( d3d11::create_constant_buffer(device, shader_samples_vs_constant_buffer::size) );
        }

    }
}

#endif
