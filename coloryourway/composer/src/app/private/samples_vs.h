#ifndef __COMPOSER_SAMPLES_VS_H__
#define __COMPOSER_SAMPLES_VS_H__

#include <cstdint>
#include <future>

#include <d3d11/d3d11_error.h>
#include <d3d11/d3d11_pointers.h>

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

        std::future< shader_samples_vs > create_shader_samples_vs_async(ID3D11Device* device)
        {
            return std::async(std::launch::async, create_shader_samples_vs, device);
        }
    }
}

#endif
