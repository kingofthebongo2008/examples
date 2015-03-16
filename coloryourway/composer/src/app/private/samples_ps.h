#ifndef __COMPOSER_SAMPLES_PS_H__
#define __COMPOSER_SAMPLES_PS_H__

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
            inline d3d11::ipixelshader_ptr   create_shader_samples_ps(ID3D11Device* device)
            {
                d3d11::ipixelshader_ptr   shader;

                using namespace os::windows;

                //strange? see in the hlsl file
                static
                #include "samples_ps_compiled.hlsl"

                    //load, compile and create a pixel shader with the code in the hlsl file, might get slow (this is a compilation), consider offloading to another thread
                    throw_if_failed<d3d11::create_geometry_shader>(device->CreatePixelShader(g_pixel_main, sizeof(g_pixel_main), nullptr, &shader));
                return shader;
            }
        }

        class shader_samples_ps final
        {

        public:
            shader_samples_ps()
            {

            }

            explicit shader_samples_ps(d3d11::ipixelshader_ptr shader) : m_shader(shader)
            {

            }

            explicit shader_samples_ps(shader_samples_ps&& o) : m_shader(std::move(o.m_shader))
            {

            }

            operator ID3D11PixelShader* () const
            {
                return m_shader.get();
            }

            shader_samples_ps& operator=(shader_samples_ps&& o)
            {
                m_shader = std::move(o.m_shader);
                return *this;
            }

            d3d11::ipixelshader_ptr     m_shader;
        };

        inline shader_samples_ps   create_shader_samples_ps(ID3D11Device* device)
        {
            return shader_samples_ps(std::move(details::create_shader_samples_ps(device)));
        }

        inline std::future< shader_samples_ps> create_shader_samples_ps_async(ID3D11Device* device)
        {
            return std::async(std::launch::async, create_shader_samples_ps, device);
        }
    }
}

#endif
