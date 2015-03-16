#ifndef __COMPOSER_SAMPLES_GS_H__
#define __COMPOSER_SAMPLES_GS_H__

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
            inline d3d11::igeometryshader_ptr   create_shader_samples_gs(ID3D11Device* device)
            {
                d3d11::igeometryshader_ptr   shader;

                using namespace os::windows;

                //strange? see in the hlsl file
                static
                #include "samples_gs_compiled.hlsl"

                    //load, compile and create a pixel shader with the code in the hlsl file, might get slow (this is a compilation), consider offloading to another thread
                    throw_if_failed<d3d11::create_geometry_shader>(device->CreateGeometryShader(g_geometry_main, sizeof(g_geometry_main), nullptr, &shader));
                return shader;
            }
        }

        class shader_samples_gs final
        {
        public:

            shader_samples_gs()
            {

            }

            explicit shader_samples_gs(d3d11::igeometryshader_ptr shader) : m_shader(shader)
            {

            }


            shader_samples_gs(shader_samples_gs&&  o) : m_shader(std::move(o.m_shader) )
            {

            }

            operator ID3D11GeometryShader* () const
            {
                return m_shader.get();
            }


            shader_samples_gs& operator=(shader_samples_gs&& o)
            {
                m_shader = std::move(o.m_shader);
                return *this;
            }

            d3d11::igeometryshader_ptr     m_shader;
        };

        inline shader_samples_gs   create_shader_samples_gs(ID3D11Device* device)
        {
            return shader_samples_gs(std::move(details::create_shader_samples_gs(device)));
        }

        inline std::future< shader_samples_gs > create_shader_samples_gs_async(ID3D11Device* device)
        {
            return std::async(std::launch::async, create_shader_samples_gs, device );
        }

    }
}

#endif
