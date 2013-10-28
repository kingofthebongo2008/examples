#ifndef __GX_SHADERS_CLEAR_LIGHT_ACCUMULATION_CS_H__
#define __GX_SHADERS_CLEAR_LIGHT_ACCUMULATION_CS_H__

#include <cstdint>
#include <d3d11/d3d11_error.h>
#include <d3d11/d3d11_pointers.h>

namespace lscm
{
    class shader_clear_light_accumulation_cs final
    {
        public:
            explicit shader_clear_light_accumulation_cs(ID3D11Device* device)
        {
            using namespace os::windows;

            //strange? see in the hlsl file
            static 
            #include "gx_shader_clear_light_accumulation_cs_compiled.hlsl"

            //load, compile and create a pixel shader with the code in the hlsl file, might get slow (this is a compilation), consider offloading to another thread
            throw_if_failed<d3d11::create_pixel_shader>(device->CreateComputeShader(gx_shader_clear_light_accumulation_cs, sizeof(gx_shader_clear_light_accumulation_cs), nullptr, &m_shader));
            m_code = &gx_shader_clear_light_accumulation_cs[0];
            m_code_size = sizeof(gx_shader_clear_light_accumulation_cs);
        }

        operator ID3D11ComputeShader* () const
        {
            return m_shader.get();
        }

        d3d11::icomputeshader_ptr   m_shader;
        const void*                 m_code;
        uint32_t                    m_code_size;
    };
}

#endif