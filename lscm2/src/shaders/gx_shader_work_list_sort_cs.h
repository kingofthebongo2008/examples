#ifndef __GX_SHADERS_WORK_LIST_SORT_CS_H__
#define __GX_SHADERS_WORK_LIST_SORT_CS_H__

#include <cstdint>
#include <future>

#include <d3d11/d3d11_error.h>
#include <d3d11/d3d11_pointers.h>

namespace lscm
{
    inline d3d11::icomputeshader_ptr   create_shader_work_list_sort_cs(ID3D11Device* device)
    {
        d3d11::icomputeshader_ptr   shader;

        using namespace os::windows;

        //strange? see in the hlsl file
        static
        #include "gx_shader_work_list_sort_cs_compiled.hlsl"

        //load, compile and create a pixel shader with the code in the hlsl file, might get slow (this is a compilation), consider offloading to another thread
        throw_if_failed<d3d11::create_pixel_shader>(device->CreateComputeShader(gx_shader_work_list_sort_cs, sizeof(gx_shader_work_list_sort_cs), nullptr, &shader));

        return shader;
    }

    std::future< d3d11::icomputeshader_ptr> create_shader_work_list_sort_cs_async(ID3D11Device* device)
    {
        return std::async(std::launch::async, create_shader_work_list_sort_cs, device);
    }


    class shader_work_list_sort_cs final
    {
        public:
        explicit shader_work_list_sort_cs(ID3D11Device* device) : m_shader(create_shader_work_list_sort_cs ( device) )
        {

        }

        explicit shader_work_list_sort_cs(d3d11::icomputeshader_ptr   shader) : m_shader(shader)
        {

        }

        operator ID3D11ComputeShader* () const
        {
            return m_shader.get();
        }

        d3d11::icomputeshader_ptr   m_shader;
    };
}

#endif