#ifndef __GX_COMPUTE_RESOURCE_H__
#define __GX_COMPUTE_RESOURCE_H__

#include <cstdint>
#include <limits>

#include <d3d11/d3d11_helpers.h>
#include <d3d11/dxgi_helpers.h>

namespace gx
{
    class compute_resource
    {
        public:

            compute_resource(
                                    d3d11::ibuffer_ptr                  resource,
                                    d3d11::iunordered_access_view_ptr   resource_uav,
                                    d3d11::ishaderresourceview_ptr	    resource_srv
                                    ) : m_resource(resource), m_resource_uav(resource_uav), m_resource_srv(resource_srv)
        {

        }

            operator ID3D11Buffer* ()
        {
            return m_resource.get();
        }

            operator const ID3D11Buffer* () const
        {
            return m_resource.get();
        }

        operator ID3D11UnorderedAccessView* ()
        {
            return m_resource_uav.get();
        }

        operator const ID3D11UnorderedAccessView* () const
        {
            return m_resource_uav.get();
        }

        operator ID3D11ShaderResourceView* ()
        {
            return m_resource_srv.get();
        }

        operator const ID3D11ShaderResourceView* () const
        {
            return m_resource_srv.get();
        }

        d3d11::ibuffer_ptr                  m_resource;
        d3d11::iunordered_access_view_ptr   m_resource_uav;
        d3d11::ishaderresourceview_ptr      m_resource_srv;

    };

    inline compute_resource create_structured_compute_resource( ID3D11Device* device, uint32_t structure_count, uint32_t structure_size )
    {
        auto resource = d3d11::create_unordered_access_structured_buffer(device, structure_count,  structure_size );

        return compute_resource(resource, d3d11::create_unordered_access_view_structured(device, resource), d3d11::create_shader_resource_view(device, resource));
    }

    inline compute_resource create_structured_compute_resource(ID3D11Device* device, uint32_t structure_count, uint32_t structure_size, const void* initial_data)
    {
        auto resource = d3d11::create_unordered_access_structured_buffer(device, structure_count, structure_size, initial_data);

        return compute_resource(resource, d3d11::create_unordered_access_view_structured(device, resource), d3d11::create_shader_resource_view(device, resource));
    }
}



#endif

