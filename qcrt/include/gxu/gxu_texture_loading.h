#ifndef __GXU_TEXTURE_LOADING_H__
#define __GXU_TEXTURE_LOADING_H__

#include <cstdint>

#include <d3d11/d3d11_pointers.h>

#include <directxtk/inc/ddstextureloader.h>
#include <directxtk/inc/wictextureloader.h>

namespace gxu
{
    inline d3d11::itexture2d_ptr load_texture_dds(ID3D11Device* device, const wchar_t* file_name) throw()
    {
        d3d11::itexture2d_ptr result;
        d3d11::iresource_ptr  resource;

        HRESULT hresult = DirectX::CreateDDSTextureFromFile(device, file_name, &resource, 0);

        if (hresult == S_OK)
        {
            hresult = resource->QueryInterface(__uuidof(ID3D11Texture2D),(void**) &result);

            if (hresult == S_OK)
            {
                return std::move(result);
            }
        }

        return result;
    }

    inline d3d11::itexture2d_ptr load_texture_wic(ID3D11Device* device, ID3D11DeviceContext* device_context, const wchar_t* file_name)  throw()
    {
        d3d11::itexture2d_ptr result;
        d3d11::iresource_ptr  resource;


        HRESULT hresult = DirectX::CreateWICTextureFromFile(device, device_context,  file_name, &resource, 0);

        if (hresult == S_OK)
        {
            hresult = resource->QueryInterface(__uuidof(ID3D11Texture2D),(void**) &result);

            if (hresult == S_OK)
            {
                return std::move(result);
            }
        }

        return result;
    }

    inline std::future< d3d11::itexture2d_ptr> load_texture_wic_async(ID3D11Device* device, ID3D11DeviceContext* device_context, const wchar_t* file_name )
    {
        return std::move(std::async(std::launch::async, load_texture_wic, device, device_context, file_name));
    }

    inline std::future< d3d11::itexture2d_ptr> load_texture_dds_async(ID3D11Device* device, const wchar_t* file_name)
    {
        return std::move(std::async(std::launch::async, load_texture_dds, device, file_name));
    }
}

#endif