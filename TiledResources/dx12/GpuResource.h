#pragma once

namespace TiledResources
{
    class GpuResource
    {
        public: 

        GpuResource(ID3D12Resource* resource) : m_resource(resource)
        {

        }

        operator const ID3D12Resource*() const
        {
            return m_resource.Get();
        }

        operator ID3D12Resource*()
        {
            return m_resource.Get();
        }

        D3D12_GPU_VIRTUAL_ADDRESS GetVirtualAddress() const
        {
            return m_resource->GetGPUVirtualAddress();
        }

        D3D12_RESOURCE_DESC GetDesc() const
        {
            return m_resource->GetDesc();
        }

        ID3D12Resource* GetResource() const
        {
            return m_resource.Get();
        }
        
        protected:
        Microsoft::WRL::ComPtr<ID3D12Resource> m_resource;
    };
}
