#include "pch.h"

#include "GpuResourceCreateContext.h"
#include "GpuPixelFormat.h"

namespace TiledResources
{
    namespace details
    {
        inline D3D12_RESOURCE_DESC DescribeBuffer(UINT elements, UINT elementSize = 1)
        {
            D3D12_RESOURCE_DESC desc = {};
            desc.Alignment = 0;
            desc.DepthOrArraySize = 1;
            desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Flags = D3D12_RESOURCE_FLAG_NONE;
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.Height = 1;
            desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            desc.MipLevels = 1;
            desc.SampleDesc.Count = 1;
            desc.SampleDesc.Quality = 0;
            desc.Width = elements * elementSize;
            return desc;
        }

        inline D3D12_RESOURCE_DESC DescribeColorBuffer(UINT width, UINT height, UINT d, DXGI_FORMAT format, UINT flags)
        {
            D3D12_RESOURCE_DESC Desc = {};
            Desc.Alignment = 0;
            Desc.DepthOrArraySize = (UINT16)d;
            Desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            Desc.Flags = (D3D12_RESOURCE_FLAGS)flags;
            Desc.Format = GetBaseFormat(format);
            Desc.Height = height;
            Desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            Desc.MipLevels = 1;
            Desc.SampleDesc.Count = 1;
            Desc.SampleDesc.Quality = 0;
            Desc.Width = width;
            return Desc;
        }

        inline D3D12_RESOURCE_DESC DescribeDepthBuffer(UINT width, UINT height, UINT d, DXGI_FORMAT format, UINT flags)
        {
            return DescribeColorBuffer(width, height, d, format, flags);
        }

        inline D3D12_RESOURCE_DESC DescribeColorBuffer(UINT width, UINT height, DXGI_FORMAT format, UINT flags)
        {
            return DescribeColorBuffer(width, height, 1, format, flags);
        }

        inline D3D12_RESOURCE_DESC DescribeDepthBuffer(UINT width, UINT height, DXGI_FORMAT format, UINT flags)
        {
            return DescribeColorBuffer(width, height, 1, format, flags | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        }

        inline D3D12_RESOURCE_DESC DescribeDepthBuffer(UINT width, UINT height, DXGI_FORMAT format)
        {
            return DescribeColorBuffer(width, height, 1, format, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        }

        inline Microsoft::WRL::ComPtr<ID3D12Heap> CreateUploadHeap(ID3D12Device* device, SIZE_T size, D3D12_HEAP_FLAGS flags )
        {
            D3D12_HEAP_DESC d = {};
            d.Properties.Type = D3D12_HEAP_TYPE_UPLOAD;
            d.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
            d.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
            d.Properties.VisibleNodeMask = 1;
            d.Properties.CreationNodeMask = 1;

            d.Alignment     = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
            d.SizeInBytes   = Align(size, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
            d.Flags         = flags;

            Microsoft::WRL::ComPtr<ID3D12Heap> result;

            DX::ThrowIfFailed(device->CreateHeap(&d, IID_PPV_ARGS(&result)));
            return result;
        }

        inline Microsoft::WRL::ComPtr<ID3D12Heap> CreateUploadBufferHeap( ID3D12Device* device, SIZE_T size )
        {
            return CreateUploadHeap(device, size, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS);
        }

        inline Microsoft::WRL::ComPtr<ID3D12Heap> CreateUploadTextureHeap(ID3D12Device* device, SIZE_T size)
        {
            return CreateUploadHeap(device, size, D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES);
        }

        inline Microsoft::WRL::ComPtr<ID3D12Heap> CreateReadBackHeap(ID3D12Device* device, SIZE_T size, D3D12_HEAP_FLAGS flags)
        {
            D3D12_HEAP_DESC d = {};
            d.Properties.Type = D3D12_HEAP_TYPE_READBACK;
            d.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
            d.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
            d.Properties.CreationNodeMask = 1;
            d.Properties.VisibleNodeMask = 1;

            d.Alignment     = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
            d.SizeInBytes   = Align(size, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
            d.Flags         = D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES;

            Microsoft::WRL::ComPtr<ID3D12Heap> result;

            DX::ThrowIfFailed(device->CreateHeap(&d, IID_PPV_ARGS(&result)));
            return result;
        }

        inline Microsoft::WRL::ComPtr<ID3D12Heap> CreateDefaultHeap(ID3D12Device* device, SIZE_T size, D3D12_HEAP_FLAGS flags )
        {
            D3D12_HEAP_DESC d = {};
            d.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
            d.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
            d.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
            d.Properties.CreationNodeMask = 1;
            d.Properties.VisibleNodeMask = 1;

            d.Alignment     = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
            d.SizeInBytes   = Align(size, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
            d.Flags         = flags;

            Microsoft::WRL::ComPtr<ID3D12Heap> result;

            DX::ThrowIfFailed(device->CreateHeap(&d, IID_PPV_ARGS(&result)));
            return result;
        }

        PlacementHeapAllocator::PlacementHeapAllocator()
        {

        }

        PlacementHeapAllocator::PlacementHeapAllocator(ID3D12Device* device, Microsoft::WRL::ComPtr<ID3D12Heap> heap, SIZE_T size) :
            m_device(device)
            , m_heap(heap)
            , m_heapOffset(0)
            , m_size(size)
        {

        }

        void PlacementHeapAllocator::CreatePlacedResource(const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES initialState, const D3D12_CLEAR_VALUE *optimizedClearValue, REFIID riid, void **resource)
        {
            auto  alignment = desc->Alignment < D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT ? D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT : desc->Alignment;
            auto  info = m_device->GetResourceAllocationInfo(1, 1, desc);

            if (Align(m_heapOffset, info.Alignment) + info.SizeInBytes < m_size)
            {
                DX::ThrowIfFailed(m_device->CreatePlacedResource(m_heap.Get(), m_heapOffset, desc, initialState, optimizedClearValue, riid, resource));
                m_heapOffset = Align(m_heapOffset, info.Alignment);
                m_heapOffset += info.SizeInBytes;
            }
        }

        inline void PlacementHeapAllocator::CreatePlacedResource(const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES initialState, REFIID riid, void **resource)
        {
            return CreatePlacedResource(desc, initialState, nullptr, riid, resource);
        }

        static inline PlacementHeapAllocator CreateUploadBuffersAllocator(ID3D12Device* d, SIZE_T size)
        {
            return PlacementHeapAllocator(d, CreateUploadHeap(d, size, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS), size);
        }

        static inline PlacementHeapAllocator CreateUploadTexturesAllocator(ID3D12Device* d, SIZE_T size)
        {
            return PlacementHeapAllocator(d, CreateUploadHeap(d, size, D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES), size);
        }

        static inline PlacementHeapAllocator CreateReadbackTexturesAllocator(ID3D12Device* d, SIZE_T size)
        {
            return PlacementHeapAllocator(d, CreateReadBackHeap(d, size, D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES), size);
        }
        static inline PlacementHeapAllocator CreateDefaultHeapTexturesAllocator(ID3D12Device* d, SIZE_T size)
        {
            return PlacementHeapAllocator(d, CreateDefaultHeap(d, size, D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES), size);
        }
    }

    GpuResourceCreateContext::GpuResourceCreateContext(ID3D12Device* device) :
        m_device(device)
        , m_texturesDescriptorHeap(device, 256)
        , m_pixelBufferDescriptorHeap(device, 256)
        , m_depthBufferDescriptorHeap(device, 256)
        , m_frameIndex(0)
    {
        m_uploadAllocator[0] = details::CreateUploadBuffersAllocator(device, details::MB(32));
        m_uploadAllocator[1] = details::CreateUploadBuffersAllocator(device, details::MB(32));
        m_uploadAllocator[2] = details::CreateUploadBuffersAllocator(device, details::MB(32));

        m_readBackAllocator[0] = details::CreateReadbackTexturesAllocator(device, details::MB(32));
        m_readBackAllocator[1] = details::CreateReadbackTexturesAllocator(device, details::MB(32));
        m_readBackAllocator[2] = details::CreateReadbackTexturesAllocator(device, details::MB(32));

        m_renderTargetAllocator     = details::CreateDefaultHeapTexturesAllocator(device, details::MB(32));
        m_tiledResourcesAllocator   = details::CreateDefaultHeapTexturesAllocator(device, details::MB(32));

    }

    GpuTexture2D GpuResourceCreateContext::CreateTexture2D()
    {
        return GpuTexture2D(nullptr, m_texturesDescriptorHeap.Allocate(), m_texturesDescriptorHeap.Allocate());
    }

    GpuUploadBuffer GpuResourceCreateContext::CreateUploadBuffer(SIZE_T size)
    {
        auto desc   = details::DescribeBuffer(size);
        Microsoft::WRL::ComPtr<ID3D12Resource>  resource;
        auto allocator = GetUploadAllocator();

        allocator->CreatePlacedResource(&desc, D3D12_RESOURCE_STATE_GENERIC_READ, IID_PPV_ARGS(&resource) );
        return GpuUploadBuffer(resource.Get());
    }

    GpuReadBackBuffer GpuResourceCreateContext::CreateReadBackBuffer(SIZE_T size)
    {
        auto desc = details::DescribeBuffer(size);
        Microsoft::WRL::ComPtr<ID3D12Resource>  resource;
        auto allocator = GetReadBackAllocator();

        allocator->CreatePlacedResource(&desc, D3D12_RESOURCE_STATE_COPY_DEST, IID_PPV_ARGS(&resource));
        return GpuReadBackBuffer(resource.Get());
    }

    GpuTiledCubeTexture GpuResourceCreateContext::CreateTiledCubeTexture()
    {
        return GpuTiledCubeTexture(nullptr, m_texturesDescriptorHeap.Allocate(), m_texturesDescriptorHeap.Allocate());
    }

    GpuColorBuffer  GpuResourceCreateContext::CreateColorBuffer(UINT width, UINT height, DXGI_FORMAT format)
    {
        D3D12_CLEAR_VALUE v = {};
        v.Format = format;

        auto desc = details::DescribeColorBuffer(width, height, format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS );

        Microsoft::WRL::ComPtr<ID3D12Resource>  resource;
        auto allocator = GetReadBackAllocator();

        D3D12_RENDER_TARGET_VIEW_DESC rtv = {};
        rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        rtv.Texture2D.MipSlice = 0;

        allocator->CreatePlacedResource(&desc, D3D12_RESOURCE_STATE_COMMON, &v, IID_PPV_ARGS(&resource));

        auto handle = m_pixelBufferDescriptorHeap.Allocate();
        m_device->CreateRenderTargetView( resource.Get() , &rtv, handle);

        return GpuColorBuffer(resource.Get(), handle, m_texturesDescriptorHeap.Allocate(), m_texturesDescriptorHeap.Allocate() );
    }

    //Depth Buffer
    GpuDepthBuffer  GpuResourceCreateContext::CreateDepthBuffer(UINT width, UINT height, DXGI_FORMAT format)
    {
        D3D12_CLEAR_VALUE v = {};
        v.Format = format;

        auto desc = details::DescribeColorBuffer(width, height, format, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL );

        Microsoft::WRL::ComPtr<ID3D12Resource>  resource;

        m_renderTargetAllocator.CreatePlacedResource(&desc, D3D12_RESOURCE_STATE_COMMON, &v, IID_PPV_ARGS(&resource));

        // Create the shader resource view
        D3D12_SHADER_RESOURCE_VIEW_DESC viewDesc = {};
        viewDesc.Format                          = GetDepthFormat(format);
        viewDesc.ViewDimension                   = D3D12_SRV_DIMENSION_TEXTURE2D;
        viewDesc.Shader4ComponentMapping         = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        viewDesc.Texture2D.MipLevels             = 1;

        DescriptorHandle srvDepth                = m_texturesDescriptorHeap.Allocate();
        DescriptorHandle srvStencil              = m_texturesDescriptorHeap.Allocate();;

        m_device->CreateShaderResourceView(resource.Get(), &viewDesc, srvDepth);

        // If stencil format is something else, use it
        DXGI_FORMAT stencilFormat = GetStencilFormat(format);
        if ( stencilFormat != DXGI_FORMAT_UNKNOWN )
        {
            viewDesc.Format = stencilFormat;
        }

        m_device->CreateShaderResourceView(resource.Get(), &viewDesc, srvStencil);

        DescriptorHandle srv[2] = { srvDepth, srvStencil };

        //Create depth stencil view
        DescriptorHandle dsvReadWrite            = m_depthBufferDescriptorHeap.Allocate();
        DescriptorHandle dsvReadDepth            = m_depthBufferDescriptorHeap.Allocate();
        DescriptorHandle dsvReadStencil          = m_depthBufferDescriptorHeap.Allocate();
        DescriptorHandle dsvReadDepthStencil     = m_depthBufferDescriptorHeap.Allocate();

        D3D12_DEPTH_STENCIL_VIEW_DESC desc2      = {};
        desc2.Format                             = format;
        desc2.ViewDimension                      = D3D12_DSV_DIMENSION_TEXTURE2D;
        desc2.Texture2D.MipSlice                 = 0;
        desc2.Flags                              = D3D12_DSV_FLAG_NONE;

        m_device->CreateDepthStencilView(resource.Get(), &desc2, dsvReadWrite);

        desc2.Flags                              = D3D12_DSV_FLAG_READ_ONLY_DEPTH;
        m_device->CreateDepthStencilView(resource.Get(), &desc2, dsvReadDepth);

        desc2.Flags                              = D3D12_DSV_FLAG_READ_ONLY_DEPTH | D3D12_DSV_FLAG_READ_ONLY_STENCIL;
        m_device->CreateDepthStencilView(resource.Get(), &desc2, dsvReadDepth);

        if (stencilFormat != DXGI_FORMAT_UNKNOWN)
        {
            desc2.Flags = D3D12_DSV_FLAG_READ_ONLY_STENCIL;
            m_device->CreateDepthStencilView(resource.Get(), &desc2, dsvReadStencil);
        }
        else
        {
            dsvReadStencil = dsvReadDepthStencil;
        }

        DescriptorHandle dsv[4]                  = { dsvReadWrite, dsvReadDepth, dsvReadStencil, dsvReadDepthStencil };

        return GpuDepthBuffer(resource.Get(), srv, dsv);
    }

    GpuBackBuffer GpuResourceCreateContext::CreateBackBuffer( ID3D12Resource* resource)
    {
        D3D12_RENDER_TARGET_VIEW_DESC rtv = {};
        rtv.ViewDimension                 = D3D12_RTV_DIMENSION_TEXTURE2D;
        rtv.Texture2D.MipSlice            = 0;

        auto handle                       = m_pixelBufferDescriptorHeap.Allocate();
        m_device->CreateRenderTargetView(resource, &rtv, handle);

        return GpuBackBuffer(resource, handle);
    }

    void GpuResourceCreateContext::Sync()
    {
        m_frameIndex++;
        m_frameIndex %= 3;
    }
}