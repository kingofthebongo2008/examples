#include "pch.h"

#include "GpuResourceCreateContext.h"

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

		inline Microsoft::WRL::ComPtr<ID3D12Heap> CreateUploadHeap(ID3D12Device* device, SIZE_T size)
		{
			D3D12_HEAP_DESC d = {};
			d.Properties.Type = D3D12_HEAP_TYPE_UPLOAD;
			d.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE;
			d.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
			d.Properties.CreationNodeMask = 1;
			d.Properties.VisibleNodeMask = 1;

			d.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
			d.SizeInBytes = Align(size, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);

			Microsoft::WRL::ComPtr<ID3D12Heap> result;

			DX::ThrowIfFailed(device->CreateHeap(&d, IID_PPV_ARGS(&result)));
			return result;
		}

		inline Microsoft::WRL::ComPtr<ID3D12Heap> CreateReadBackHeap(ID3D12Device* device, SIZE_T size)
		{
			D3D12_HEAP_DESC d = {};
			d.Properties.Type = D3D12_HEAP_TYPE_READBACK;
			d.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
			d.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
			d.Properties.CreationNodeMask = 1;
			d.Properties.VisibleNodeMask = 1;

			d.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
			d.SizeInBytes = Align(size, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);

			Microsoft::WRL::ComPtr<ID3D12Heap> result;

			DX::ThrowIfFailed(device->CreateHeap(&d, IID_PPV_ARGS(&result)));
			return result;
		}

		FramePlacementHeapAllocator::FramePlacementHeapAllocator()
		{

		}

		FramePlacementHeapAllocator::FramePlacementHeapAllocator(ID3D12Device* device, Microsoft::WRL::ComPtr<ID3D12Heap> heap, SIZE_T size) :
			m_device(device)
			, m_heap(heap)
			, m_heapOffset(0)
			, m_size(size)
		{

		}

		void FramePlacementHeapAllocator::CreatePlacedResource(const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES initialState, const D3D12_CLEAR_VALUE *optimizedClearValue, REFIID riid, void **resource)
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

		inline void FramePlacementHeapAllocator::CreatePlacedResource(const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES initialState, REFIID riid, void **resource)
		{
			return CreatePlacedResource(desc, initialState, nullptr, riid, resource);
		}

		static inline FramePlacementHeapAllocator CreateUploadAllocator(ID3D12Device* d, SIZE_T size)
		{
			return FramePlacementHeapAllocator(d, CreateUploadHeap(d, size), size);
		}

		static inline FramePlacementHeapAllocator CreateReadbackAllocator(ID3D12Device* d, SIZE_T size)
		{
			return FramePlacementHeapAllocator(d, CreateReadBackHeap(d, size), size);
		}
	}

	GpuResourceCreateContext::GpuResourceCreateContext(ID3D12Device* device) :
		m_device(device)
		, m_texturesDescriptorHeap(device, 256)
		, m_frameIndex(0)
	{
		m_uploadAllocator[0] = details::CreateUploadAllocator(device, details::MB(32));
		m_uploadAllocator[1] = details::CreateUploadAllocator(device, details::MB(32));
		m_uploadAllocator[2] = details::CreateUploadAllocator(device, details::MB(32));

		m_readbackAllocator[0] = details::CreateReadbackAllocator(device, details::MB(32));
		m_readbackAllocator[1] = details::CreateReadbackAllocator(device, details::MB(32));
		m_readbackAllocator[2] = details::CreateReadbackAllocator(device, details::MB(32));
	}

	GpuTexture2D GpuResourceCreateContext::CreateTexture2D()
	{
		return GpuTexture2D(nullptr, m_texturesDescriptorHeap.Allocate(), m_texturesDescriptorHeap.Allocate());
	}

	GpuUploadBuffer GpuResourceCreateContext::CreateUploadBuffer(SIZE_T size)
	{
		auto desc = details::DescribeBuffer(size);
		Microsoft::WRL::ComPtr<ID3D12Resource>  m_resource;

		return GpuUploadBuffer(nullptr);
	}

	GpuReadBackBuffer GpuResourceCreateContext::CreateReadBackBuffer(SIZE_T size)
	{
		return GpuReadBackBuffer(nullptr);
	}

	GpuTiledCubeTexture GpuResourceCreateContext::CreateTiledCubeTexture()
	{
		return GpuTiledCubeTexture(nullptr, m_texturesDescriptorHeap.Allocate(), m_texturesDescriptorHeap.Allocate());
	}

	void GpuResourceCreateContext::Sync()
	{
		m_frameIndex++;
		m_frameIndex %= 3;
	}
}