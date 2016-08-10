#pragma once

#pragma once

#if defined(_PC)
#include <d3d11/platforms/pc/pc_d3d11_helpers.h>
#endif

#include <cstdint>
#include <tuple>

#include <os/windows/com_error.h>
#include <os/windows/dxgi_pointers.h>

#include <d3d11/d3d11_exception.h>
#include <d3d11/d3d11_pointers.h>
#include <d3d11/d3d11_helpers_types.h>

namespace d3d11
{
    namespace helpers
    {
        inline std::tuple<d3d11::device, d3d11::device_context> create_device( _In_opt_ IDXGIAdapter* pAdapter, D3D_DRIVER_TYPE DriverType, HMODULE Software, UINT Flags, _In_reads_opt_(FeatureLevels) CONST D3D_FEATURE_LEVEL* pFeatureLevels, UINT FeatureLevels, UINT SDKVersion,  _Out_opt_ D3D_FEATURE_LEVEL* pFeatureLevel )
        {
            using namespace os::windows;
            using namespace d3d11;

            device         d;
            device_context ctx;
            throw_if_failed(D3D11CreateDevice(pAdapter, DriverType, Software, Flags, pFeatureLevels, FeatureLevels, SDKVersion,  &d, pFeatureLevel,  &ctx));
            return std::make_tuple(d, ctx);
        }

        inline std::tuple<d3d11::device, d3d11::device_context> create_device(_In_opt_ IDXGIAdapter* adapter = nullptr, D3D_FEATURE_LEVEL minimumFeatureLevel = D3D_FEATURE_LEVEL_11_0 )
        {
            using namespace os::windows;
            using namespace d3d11;
            auto flags = D3D11_CREATE_DEVICE_DEBUG;// | D3D11_CREATE_DEVICE_BGRA_SUPPORT;
            D3D_FEATURE_LEVEL   level_out = {};

            return  create_device (adapter, D3D_DRIVER_TYPE_UNKNOWN, 0, flags, &minimumFeatureLevel, 1, D3D11_SDK_VERSION, &level_out );
        }

        inline d3d11::texture2d create_texture_2d(ID3D11Device* d, const D3D11_TEXTURE2D_DESC *pDesc, const D3D11_SUBRESOURCE_DATA *pInitialData= nullptr)
        {
            d3d11::texture2d t;
            throw_if_failed(d->CreateTexture2D(pDesc, pInitialData, &t));
            return t;
        }

        inline d3d11::buffer create_buffer(ID3D11Device* d, const D3D11_BUFFER_DESC *pDesc, const D3D11_SUBRESOURCE_DATA *pInitialData = nullptr)
        {
            d3d11::buffer b;
            throw_if_failed(d->CreateBuffer(pDesc, pInitialData, &b));
            return b;
        }

        inline d3d11::buffer create_vertex_buffer(ID3D11Device* d, size_t vertex_size, size_t vertex_count, const void* vertices = nullptr )
        {
            D3D11_BUFFER_DESC r = {};

            r.ByteWidth           = static_cast<UINT>(vertex_count * vertex_size);
            r.StructureByteStride = static_cast<UINT>(vertex_size);
            r.Usage               = D3D11_USAGE_DEFAULT;
            r.BindFlags           = D3D11_BIND_VERTEX_BUFFER;

            if (vertices)
            {
                D3D11_SUBRESOURCE_DATA data;
                data.pSysMem = vertices;
                data.SysMemPitch = static_cast<UINT>(r.ByteWidth);
                return create_buffer(d, &r, &data);
            }
            else
            {
                return create_buffer(d, &r, nullptr);
            }
        }

        inline d3d11::buffer create_index_buffer(ID3D11Device* d, size_t index_size, size_t index_count, const void* indices = nullptr)
        {
            D3D11_BUFFER_DESC r = {};

            r.ByteWidth = static_cast<UINT>(index_count * index_size);
            r.StructureByteStride = static_cast<UINT>(index_size);
            r.Usage = D3D11_USAGE_DEFAULT;
            r.BindFlags = D3D11_BIND_INDEX_BUFFER;

            if (indices)
            {
                D3D11_SUBRESOURCE_DATA data;
                data.pSysMem = indices;
                data.SysMemPitch = static_cast<UINT>(r.ByteWidth);
                return create_buffer(d, &r, &data);
            }
            else
            {
                return create_buffer(d, &r, nullptr);
            }
        }

        inline d3d11::input_layout create_input_layout(ID3D11Device* d, const D3D11_INPUT_ELEMENT_DESC *pInputElementDescs, UINT NumElements, const void *pShaderBytecodeWithInputSignature, SIZE_T BytecodeLength)
        {
            input_layout r;

            throw_if_failed(d->CreateInputLayout(pInputElementDescs, NumElements, pShaderBytecodeWithInputSignature, BytecodeLength, &r));

            return r;
        }

        inline d3d11::buffer create_constant_buffer( ID3D11Device* d, size_t size)
        {
            D3D11_BUFFER_DESC r = {};

            r.ByteWidth = static_cast<UINT>(size);
            r.StructureByteStride = 1;
            r.Usage = D3D11_USAGE_DYNAMIC;
            r.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            r.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

            return create_buffer(d, &r, nullptr);
        }

        inline d3d11::texture2d create_tiled_texture_2d(ID3D11Device* device, uint32_t width, uint32_t height, uint32_t mip_levels = 1)
        {
            D3D11_TEXTURE2D_DESC d = {};

            d.Width         = width;
            d.Height        = height;
            d.MipLevels     = mip_levels;
            d.ArraySize     = 1;
            d.SampleDesc.Count = 1;
            d.Format        = DXGI_FORMAT_B8G8R8A8_UNORM;
            d.MiscFlags     = D3D11_RESOURCE_MISC_TILED;
            d.BindFlags     = D3D11_BIND_SHADER_RESOURCE;
            
            return create_texture_2d(device, &d );
        }

        inline d3d11::texture2d create_staging_texture_2d(ID3D11Device* device, uint32_t width, uint32_t height, DXGI_FORMAT f)
        {
            D3D11_TEXTURE2D_DESC d = {};

            d.Width     = width;
            d.Height    = height;
            d.MipLevels = 1;
            d.ArraySize = 1;
            d.SampleDesc.Count = 1;
            d.Format    = f;
            d.MiscFlags = 0;
            d.BindFlags = 0;
            d.Usage = D3D11_USAGE_STAGING;
            d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

            return create_texture_2d(device, &d);
        }

        inline d3d11::buffer create_tiled_pool(ID3D11Device* device, uint32_t tile_count )
        {
            D3D11_BUFFER_DESC d = {};
            
            d.ByteWidth         = tile_count * 65536;
            d.MiscFlags         = D3D11_RESOURCE_MISC_TILE_POOL;
            return create_buffer(device, &d);
        }


        inline d3d11::vertex_shader create_vertex_shader(ID3D11Device* d, const void* code, size_t size)
        {
            d3d11::vertex_shader v;

            throw_if_failed(d->CreateVertexShader(code, size, nullptr, &v));
            return v;
        }

        inline d3d11::pixel_shader create_pixel_shader(ID3D11Device* d, const void* code, size_t size)
        {
            d3d11::pixel_shader v;

            throw_if_failed(d->CreatePixelShader(code, size, nullptr, &v));
            return v;
        }

        inline d3d11::render_target_view create_render_target_view(ID3D11Device* d, ID3D11Resource* resource, const D3D11_RENDER_TARGET_VIEW_DESC* description = nullptr)
        {
            d3d11::render_target_view v;
            throw_if_failed(d->CreateRenderTargetView( resource, description,&v));
            return v;
        }

        inline d3d11::rasterizer_state create_rasterizer_state(ID3D11Device* device, D3D11_RASTERIZER_DESC* d)
        {
            d3d11::rasterizer_state s;

            throw_if_failed(device->CreateRasterizerState(d, &s));
            return s;
        }

        inline d3d11::blend_state create_blend_state(ID3D11Device* device, D3D11_BLEND_DESC* d)
        {
            d3d11::blend_state s;

            throw_if_failed(device->CreateBlendState(d, &s));
            return s;
        }

        inline d3d11::depth_stencil_state create_depth_stencil_state(ID3D11Device* device, D3D11_DEPTH_STENCIL_DESC* d)
        {
            d3d11::depth_stencil_state s;

            throw_if_failed(device->CreateDepthStencilState(d, &s));
            return s;
        }

        inline d3d11::depth_stencil_view create_depth_stencil_view(ID3D11Device* device, ID3D11Resource* r, D3D11_DEPTH_STENCIL_VIEW_DESC* d)
        {
            d3d11::depth_stencil_view s;

            throw_if_failed(device->CreateDepthStencilView(r,d, &s));
            return s;
        }

        inline d3d11::sampler_state create_sampler_state(ID3D11Device* device, D3D11_SAMPLER_DESC* d)
        {
            d3d11::sampler_state s;

            throw_if_failed(device->CreateSamplerState(d, &s));
            return s;
        }

        inline d3d11::shader_resource_view create_shader_resource_view(ID3D11Device* device, ID3D11Resource* r, D3D11_SHADER_RESOURCE_VIEW_DESC* d = nullptr)
        {
            d3d11::shader_resource_view s;

            throw_if_failed(device->CreateShaderResourceView(r, d, &s));
            return s;
        }

        inline void update_tile_mappings
        (
            ID3D11DeviceContext2* ctx, /* [annotation] */
            _In_  ID3D11Resource *pTiledResource,
            /* [annotation] */
            _In_  UINT NumTiledResourceRegions,
            /* [annotation] */
            _In_reads_opt_(NumTiledResourceRegions)  const D3D11_TILED_RESOURCE_COORDINATE *pTiledResourceRegionStartCoordinates,
            /* [annotation] */
            _In_reads_opt_(NumTiledResourceRegions)  const D3D11_TILE_REGION_SIZE *pTiledResourceRegionSizes,
            /* [annotation] */
            _In_opt_  ID3D11Buffer *pTilePool,
            /* [annotation] */
            _In_  UINT NumRanges,
            /* [annotation] */
            _In_reads_opt_(NumRanges)  const UINT *pRangeFlags,
            /* [annotation] */
            _In_reads_opt_(NumRanges)  const UINT *pTilePoolStartOffsets,
            /* [annotation] */
            _In_reads_opt_(NumRanges)  const UINT *pRangeTileCounts,
            /* [annotation] */
            _In_  UINT Flags)
        {
            throw_if_failed(ctx->UpdateTileMappings(pTiledResource, NumTiledResourceRegions, pTiledResourceRegionStartCoordinates, pTiledResourceRegionSizes, pTilePool, NumRanges, pRangeFlags, pTilePoolStartOffsets, pRangeTileCounts, Flags));
        }
    }
}
