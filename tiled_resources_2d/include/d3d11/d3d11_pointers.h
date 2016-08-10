#pragma once

#include <os/windows/com_pointers.h>

#if defined(_PC)
    #include <d3d11/platforms/pc/pc_d3d11_pointers.h>
#endif

namespace d3d11
{
    using device                = os::windows::com_ptr<ID3D11Device>;
    using device1               = os::windows::com_ptr<ID3D11Device1>;
    using device2               = os::windows::com_ptr<ID3D11Device2>;
    using blob                  = os::windows::com_ptr<ID3DBlob>;
    using resource              = os::windows::com_ptr<ID3D11Resource>;
    using device_context        = os::windows::com_ptr<ID3D11DeviceContext>;
    using device_context1       = os::windows::com_ptr<ID3D11DeviceContext1>;
    using device_context2       = os::windows::com_ptr<ID3D11DeviceContext2>;
    using input_layout          = os::windows::com_ptr<ID3D11InputLayout>;

    using texture2d             = os::windows::com_ptr<ID3D11Texture2D>;
    using buffer                = os::windows::com_ptr<ID3D11Buffer>;

    using vertex_shader         = os::windows::com_ptr<ID3D11VertexShader>;
    using pixel_shader          = os::windows::com_ptr<ID3D11PixelShader>;

    using render_target_view    = os::windows::com_ptr<ID3D11RenderTargetView>;
    using shader_resource_view  = os::windows::com_ptr<ID3D11ShaderResourceView>;
    using depth_stencil_view    = os::windows::com_ptr<ID3D11DepthStencilView>;

    using depth_stencil_state   = os::windows::com_ptr<ID3D11DepthStencilState>;
    using rasterizer_state      = os::windows::com_ptr<ID3D11RasterizerState>;
    using rasterizer_state1     = os::windows::com_ptr<ID3D11RasterizerState1>;
    using sampler_state         = os::windows::com_ptr<ID3D11SamplerState>;
    using blend_state           = os::windows::com_ptr<ID3D11BlendState>;
    using blend_state1          = os::windows::com_ptr<ID3D11BlendState1>;
}
