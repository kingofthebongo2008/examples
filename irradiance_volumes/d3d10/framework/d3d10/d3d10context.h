//-----------------------------------------------------------------------------
// File: Framework\D3D10\D3D10Context.h
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------





#ifndef _D3D10CONTEXT_H_
#define _D3D10CONTEXT_H_

#include <d3d11.h>
#include <d3dx11.h>
#include <d3dx11effect.h>

#define SAFE_RELEASE(p) { if (p){ p->Release(); p = NULL; } }

#include "../../Framework/Math/Vector.h"

// LoadEffect flags
#define ASSEMBLY 0x1
#define DUMP_TO_FILE 0x2
#define ASSEMBLY_TO_FILE (ASSEMBLY | DUMP_TO_FILE)

// VendorIDs
#define VENDOR_ATI    0x1002
#define VENDOR_NVIDIA 0x10DE

class D3D10Context
{
public:
	D3D10Context();
	virtual ~D3D10Context();

	// Creates and destroys this rendering context
	bool Create(const TCHAR *windowTitle, const DXGI_FORMAT backBufferFormat, const DXGI_FORMAT depthBufferFormat, const int width, const int height, const int msaaSamples, const bool fullscreen);
	void Destroy();

	// Functions for controling the context
	bool ToggleFullscreen();
	bool SetMode(const DXGI_FORMAT backBufferFormat, const DXGI_FORMAT depthBufferFormat, const int width, const int height, const int msaaSamples, const bool fullscreen);
	bool Resize(const int width, const int height);
	void SetPosition(const int x, const int y);

	// Render target helper functions
	ID3D11Texture2D *CreateRenderTarget2D(const DXGI_FORMAT format, const int width, const int height, const int arraySize, const int samples, const int mipmapCount,
		ID3D11RenderTargetView **rtRTV = NULL, ID3D11ShaderResourceView **rtSRV = NULL, ID3D11ShaderResourceView **rtSliceSRVs = NULL, const unsigned int flags = 0);
	ID3D11Texture3D *CreateRenderTarget3D(const DXGI_FORMAT format, const int width, const int height, const int depth, const int mipmapCount, ID3D11RenderTargetView **rtRTV = NULL, ID3D11ShaderResourceView **rtSRV = NULL, const unsigned int flags = 0);
	ID3D11Texture2D *CreateRenderTargetCube(const DXGI_FORMAT format, const int size, const int samples, const int mipmapCount, ID3D11RenderTargetView **rtArrayRTV = NULL,
		ID3D11RenderTargetView *rtFaceRTVs[6] = NULL, ID3D11ShaderResourceView **rtSRV = NULL, const unsigned int flags = 0);

	// Depth-stencil target helper functions
	ID3D11Texture2D *CreateDepthTarget2D(const DXGI_FORMAT format, const int width, const int height, const int arraySize, const int samples, const int mipmapCount,
		ID3D11DepthStencilView **rtArrayDSV = NULL, ID3D11DepthStencilView **rtSliceDSVs = NULL, ID3D11ShaderResourceView **rtArraySRV = NULL,
		ID3D11ShaderResourceView **rtSliceSRVs = NULL, const unsigned int flags = 0);
	ID3D11Texture2D *CreateDepthTargetCube(const DXGI_FORMAT format, const int size, const int samples, const int mipmapCount, ID3D11DepthStencilView **rtArrayDSV = NULL,
		ID3D11DepthStencilView *rtFaceDSVs[6] = NULL, const unsigned int flags = 0);

	// Buffer helper functions
	ID3D11Buffer *CreateVertexBuffer(const int size, const D3D11_USAGE usage, const void *data = NULL);
	ID3D11Buffer *CreateIndexBuffer(const int size, const D3D11_USAGE usage, const void *data = NULL);
	ID3D11Buffer *CreateConstantBuffer(const int size, const D3D11_USAGE usage, const void *data = NULL);

	ID3D11InputLayout *CreateInputLayout(ID3DX11EffectPass *effectPass, const D3D11_INPUT_ELEMENT_DESC *layout, const int elementCount);

	// Render target view helper functions
	ID3D11RenderTargetView *CreateRenderTargetView2D(ID3D11Texture2D *renderTarget, const DXGI_FORMAT format, const int firstSlice = 0, const int arraySize = 0);
	ID3D11RenderTargetView *CreateRenderTargetView3D(ID3D11Texture3D *renderTarget, const DXGI_FORMAT format, const int firstSlice = 0, const int arraySize = 0);
	ID3D11DepthStencilView *CreateDepthStencilView2D(ID3D11Texture2D *depthTarget, const DXGI_FORMAT format, const int firstSlice = 0, const int arraySize = 0);

	// Shader resource view helper functions
	ID3D11ShaderResourceView *CreateDefaultSRV(ID3D11Resource *resource);

	// State object helper functions
	ID3D11SamplerState *CreateSamplerState(const D3D11_FILTER filter, D3D11_TEXTURE_ADDRESS_MODE addressMode, const float maxLOD = D3D11_FLOAT32_MAX);
	ID3D11BlendState *CreateBlendState(const D3D11_BLEND src, const D3D11_BLEND dst, const D3D11_BLEND_OP op = D3D11_BLEND_OP_ADD, const UINT8 mask = D3D11_COLOR_WRITE_ENABLE_ALL, const int count = 1);
	ID3D11RasterizerState *CreateRasterizerState(const D3D11_CULL_MODE cullMode, const D3D11_FILL_MODE fillMode = D3D11_FILL_SOLID, const bool multisampleEnable = true,
		const bool depthClipEnable = true, const int depthBias = 0, const float slopeScaledDepthBias = 0.0f, const bool scissorEnable = false);
	ID3D11DepthStencilState *CreateDepthStencilState(const bool depthEnable, const bool depthWriteEnable, const D3D11_COMPARISON_FUNC depthFunc = D3D11_COMPARISON_LESS_EQUAL);

	// Texture helper functions
	ID3D11Texture1D *CreateTexture1D(const void *data, const DXGI_FORMAT format, const int width, const int arraySize, ID3D11ShaderResourceView **resourceView = NULL, const unsigned int flags = 0);
	ID3D11Texture2D *CreateTexture2D(const void *data, const DXGI_FORMAT format, const int width, const int height, const int arraySize, ID3D11ShaderResourceView **resourceView = NULL, const unsigned int flags = 0);
	ID3D11Texture3D *CreateTexture3D(const void *data, const DXGI_FORMAT format, const int width, const int height, const int depth, ID3D11ShaderResourceView **resourceView = NULL, const unsigned int flags = 0);
	ID3D11Texture2D *CreateTextureCube(const void *data, const DXGI_FORMAT format, const int size, ID3D11ShaderResourceView **resourceView = NULL, const unsigned int flags = 0);
	ID3D11Resource *LoadTexture(const TCHAR *fileName, ID3D11ShaderResourceView **resourceView = NULL, const unsigned int flags = 0);

	// Effect helper functions
	ID3DX11Effect *LoadEffect(const TCHAR *fileName, const D3D_SHADER_MACRO *defines = NULL, const unsigned int flags = 0);
	ID3D11Buffer *CreateEffectConstantBuffer(ID3DX11Effect *effect, const char *name);

	void SetEffect(ID3DX11Effect *effect);
	void SetTexture(const char *textureName, ID3D11ShaderResourceView *resourceView);
	void SetConstant(const char *constantName, const float value);
	void SetConstant(const char *constantName, const float2 &value);
	void SetConstant(const char *constantName, const float3 &value);
	void SetConstant(const char *constantName, const float4 &value);
	void SetConstant(const char *constantName, const float4x4 &value);
	void SetConstantArray(const char *constantName, const float4 *value, const int count);
	void Apply(const char *techniqueName, const char *passName);
	void Apply(const char *techniqueName, const int passIndex);
	void Apply(const int techniqueIndex, const int passIndex);

	// Get the device pointer
	ID3D11Device *GetDevice() const { return m_device; }
	ID3D11DeviceContext *GetDeviceContext() const { return m_device_context; }

	// Get buffers
	ID3D11Texture2D *GetBackBuffer() const { return m_backBuffer; }
	ID3D11Texture2D *GetDepthBuffer() const { return m_depthBuffer; }

	// Utility functions
	void Clear(const float *clearColor = NULL, const UINT dsClearFlags = 0, const float depth = 1.0f, const UINT8 stencil = 0);
	void Present();

	void SetRenderTarget(const int width, const int height, ID3D11RenderTargetView **renderTargets, const int renderTargetCount, ID3D11DepthStencilView *depthTarget);
	void SetRTToBackBuffer();

	// Get window information
	int GetWidth() const { return m_width; }
	int GetHeight() const { return m_height; }
	HWND GetWindow() const { return m_hwnd; }

	// Save framebuffer contents to file
	bool SaveScreenshot(const TCHAR *name, const D3DX11_IMAGE_FILE_FORMAT format = D3DX11_IFF_DDS);

	// Get adapter information
	uint GetDeviceID() const { return m_adapterDesc.DeviceId; }
	uint GetVendorID() const { return m_adapterDesc.VendorId; }

protected:
	HWND m_hwnd;

	// Standard D3D10 resources
	IDXGISwapChain *m_swapChain;
	ID3D11DeviceContext* m_device_context;
	ID3D11Device *m_device;
	ID3D11Texture2D *m_backBuffer;
	ID3D11Texture2D *m_depthBuffer;
	ID3D11RenderTargetView *m_backBufferView;
	ID3D11DepthStencilView *m_depthBufferView;

	// Window and context variables
	DXGI_FORMAT m_backBufferFormat;
	DXGI_FORMAT m_depthBufferFormat;
	int m_width, m_height, m_msaaSamples;
	int m_windowedX, m_windowedY;
	bool m_fullscreen;

private:
	// Internal functions
	bool InitializeBuffers();
	bool ReleaseBuffers();

	ID3DX11Effect *m_currentEffect;

	DXGI_ADAPTER_DESC m_adapterDesc;
};

#endif // _D3D10CONTEXT_H_
