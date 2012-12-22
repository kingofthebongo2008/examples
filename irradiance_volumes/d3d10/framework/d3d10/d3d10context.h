//-----------------------------------------------------------------------------
// File: Framework\D3D10\D3D10Context.h
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------





#ifndef _D3D10CONTEXT_H_
#define _D3D10CONTEXT_H_

#include <d3d10.h>
#include <d3dx10.h>

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
	ID3D10Texture2D *CreateRenderTarget2D(const DXGI_FORMAT format, const int width, const int height, const int arraySize, const int samples, const int mipmapCount,
		ID3D10RenderTargetView **rtRTV = NULL, ID3D10ShaderResourceView **rtSRV = NULL, ID3D10ShaderResourceView **rtSliceSRVs = NULL, const unsigned int flags = 0);
	ID3D10Texture3D *CreateRenderTarget3D(const DXGI_FORMAT format, const int width, const int height, const int depth, const int mipmapCount, ID3D10RenderTargetView **rtRTV = NULL, ID3D10ShaderResourceView **rtSRV = NULL, const unsigned int flags = 0);
	ID3D10Texture2D *CreateRenderTargetCube(const DXGI_FORMAT format, const int size, const int samples, const int mipmapCount, ID3D10RenderTargetView **rtArrayRTV = NULL,
		ID3D10RenderTargetView *rtFaceRTVs[6] = NULL, ID3D10ShaderResourceView **rtSRV = NULL, const unsigned int flags = 0);

	// Depth-stencil target helper functions
	ID3D10Texture2D *CreateDepthTarget2D(const DXGI_FORMAT format, const int width, const int height, const int arraySize, const int samples, const int mipmapCount,
		ID3D10DepthStencilView **rtArrayDSV = NULL, ID3D10DepthStencilView **rtSliceDSVs = NULL, ID3D10ShaderResourceView **rtArraySRV = NULL,
		ID3D10ShaderResourceView **rtSliceSRVs = NULL, const unsigned int flags = 0);
	ID3D10Texture2D *CreateDepthTargetCube(const DXGI_FORMAT format, const int size, const int samples, const int mipmapCount, ID3D10DepthStencilView **rtArrayDSV = NULL,
		ID3D10DepthStencilView *rtFaceDSVs[6] = NULL, const unsigned int flags = 0);

	// Buffer helper functions
	ID3D10Buffer *CreateVertexBuffer(const int size, const D3D10_USAGE usage, const void *data = NULL);
	ID3D10Buffer *CreateIndexBuffer(const int size, const D3D10_USAGE usage, const void *data = NULL);
	ID3D10Buffer *CreateConstantBuffer(const int size, const D3D10_USAGE usage, const void *data = NULL);

	ID3D10InputLayout *CreateInputLayout(ID3D10EffectPass *effectPass, const D3D10_INPUT_ELEMENT_DESC *layout, const int elementCount);

	// Render target view helper functions
	ID3D10RenderTargetView *CreateRenderTargetView2D(ID3D10Texture2D *renderTarget, const DXGI_FORMAT format, const int firstSlice = 0, const int arraySize = 0);
	ID3D10RenderTargetView *CreateRenderTargetView3D(ID3D10Texture3D *renderTarget, const DXGI_FORMAT format, const int firstSlice = 0, const int arraySize = 0);
	ID3D10DepthStencilView *CreateDepthStencilView2D(ID3D10Texture2D *depthTarget, const DXGI_FORMAT format, const int firstSlice = 0, const int arraySize = 0);

	// Shader resource view helper functions
	ID3D10ShaderResourceView *CreateDefaultSRV(ID3D10Resource *resource);

	// State object helper functions
	ID3D10SamplerState *CreateSamplerState(const D3D10_FILTER filter, D3D10_TEXTURE_ADDRESS_MODE addressMode, const float maxLOD = D3D10_FLOAT32_MAX);
	ID3D10BlendState *CreateBlendState(const D3D10_BLEND src, const D3D10_BLEND dst, const D3D10_BLEND_OP op = D3D10_BLEND_OP_ADD, const UINT8 mask = D3D10_COLOR_WRITE_ENABLE_ALL, const int count = 1);
	ID3D10RasterizerState *CreateRasterizerState(const D3D10_CULL_MODE cullMode, const D3D10_FILL_MODE fillMode = D3D10_FILL_SOLID, const bool multisampleEnable = true,
		const bool depthClipEnable = true, const int depthBias = 0, const float slopeScaledDepthBias = 0.0f, const bool scissorEnable = false);
	ID3D10DepthStencilState *CreateDepthStencilState(const bool depthEnable, const bool depthWriteEnable, const D3D10_COMPARISON_FUNC depthFunc = D3D10_COMPARISON_LESS_EQUAL);

	// Texture helper functions
	ID3D10Texture1D *CreateTexture1D(const void *data, const DXGI_FORMAT format, const int width, const int arraySize, ID3D10ShaderResourceView **resourceView = NULL, const unsigned int flags = 0);
	ID3D10Texture2D *CreateTexture2D(const void *data, const DXGI_FORMAT format, const int width, const int height, const int arraySize, ID3D10ShaderResourceView **resourceView = NULL, const unsigned int flags = 0);
	ID3D10Texture3D *CreateTexture3D(const void *data, const DXGI_FORMAT format, const int width, const int height, const int depth, ID3D10ShaderResourceView **resourceView = NULL, const unsigned int flags = 0);
	ID3D10Texture2D *CreateTextureCube(const void *data, const DXGI_FORMAT format, const int size, ID3D10ShaderResourceView **resourceView = NULL, const unsigned int flags = 0);
	ID3D10Resource *LoadTexture(const TCHAR *fileName, ID3D10ShaderResourceView **resourceView = NULL, const unsigned int flags = 0);

	// Effect helper functions
	ID3D10Effect *LoadEffect(const TCHAR *fileName, const D3D10_SHADER_MACRO *defines = NULL, const unsigned int flags = 0);
	ID3D10Buffer *CreateEffectConstantBuffer(ID3D10Effect *effect, const char *name);

	void SetEffect(ID3D10Effect *effect);
	void SetTexture(const char *textureName, ID3D10ShaderResourceView *resourceView);
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
	ID3D10Device *GetDevice() const { return m_device; }

	// Get buffers
	ID3D10Texture2D *GetBackBuffer() const { return m_backBuffer; }
	ID3D10Texture2D *GetDepthBuffer() const { return m_depthBuffer; }

	// Utility functions
	void Clear(const float *clearColor = NULL, const UINT dsClearFlags = 0, const float depth = 1.0f, const UINT8 stencil = 0);
	void Present();

	void SetRenderTarget(const int width, const int height, ID3D10RenderTargetView **renderTargets, const int renderTargetCount, ID3D10DepthStencilView *depthTarget);
	void SetRTToBackBuffer();

	// Get window information
	int GetWidth() const { return m_width; }
	int GetHeight() const { return m_height; }
	HWND GetWindow() const { return m_hwnd; }

	// Save framebuffer contents to file
	bool SaveScreenshot(const TCHAR *name, const D3DX10_IMAGE_FILE_FORMAT format = D3DX10_IFF_DDS);

	// Get adapter information
	uint GetDeviceID() const { return m_adapterDesc.DeviceId; }
	uint GetVendorID() const { return m_adapterDesc.VendorId; }

protected:
	HWND m_hwnd;

	// Standard D3D10 resources
	IDXGISwapChain *m_swapChain;
	ID3D10Device *m_device;
	ID3D10Texture2D *m_backBuffer;
	ID3D10Texture2D *m_depthBuffer;
	ID3D10RenderTargetView *m_backBufferView;
	ID3D10DepthStencilView *m_depthBufferView;

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

	ID3D10Effect *m_currentEffect;

	DXGI_ADAPTER_DESC m_adapterDesc;
};

#endif // _D3D10CONTEXT_H_
