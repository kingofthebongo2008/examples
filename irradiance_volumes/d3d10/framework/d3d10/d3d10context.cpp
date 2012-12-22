//-----------------------------------------------------------------------------
// File: Framework\D3D10\D3D10Context.cpp
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------





#include "D3D10Context.h"
#include <stdio.h>

#pragma comment (lib, "dxgi.lib")

// Texture format description. Used for various messages.
struct FormatDescription
{
	const int bpp;
	const char *name;
};

static const FormatDescription g_formats[] =
{
	{ 0,   "UNKNOWN" },
	{ 128, "R32G32B32A32_TYPELESS" },
	{ 128, "R32G32B32A32_FLOAT" },
	{ 128, "R32G32B32A32_UINT" },
	{ 128, "R32G32B32A32_SINT" },
	{ 96,  "R32G32B32_TYPELESS" },
	{ 96,  "R32G32B32_FLOAT" },
	{ 96,  "R32G32B32_UINT" },
	{ 96,  "R32G32B32_SINT" },
	{ 64,  "R16G16B16A16_TYPELESS" },
	{ 64,  "R16G16B16A16_FLOAT" },
	{ 64,  "R16G16B16A16_UNORM" },
	{ 64,  "R16G16B16A16_UINT" },
	{ 64,  "R16G16B16A16_SNORM" },
	{ 64,  "R16G16B16A16_SINT" },
	{ 64,  "R32G32_TYPELESS" },
	{ 64,  "R32G32_FLOAT" },
	{ 64,  "R32G32_UINT" },
	{ 64,  "R32G32_SINT" },
	{ 64,  "R32G8X24_TYPELESS" },
	{ 64,  "D32_FLOAT_S8X24_UINT" },
	{ 64,  "R32_FLOAT_X8X24_TYPELESS" },
	{ 64,  "X32_TYPELESS_G8X24_UINT" },
	{ 32,  "R10G10B10A2_TYPELESS" },
	{ 32,  "R10G10B10A2_UNORM" },
	{ 32,  "R10G10B10A2_UINT" },
	{ 32,  "R11G11B10_FLOAT" },
	{ 32,  "R8G8B8A8_TYPELESS" },
	{ 32,  "R8G8B8A8_UNORM" },
	{ 32,  "R8G8B8A8_UNORM_SRGB" },
	{ 32,  "R8G8B8A8_UINT" },
	{ 32,  "R8G8B8A8_SNORM" },
	{ 32,  "R8G8B8A8_SINT" },
	{ 32,  "R16G16_TYPELESS" },
	{ 32,  "R16G16_FLOAT" },
	{ 32,  "R16G16_UNORM" },
	{ 32,  "R16G16_UINT" },
	{ 32,  "R16G16_SNORM" },
	{ 32,  "R16G16_SINT" },
	{ 32,  "R32_TYPELESS" },
	{ 32,  "D32_FLOAT" },
	{ 32,  "R32_FLOAT" },
	{ 32,  "R32_UINT" },
	{ 32,  "R32_SINT" },
	{ 32,  "R24G8_TYPELESS" },
	{ 32,  "D24_UNORM_S8_UINT" },
	{ 32,  "R24_UNORM_X8_TYPELESS" },
	{ 32,  "X24_TYPELESS_G8_UINT" },
	{ 16,  "R8G8_TYPELESS" },
	{ 16,  "R8G8_UNORM" },
	{ 16,  "R8G8_UINT" },
	{ 16,  "R8G8_SNORM" },
	{ 16,  "R8G8_SINT" },
	{ 16,  "R16_TYPELESS" },
	{ 16,  "R16_FLOAT" },
	{ 16,  "D16_UNORM" },
	{ 16,  "R16_UNORM" },
	{ 16,  "R16_UINT" },
	{ 16,  "R16_SNORM" },
	{ 16,  "R16_SINT" },
	{ 8,   "R8_TYPELESS" },
	{ 8,   "R8_UNORM" },
	{ 8,   "R8_UINT" },
	{ 8,   "R8_SNORM" },
	{ 8,   "R8_SINT" },
	{ 8,   "A8_UNORM" },
	{ 1,   "R1_UNORM" },
	{ 32,  "R9G9B9E5_SHAREDEXP" },
	{ 16,  "R8G8_B8G8_UNORM" },
	{ 16,  "G8R8_G8B8_UNORM" },
	{ 4,   "BC1_TYPELESS" },
	{ 4,   "BC1_UNORM" },
	{ 4,   "BC1_UNORM_SRGB" },
	{ 8,   "BC2_TYPELESS" },
	{ 8,   "BC2_UNORM" },
	{ 8,   "BC2_UNORM_SRGB" },
	{ 8,   "BC3_TYPELESS" },
	{ 8,   "BC3_UNORM" },
	{ 8,   "BC3_UNORM_SRGB" },
	{ 4,   "BC4_TYPELESS" },
	{ 4,   "BC4_UNORM" },
	{ 4,   "BC4_SNORM" },
	{ 8,   "BC5_TYPELESS" },
	{ 8,   "BC5_UNORM" },
	{ 8,   "BC5_SNORM" },
	{ 16,  "B5G6R5_UNORM" },
	{ 16,  "B5G5R5A1_UNORM" },
	{ 32,  "B8G8R8A8_UNORM" },
	{ 32,  "B8G8R8X8_UNORM" },
};

D3D10Context::D3D10Context()
{
	// Initialize all pointers to NULL
	m_backBuffer = NULL;
	m_backBufferView = NULL;
	m_depthBuffer = NULL;
	m_depthBufferView = NULL;

	m_swapChain = NULL;
	m_device = NULL;

	m_backBufferFormat = DXGI_FORMAT_UNKNOWN;
	m_depthBufferFormat = DXGI_FORMAT_UNKNOWN;
	m_width  = 0;
	m_height = 0;
	m_msaaSamples = 1;
	m_windowedX = 0;
	m_windowedY = 0;
	m_fullscreen = false;
}

D3D10Context::~D3D10Context()
{

}

bool D3D10Context::Create(const TCHAR *windowTitle, const DXGI_FORMAT backBufferFormat, const DXGI_FORMAT depthBufferFormat, const int width, const int height, const int msaaSamples, const bool fullscreen)
{
	// Save context information
	m_backBufferFormat = backBufferFormat;
	m_depthBufferFormat = depthBufferFormat;
	m_width  = width;
	m_height = height;
	m_msaaSamples = msaaSamples;
	m_fullscreen = fullscreen;

	// Place window in the middle of the work area and ensure that it's not spanning outside it
	RECT wRect;
	wRect.left = 0;
	wRect.right = width;
	wRect.top = 0;
	wRect.bottom = height;

	DWORD wStyle = WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE;
	DWORD fStyle = WS_POPUP | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE;
	DWORD style = fullscreen? fStyle : wStyle;
	AdjustWindowRect(&wRect, wStyle, FALSE);

	int w = wRect.right - wRect.left;
	int h = wRect.bottom - wRect.top;

	m_windowedX = (GetSystemMetrics(SM_CXSCREEN) - w) / 2;
	m_windowedY = (GetSystemMetrics(SM_CYSCREEN) - h) / 2;

	int x, y;
	if (fullscreen)
	{
		w = width;
		h = height;
		x = 0;
		y = 0;
	}
	else
	{
		x = m_windowedX;
		y = m_windowedY;
	}

	m_hwnd = CreateWindowEx(0, _T("D3D10App"), windowTitle, style, x, y, w, h, HWND_DESKTOP, NULL, NULL, NULL);


	// Find the size of our client area
	RECT rect;
	GetClientRect(m_hwnd, &rect);

	// We don't get an initial notification of window size automatically, so we'll have to send one of our own
	WINDOWPOS wPos;
	wPos.cx = rect.right;
	wPos.cy = rect.bottom;
	wPos.flags = SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER;
	wPos.hwnd = m_hwnd;
	SendMessage(m_hwnd, WM_WINDOWPOSCHANGED, 0, (LPARAM) &wPos);


	// Create DXGI factory
	IDXGIFactory *dxgiFactory;
	if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), (void **) &dxgiFactory)))
	{
		MessageBoxA(m_hwnd, "DXGI factory creation failed", "Error", MB_OK | MB_ICONERROR);
		return false;
	}

	// Use first adapter
	IDXGIAdapter *adapter;
	if (dxgiFactory->EnumAdapters(0, &adapter) == DXGI_ERROR_NOT_FOUND)
	{
		MessageBoxA(m_hwnd, "No adapters found", "Error", MB_OK | MB_ICONERROR);
		return false;
	}

	adapter->GetDesc(&m_adapterDesc);



	// Create device and swap chain
	DXGI_SWAP_CHAIN_DESC sd;
	memset(&sd, 0, sizeof(sd));
	sd.BufferDesc.Width  = rect.right;
	sd.BufferDesc.Height = rect.bottom;
	sd.BufferDesc.Format = backBufferFormat;
	sd.SampleDesc.Count = msaaSamples;
	sd.SampleDesc.Quality = 0;
	sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	sd.BufferCount = 1;
	sd.OutputWindow = m_hwnd;
	sd.Windowed = (BOOL) (!fullscreen);
	sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
	sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

	DWORD flags = D3D10_CREATE_DEVICE_SINGLETHREADED;
#ifdef _DEBUG
    flags |= D3D10_CREATE_DEVICE_DEBUG;
#endif

	if (FAILED(D3D10CreateDeviceAndSwapChain(adapter, D3D10_DRIVER_TYPE_HARDWARE,  NULL, flags, D3D10_SDK_VERSION, &sd, &m_swapChain, &m_device)))
//	if (FAILED(D3D10CreateDeviceAndSwapChain(adapter, D3D10_DRIVER_TYPE_REFERENCE, NULL, flags, D3D10_SDK_VERSION, &sd, &m_swapChain, &m_device)))
	{
		MessageBoxA(m_hwnd, "D3D device creation failed", "Error", MB_OK | MB_ICONERROR);
		return false;
	}

	adapter->Release();
	dxgiFactory->Release();

	if (!InitializeBuffers()) return false;

	return true;
}

void D3D10Context::Destroy()
{
	ReleaseBuffers();

	if (m_swapChain)
	{
		// Restore video mode
		if (m_fullscreen) m_swapChain->SetFullscreenState(false, NULL);
		m_swapChain->Release();
	}

	if (m_device)
	{
		// Release device and check for unreleased references
		ULONG count = m_device->Release();
#ifdef _DEBUG
		if (count)
		{
			char str[512];
			sprintf(str, "\n"
				"+--------------------------------------------------------------+\n"
				"|                                                              |\n"
				"|  There are %2d unreleased references left on the D3D device!  |\n"
				"|                                                              |\n"
				"+--------------------------------------------------------------+\n\n", count);
			OutputDebugStringA(str);
		}
#endif
	}
}

bool D3D10Context::ToggleFullscreen()
{
	return SetMode(m_backBufferFormat, m_depthBufferFormat, m_width, m_height, m_msaaSamples, !m_fullscreen);
}

bool D3D10Context::SetMode(const DXGI_FORMAT backBufferFormat, const DXGI_FORMAT depthBufferFormat, const int width, const int height, const int msaaSamples, const bool fullscreen)
{
	// Save new context information
	m_backBufferFormat  = backBufferFormat;
	m_depthBufferFormat = depthBufferFormat;
	m_width  = width;
	m_height = height;
	m_msaaSamples = msaaSamples;
	m_fullscreen  = fullscreen;

	ReleaseBuffers();

	RECT wRect;
	wRect.left = 0;
	wRect.right = width;
	wRect.top = 0;
	wRect.bottom = height;

	DWORD style = fullscreen? (WS_POPUP | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE) : (WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE);
	AdjustWindowRect(&wRect, style, FALSE);

	// Update window style
	SetWindowLong(m_hwnd, GWL_STYLE, style);

	// Move window with SWP_NOSENDCHANGING flag to avoid receiving window size change notifications since we're taking care of all that here
	int x = fullscreen? 0 : m_windowedX;
	int y = fullscreen? 0 : m_windowedY;
	int w = wRect.right - wRect.left;
	int h = wRect.bottom - wRect.top;
	SetWindowPos(m_hwnd, HWND_TOP, x, y, w, h, SWP_NOSENDCHANGING);

	m_swapChain->ResizeBuffers(1, width, height, backBufferFormat, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH);

	bool result = InitializeBuffers();

	m_swapChain->SetFullscreenState(fullscreen, NULL);

	return result;
}

bool D3D10Context::Resize(const int width, const int height)
{
	// Sanity check
	if (m_backBuffer == NULL) return false;
	if (width == m_width && height == m_height) return true;

	m_width  = width;
	m_height = height;

	ReleaseBuffers();

	m_swapChain->ResizeBuffers(1, width, height, m_backBufferFormat, 0);

	if (!InitializeBuffers()) return false;

	return true;
}

void D3D10Context::SetPosition(const int x, const int y)
{
	m_windowedX = x;
	m_windowedY = y;
}

ID3D10Texture2D *D3D10Context::CreateRenderTarget2D(const DXGI_FORMAT format, const int width, const int height, const int arraySize, const int samples, const int mipmapCount,
	ID3D10RenderTargetView **rtRTV, ID3D10ShaderResourceView **rtSRV, ID3D10ShaderResourceView **rtSliceSRVs, const unsigned int flags)
{
	ID3D10Texture2D *renderTarget;

	// Setup texture description
	D3D10_TEXTURE2D_DESC desc;
	desc.Width  = width;
	desc.Height = height;
	desc.MipLevels = mipmapCount;
	desc.ArraySize = arraySize;
	desc.SampleDesc.Count = samples;
	desc.SampleDesc.Quality = 0;
	desc.Format = format;
	desc.Usage = D3D10_USAGE_DEFAULT;
	desc.BindFlags = D3D10_BIND_RENDER_TARGET | D3D10_BIND_SHADER_RESOURCE;
	desc.CPUAccessFlags = 0;
	desc.MiscFlags = 0;
	if (FAILED(m_device->CreateTexture2D(&desc, NULL, &renderTarget)))
	{
		char str[256];
		sprintf(str, "Couldn't create render target (%s, %dx%d, %d, %d, %d)", g_formats[format].name, width, height, arraySize, samples, mipmapCount);
		MessageBoxA(m_hwnd, str, "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	// Create render target view if requested
	if (rtRTV)
	{
		D3D10_RENDER_TARGET_VIEW_DESC rtDesc;
		rtDesc.Format = format;
		if (arraySize > 1)
		{
			rtDesc.ViewDimension = D3D10_RTV_DIMENSION_TEXTURE2DARRAY;
			rtDesc.Texture2DArray.FirstArraySlice = 0;
			rtDesc.Texture2DArray.ArraySize = arraySize;
			rtDesc.Texture2DArray.MipSlice = 0;
		}
		else
		{
			rtDesc.ViewDimension = (samples > 1)? D3D10_RTV_DIMENSION_TEXTURE2DMS : D3D10_RTV_DIMENSION_TEXTURE2D;
			rtDesc.Texture2D.MipSlice = 0;
		}
		if (FAILED(m_device->CreateRenderTargetView(renderTarget, &rtDesc, rtRTV)))
		{
			MessageBoxA(m_hwnd, "CreateRenderTargetView failed", "Error", MB_OK | MB_ICONERROR);
			return NULL;
		}
	}

	// Create shader resource view if requested
	if (rtSRV)
	{
		D3D10_SHADER_RESOURCE_VIEW_DESC srvDesc;
		srvDesc.Format = format;
		if (arraySize > 1)
		{
			srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2DARRAY;
			srvDesc.Texture2DArray.FirstArraySlice = 0;
			srvDesc.Texture2DArray.ArraySize = arraySize;
			srvDesc.Texture2DArray.MostDetailedMip = 0;
			srvDesc.Texture2DArray.MipLevels = mipmapCount;
		}
		else
		{
			srvDesc.ViewDimension = (samples > 1)? D3D10_SRV_DIMENSION_TEXTURE2DMS : D3D10_SRV_DIMENSION_TEXTURE2D;
			srvDesc.Texture2D.MostDetailedMip = 0;
			srvDesc.Texture2D.MipLevels = mipmapCount;
		}
		if (FAILED(m_device->CreateShaderResourceView(renderTarget, &srvDesc, rtSRV)))
		{
			MessageBoxA(m_hwnd, "CreateShaderResourceView failed", "Error", MB_OK | MB_ICONERROR);
			return NULL;
		}
	}

	// Create shader resource view for each slice if requested
	if (rtSliceSRVs)
	{
		D3D10_SHADER_RESOURCE_VIEW_DESC srvDesc;
		srvDesc.Format = format;
		srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2DARRAY;
		srvDesc.Texture2DArray.ArraySize = 1;
		srvDesc.Texture2DArray.MostDetailedMip = 0;
		srvDesc.Texture2DArray.MipLevels = mipmapCount;
		for (int i = 0; i < arraySize; i++)
		{
			srvDesc.Texture2DArray.FirstArraySlice = i;

			if (FAILED(m_device->CreateShaderResourceView(renderTarget, &srvDesc, &rtSliceSRVs[i])))
			{
				MessageBoxA(m_hwnd, "CreateShaderResourceView failed", "Error", MB_OK | MB_ICONERROR);
				return NULL;
			}
		}
	}

	return renderTarget;
}

ID3D10Texture3D *D3D10Context::CreateRenderTarget3D(const DXGI_FORMAT format, const int width, const int height, const int depth, const int mipmapCount, ID3D10RenderTargetView **rtRTV, ID3D10ShaderResourceView **rtSRV, const unsigned int flags)
{
	ID3D10Texture3D *renderTarget;

	// Setup texture description
	D3D10_TEXTURE3D_DESC desc;
	desc.Width  = width;
	desc.Height = height;
	desc.Depth  = depth;
	desc.MipLevels = mipmapCount;
	desc.Format = format;
	desc.Usage = D3D10_USAGE_DEFAULT;
	desc.BindFlags = D3D10_BIND_RENDER_TARGET | D3D10_BIND_SHADER_RESOURCE;
	desc.CPUAccessFlags = 0;
	desc.MiscFlags = 0;
	if (FAILED(m_device->CreateTexture3D(&desc, NULL, &renderTarget)))
	{
		char str[256];
		sprintf(str, "Couldn't create render target (%s, %dx%dx%d, %d)", g_formats[format].name, width, height, depth, mipmapCount);
		MessageBoxA(m_hwnd, str, "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	// Create render target view if requested
	if (rtRTV)
	{
		D3D10_RENDER_TARGET_VIEW_DESC rtDesc;
		rtDesc.Format = format;
		rtDesc.ViewDimension = D3D10_RTV_DIMENSION_TEXTURE3D;
		rtDesc.Texture3D.FirstWSlice = 0;
		rtDesc.Texture3D.WSize = -1;
		rtDesc.Texture3D.MipSlice = 0;
		if (FAILED(m_device->CreateRenderTargetView(renderTarget, &rtDesc, rtRTV)))
		{
			MessageBoxA(m_hwnd, "CreateRenderTargetView failed", "Error", MB_OK | MB_ICONERROR);
			return NULL;
		}
	}

	// Create shader resource view if requested
	if (rtSRV)
	{
		D3D10_SHADER_RESOURCE_VIEW_DESC srvDesc;
		srvDesc.Format = format;
		srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE3D;
		srvDesc.Texture3D.MostDetailedMip = 0;
		srvDesc.Texture3D.MipLevels = mipmapCount;
		if (FAILED(m_device->CreateShaderResourceView(renderTarget, &srvDesc, rtSRV)))
		{
			MessageBoxA(m_hwnd, "CreateShaderResourceView failed", "Error", MB_OK | MB_ICONERROR);
			return NULL;
		}
	}

	return renderTarget;
}

ID3D10Texture2D *D3D10Context::CreateRenderTargetCube(const DXGI_FORMAT format, const int size, const int samples, const int mipmapCount,
	ID3D10RenderTargetView **rtArrayRTV, ID3D10RenderTargetView *rtFaceRTVs[6], ID3D10ShaderResourceView **rtSRV, const unsigned int flags)
{
	ID3D10Texture2D *renderTarget;

	// Setup texture description
	D3D10_TEXTURE2D_DESC desc;
	desc.Width  = size;
	desc.Height = size;
	desc.MipLevels = mipmapCount;
	desc.ArraySize = 6;
	desc.SampleDesc.Count = samples;
	desc.SampleDesc.Quality = 0;
	desc.Format = format;
	desc.Usage = D3D10_USAGE_DEFAULT;
	desc.BindFlags = D3D10_BIND_RENDER_TARGET | D3D10_BIND_SHADER_RESOURCE;
	desc.CPUAccessFlags = 0;
	desc.MiscFlags = D3D10_RESOURCE_MISC_TEXTURECUBE;
	if (FAILED(m_device->CreateTexture2D(&desc, NULL, &renderTarget)))
	{
		char str[256];
		sprintf(str, "Couldn't create render target (%s, %d, %d, %d)", g_formats[format].name, size, samples, mipmapCount);
		MessageBoxA(m_hwnd, str, "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	// Create render target view if requested
	if (rtArrayRTV)
	{
		D3D10_RENDER_TARGET_VIEW_DESC rtDesc;
		rtDesc.Format = format;
		rtDesc.ViewDimension = D3D10_RTV_DIMENSION_TEXTURE2DARRAY;
		rtDesc.Texture2DArray.FirstArraySlice = 0;
		rtDesc.Texture2DArray.ArraySize = 6;
		rtDesc.Texture2DArray.MipSlice = 0;
		if (FAILED(m_device->CreateRenderTargetView(renderTarget, &rtDesc, rtArrayRTV)))
		{
			MessageBoxA(m_hwnd, "CreateRenderTargetView failed", "Error", MB_OK | MB_ICONERROR);
			return NULL;
		}
	}

	// Create render target view of individual faces if requested
	if (rtFaceRTVs)
	{
		D3D10_RENDER_TARGET_VIEW_DESC rtDesc;
		rtDesc.Format = format;
		rtDesc.ViewDimension = D3D10_RTV_DIMENSION_TEXTURE2DARRAY;
		rtDesc.Texture2DArray.ArraySize = 1;
		rtDesc.Texture2DArray.MipSlice = 0;
		for (int i = 0; i < 6; i++)
		{
			rtDesc.Texture2DArray.FirstArraySlice = i;
			if (FAILED(m_device->CreateRenderTargetView(renderTarget, &rtDesc, &rtFaceRTVs[i])))
			{
				MessageBoxA(m_hwnd, "CreateRenderTargetView failed", "Error", MB_OK | MB_ICONERROR);
				return NULL;
			}
		}
	}

	// Create shader resource view if requested
	if (rtSRV)
	{
		D3D10_SHADER_RESOURCE_VIEW_DESC srvDesc;
		srvDesc.Format = format;
		srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURECUBE;
		srvDesc.TextureCube.MostDetailedMip = 0;
		srvDesc.TextureCube.MipLevels = mipmapCount;
		if (FAILED(m_device->CreateShaderResourceView(renderTarget, &srvDesc, rtSRV)))
		{
			MessageBoxA(m_hwnd, "CreateShaderResourceView failed", "Error", MB_OK | MB_ICONERROR);
			return NULL;
		}
	}

	return renderTarget;
}

ID3D10Texture2D *D3D10Context::CreateDepthTarget2D(const DXGI_FORMAT format, const int width, const int height, const int arraySize, const int samples, const int mipmapCount,
	ID3D10DepthStencilView **rtArrayDSV, ID3D10DepthStencilView **rtSliceDSVs,  ID3D10ShaderResourceView **rtArraySRV, ID3D10ShaderResourceView **rtSliceSRVs, const unsigned int flags)
{
	ID3D10Texture2D *depthTarget;

	// Setup depth stencil texture description
	D3D10_TEXTURE2D_DESC desc;
	desc.Width  = width;
	desc.Height = height;
	desc.MipLevels = mipmapCount;
	desc.ArraySize = arraySize;
	desc.Format = format;
	desc.SampleDesc.Count = samples;
	desc.SampleDesc.Quality = 0;
	desc.Usage = D3D10_USAGE_DEFAULT;
	desc.BindFlags = D3D10_BIND_DEPTH_STENCIL;
	desc.CPUAccessFlags = 0;
	desc.MiscFlags = 0;

	DXGI_FORMAT srvFormat = DXGI_FORMAT_UNKNOWN;
	DXGI_FORMAT dsvFormat = DXGI_FORMAT_UNKNOWN;
	if (rtArraySRV || rtSliceSRVs)
	{
		desc.BindFlags |= D3D10_BIND_SHADER_RESOURCE;
		if (format == DXGI_FORMAT_R16_TYPELESS)
		{
			srvFormat = DXGI_FORMAT_R16_UNORM;
			dsvFormat = DXGI_FORMAT_D16_UNORM;
		}
		else if (format == DXGI_FORMAT_R32_TYPELESS)
		{
			srvFormat = DXGI_FORMAT_R32_FLOAT;
			dsvFormat = DXGI_FORMAT_D32_FLOAT;
		}
	}

	if (FAILED(m_device->CreateTexture2D(&desc, NULL, &depthTarget)))
	{
		char str[256];
		sprintf(str, "Couldn't create depth target (%s, %dx%d, %d, %d, %d)", g_formats[format].name, width, height, arraySize, samples, mipmapCount);
		MessageBoxA(m_hwnd, str, "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	// Create depth stencil view if requested
	if (rtArrayDSV)
	{
		D3D10_DEPTH_STENCIL_VIEW_DESC dsvDesc;
		dsvDesc.Format = dsvFormat;
		if (arraySize > 1)
		{
			dsvDesc.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2DARRAY;
			dsvDesc.Texture2DArray.FirstArraySlice = 0;
			dsvDesc.Texture2DArray.ArraySize = arraySize;
			dsvDesc.Texture2DArray.MipSlice = 0;
		}
		else
		{
			dsvDesc.ViewDimension = (samples > 1)? D3D10_DSV_DIMENSION_TEXTURE2DMS : D3D10_DSV_DIMENSION_TEXTURE2D;
			dsvDesc.Texture2D.MipSlice = 0;
		}
		if (FAILED(m_device->CreateDepthStencilView(depthTarget, &dsvDesc, rtArrayDSV)))
		{
			MessageBoxA(m_hwnd, "CreateDepthStencilView failed", "Error", MB_OK | MB_ICONERROR);
			return NULL;
		}
	}

	// Create depth stencil view for each slice if requested
	if (rtSliceDSVs)
	{
		D3D10_DEPTH_STENCIL_VIEW_DESC dsvDesc;
		dsvDesc.Format = dsvFormat;
		dsvDesc.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2DARRAY;
		dsvDesc.Texture2DArray.ArraySize = 1;
		dsvDesc.Texture2DArray.MipSlice = 0;
		for (int i = 0; i < arraySize; i++)
		{
			dsvDesc.Texture2DArray.FirstArraySlice = i;
			if (FAILED(m_device->CreateDepthStencilView(depthTarget, &dsvDesc, &rtSliceDSVs[i])))
			{
				MessageBoxA(m_hwnd, "CreateDepthStencilView failed", "Error", MB_OK | MB_ICONERROR);
				return NULL;
			}
		}
	}

	// Create shader resource view if requested
	if (rtArraySRV)
	{
		D3D10_SHADER_RESOURCE_VIEW_DESC srvDesc;
		srvDesc.Format = srvFormat;
		if (arraySize > 1)
		{
			srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2DARRAY;
			srvDesc.Texture2DArray.FirstArraySlice = 0;
			srvDesc.Texture2DArray.ArraySize = arraySize;
			srvDesc.Texture2DArray.MostDetailedMip = 0;
			srvDesc.Texture2DArray.MipLevels = mipmapCount;
		}
		else
		{
			srvDesc.ViewDimension = (samples > 1)? D3D10_SRV_DIMENSION_TEXTURE2DMS : D3D10_SRV_DIMENSION_TEXTURE2D;
			srvDesc.Texture2D.MipLevels = mipmapCount;
			srvDesc.Texture2D.MostDetailedMip = 0;
		}
		if (FAILED(m_device->CreateShaderResourceView(depthTarget, &srvDesc, rtArraySRV)))
		{
			MessageBoxA(m_hwnd, "CreateShaderResourceView failed", "Error", MB_OK | MB_ICONERROR);
			return NULL;
		}
	}

	// Create shader resource view for each slice if requested
	if (rtSliceSRVs)
	{
		D3D10_SHADER_RESOURCE_VIEW_DESC srvDesc;
		srvDesc.Format = srvFormat;
		srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2DARRAY;
		srvDesc.Texture2DArray.ArraySize = 1;
		srvDesc.Texture2DArray.MostDetailedMip = 0;
		srvDesc.Texture2DArray.MipLevels = mipmapCount;

		for (int i = 0; i < arraySize; i++)
		{
			srvDesc.Texture2DArray.FirstArraySlice = i;
			if (FAILED(m_device->CreateShaderResourceView(depthTarget, &srvDesc, &rtSliceSRVs[i])))
			{
				MessageBoxA(m_hwnd, "CreateShaderResourceView failed", "Error", MB_OK | MB_ICONERROR);
				return NULL;
			}
		}
	}

	return depthTarget;
}

ID3D10Texture2D *D3D10Context::CreateDepthTargetCube(const DXGI_FORMAT format, const int size, const int samples, const int mipmapCount,
	ID3D10DepthStencilView **rtArrayDSV, ID3D10DepthStencilView *rtFaceDSVs[6], const unsigned int flags)
{
	ID3D10Texture2D *depthTarget;

	// Setup depth stencil texture description
	D3D10_TEXTURE2D_DESC desc;
	desc.Width  = size;
	desc.Height = size;
	desc.MipLevels = mipmapCount;
	desc.ArraySize = 6;
	desc.Format = format;
	desc.SampleDesc.Count = samples;
	desc.SampleDesc.Quality = 0;
	desc.Usage = D3D10_USAGE_DEFAULT;
	desc.BindFlags = D3D10_BIND_DEPTH_STENCIL;
	desc.CPUAccessFlags = 0;
	desc.MiscFlags = D3D10_RESOURCE_MISC_TEXTURECUBE;
	if (FAILED(m_device->CreateTexture2D(&desc, NULL, &depthTarget)))
	{
		char str[256];
		sprintf(str, "Couldn't create depth target (%s, %d, %d, %d)", g_formats[format].name, size, samples, mipmapCount);
		MessageBoxA(m_hwnd, str, "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	// Create depth stencil view if requested
	if (rtArrayDSV)
	{
		D3D10_DEPTH_STENCIL_VIEW_DESC dsvDesc;
		dsvDesc.Format = format;
		dsvDesc.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2DARRAY;
		dsvDesc.Texture2DArray.FirstArraySlice = 0;
		dsvDesc.Texture2DArray.ArraySize = 6;
		dsvDesc.Texture2DArray.MipSlice = 0;
		if (FAILED(m_device->CreateDepthStencilView(depthTarget, &dsvDesc, rtArrayDSV)))
		{
			MessageBoxA(m_hwnd, "CreateDepthStencilView failed", "Error", MB_OK | MB_ICONERROR);
			return NULL;
		}
	}

	// Create depth stencil view for each face if requested
	if (rtFaceDSVs)
	{
		D3D10_DEPTH_STENCIL_VIEW_DESC dsvDesc;
		dsvDesc.Format = format;
		dsvDesc.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2DARRAY;
		dsvDesc.Texture2DArray.ArraySize = 1;
		dsvDesc.Texture2DArray.MipSlice = 0;
		for (int i = 0; i < 6; i++)
		{
			dsvDesc.Texture2DArray.FirstArraySlice = i;
			if (FAILED(m_device->CreateDepthStencilView(depthTarget, &dsvDesc, &rtFaceDSVs[i])))
			{
				MessageBoxA(m_hwnd, "CreateDepthStencilView failed", "Error", MB_OK | MB_ICONERROR);
				return NULL;
			}
		}
	}

	return depthTarget;
}


ID3D10Buffer *D3D10Context::CreateVertexBuffer(const int size, const D3D10_USAGE usage, const void *data)
{
	ID3D10Buffer *vertexBuffer;

	// Setup vertex buffer description
	D3D10_BUFFER_DESC bd;
	bd.Usage = usage;
	bd.ByteWidth = size;
	bd.BindFlags = D3D10_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = (usage == D3D10_USAGE_IMMUTABLE)? 0 : D3D10_CPU_ACCESS_WRITE;
	bd.MiscFlags = 0;

	D3D10_SUBRESOURCE_DATA srData;
	srData.pSysMem = data;
	srData.SysMemPitch = 0;
	srData.SysMemSlicePitch = 0;
	if (FAILED(m_device->CreateBuffer(&bd, data? &srData : NULL, &vertexBuffer)))
	{
		MessageBoxA(m_hwnd, "Vertex buffer creation failed", "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	return vertexBuffer;
}

ID3D10Buffer *D3D10Context::CreateIndexBuffer(const int size, const D3D10_USAGE usage, const void *data)
{
	ID3D10Buffer *indexBuffer;

	// Setup index buffer description
	D3D10_BUFFER_DESC bd;
	bd.Usage = usage;
	bd.ByteWidth = size;
	bd.BindFlags = D3D10_BIND_INDEX_BUFFER;
	bd.CPUAccessFlags = (usage == D3D10_USAGE_IMMUTABLE)? 0 : D3D10_CPU_ACCESS_WRITE;
	bd.MiscFlags = 0;

	D3D10_SUBRESOURCE_DATA srData;
	srData.pSysMem = data;
	srData.SysMemPitch = 0;
	srData.SysMemSlicePitch = 0;
	if (FAILED(m_device->CreateBuffer(&bd, data? &srData : NULL, &indexBuffer)))
	{
		MessageBoxA(m_hwnd, "Index buffer creation failed", "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	return indexBuffer;
}

ID3D10Buffer *D3D10Context::CreateConstantBuffer(const int size, const D3D10_USAGE usage, const void *data)
{
	ID3D10Buffer *constantBuffer;

	// Setup constant buffer description
	D3D10_BUFFER_DESC bd;
	bd.Usage = usage;
	bd.ByteWidth = size;
	bd.BindFlags = D3D10_BIND_CONSTANT_BUFFER;
	bd.CPUAccessFlags = (usage == D3D10_USAGE_IMMUTABLE)? 0 : D3D10_CPU_ACCESS_WRITE;
	bd.MiscFlags = 0;

	D3D10_SUBRESOURCE_DATA srData;
	srData.pSysMem = data;
	srData.SysMemPitch = 0;
	srData.SysMemSlicePitch = 0;
	if (FAILED(m_device->CreateBuffer(&bd, data? &srData : NULL, &constantBuffer)))
	{
		MessageBoxA(m_hwnd, "Constant buffer creation failed", "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	return constantBuffer;
}

ID3D10Buffer *D3D10Context::CreateEffectConstantBuffer(ID3D10Effect *effect, const char *name)
{
	// Get constant buffer variable from the effect
	ID3D10EffectConstantBuffer *cbVar = effect->GetConstantBufferByName(name);
	if (!cbVar->IsValid())
	{
		char str[256];
		sprintf(str, "\"%s\" is not a valid constant buffer");
		MessageBoxA(m_hwnd, str, "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	// Get the actual constant buffer ...
	ID3D10Buffer *cb;
	cbVar->GetConstantBuffer(&cb);

	// ... get its info ...
	D3D10_BUFFER_DESC desc;
	cb->GetDesc(&desc);

	// ... delete it ...
	cb->Release();

	// ... and replace with a better one
	desc.Usage = D3D10_USAGE_DYNAMIC;
	desc.CPUAccessFlags = D3D10_CPU_ACCESS_WRITE;
	if (FAILED(m_device->CreateBuffer(&desc, NULL, &cb)))
	{
		MessageBoxA(m_hwnd, "Constant buffer creation failed", "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	cbVar->SetConstantBuffer(cb);

	return cb;
}

ID3D10InputLayout *D3D10Context::CreateInputLayout(ID3D10EffectPass *effectPass, const D3D10_INPUT_ELEMENT_DESC *layout, const int elementCount)
{
	D3D10_PASS_DESC passDesc;
	if (FAILED(effectPass->GetDesc(&passDesc))) return NULL;

	ID3D10InputLayout *inputLayout;
	if (FAILED(m_device->CreateInputLayout(layout, elementCount, passDesc.pIAInputSignature, passDesc.IAInputSignatureSize, &inputLayout)))
	{
		MessageBoxA(m_hwnd, "Input layout creation failed", "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	return inputLayout;
}

ID3D10RenderTargetView *D3D10Context::CreateRenderTargetView2D(ID3D10Texture2D *renderTarget, const DXGI_FORMAT format, const int firstSlice, const int arraySize)
{
	D3D10_RENDER_TARGET_VIEW_DESC rtDesc;
	rtDesc.Format = format;
	if (arraySize > 0)
	{
		rtDesc.ViewDimension = D3D10_RTV_DIMENSION_TEXTURE2DARRAY;
		rtDesc.Texture2DArray.FirstArraySlice = firstSlice;
		rtDesc.Texture2DArray.ArraySize = arraySize;
		rtDesc.Texture2DArray.MipSlice = 0;
	}
	else
	{
		rtDesc.ViewDimension = D3D10_RTV_DIMENSION_TEXTURE2D;
		rtDesc.Texture2D.MipSlice = 0;
	}

	ID3D10RenderTargetView *rtv;
	if (FAILED(m_device->CreateRenderTargetView(renderTarget, &rtDesc, &rtv)))
	{
		MessageBoxA(m_hwnd, "CreateRenderTargetView failed", "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	return rtv;
}

ID3D10RenderTargetView *D3D10Context::CreateRenderTargetView3D(ID3D10Texture3D *renderTarget, const DXGI_FORMAT format, const int firstSlice, const int arraySize)
{
	D3D10_RENDER_TARGET_VIEW_DESC rtDesc;
	rtDesc.Format = format;
	rtDesc.ViewDimension = D3D10_RTV_DIMENSION_TEXTURE3D;
	rtDesc.Texture3D.FirstWSlice = firstSlice;
	rtDesc.Texture3D.WSize = arraySize;
	rtDesc.Texture3D.MipSlice = 0;

	ID3D10RenderTargetView *rtv;
	if (FAILED(m_device->CreateRenderTargetView(renderTarget, &rtDesc, &rtv)))
	{
		MessageBoxA(m_hwnd, "CreateRenderTargetView failed", "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	return rtv;
}

ID3D10DepthStencilView *D3D10Context::CreateDepthStencilView2D(ID3D10Texture2D *depthTarget, const DXGI_FORMAT format, const int firstSlice, const int arraySize)
{
	D3D10_DEPTH_STENCIL_VIEW_DESC dsvDesc;
	dsvDesc.Format = format;
	if (arraySize > 0)
	{
		dsvDesc.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2DARRAY;
		dsvDesc.Texture2DArray.FirstArraySlice = firstSlice;
		dsvDesc.Texture2DArray.ArraySize = arraySize;
		dsvDesc.Texture2DArray.MipSlice = 0;
	}
	else
	{
		dsvDesc.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2D;
		dsvDesc.Texture2D.MipSlice = 0;
	}

	ID3D10DepthStencilView *dsv;
	if (FAILED(m_device->CreateDepthStencilView(depthTarget, &dsvDesc, &dsv)))
	{
		MessageBoxA(m_hwnd, "CreateDepthStencilView failed", "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	return dsv;
}


ID3D10ShaderResourceView *D3D10Context::CreateDefaultSRV(ID3D10Resource *resource)
{
	// Check what kind of texture it is
	D3D10_RESOURCE_DIMENSION type;
	resource->GetType(&type);

	D3D10_SHADER_RESOURCE_VIEW_DESC srvDesc;
	ID3D10ShaderResourceView *srv;

	// Fill shader resrouce view description based on input texture format and type
	switch (type)
	{
		case D3D10_RESOURCE_DIMENSION_TEXTURE1D:
			D3D10_TEXTURE1D_DESC desc1d;
			((ID3D10Texture1D *) resource)->GetDesc(&desc1d);

			srvDesc.Format = desc1d.Format;
			if (desc1d.ArraySize > 1)
			{
				srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2DARRAY;
				srvDesc.Texture1DArray.FirstArraySlice = 0;
				srvDesc.Texture1DArray.ArraySize = desc1d.ArraySize;
				srvDesc.Texture1DArray.MostDetailedMip = 0;
				srvDesc.Texture1DArray.MipLevels = desc1d.MipLevels;
			}
			else
			{
				srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE1D;
				srvDesc.Texture1D.MostDetailedMip = 0;
				srvDesc.Texture1D.MipLevels = desc1d.MipLevels;
			}
			break;
		case D3D10_RESOURCE_DIMENSION_TEXTURE2D:
			D3D10_TEXTURE2D_DESC desc2d;
			((ID3D10Texture2D *) resource)->GetDesc(&desc2d);

			srvDesc.Format = desc2d.Format;
			if (desc2d.ArraySize > 1)
			{
				if (desc2d.MiscFlags & D3D10_RESOURCE_MISC_TEXTURECUBE)
				{
					srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURECUBE;
					srvDesc.TextureCube.MostDetailedMip = 0;
					srvDesc.TextureCube.MipLevels = desc2d.MipLevels;
				}
				else
				{
					srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2DARRAY;
					srvDesc.Texture2DArray.FirstArraySlice = 0;
					srvDesc.Texture2DArray.ArraySize = desc2d.ArraySize;
					srvDesc.Texture2DArray.MostDetailedMip = 0;
					srvDesc.Texture2DArray.MipLevels = desc2d.MipLevels;
				}
			}
			else
			{
				srvDesc.ViewDimension = (desc2d.SampleDesc.Count > 1)? D3D10_SRV_DIMENSION_TEXTURE2DMS : D3D10_SRV_DIMENSION_TEXTURE2D;
				srvDesc.Texture2D.MostDetailedMip = 0;
				srvDesc.Texture2D.MipLevels = desc2d.MipLevels;
			}
			break;
		case D3D10_RESOURCE_DIMENSION_TEXTURE3D:
			D3D10_TEXTURE3D_DESC desc3d;
			((ID3D10Texture3D *) resource)->GetDesc(&desc3d);

			srvDesc.Format = desc3d.Format;
			srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE3D;
			srvDesc.Texture3D.MostDetailedMip = 0;
			srvDesc.Texture3D.MipLevels = desc3d.MipLevels;
			break;
		default:
			MessageBoxA(m_hwnd, "Unsupported type", "Error", MB_OK | MB_ICONERROR);
			return NULL;
	}

	if (FAILED(m_device->CreateShaderResourceView(resource, &srvDesc, &srv)))
	{
		MessageBoxA(m_hwnd, "CreateShaderResourceView failed", "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	return srv;
}

ID3D10SamplerState *D3D10Context::CreateSamplerState(const D3D10_FILTER filter, D3D10_TEXTURE_ADDRESS_MODE addressMode, const float maxLOD)
{
	// Fill sampler state description
	D3D10_SAMPLER_DESC desc;
	desc.Filter = filter;
	desc.AddressU = addressMode;
	desc.AddressV = addressMode;
	desc.AddressW = addressMode;
	desc.MipLODBias = 0;
	desc.MaxAnisotropy = (filter == D3D10_FILTER_ANISOTROPIC || filter == D3D10_FILTER_COMPARISON_ANISOTROPIC)? 16 : 1;
	desc.ComparisonFunc = D3D10_COMPARISON_LESS;
	desc.BorderColor[0] = 0;
	desc.BorderColor[1] = 0;
	desc.BorderColor[2] = 0;
	desc.BorderColor[3] = 0;
	desc.MinLOD = 0;
	desc.MaxLOD = maxLOD;

	ID3D10SamplerState *samplerState;
	if (FAILED(m_device->CreateSamplerState(&desc, &samplerState)))
	{
		MessageBoxA(m_hwnd, "CreateSamplerState failed", "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	return samplerState;
}

ID3D10BlendState *D3D10Context::CreateBlendState(const D3D10_BLEND src, const D3D10_BLEND dst, const D3D10_BLEND_OP op, const UINT8 mask, const int count)
{
	BOOL blendEnable = (src != D3D10_BLEND_ONE || dst != D3D10_BLEND_ZERO);

	// Fill blendstate description
	D3D10_BLEND_DESC desc;
	desc.AlphaToCoverageEnable = false;
	desc.BlendOp = op;
	desc.SrcBlend = src;
	desc.DestBlend = dst;
	desc.BlendOpAlpha = op;
	desc.SrcBlendAlpha = src;
	desc.DestBlendAlpha = dst;

	// Enable blend and set the mask for the provided number of MRTs
	memset(&desc.BlendEnable, 0, sizeof(desc.BlendEnable));
	memset(&desc.RenderTargetWriteMask, 0, sizeof(desc.RenderTargetWriteMask));
	for (int i = 0; i < count; i++)
	{
		desc.BlendEnable[i] = blendEnable;
		desc.RenderTargetWriteMask[i] = mask;
	}

	ID3D10BlendState *blendState;
	if (FAILED(m_device->CreateBlendState(&desc, &blendState)))
	{
		MessageBoxA(m_hwnd, "CreateBlendState failed", "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	return blendState;
}

ID3D10RasterizerState *D3D10Context::CreateRasterizerState(const D3D10_CULL_MODE cullMode, const D3D10_FILL_MODE fillMode, const bool multisampleEnable,
	const bool depthClipEnable, const int depthBias, const float slopeScaledDepthBias, const bool scissorEnable)
{
	// Fill in the rasterizer state description
	D3D10_RASTERIZER_DESC desc;
	desc.AntialiasedLineEnable = false;
	desc.CullMode = cullMode;
	desc.DepthBias = depthBias;
	desc.DepthBiasClamp = 0.0f;
	desc.DepthClipEnable = depthClipEnable;
	desc.FillMode = fillMode;
	desc.FrontCounterClockwise = false;
	desc.MultisampleEnable = multisampleEnable;
	desc.ScissorEnable = scissorEnable;
	desc.SlopeScaledDepthBias = slopeScaledDepthBias;

	ID3D10RasterizerState *rasterizerState;
	if (FAILED(m_device->CreateRasterizerState(&desc, &rasterizerState)))
	{
		MessageBoxA(m_hwnd, "CreateRasterizerState failed", "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	return rasterizerState;
}

ID3D10DepthStencilState *D3D10Context::CreateDepthStencilState(const bool depthEnable, const bool depthWriteEnable, const D3D10_COMPARISON_FUNC depthFunc)
{
	// Fill in the depth stencil state description
	D3D10_DEPTH_STENCIL_DESC desc;
	desc.DepthEnable = depthEnable;
	desc.DepthWriteMask = depthWriteEnable? D3D10_DEPTH_WRITE_MASK_ALL : D3D10_DEPTH_WRITE_MASK_ZERO;
	desc.DepthFunc = depthFunc;
	desc.StencilEnable = false;
	desc.StencilReadMask = 0;
	desc.StencilWriteMask = 0;
	desc.BackFace.StencilFunc = D3D10_COMPARISON_ALWAYS;
	desc.BackFace.StencilDepthFailOp = D3D10_STENCIL_OP_KEEP;
	desc.BackFace.StencilFailOp = D3D10_STENCIL_OP_KEEP;
	desc.BackFace.StencilPassOp = D3D10_STENCIL_OP_KEEP;
	desc.FrontFace.StencilFunc = D3D10_COMPARISON_ALWAYS;
	desc.FrontFace.StencilDepthFailOp = D3D10_STENCIL_OP_KEEP;
	desc.FrontFace.StencilFailOp = D3D10_STENCIL_OP_KEEP;
	desc.FrontFace.StencilPassOp = D3D10_STENCIL_OP_KEEP;

	ID3D10DepthStencilState *depthStencilState;
	if (FAILED(m_device->CreateDepthStencilState(&desc, &depthStencilState)))
	{
		MessageBoxA(m_hwnd, "CreateDepthStencilState failed", "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	return depthStencilState;	
}

ID3D10Texture1D *D3D10Context::CreateTexture1D(const void *data, const DXGI_FORMAT format, const int width, const int arraySize, ID3D10ShaderResourceView **resourceView, const unsigned int flags)
{
	// Setup the texture description
	D3D10_TEXTURE1D_DESC desc;
	desc.Width  = width;
	desc.Format = format;
	desc.MipLevels = 1;
	desc.ArraySize = arraySize;
	desc.Usage = D3D10_USAGE_IMMUTABLE;
	desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
	desc.CPUAccessFlags = 0;
	desc.MiscFlags = 0;

	// Setup data pointers
	D3D10_SUBRESOURCE_DATA *texData = new D3D10_SUBRESOURCE_DATA[arraySize];
	int size = width * g_formats[format].bpp / 8;

	for (int i = 0; i < arraySize; i++)
	{
		texData[i].pSysMem = ((ubyte *) data) + i * size;
	}

	ID3D10Texture1D *texture;
	HRESULT hr = m_device->CreateTexture1D(&desc, texData, &texture);
	delete texData;

	if (FAILED(hr))
	{
		MessageBoxA(m_hwnd, "CreateTexture1D failed", "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	// Return a shader resource view if requested
	if (resourceView)
	{
		*resourceView = CreateDefaultSRV(texture);
	}

	return texture;
}

ID3D10Texture2D *D3D10Context::CreateTexture2D(const void *data, const DXGI_FORMAT format, const int width, const int height, const int arraySize, ID3D10ShaderResourceView **resourceView, const unsigned int flags)
{
	// Setup the texture description
	D3D10_TEXTURE2D_DESC desc;
	desc.Width  = width;
	desc.Height = height;
	desc.Format = format;
	desc.MipLevels = 1;
	desc.ArraySize = arraySize;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.Usage = D3D10_USAGE_IMMUTABLE;
	desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
	desc.CPUAccessFlags = 0;
	desc.MiscFlags = 0;

	// Setup data pointers
	D3D10_SUBRESOURCE_DATA *texData = new D3D10_SUBRESOURCE_DATA[arraySize];
	int lineSize = width * g_formats[format].bpp / 8;
	if (format >= DXGI_FORMAT_BC1_TYPELESS && format <= DXGI_FORMAT_BC5_SNORM) lineSize *= 4;
	int sliceSize = height * lineSize;

	for (int i = 0; i < arraySize; i++)
	{
		texData[i].pSysMem = ((ubyte *) data) + i * sliceSize;
		texData[i].SysMemPitch = lineSize;
	}

	ID3D10Texture2D *texture;
	HRESULT hr = m_device->CreateTexture2D(&desc, texData, &texture);
	delete texData;

	if (FAILED(hr))
	{
		MessageBoxA(m_hwnd, "CreateTexture2D failed", "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	// Return a shader resource view if requested
	if (resourceView)
	{
		*resourceView = CreateDefaultSRV(texture);
	}

	return texture;
}

ID3D10Texture3D *D3D10Context::CreateTexture3D(const void *data, const DXGI_FORMAT format, const int width, const int height, const int depth, ID3D10ShaderResourceView **resourceView, const unsigned int flags)
{
	// Setup the texture description
	D3D10_TEXTURE3D_DESC desc;
	desc.Width  = width;
	desc.Height = height;
	desc.Depth  = depth;
	desc.Format = format;
	desc.MipLevels = 1;
	desc.Usage = D3D10_USAGE_IMMUTABLE;
	desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
	desc.CPUAccessFlags = 0;
	desc.MiscFlags = 0;

	// Setup data pointer
	D3D10_SUBRESOURCE_DATA texData;
	texData.pSysMem = data;
	texData.SysMemPitch = width * g_formats[format].bpp / 8;
	texData.SysMemSlicePitch = texData.SysMemPitch * height;

	ID3D10Texture3D *texture;
	if (FAILED(m_device->CreateTexture3D(&desc, &texData, &texture)))
	{
		MessageBoxA(m_hwnd, "CreateTexture3D failed", "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	// Return a shader resource view if requested
	if (resourceView)
	{
		*resourceView = CreateDefaultSRV(texture);
	}

	return texture;
}

ID3D10Texture2D *D3D10Context::CreateTextureCube(const void *data, const DXGI_FORMAT format, const int size, ID3D10ShaderResourceView **resourceView, const unsigned int flags)
{
	// Setup the texture description
	D3D10_TEXTURE2D_DESC desc;
	desc.Width  = size;
	desc.Height = size;
	desc.Format = format;
	desc.MipLevels = 1;
	desc.ArraySize = 6;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.Usage = D3D10_USAGE_IMMUTABLE;
	desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
	desc.CPUAccessFlags = 0;
	desc.MiscFlags = D3D10_RESOURCE_MISC_TEXTURECUBE;

	D3D10_SUBRESOURCE_DATA texData[6];
	int lineSize  = size * g_formats[format].bpp / 8;
	int sliceSize = size * lineSize;
	for (int i = 0; i < 6; i++)
	{
		texData[i].pSysMem = ((ubyte *) data) + i * sliceSize;
		texData[i].SysMemPitch = lineSize;
	}

	ID3D10Texture2D *texture;
	if (FAILED(m_device->CreateTexture2D(&desc, texData, &texture)))
	{
		MessageBoxA(m_hwnd, "CreateTexture2D failed", "Error", MB_OK | MB_ICONERROR);
		return NULL;
	}

	// Return a shader resource view if requested
	if (resourceView)
	{
		D3D10_SHADER_RESOURCE_VIEW_DESC srvDesc;
		ZeroMemory(&srvDesc, sizeof(srvDesc));
		srvDesc.Format = desc.Format;
		srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURECUBE;
		srvDesc.TextureCube.MostDetailedMip = 0;
		srvDesc.TextureCube.MipLevels = 1;
		if (FAILED(m_device->CreateShaderResourceView(texture, &srvDesc, resourceView)))
		{
			MessageBoxA(m_hwnd, "CreateShaderResourceView failed", "Error", MB_OK | MB_ICONERROR);
			return NULL;
		}
	}

	return texture;
}

ID3D10Resource *D3D10Context::LoadTexture(const TCHAR *fileName, ID3D10ShaderResourceView **resourceView, const unsigned int flags)
{
	TCHAR str[256];
/*
	FILE *file = _tfopen(fileName, _T("rb"));
	if (file == NULL)
	{
		_stprintf(str, _T("Couldn't load \"%s\""), fileName);
		MessageBox(m_hwnd, str, _T("Error"), MB_OK | MB_ICONERROR);

		return NULL;
	}

	fseek(file, 0, SEEK_END);
	long size = ftell(file);
	fseek(file, 0, SEEK_SET);

	ubyte *mem = new ubyte[size];

	fread(mem, 1, size, file);
	fclose(file);

	D3DX10_IMAGE_INFO info;
	D3DX10GetImageInfoFromMemory(mem, size, NULL, &info, NULL);
	if (info.Height == 1)
	{
		info.ResourceDimension = D3D10_RESOURCE_DIMENSION_TEXTURE1D;
		info.Height = 0;
		info.Depth = 0;
	}

	D3DX10_IMAGE_LOAD_INFO loadInfo;
	loadInfo.BindFlags = D3D10_BIND_SHADER_RESOURCE;
	loadInfo.CpuAccessFlags = 0;
	loadInfo.Depth = info.Depth;
	loadInfo.Filter = D3DX10_FILTER_NONE;
	loadInfo.FirstMipLevel = 0;
	loadInfo.Format = info.Format;
	loadInfo.Height = info.Height;
	loadInfo.MipFilter = D3DX10_FILTER_NONE;//D3DX10_FILTER_BOX;
	loadInfo.MipLevels = info.MipLevels;
	loadInfo.MiscFlags = info.MiscFlags;
	loadInfo.pSrcInfo = &info;
	loadInfo.Usage = D3D10_USAGE_IMMUTABLE;
	loadInfo.Width = info.Width;

	ID3D10Resource *texture;
	HRESULT hr = D3DX10CreateTextureFromMemory(m_device, mem, size, &loadInfo, NULL, &texture, NULL);

	delete mem;

	if (FAILED(hr))
*/

	ID3D10Resource *texture;
	if (FAILED(D3DX10CreateTextureFromFile(m_device, fileName, NULL, NULL, &texture, NULL)))
	{
		_stprintf(str, _T("Couldn't load \"%s\""), fileName);
		MessageBox(m_hwnd, str, _T("Error"), MB_OK | MB_ICONERROR);

		return NULL;
	}

	// Return a shader resource view if requested
	if (resourceView)
	{
		*resourceView = CreateDefaultSRV(texture);
	}

	return texture;
}

ID3D10Effect *D3D10Context::LoadEffect(const TCHAR *fileName, const D3D10_SHADER_MACRO *defines, const unsigned int flags)
{
	TCHAR str[256];

	ID3D10Effect *effect = NULL;
	ID3D10Blob *errors = NULL;

	// Load the effect
	HRESULT hr = D3DX10CreateEffectFromFile(fileName, defines, NULL, "fx_4_0", D3D10_SHADER_ENABLE_STRICTNESS | D3D10_SHADER_PACK_MATRIX_ROW_MAJOR, D3D10_EFFECT_SINGLE_THREADED, m_device, NULL, NULL, &effect, &errors, NULL);
	if (SUCCEEDED(hr))
	{
		// Output assembly if requested
		if (flags & ASSEMBLY)
		{
			ID3D10Blob *assembly = NULL;
			D3D10DisassembleEffect(effect, FALSE, &assembly);
			if (assembly)
			{
				if (flags & DUMP_TO_FILE)
				{
					// Write the assembly to file
					_tcscpy(str, fileName);
					TCHAR *ext = _tcsrchr(str, _T('.'));
					if (ext)
					{
						_tcscpy(ext, _T("Asm.txt"));
						FILE *file = _tfopen(str, _T("wb"));
						if (file)
						{
							fwrite(assembly->GetBufferPointer(), assembly->GetBufferSize(), 1, file);
							fclose(file);
						}
					}
				}
				else
				{
					// Write the assembly to debug output
					OutputDebugStringA((char *) assembly->GetBufferPointer());
				}
				assembly->Release();
			}
		}
	}
	else
	{
		if (errors)
		{
			MessageBoxA(m_hwnd, (LPCSTR) errors->GetBufferPointer(), "Error", MB_OK | MB_ICONERROR);
		}
		else
		{
			_stprintf(str, _T("Error loading \"%s\""), fileName);
			MessageBox(m_hwnd, str, _T("Error"), MB_OK | MB_ICONERROR);
		}
	}
	if (errors)
	{
		OutputDebugStringA((LPCSTR) errors->GetBufferPointer());
		errors->Release();
	}

	return effect;
}

void D3D10Context::SetEffect(ID3D10Effect *effect)
{
	m_currentEffect = effect;
}

#ifdef _DEBUG
#define CHECK_VAR(var, type, name)\
	if (!var->IsValid()){\
		char str[256];\
		sprintf(str, type " is not valid!\n", name);\
		OutputDebugStringA(str);\
		return;\
	}
#else
#define CHECK_VAR(var, type, name)
#endif

void D3D10Context::SetTexture(const char *textureName, ID3D10ShaderResourceView *resourceView)
{
	ID3D10EffectVariable *var = m_currentEffect->GetVariableByName(textureName);
	CHECK_VAR(var, "Texture \"%s\"", textureName);
	var->AsShaderResource()->SetResource(resourceView);
}

void D3D10Context::SetConstant(const char *constantName, const float value)
{
	ID3D10EffectScalarVariable *var = m_currentEffect->GetVariableByName(constantName)->AsScalar();
	CHECK_VAR(var, "Constant \"%s\" as float", constantName);
	var->SetFloat(value);
}

void D3D10Context::SetConstant(const char *constantName, const float2 &value)
{
	ID3D10EffectVectorVariable *var = m_currentEffect->GetVariableByName(constantName)->AsVector();
	CHECK_VAR(var, "Constant \"%s\" as float2", constantName);
	var->SetRawValue((void *) &value, 0, sizeof(value));
}

void D3D10Context::SetConstant(const char *constantName, const float3 &value)
{
	ID3D10EffectVectorVariable *var = m_currentEffect->GetVariableByName(constantName)->AsVector();
	CHECK_VAR(var, "Constant \"%s\" as float3", constantName);
	var->SetRawValue((void *) &value, 0, sizeof(value));
}

void D3D10Context::SetConstant(const char *constantName, const float4 &value)
{
	ID3D10EffectVectorVariable *var = m_currentEffect->GetVariableByName(constantName)->AsVector();
	CHECK_VAR(var, "Constant \"%s\" as float4", constantName);
	var->SetRawValue((void *) &value, 0, sizeof(value));
}

void D3D10Context::SetConstant(const char *constantName, const float4x4 &value)
{
	ID3D10EffectMatrixVariable *var = m_currentEffect->GetVariableByName(constantName)->AsMatrix();
	CHECK_VAR(var, "Constant \"%s\" as float4x4", constantName);
	var->SetMatrix((float *) &value);
}

void D3D10Context::SetConstantArray(const char *constantName, const float4 *value, const int count)
{
	ID3D10EffectVectorVariable *var = m_currentEffect->GetVariableByName(constantName)->AsVector();
	CHECK_VAR(var, "Constant \"%s\" as float4 array", constantName);
	var->SetFloatVectorArray((float *) value, 0, count);
}

void D3D10Context::Apply(const char *techniqueName, const char *passName)
{
	ID3D10EffectTechnique *tech = m_currentEffect->GetTechniqueByName(techniqueName);
	CHECK_VAR(tech, "Technique \"%s\"", techniqueName);

	ID3D10EffectPass *p = tech->GetPassByName(passName);
	CHECK_VAR(p, "Pass \"%s\"", passName);
	p->Apply(0);
}

void D3D10Context::Apply(const char *techniqueName, const int passIndex)
{
	ID3D10EffectTechnique *tech = m_currentEffect->GetTechniqueByName(techniqueName);
	CHECK_VAR(tech, "Technique \"%s\"", techniqueName);

	ID3D10EffectPass *p = tech->GetPassByIndex(passIndex);
	CHECK_VAR(p, "Pass %d", passIndex);
	p->Apply(0);
}

void D3D10Context::Apply(const int techniqueIndex, const int passIndex)
{
	ID3D10EffectTechnique *tech = m_currentEffect->GetTechniqueByIndex(techniqueIndex);
	CHECK_VAR(tech, "Technique %d", techniqueIndex);

	ID3D10EffectPass *p = tech->GetPassByIndex(passIndex);
	CHECK_VAR(p, "Pass %d", passIndex);
	p->Apply(0);
}

void D3D10Context::Clear(const float *clearColor, const UINT dsClearFlags, const float depth, const UINT8 stencil)
{
	// Grab the list of render target targets and the depth buffer
	ID3D10RenderTargetView *rtViews[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
	ID3D10DepthStencilView *dsView = NULL;
	m_device->OMGetRenderTargets(clearColor? 8 : 0, rtViews, dsClearFlags? &dsView : NULL);

	// Clear all active render target
	for (int i = 0; i < 8; i++)
	{
		if (rtViews[i])
		{
			m_device->ClearRenderTargetView(rtViews[i], clearColor);
			rtViews[i]->Release();
		}
	}

	// Clear depth stencil buffer
	if (dsView)
	{
		m_device->ClearDepthStencilView(dsView, dsClearFlags, depth, stencil);
		dsView->Release();
	}
}

void D3D10Context::Present()
{
	// Make a clean exit if the device gets removed for some reason
	if (FAILED(m_device->GetDeviceRemovedReason()))
	{
		PostMessage(m_hwnd, WM_CLOSE, 0, 0);
		return;
	}

	// TODO: Implement a better way to yield when occluded ...
	// This cuts down the CPU usage, but still renders when occluded.
	DWORD sleepTime = 0;
	if (m_swapChain->Present(0, 0) == DXGI_STATUS_OCCLUDED)
	{
		sleepTime = 200;
	}

	Sleep(sleepTime);
}

void D3D10Context::SetRenderTarget(const int width, const int height, ID3D10RenderTargetView **renderTargets, const int renderTargetCount, ID3D10DepthStencilView *depthTarget)
{
	m_device->OMSetRenderTargets(renderTargetCount, renderTargets, depthTarget);

	D3D10_VIEWPORT viewport;
	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;
	viewport.Width  = width;
	viewport.Height = height;
	viewport.MinDepth = 0;
	viewport.MaxDepth = 1;
	m_device->RSSetViewports(1, &viewport);
}

void D3D10Context::SetRTToBackBuffer()
{
	m_device->OMSetRenderTargets(1, &m_backBufferView, m_depthBufferView);

	D3D10_VIEWPORT viewport;
	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;
	viewport.Width  = m_width;
	viewport.Height = m_height;
	viewport.MinDepth = 0;
	viewport.MaxDepth = 1;
	m_device->RSSetViewports(1, &viewport);
}

bool D3D10Context::InitializeBuffers()
{
	if (FAILED(m_swapChain->GetBuffer(0, __uuidof(ID3D10Texture2D), (LPVOID *) &m_backBuffer))) return false;

	if (FAILED(m_device->CreateRenderTargetView(m_backBuffer, NULL, &m_backBufferView)))
	{
		MessageBoxA(m_hwnd, "CreateRenderTargetView failed", "Error", MB_OK | MB_ICONERROR);
		return false;
	}

	// Create depth stencil texture
	D3D10_TEXTURE2D_DESC descDepth;
	descDepth.Width  = m_width;
	descDepth.Height = m_height;
	descDepth.MipLevels = 1;
	descDepth.ArraySize = 1;
	descDepth.Format = m_depthBufferFormat;
	descDepth.SampleDesc.Count = m_msaaSamples;
	descDepth.SampleDesc.Quality = 0;
	descDepth.Usage = D3D10_USAGE_DEFAULT;
	descDepth.BindFlags = D3D10_BIND_DEPTH_STENCIL;
	descDepth.CPUAccessFlags = 0;
	descDepth.MiscFlags = 0;
	if (FAILED(m_device->CreateTexture2D(&descDepth, NULL, &m_depthBuffer)))
	{
		MessageBoxA(m_hwnd, "Couldn't create main depth buffer", "Error", MB_OK | MB_ICONERROR);
		return false;
	}

	// Create the depth stencil view
	D3D10_DEPTH_STENCIL_VIEW_DESC descDSV;
	descDSV.Format = descDepth.Format;
	if (m_msaaSamples > 1)
	{
		descDSV.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2DMS;
	}
	else
	{
		descDSV.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2D;
		descDSV.Texture2D.MipSlice = 0;
	}
	if (FAILED(m_device->CreateDepthStencilView(m_depthBuffer, &descDSV, &m_depthBufferView)))
	{
		MessageBoxA(m_hwnd, "CreateDepthStencilView failed", "Error", MB_OK | MB_ICONERROR);
		return false;
	}

	m_device->OMSetRenderTargets(1, &m_backBufferView, m_depthBufferView);

	// Setup the viewport
	D3D10_VIEWPORT vp;
	vp.Width    = m_width;
	vp.Height   = m_height;
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	m_device->RSSetViewports(1, &vp);

	return true;
}

bool D3D10Context::ReleaseBuffers()
{
	if (m_device) m_device->OMSetRenderTargets(0, NULL, NULL);

	SAFE_RELEASE(m_backBuffer);
	SAFE_RELEASE(m_backBufferView);
	SAFE_RELEASE(m_depthBuffer);
	SAFE_RELEASE(m_depthBufferView);

	return true;
}

#include <io.h>
#include <errno.h>

bool D3D10Context::SaveScreenshot(const TCHAR *name, const D3DX10_IMAGE_FILE_FORMAT format)
{
	TCHAR fileName[256];
	TCHAR *ext = NULL;

	// Select file extension
	switch (format)
	{
	case D3DX10_IFF_DDS:
		ext = _T("dds");
		break;
	case D3DX10_IFF_BMP:
		ext = _T("bmp");
		break;
	case D3DX10_IFF_PNG:
		ext = _T("png");
		break;
	case D3DX10_IFF_JPG:
		ext = _T("jpg");
		break;
	case D3DX10_IFF_TIFF:
		ext = _T("tiff");
		break;
	case D3DX10_IFF_GIF:
		ext = _T("gif");
		break;
	case D3DX10_IFF_WMP:
		ext = _T("wmp");
		break;
	default:
		return false;
	};



	// Find first free filename
	int i = 0;
	do
	{
		_stprintf(fileName, _T("%s%i.%s"), name, i++, ext);
	}
	while (_taccess_s(fileName, 00) != ENOENT);

	// If the backbuffer if multisampled we need to resolve it
	if (m_msaaSamples > 1)
	{
		// Create temporary texture of the same type as the backbuffer
		D3D10_TEXTURE2D_DESC desc;
		m_backBuffer->GetDesc(&desc);
		desc.SampleDesc.Count = 1;
		desc.SampleDesc.Quality = 0;

		ID3D10Texture2D *texture;
		if (SUCCEEDED(m_device->CreateTexture2D(&desc, NULL, &texture)))
		{
			// Resolve buckbuffer into our temporary texture
			m_device->ResolveSubresource(texture, 0, m_backBuffer, 0, m_backBufferFormat);

			HRESULT hr = D3DX10SaveTextureToFile(texture, format, fileName);

			// Remove the temporary texture
			texture->Release();

			return (hr == S_OK);
		}
	}
	else
	{
		// Write out backbuffer directly
		if (SUCCEEDED(D3DX10SaveTextureToFile(m_backBuffer, format, fileName))) return true;
	}

	return false;
}
