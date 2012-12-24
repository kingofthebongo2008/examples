//-----------------------------------------------------------------------------
// File: Framework\D3D10\D3D10App.cpp
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------





#include "D3D10App.h"
#include <stdio.h>

#pragma comment (lib, "d3d11.lib")
#pragma comment (lib, "d3dx11.lib")
#pragma comment (lib, "dxguid.lib")
#pragma comment (lib, "d3dcompiler.lib")
#pragma comment (lib, "Effects11.lib")

// Vertex structure for the tool functions
struct Pos2Tex3
{
	float2 pos;
	float3 tex;
};

D3D10App::D3D10App()
{
	// Initialize all variables to defaults
	m_context = NULL;
	m_toolsEffect = NULL;
	m_toolsVB = NULL;
	m_toolsVsCB = NULL;
	m_toolsPsCB = NULL;
	m_toolsVBSize = 0;
	m_pos3Layout = NULL;
	m_pos2Tex3Layout = NULL;

	m_time = 0;
	m_frameTime = 0;

	memset(m_keys, 0, sizeof(m_keys));
	m_mouseCapture = false;

	m_showFPS = true;

	m_displayPath = true;
	m_displayPathLooping = false;
	m_displaySmooth = false;
	m_benchMark = false;
	m_benchMarkTime = 0;
}

D3D10App::~D3D10App()
{
}

bool D3D10App::Create()
{
	// If the sample didn't create a context already, the framework creates a default one
	if (m_context == NULL)
	{
		m_context = new D3D10Context();
		if (!m_context->Create(_T("Sample"), DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_D16_UNORM, 800, 600, 1, false)) return false;
	}

	// Load the font
	if (FAILED(m_mainFont.LoadFont(m_context->GetDevice(), FONT_PATH _T("Future.dds"), FONT_PATH _T("Future.font")))) return false;

	// Tool resources
	if ((m_toolsEffect = m_context->LoadEffect(SHADER_PATH _T("Tools.fx"))) == NULL) return false;
	if ((m_toolsVsCB = m_context->CreateEffectConstantBuffer(m_toolsEffect, "MainVS")) == NULL) return false;
	if ((m_toolsPsCB = m_context->CreateEffectConstantBuffer(m_toolsEffect, "MainPS")) == NULL) return false;

	D3D11_INPUT_ELEMENT_DESC layout0[] =
	{
		{ "SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	};
	if ((m_pos3Layout = m_context->CreateInputLayout(m_toolsEffect->GetTechniqueByIndex(0)->GetPassByIndex(0), layout0, elementsOf(layout0))) == NULL) return false;

	D3D11_INPUT_ELEMENT_DESC layout1[] =
	{
		{ "SV_Position", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TexCoord",    0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 8, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	};
	if ((m_pos2Tex3Layout = m_context->CreateInputLayout(m_toolsEffect->GetTechniqueByIndex(2)->GetPassByIndex(0), layout1, elementsOf(layout1))) == NULL) return false;

	ResetCamera();

	return true;
}

void D3D10App::Destroy()
{
	// Release our resources
	if (m_context)
	{
		m_mainFont.Release();

		SAFE_RELEASE(m_toolsEffect);
		SAFE_RELEASE(m_toolsVB);
		SAFE_RELEASE(m_toolsVsCB);
		SAFE_RELEASE(m_toolsPsCB);
		SAFE_RELEASE(m_pos3Layout);
		SAFE_RELEASE(m_pos2Tex3Layout);

		m_context->Destroy();
		delete m_context;
	}
}

void D3D10App::CaptureMouse(const bool value)
{
	if (m_mouseCapture != value)
	{
		static POINT point;

		if (value)
		{
			// Store the current position and set cursor to the middle of the window
			GetCursorPos(&point);

			POINT p = { m_context->GetWidth() / 2, m_context->GetHeight() / 2 };
			ClientToScreen(m_context->GetWindow(), &p);
			SetCursorPos(p.x, p.y);
		}
		else
		{
			// Restore the cursor position
			SetCursorPos(point.x, point.y);
		}
		// Set cursor visibility
		ShowCursor((BOOL) m_mouseCapture);

		m_mouseCapture = value;
	}
}

void D3D10App::UpdateCameraPosition(const float speed)
{
	if (m_benchMark)
	{
		if (m_benchMarkTime >= m_camera.GetPathNodeCount())
		{
			m_benchMark = false;
		}
		m_camera.SetCameraToPathTime(m_benchMarkTime, false);
		m_benchMarkTime += m_frameTime;
	}
	else
	{
		if (m_keys[VK_RETURN])
		{
			ResetCamera();
		}
		else
		{
			// Get current orientation
			float3 dx, dy, dz;
			m_camera.GetViewBaseVectorsXY(dx, dy, dz);

			// Camera control
			float3 dir(0, 0, 0);
			if (m_keys[VK_LEFT]  || m_keys['A']) dir -= dx;
			if (m_keys[VK_RIGHT] || m_keys['D']) dir += dx;
			if (m_keys[VK_SHIFT])   dir -= dy;
			if (m_keys[VK_CONTROL]) dir += dy;
			if (m_keys[VK_DOWN] || m_keys['S']) dir -= dz;
			if (m_keys[VK_UP]   || m_keys['W']) dir += dz;

			// Only update if camera actually moved
			float l = dot(dir, dir);
			if (l > 0.01f)
			{
				float s = speed / sqrtf(l);
				// Numpad0 controls sneak mode
				if (m_keys[VK_NUMPAD0]) s *= (1.0f / 16.0f);
				m_camera.SetPosition(m_camera.GetPosition() + s * dir);
			}
		}
	}
}

void D3D10App::ResetCamera()
{
	// Camera defaults. May be overriden by the sample.
	m_camera.SetPosition(float3(0, 0, 0));
	m_camera.SetRotation(0, 0, 0);
}

bool D3D10App::OnMouseClick(HWND hwnd, const int x, const int y, const MouseButton button, const bool pressed)
{
	// Left clicking in the window captures the mouse for free flying
	if (button == MOUSE_LEFT)
	{
		CaptureMouse(true);
		return true;
	}

	return false;
}

bool D3D10App::OnMouseMove(HWND hwnd, const int x, const int y, const bool lButton, const bool mButton, const bool rButton)
{
	if (m_mouseCapture)
	{
		// Only update on every other mouse move event. This is becase every time we move the mouse with SetCursorPos Windows
		// will send the application a WM_MOUSEMOVE triggering this call. We are only interested in actual physical mouse moves.
		static bool changed = false;
		if (changed = !changed)
		{
			const float mouseSensibility = 0.003f;
			int xMid = m_context->GetWidth() / 2;
			int yMid = m_context->GetHeight() / 2;

			m_camera.UpdateRotation(mouseSensibility * (y - yMid), mouseSensibility * (xMid - x));

			// Set cursor back to the middle of the window
			POINT p = { xMid, yMid };
			ClientToScreen(m_context->GetWindow(), &p);
			SetCursorPos(p.x, p.y);
		}

		return true;
	}

	return false;
}

bool D3D10App::OnMouseWheel(HWND hwnd, const int x, const int y, const int scroll)
{

	return false;
}

bool D3D10App::OnKeyPress(HWND hwnd, const unsigned int key, const bool pressed)
{
	if (key < 256) // Sanity check
	{
		// Track key states
		m_keys[key] = pressed;
	}

	if (pressed)
	{
		switch (key)
		{
		// Toggle FPS display
		case VK_SPACE:
			m_showFPS = !m_showFPS;
			break;
		// Tools for creating camera paths
		case VK_INSERT:
			m_camera.AddPathNode();
			break;
		case VK_DELETE:
			m_camera.RemovePathNode();
			break;
		// Fly through the current camera path
		case 'B':
			if (m_benchMark)
			{
				m_benchMark = false;
			}
			else
			{
				m_benchMark = true;
				m_benchMarkTime = 0;
			}
			break;
		// Control display of the camera path
		case 'C':
			m_displaySmooth = !m_displaySmooth;
			break;
		case 'V':
			m_displayPath = !m_displayPath;
			break;
		case 'X':
			m_displayPathLooping = !m_displayPathLooping;
			break;
		// Capture a screenshot
		case VK_F9:
			m_context->SaveScreenshot(_T("Screenshot"));
			break;
#ifdef _DEBUG
		// Store the current camera position and orientation to clipboard so it can easily be pasted into source code.
		case VK_F12:
			if (OpenClipboard(hwnd))
			{
				EmptyClipboard();

				float3 camPos = m_camera.GetPosition();
				float wx = m_camera.GetRotationX();
				float wy = m_camera.GetRotationY();
				float wz = m_camera.GetRotationZ();

				char str[256];
				int len = sprintf(str, "m_camera.SetPosition(float3(%.15ff, %.15ff, %.15ff));\r\nm_camera.SetRotation(%.15ff, %.15ff, %.15ff);\r\n", camPos.x, camPos.y, camPos.z, wx, wy, wz);

				HGLOBAL handle = GlobalAlloc(GMEM_MOVEABLE | GMEM_DDESHARE, len + 1);
				char *mem = (char *) GlobalLock(handle);
				if (mem != NULL)
				{
					strcpy(mem, str);
					GlobalUnlock(handle);
					SetClipboardData(CF_TEXT, handle);
				}
				CloseClipboard();
			}
			break;
#endif
		default:
			return false;
		};

		return true;
	}

	return false;
}

void D3D10App::OnSize(HWND hwnd, const int w, const int h)
{
	m_context->Resize(w, h);

	m_camera.SetViewport(w, h);
}

void D3D10App::OnPosition(HWND hwnd, const int x, const int y)
{
	m_context->SetPosition(x, y);
}

void D3D10App::RenderGUI()
{
	if (m_showFPS)
	{
		static float accTime = 0.1f;
		static char str[16];
		static int nFrames = 0;

		// Update the fps counter 10 times / sec to get reasonably stable values.
		if (accTime >= 0.1f)
		{
			float fps = nFrames / accTime;
			if (m_keys[VK_DECIMAL])
			{
				sprintf(str, "%.2f", fps);
			}
			else if (fps < 9.95f) // For low fps we use one decimal
			{
				sprintf(str, "%.1f", fps);
			}
			else
			{
				sprintf(str, "%d", (int) (fps + 0.5f));
			}
			nFrames = 0;
			accTime = 0;
		}
		accTime += m_frameTime;
		nFrames++;

		// Render in upper left corner of the window
		int w = m_context->GetWidth();
		int h = m_context->GetHeight();
		m_mainFont.DrawText(m_context->GetDeviceContext(), str, 10.0f / w - 1, 1 - 10.0f / h, 80.0f / w, 60.0f / h, HA_LEFT, VA_TOP);
	}
}

void D3D10App::RenderBillboards(const float3 *position, const int count, const float size, const float4 &color)
{
	// Make sure we have enough room in the tool vertex buffer
	SetToolsVBSize(count * 6 * sizeof(float3));

	float3 dx, dy;
	m_camera.GetBaseVectors(&dx, &dy, NULL);

	// Fill vertex buffer
	float3 *dest;
	ID3D11DeviceContext* context = m_context->GetDeviceContext();

	D3D11_MAPPED_SUBRESOURCE resource;

	context->Map( m_toolsVB, 0, D3D11_MAP_WRITE_DISCARD, 0, &resource);

	dest = reinterpret_cast<float3*> ( resource.pData );
	for (int i = 0; i < count; i++)
	{
		dest[6 * i + 0] = position[i] + size * (-dx + dy);
		dest[6 * i + 1] = position[i] + size * ( dx + dy);
		dest[6 * i + 2] = position[i] + size * (-dx - dy);
		dest[6 * i + 3] = position[i] + size * (-dx - dy);
		dest[6 * i + 4] = position[i] + size * ( dx + dy);
		dest[6 * i + 5] = position[i] + size * ( dx - dy);
	}
	context->Unmap(m_toolsVB, 0 );


	ID3D11DeviceContext *dev = m_context->GetDeviceContext();

	// Set constants
	float4x4 *mvp;

	dev->Map(m_toolsVsCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &resource);
	mvp = reinterpret_cast<float4x4*> ( resource.pData );
		*mvp = m_camera.GetModelViewProjection();
	dev->Unmap(m_toolsVsCB, 0);

	float4 *col;

	dev->Map( m_toolsPsCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &resource);
		col = reinterpret_cast<float4*> ( resource.pData );
		*col = color;
	dev->Unmap( m_toolsPsCB, 0 );

	// Setup effect
	m_context->SetEffect(m_toolsEffect);
	m_context->Apply(1, 0);

	dev->IASetInputLayout(m_pos3Layout);

	UINT stride = sizeof(float3);
	UINT offset = 0;
	dev->IASetVertexBuffers(0, 1, &m_toolsVB, &stride, &offset);

	// Render the quads
	dev->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	dev->Draw(6 * count, 0);
}

void D3D10App::RenderCameraPath()
{
	if (!m_displayPath) return;

	uint vertexCount = m_camera.GetPathNodeCount();
	if (vertexCount)
	{
		const int smoothCount = 12;

		// Compute vertex count needed
		if (m_displayPathLooping) vertexCount++;
		if (m_displaySmooth) vertexCount *= smoothCount;
		uint size = vertexCount * sizeof(float3);

		// Make sure we have enough from in the vertex buffer
		SetToolsVBSize(size);

		// Fill vertex buffer
		float3 *dest;
		ID3D11DeviceContext* context = m_context->GetDeviceContext();

		D3D11_MAPPED_SUBRESOURCE resource;
		context->Map(m_toolsVB, 0, D3D11_MAP_WRITE_DISCARD, 0, &resource);
		dest = reinterpret_cast<float3*> ( resource.pData ) ;
		if (m_displaySmooth)
		{
			float d = 1.0f / smoothCount;
			for (uint i = 0; i < vertexCount; i++)
			{
				m_camera.GetNodeAt(i * d, &dest[i], NULL, m_displayPathLooping);
			}
		}
		else
		{
			uint count = vertexCount;
			if (m_displayPathLooping) count--;
			for (uint i = 0; i < count; i++)
			{
				dest[i] = m_camera.GetPathNodePosition(i);
			}
			if (m_displayPathLooping) dest[count] = m_camera.GetPathNodePosition(0);
		}
		context->Unmap(m_toolsVB, 0);


		ID3D11DeviceContext *dev = m_context->GetDeviceContext();

		// Set constants
		float4x4 *mvp;

		dev->Map( m_toolsVsCB, 0, D3D11_MAP_WRITE_DISCARD, 0,  &resource);
			mvp = reinterpret_cast<float4x4*> ( resource.pData ) ;
			*mvp = m_camera.GetModelViewProjection();
			dev->Unmap(m_toolsVsCB, 0 );

		// Setup effect
		m_context->SetEffect(m_toolsEffect);
		m_context->Apply(0, 0);

		dev->IASetInputLayout(m_pos3Layout);

		UINT stride = sizeof(float3);
		UINT offset = 0;
		dev->IASetVertexBuffers(0, 1, &m_toolsVB, &stride, &offset);

		// Render the camera path
		dev->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP);
		dev->Draw(vertexCount, 0);
	}
}

void D3D10App::DebugViewTexture2D(ID3D11ShaderResourceView *srv, const float x, const float y, const float width, const float height, const int slice)
{
	// Make sure we have enough space in the vertex buffer
	SetToolsVBSize(4 * sizeof(Pos2Tex3));

	// Fill vertex buffer
	Pos2Tex3 *dest;
	ID3D11DeviceContext* context = m_context->GetDeviceContext();
	D3D11_MAPPED_SUBRESOURCE resource;
	context->Map(m_toolsVB, 0, D3D11_MAP_WRITE_DISCARD, 0, &resource);
	dest = reinterpret_cast<Pos2Tex3*> ( resource.pData );
		dest[0].pos = float2(x, y + height);
		dest[0].tex = float3(0, 0, (float) slice);
		dest[1].pos = float2(x + width, y + height);
		dest[1].tex = float3(1, 0, (float) slice);
		dest[2].pos = float2(x, y);
		dest[2].tex = float3(0, 1, (float) slice);
		dest[3].pos = float2(x + width, y);
		dest[3].tex = float3(1, 1, (float) slice);
	context->Unmap(m_toolsVB, 0);


	ID3D11DeviceContext *dev = m_context->GetDeviceContext();

	// Setup the effect
	m_context->SetEffect(m_toolsEffect);
	if (slice < 0)
	{
		m_context->SetTexture("tex2d", srv);
		m_context->Apply(2, 0);
	}
	else
	{
		m_context->SetTexture("texArray", srv);
		m_context->Apply(2, 1);
	}

	dev->IASetInputLayout(m_pos2Tex3Layout);

	UINT stride = sizeof(Pos2Tex3);
	UINT offset = 0;
	dev->IASetVertexBuffers(0, 1, &m_toolsVB, &stride, &offset);

	// Render a textured quad
	dev->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	dev->Draw(4, 0);
}

void D3D10App::DebugViewTexture3D(ID3D11ShaderResourceView *srv, const float x, const float y, const float width, const float height, const float z)
{
	// Make sure we have enough space in the vertex buffer
	SetToolsVBSize(4 * sizeof(Pos2Tex3));

	ID3D11DeviceContext* context = m_context->GetDeviceContext();

	// Fill vertex buffer
	Pos2Tex3 *dest;
	D3D11_MAPPED_SUBRESOURCE resource;	
	context->Map( m_toolsVB, 0, D3D11_MAP_WRITE_DISCARD, 0, &resource);
		dest = reinterpret_cast< Pos2Tex3* > ( resource.pData );
		dest[0].pos = float2(x, y + height);
		dest[0].tex = float3(0, 0, z);
		dest[1].pos = float2(x + width, y + height);
		dest[1].tex = float3(1, 0, z);
		dest[2].pos = float2(x, y);
		dest[2].tex = float3(0, 1, z);
		dest[3].pos = float2(x + width, y);
		dest[3].tex = float3(1, 1, z);
	context->Unmap( m_toolsVB, 0);


	ID3D11DeviceContext *dev = m_context->GetDeviceContext();

	// Setup the effect
	m_context->SetEffect(m_toolsEffect);
	m_context->SetTexture("tex3d", srv);
	m_context->Apply(2, 2);

	dev->IASetInputLayout(m_pos2Tex3Layout);

	UINT stride = sizeof(Pos2Tex3);
	UINT offset = 0;
	dev->IASetVertexBuffers(0, 1, &m_toolsVB, &stride, &offset);

	// Render a texture quad
	dev->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	dev->Draw(4, 0);
}

bool D3D10App::SetToolsVBSize(const uint size)
{
	// Check if we need to resize the buffer
	if (size > m_toolsVBSize)
	{
		// Round size up to closest 4KB
		m_toolsVBSize = (size + 0xFFF) & ~0xFFF;

		if (m_toolsVB)
		{
			UINT so = 0;
			ID3D11Buffer *null = NULL;
			// Set first VB to NULL in case this buffer is bound already ...
			m_context->GetDeviceContext()->IASetVertexBuffers(0, 1, &null, &so, &so);
			// ... then delete it
			m_toolsVB->Release();
		}
		if ((m_toolsVB = m_context->CreateVertexBuffer(m_toolsVBSize, D3D11_USAGE_DYNAMIC, NULL)) == NULL)
		{
			m_toolsVBSize = 0;
			return false;
		}
	}

	return true;
}

int D3D10App::Run()
{
	MSG msg;
	msg.wParam = 0;

	if (Create())
	{
		if (Load())
		{
			// Initialize high-res timer
			LARGE_INTEGER freq, lCnt, cnt;
			QueryPerformanceFrequency(&freq);
			QueryPerformanceCounter(&cnt);

			do
			{
				while (true)
				{
					//if (activeWindow && !minimized){
						if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE) <= 0) break;
					/*} else {
						if (GetMessage(&msg, NULL, 0, 0) <= 0) break;
					}*/

					TranslateMessage(&msg);
					DispatchMessage(&msg);
				}
				if (msg.message == WM_QUIT) break;

				// Keep track of time
				lCnt = cnt;
				QueryPerformanceCounter(&cnt);
				m_frameTime = float(cnt.QuadPart - lCnt.QuadPart) / freq.QuadPart;

				// On some systems this appears to be the only "solution" to random glitches in the timer
				m_frameTime = clamp(m_frameTime, 0.0001f, 1.0f);

				m_time += m_frameTime;

				OnRender();

			} while (true);

			// Clear state first to get rid of "currently bound" warnings
			m_context->GetDeviceContext()->ClearState();

			Unload();
		}
	}

	Destroy();

	return (int) msg.wParam;
}

#define GETX(lParam) ((int) (short) LOWORD(lParam))
#define GETY(lParam) ((int) (short) HIWORD(lParam))

LRESULT D3D10App::ProcessMessage(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    PAINTSTRUCT ps;
    HDC hdc;

	switch (message)
	{
        case WM_PAINT:
            hdc = BeginPaint(hwnd, &ps);
            EndPaint(hwnd, &ps);
            break;
		case WM_MOUSEMOVE:
			OnMouseMove(hwnd, GETX(lParam), GETY(lParam), (wParam & MK_LBUTTON) != 0, (wParam & MK_MBUTTON) != 0, (wParam & MK_RBUTTON) != 0);
			break;
		case WM_KEYDOWN:
			if (wParam == VK_ESCAPE)
			{
				if (m_mouseCapture)
				{
					// Release mouse capture on escape if there is one ...
					CaptureMouse(false);
				}
				else
				{
					// ... otherwise exit the application
					PostMessage(hwnd, WM_CLOSE, 0, 0);
				}
			}
			else
			{
				OnKeyPress(hwnd, (unsigned int) wParam, true);
			}
			break;
		case WM_KEYUP:
			OnKeyPress(hwnd, (unsigned int) wParam, false);
			break;
		case WM_SYSKEYDOWN:
			// Toggle fullscreen on Alt-Enter
			if ((lParam & (1 << 29)) && wParam == VK_RETURN)
			{
				m_context->ToggleFullscreen();
			}
			break;
		case WM_LBUTTONDOWN:
			OnMouseClick(hwnd, GETX(lParam), GETY(lParam), MOUSE_LEFT, true);
			break;
		case WM_LBUTTONUP:
			OnMouseClick(hwnd, GETX(lParam), GETY(lParam), MOUSE_LEFT, false);
			break;
		case WM_RBUTTONDOWN:
			OnMouseClick(hwnd, GETX(lParam), GETY(lParam), MOUSE_RIGHT, true);
			break;
		case WM_RBUTTONUP:
			OnMouseClick(hwnd, GETX(lParam), GETY(lParam), MOUSE_RIGHT, false);
			break;
		case WM_MBUTTONDOWN:
			OnMouseClick(hwnd, GETX(lParam), GETY(lParam), MOUSE_MIDDLE, true);
			break;
		case WM_MBUTTONUP:
			OnMouseClick(hwnd, GETX(lParam), GETY(lParam), MOUSE_MIDDLE, false);
			break;
		case WM_WINDOWPOSCHANGED:
			WINDOWPOS *p;
			p = (WINDOWPOS *) lParam;

			// Ignore events with SWP_NOSENDCHANGING flag
			if (p->flags & SWP_NOSENDCHANGING) break;

			if ((p->flags & SWP_NOMOVE) == 0)
			{
				OnPosition(hwnd, p->x, p->y);
			}
			if ((p->flags & SWP_NOSIZE) == 0)
			{
				RECT rect;
				GetClientRect(hwnd, &rect);
				OnSize(hwnd, rect.right - rect.left, rect.bottom - rect.top);
			}
			break;
		case WM_CREATE:
			ShowWindow(hwnd, SW_SHOW);
			break;
		case WM_CLOSE:
			DestroyWindow(hwnd);
			break;
		case WM_DESTROY:
			PostQuitMessage(0);
			break;
		default:
			return DefWindowProc(hwnd, message, wParam, lParam);
	}
	return 0;
}
