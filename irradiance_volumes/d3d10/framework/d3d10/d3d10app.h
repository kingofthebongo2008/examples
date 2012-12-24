//-----------------------------------------------------------------------------
// File: Framework\D3D10\D3D10App.h
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------





#ifndef _D3D10APP_H_
#define _D3D10APP_H_

#include "D3D10Context.h"
#include "../GUI.h"
#include "../Math/Camera.h"

#include <d3d11.h>

/** \file
    D3D10 base application class.
*/

enum MouseButton
{
	MOUSE_LEFT,
	MOUSE_MIDDLE,
	MOUSE_RIGHT,
};


/** Base application class */
class D3D10App
{
public:
	D3D10App();
	virtual ~D3D10App();

    /** This pure virutal function must be implemented by each sample application.
        \return The relative path home directory of the sample application as a char pointer
    */
	virtual const TCHAR *GetHomeDirectory() = 0;

	// Initializes and destroys the window and D3D10 context
	virtual bool Create();
	virtual void Destroy();

	// Loads and unloads resources
	virtual bool Load(){ return true; }
	virtual void Unload(){}

	// Reloads resources
	virtual bool Reload(){
		Unload();
		return Load();
	}

    /** This pure virutal function must be implemented by each sample application.
        It will be called once per frame for rendering. The function must call Present() itself.
    */
	virtual void OnRender() = 0;

	// Camera handling functions
	void CaptureMouse(const bool value);
	void UpdateCameraPosition(const float speed);
	virtual void ResetCamera();

	// Event handlers. May optionally be overridden.
	virtual bool OnMouseClick(HWND hwnd, const int x, const int y, const MouseButton button, const bool pressed);
	virtual bool OnMouseMove(HWND hwnd, const int x, const int y, const bool lButton, const bool mButton, const bool rButton);
	virtual bool OnMouseWheel(HWND hwnd, const int x, const int y, const int scroll);
	virtual bool OnKeyPress(HWND hwnd, const unsigned int key, const bool pressed);
	virtual void OnSize(HWND hwnd, const int w, const int h);
	virtual void OnPosition(HWND hwnd, const int x, const int y);

	// Main application loop
	int Run();
	LRESULT ProcessMessage(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);
protected:
	void RenderGUI();

	// Various rendering tools
	void RenderBillboards(const float3 *position, const int count, const float size, const float4 &color);
	void RenderCameraPath();

	// Various debugging tools
	void DebugViewTexture2D(ID3D11ShaderResourceView *srv, const float x, const float y, const float width, const float height, const int slice = -1);
	void DebugViewTexture3D(ID3D11ShaderResourceView *srv, const float x, const float y, const float width, const float height, const float z);

	bool SetToolsVBSize(const uint size);

	// Main rendering context
	D3D10Context *m_context;
	// Main font
	TexFont m_mainFont;
	// Main camera
	Camera m_camera;

	// Total running time and current frame duration in seconds
	float m_time, m_frameTime;

	// Array that keeps track of key states
	bool m_keys[256];
	// Whether the mouse is currently captured
	bool m_mouseCapture;
	// Whether to show the framerate
	bool m_showFPS;

private:
	// Internal variables for the rendering and debugging tool functions
	ID3DX11Effect *m_toolsEffect;
	ID3D11Buffer *m_toolsVB;
	ID3D11Buffer *m_toolsVsCB, *m_toolsPsCB;
	uint m_toolsVBSize;
	ID3D11InputLayout *m_pos3Layout;
	ID3D11InputLayout *m_pos2Tex3Layout;

	// For displaying and debuggint camera paths
	bool m_displayPath;
	bool m_displayPathLooping;
	bool m_displaySmooth;
	bool m_benchMark;
	float m_benchMarkTime;
};

#endif
