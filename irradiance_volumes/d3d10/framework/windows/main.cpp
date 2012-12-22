//-----------------------------------------------------------------------------
// File: Framework\Windows\Main.cpp
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------





#include "../D3D10/D3D10App.h"
#include "Resource.h"

#ifdef _DEBUG
#include <crtdbg.h>
#endif

extern D3D10App *app;

LRESULT CALLBACK WinProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	// Pass on Windows messages to the sample
	return app->ProcessMessage(hwnd, message, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hThisInstance, HINSTANCE hLastInstance, LPSTR lpszCmdLine, int nCmdShow)
{
#ifdef _DEBUG
	int flag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
	flag |= _CRTDBG_LEAK_CHECK_DF;   // Turn on leak checks
	flag |= _CRTDBG_CHECK_ALWAYS_DF; // Turn on memory checks
	_CrtSetDbgFlag(flag);
#endif

	/*
		Some dual core systems have a problem where the different CPUs return different
		QueryPerformanceCounter values. So when this thread is schedule on the other CPU
		in a later frame, we could even get a negative frameTime. To solve this we force
		the main thread to always run on CPU 0.
	*/
	SetThreadAffinityMask(GetCurrentThread(), 1);

	// Make sure we're starting in the exe's directory ...
	TCHAR path[MAX_PATH];
	if (GetModuleFileName(NULL, path, sizeof(path)))
	{
		TCHAR *slash = wcsrchr(path, '\\');
		if (slash) *slash = '\0';
		SetCurrentDirectory(path);
	}
	// ... then move to the relative path of the home directory for the app
	SetCurrentDirectory(app->GetHomeDirectory());

	WNDCLASS wincl;
	wincl.hInstance = hThisInstance;
	wincl.lpszClassName = _T("D3D10App");
	wincl.lpfnWndProc = WinProc;
	wincl.style = 0;
	wincl.hIcon = LoadIcon(hThisInstance, MAKEINTRESOURCE(IDI_MAINICON));
	wincl.hCursor = LoadCursor(NULL, IDI_APPLICATION);
	wincl.lpszMenuName = NULL;
	wincl.cbClsExtra = 0;
	wincl.cbWndExtra = 0;
	wincl.hbrBackground = NULL;

	if (!RegisterClass(&wincl)) return 0;

	// Enter main application loop
	int result = app->Run();

	delete app;

	return result;
}
