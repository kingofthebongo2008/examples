
// opengl_mfc_example.h : main header file for the opengl_mfc_example application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols



// Copengl_mfc_exampleApp:
// See opengl_mfc_example.cpp for the implementation of this class
//

class Copengl_mfc_exampleApp : public CWinAppEx
{
public:
	Copengl_mfc_exampleApp();


// Overrides
public:
	virtual BOOL InitInstance();
	virtual int	 ExitInstance();

// Implementation
	UINT  m_nAppLook;
	BOOL  m_bHiColorIcons;

	virtual void PreLoadState();
	virtual void LoadCustomState();
	virtual void SaveCustomState();

	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()


};

extern Copengl_mfc_exampleApp theApp;
