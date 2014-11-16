
// MainFrm.h : interface of the CMainFrame class
//

#pragma once

#include <gl/glew.h>

#include "FileView.h"
#include "ClassView.h"
#include "OutputWnd.h"
#include "PropertiesWnd.h"
#include "opengl_helper.h"

class CMainFrame : public CMDIFrameWndEx
{
	DECLARE_DYNAMIC(CMainFrame)
public:
	CMainFrame();

// Attributes
public:

// Operations
public:

// Overrides
public:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
	virtual BOOL LoadFrame(UINT nIDResource, DWORD dwDefaultStyle = WS_OVERLAPPEDWINDOW | FWS_ADDTOTITLE, CWnd* pParentWnd = NULL, CCreateContext* pContext = NULL);

// Implementation
public:
	virtual ~CMainFrame();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:  // control bar embedded members
	CMFCMenuBar       m_wndMenuBar;
	CMFCToolBar       m_wndToolBar;
	CMFCStatusBar     m_wndStatusBar;
	CMFCToolBarImages m_UserImages;
	CFileView         m_wndFileView;
	CClassView        m_wndClassView;
	COutputWnd        m_wndOutput;
	CPropertiesWnd    m_wndProperties;

// Generated message map functions
protected:
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnWindowManager();
	afx_msg void OnViewCustomize();
	afx_msg LRESULT OnToolbarCreateNew(WPARAM wp, LPARAM lp);
	afx_msg void OnApplicationLook(UINT id);
	afx_msg void OnUpdateApplicationLook(CCmdUI* pCmdUI);
	afx_msg void OnSettingChange(UINT uFlags, LPCTSTR lpszSection);
	afx_msg void OnDestroy();
	DECLARE_MESSAGE_MAP()

	BOOL CreateDockingWindows();
	void SetDockingWindowIcons(BOOL bHiColorIcons);

private:

	HGLRC	m_glContext;
	GLuint  m_program;
	GLuint  m_fs;
	GLuint  m_vs;

public:

	GLuint  GetVertexShader() const
	{
		return m_vs;
	}

	GLuint  GetFragmentShader() const
	{
		return m_fs;
	}

	GLuint  GetProgram() const
	{
		return m_program;
	}

	void MakeGlContextCurrent()
	{
		ogl::throw_if_failed<ogl::windows_exception>(wglMakeCurrent(this->GetWindowDC()->m_hDC, m_glContext));
	}

	void ResetGlContext()
	{
		ogl::throw_if_failed<ogl::windows_exception>(wglMakeCurrent(NULL, NULL));
	}

	HGLRC GetShareContext() const
	{
		return m_glContext;
	}
};


