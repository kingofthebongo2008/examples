
// opengl_mfc_exampleView.h : interface of the Copengl_mfc_exampleView class
//

#pragma once


#include <gl/glew.h>


class Copengl_mfc_exampleView : public CView
{
protected: // create from serialization only
	Copengl_mfc_exampleView();
	DECLARE_DYNCREATE(Copengl_mfc_exampleView)

// Attributes
public:
	Copengl_mfc_exampleDoc* GetDocument() const;

// Operations
public:

// Overrides
public:
	virtual void OnDraw(CDC* pDC);  // overridden to draw this view
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);


// Implementation
public:
	virtual ~Copengl_mfc_exampleView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	afx_msg void OnFilePrintPreview();
	afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnContextMenu(CWnd* pWnd, CPoint point);
	afx_msg int  OnCreate(LPCREATESTRUCT lpCreateStruct );
	afx_msg void OnDestroy();
	afx_msg void OnClose();
	afx_msg BOOL OnEraseBkgnd(CDC* pDC);
	DECLARE_MESSAGE_MAP()


private:
	HGLRC	m_glContext;
	//this simulates our document data, graphic, textures, etc.
	GLuint	m_vao;
};

#ifndef _DEBUG  // debug version in opengl_mfc_exampleView.cpp
inline Copengl_mfc_exampleDoc* Copengl_mfc_exampleView::GetDocument() const
   { return reinterpret_cast<Copengl_mfc_exampleDoc*>(m_pDocument); }
#endif

