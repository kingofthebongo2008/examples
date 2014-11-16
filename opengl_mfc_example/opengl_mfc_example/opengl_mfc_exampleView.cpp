
// opengl_mfc_exampleView.cpp : implementation of the Copengl_mfc_exampleView class
//

#include "stdafx.h"
// SHARED_HANDLERS can be defined in an ATL project implementing preview, thumbnail
// and search filter handlers and allows sharing of document code with that project.
#ifndef SHARED_HANDLERS
#include "opengl_mfc_example.h"
#endif

#include <gl/glew.h>
#include <gl/wglew.h>

#include "opengl_mfc_exampleDoc.h"
#include "opengl_mfc_exampleView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#include "opengl_exception.h"
#include "opengl_helper.h"
#include "opengl_mfc_example_gl_repository_helper.h"


// Copengl_mfc_exampleView

IMPLEMENT_DYNCREATE(Copengl_mfc_exampleView, CView)

BEGIN_MESSAGE_MAP(Copengl_mfc_exampleView, CView)
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &Copengl_mfc_exampleView::OnFilePrintPreview)
	ON_WM_CONTEXTMENU()
	ON_WM_RBUTTONUP()
	ON_WM_CREATE()
	ON_WM_DESTROY()
	ON_WM_CLOSE()
	ON_WM_ERASEBKGND()
END_MESSAGE_MAP()

// Copengl_mfc_exampleView construction/destruction

Copengl_mfc_exampleView::Copengl_mfc_exampleView()
{
	// TODO: add construction code here
}

Copengl_mfc_exampleView::~Copengl_mfc_exampleView()
{
}


void Test()
{



}

int  Copengl_mfc_exampleView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	auto dc = GetWindowDC()->m_hDC;
	
	m_glContext = ogl::create_context(dc, ogl::GetShareContext());
	ogl::throw_if_failed<ogl::windows_exception>(m_glContext != nullptr);

	ogl::scoped_draw_context  ogl(m_glContext, dc);
	OGL_CALL(glGenVertexArrays(1, &m_vao));

	return CView::OnCreate(lpCreateStruct);
}

void Copengl_mfc_exampleView::OnDestroy()
{
	{
		auto dc = GetWindowDC()->m_hDC;
		ogl::scoped_draw_context  ogl(m_glContext, dc);
		OGL_CALL(glDeleteVertexArrays(1, &m_vao));
	}
	ogl::throw_if_failed<ogl::windows_exception>(wglDeleteContext(m_glContext));
	CView::OnDestroy();
}

void Copengl_mfc_exampleView::OnClose()
{
	CView::OnClose();
}

BOOL Copengl_mfc_exampleView::OnEraseBkgnd(CDC* pDC)
{
	return true;
}

BOOL Copengl_mfc_exampleView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	// the CREATESTRUCT cs
	cs.lpszClass = ::AfxRegisterWndClass(CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS | CS_OWNDC, ::LoadCursor(NULL, IDC_ARROW), NULL, NULL);
	return CView::PreCreateWindow(cs);
}

// Copengl_mfc_exampleView drawing

void Copengl_mfc_exampleView::OnDraw(CDC* dc)
{
	Copengl_mfc_exampleDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);

	if (!pDoc)
		return;

	auto gdi = dc->m_hDC;
	ogl::scoped_draw_context  ogl(m_glContext, gdi);
	auto r = ogl::GetCode();

	const GLfloat green[] = { 0.0f, 0.25f, 0.0f, 1.0f };

	OGL_CALL(glClearBufferfv(GL_COLOR, 0, green));
	OGL_CALL(glClearDepth(1.0f));

	OGL_CALL(glViewport(0, 0, 1280, 720));

	OGL_CALL(glDisable(GL_DEPTH_TEST));	// Enables Depth Testing
	OGL_CALL(glDepthFunc(GL_LEQUAL));	// The Type Of Depth Testing To Do
	OGL_CALL(glBindVertexArray(m_vao));
 	OGL_CALL(glUseProgram(r.GetProgram()));
	OGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 3));
	OGL_CALL(SwapBuffers(gdi));

	// TODO: add draw code for native data here
}


void Copengl_mfc_exampleView::OnFilePrintPreview()
{
#ifndef SHARED_HANDLERS
	AFXPrintPreview(this);
#endif
}

BOOL Copengl_mfc_exampleView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}

void Copengl_mfc_exampleView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add extra initialization before printing
}

void Copengl_mfc_exampleView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add cleanup after printing
}

void Copengl_mfc_exampleView::OnRButtonUp(UINT /* nFlags */, CPoint point)
{
	ClientToScreen(&point);
	OnContextMenu(this, point);
}

void Copengl_mfc_exampleView::OnContextMenu(CWnd* /* pWnd */, CPoint point)
{
#ifndef SHARED_HANDLERS
	theApp.GetContextMenuManager()->ShowPopupMenu(IDR_POPUP_EDIT, point.x, point.y, this, TRUE);
#endif
}


// Copengl_mfc_exampleView diagnostics

#ifdef _DEBUG
void Copengl_mfc_exampleView::AssertValid() const
{
	CView::AssertValid();
}

void Copengl_mfc_exampleView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

Copengl_mfc_exampleDoc* Copengl_mfc_exampleView::GetDocument() const // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(Copengl_mfc_exampleDoc)));
	return (Copengl_mfc_exampleDoc*)m_pDocument;
}
#endif //_DEBUG


// Copengl_mfc_exampleView message handlers
