
// opengl_mfc_exampleDoc.cpp : implementation of the Copengl_mfc_exampleDoc class
//

#include "stdafx.h"
// SHARED_HANDLERS can be defined in an ATL project implementing preview, thumbnail
// and search filter handlers and allows sharing of document code with that project.
#ifndef SHARED_HANDLERS
#include "opengl_mfc_example.h"
#endif

#include "opengl_mfc_exampleDoc.h"
#include "opengl_mfc_example_gl_repository_helper.h"
#include "opengl_mfc_example_gl_repository_helper.h"

#include <propkey.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// Copengl_mfc_exampleDoc

IMPLEMENT_DYNCREATE(Copengl_mfc_exampleDoc, CDocument)

BEGIN_MESSAGE_MAP(Copengl_mfc_exampleDoc, CDocument)
END_MESSAGE_MAP()


// Copengl_mfc_exampleDoc construction/destruction

Copengl_mfc_exampleDoc::Copengl_mfc_exampleDoc()
{
	// TODO: add one-time construction code here

}

Copengl_mfc_exampleDoc::~Copengl_mfc_exampleDoc()
{

	

}

BOOL Copengl_mfc_exampleDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	//make context, so we can create the vbo;
	ogl::scoped_draw_context2 context;


	

	// TODO: add reinitialization code here
	// (SDI documents will reuse this document)

	return TRUE;
}

void Copengl_mfc_exampleDoc::OnCloseDocument()
{


	CDocument::OnCloseDocument();
}

// Copengl_mfc_exampleDoc serialization

void Copengl_mfc_exampleDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: add storing code here
	}
	else
	{
		// TODO: add loading code here
	}
}

#ifdef SHARED_HANDLERS

// Support for thumbnails
void Copengl_mfc_exampleDoc::OnDrawThumbnail(CDC& dc, LPRECT lprcBounds)
{
	// Modify this code to draw the document's data
	dc.FillSolidRect(lprcBounds, RGB(255, 255, 255));

	CString strText = _T("TODO: implement thumbnail drawing here");
	LOGFONT lf;

	CFont* pDefaultGUIFont = CFont::FromHandle((HFONT) GetStockObject(DEFAULT_GUI_FONT));
	pDefaultGUIFont->GetLogFont(&lf);
	lf.lfHeight = 36;

	CFont fontDraw;
	fontDraw.CreateFontIndirect(&lf);

	CFont* pOldFont = dc.SelectObject(&fontDraw);
	dc.DrawText(strText, lprcBounds, DT_CENTER | DT_WORDBREAK);
	dc.SelectObject(pOldFont);
}

// Support for Search Handlers
void Copengl_mfc_exampleDoc::InitializeSearchContent()
{
	CString strSearchContent;
	// Set search contents from document's data. 
	// The content parts should be separated by ";"

	// For example:  strSearchContent = _T("point;rectangle;circle;ole object;");
	SetSearchContent(strSearchContent);
}

void Copengl_mfc_exampleDoc::SetSearchContent(const CString& value)
{
	if (value.IsEmpty())
	{
		RemoveChunk(PKEY_Search_Contents.fmtid, PKEY_Search_Contents.pid);
	}
	else
	{
		CMFCFilterChunkValueImpl *pChunk = NULL;
		ATLTRY(pChunk = new CMFCFilterChunkValueImpl);
		if (pChunk != NULL)
		{
			pChunk->SetTextValue(PKEY_Search_Contents, value, CHUNK_TEXT);
			SetChunkValue(pChunk);
		}
	}
}

#endif // SHARED_HANDLERS

// Copengl_mfc_exampleDoc diagnostics

#ifdef _DEBUG
void Copengl_mfc_exampleDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void Copengl_mfc_exampleDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// Copengl_mfc_exampleDoc commands
