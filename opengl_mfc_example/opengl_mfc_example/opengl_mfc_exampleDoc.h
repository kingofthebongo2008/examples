
// opengl_mfc_exampleDoc.h : interface of the Copengl_mfc_exampleDoc class
//


#pragma once

#include <gl/glew.h>
#include <gl/wglew.h>

class Copengl_mfc_exampleDoc : public CDocument
{
protected: // create from serialization only
	Copengl_mfc_exampleDoc();
	DECLARE_DYNCREATE(Copengl_mfc_exampleDoc)

// Attributes
public:

// Operations
public:

// Overrides
public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);
	void	OnCloseDocument();
#ifdef SHARED_HANDLERS
	virtual void InitializeSearchContent();
	virtual void OnDrawThumbnail(CDC& dc, LPRECT lprcBounds);
#endif // SHARED_HANDLERS

// Implementation
public:
	virtual ~Copengl_mfc_exampleDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()

#ifdef SHARED_HANDLERS
	// Helper function that sets search content for a Search Handler
	void SetSearchContent(const CString& value);
#endif // SHARED_HANDLERS

private:



public:

};
