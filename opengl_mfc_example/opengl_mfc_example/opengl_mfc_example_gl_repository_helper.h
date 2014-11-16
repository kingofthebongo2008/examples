
// opengl_mfc_exampleView.h : interface of the Copengl_mfc_exampleView class
//

#pragma once

#include <afxwin.h>
#include "MainFrm.h"
#include "opengl_mfc_example_gl_repository.h"

namespace ogl
{
	//make a code repository with readonly shaders, initialized only during program startup and accessed at render time
	inline code_repository GetCode()
	{
		auto app = reinterpret_cast<CMainFrame*> (AfxGetApp()->GetMainWnd());
		return code_repository(app->GetProgram());
	}

	//context used when we createnew shaders, vbo, etc
	class scoped_draw_context2
	{
	public:
		scoped_draw_context2()
		{
			auto app = reinterpret_cast<CMainFrame*> (AfxGetApp()->GetMainWnd());
			app->MakeGlContextCurrent();
		}

		~scoped_draw_context2()
		{
			auto app = reinterpret_cast<CMainFrame*> (AfxGetApp()->GetMainWnd());
			app->ResetGlContext();
		}

	private:
		scoped_draw_context2(const scoped_draw_context2&);
		scoped_draw_context2& operator=(const scoped_draw_context2&);

	};

	//make a code repository with readonly shaders, initialized only during program startup and accessed at render time
	inline HGLRC GetShareContext()
	{
		auto app = reinterpret_cast<CMainFrame*> (AfxGetApp()->GetMainWnd());
		return app->GetShareContext();
	}
}




