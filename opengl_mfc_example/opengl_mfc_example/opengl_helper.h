#ifndef __opengl_helper_h__
#define __opengl_helper_h__

#include <exception>
#include <string>

#include <windows.h>
#include <GL/glew.h>
#include <GL/wglew.h>

#include "opengl_exception.h"

namespace ogl
{
	inline PIXELFORMATDESCRIPTOR create_pixel_format_descriptor()
	{
		PIXELFORMATDESCRIPTOR pfd = {};

		pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
		pfd.nVersion = 1;
		pfd.dwFlags = PFD_DOUBLEBUFFER | PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW;
		pfd.iPixelType = PFD_TYPE_RGBA;
		pfd.cColorBits = 32;
		pfd.cDepthBits = 32;
		pfd.iLayerType = PFD_MAIN_PLANE;

		return pfd;
	}

	inline PIXELFORMATDESCRIPTOR create_pixel_format_descriptor2()
	{
		PIXELFORMATDESCRIPTOR pfd = {};

		pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
		pfd.nVersion = 1;
		pfd.dwFlags = PFD_DOUBLEBUFFER | PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW;
		pfd.iPixelType = PFD_TYPE_RGBA;
		pfd.cColorBits = 24;
		pfd.cDepthBits = 32;
		pfd.iLayerType = PFD_MAIN_PLANE;

		return pfd;
	}


	//every window has own context, this way we can do multithreaded rendering
	inline HGLRC create_context(HDC hdc, HGLRC shareContext)
	{
		auto pfd = create_pixel_format_descriptor2();
		auto pixelFormat = ChoosePixelFormat(hdc, &pfd);

		ogl::throw_if_failed<ogl::windows_exception>( pixelFormat != 0 );
		ogl::throw_if_failed<ogl::windows_exception>(SetPixelFormat(hdc, pixelFormat, &pfd));

		//Get a GL 4,4 context
		int attribs[] =
		{
			WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
			WGL_CONTEXT_MINOR_VERSION_ARB, 3,
			WGL_CONTEXT_FLAGS_ARB, 0,
			0
		};

		auto hrc = wglCreateContextAttribsARB(hdc, shareContext, attribs);
		return hrc;
	}

	//create one initial context for the whole application
	inline HGLRC create_initial_context( HDC hdc )
	{
		auto pfd = create_pixel_format_descriptor();
		auto pixelFormat = ChoosePixelFormat(hdc, &pfd);

		ogl::throw_if_failed<ogl::windows_exception>(pixelFormat != 0);
		ogl::throw_if_failed<ogl::windows_exception>(SetPixelFormat(hdc, pixelFormat, &pfd));

		auto tempContext = wglCreateContext(hdc);
		auto previousDC	= wglMakeCurrent(hdc, tempContext);

		ogl::throw_if_failed<ogl::windows_exception>(glewInit() == GLEW_OK);

		//Get a GL 4,4 context
		int attribs[] =
		{
			WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
			WGL_CONTEXT_MINOR_VERSION_ARB, 2,
			WGL_CONTEXT_FLAGS_ARB, 0,
			0
		};

		ogl::throw_if_failed<ogl::windows_exception>(wglewIsSupported("WGL_ARB_create_context") == 1);

		auto hrc = wglCreateContextAttribsARB(hdc, 0, attribs);
		ogl::throw_if_failed<ogl::windows_exception>(wglMakeCurrent(NULL, NULL));
		ogl::throw_if_failed<ogl::windows_exception>(wglDeleteContext(tempContext));
		ogl::throw_if_failed<ogl::windows_exception>(wglMakeCurrent(hdc, hrc));

		return hrc;
	}

    inline void check_result()
    {
        auto result = glGetError();
        if (result != GL_NO_ERROR)
        {
            throw ogl::opengl_exception(result);
        }
    }

    #define OGL_CALL(call) \
    { \
    	(call); \
    	ogl::check_result(); \
    }

	class scoped_draw_context
	{
		public:
		scoped_draw_context(HGLRC context, HDC hdc)
		{
            OGL_CALL(wglMakeCurrent(hdc, context));
		}

		~scoped_draw_context()
		{
            wglMakeCurrent(NULL, NULL);
		}

		private:

		scoped_draw_context(const scoped_draw_context&);
		scoped_draw_context& operator=(const scoped_draw_context&);

	};
}

#endif