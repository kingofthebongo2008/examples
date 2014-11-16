
// opengl_mfc_exampleView.h : interface of the Copengl_mfc_exampleView class
//

#pragma once


#include <gl/glew.h>

namespace ogl
{
	class code_repository
	{
	private:

		GLuint m_program;

	public:
		code_repository(GLuint program) : m_program(program)
		{

		}

		GLuint GetProgram() const
		{
			return m_program;
		}
	};
}



