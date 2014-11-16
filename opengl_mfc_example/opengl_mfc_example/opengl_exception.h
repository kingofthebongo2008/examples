#ifndef __opengl_exception_H__
#define __opengl_exception_H__

#include <exception>
#include <string>

#include <windows.h>

namespace ogl
{
	class exception : public std::exception
	{
		public:

		exception( const std::string& m ) : m_message(m)
		{

		}

		const char * what() const override
		{
			return m_message.c_str();
		}

		std::string m_message;
	};


	class windows_exception : public std::exception
	{

	public:
		windows_exception( DWORD errorCode ) : m_windowsErrorCode(errorCode)
		{

		}

		const char * what() const override
		{
			return "windows_exception";
		}

		DWORD	m_windowsErrorCode;
	};

	class opengl_exception : public std::exception
	{

	public:
		opengl_exception(GLenum errorCode) : m_openglErrorCode(errorCode)
		{

		}

		const char * what() const override
		{
			return "opengl_exception";
		}

		GLenum	m_openglErrorCode;
	};

	template < typename exception > inline void throw_if_failed( bool result )
	{
		if (!result)
		{
			auto error = GetLastError();
			throw exception(error);
		}
	}

	template < typename exception > void throw_if_failed(BOOL result)
	{
		if (!result)
		{
			auto error = GetLastError();
			throw exception(error);
		}
	}

	template < typename exception > void throw_if_failed(void* pointer)
	{
		if (!pointer)
		{
			auto error = GetLastError();
			throw exception(error);
		}
	}

	template < typename exception > void throw_if_failed(GLenum result)
	{
		if (!result)
		{
			auto error = GetLastError();
			throw exception(error);
		}
	}

	template < > inline void throw_if_failed<opengl_exception>(bool result)
	{
		if (!result)
		{
			auto result = glGetError();
			throw opengl_exception(result);
		}
	}

	template <  typename exception, typename functor > inline void throw_if_failed( functor f )
	{
		auto result = glGetError();
		if ( result != GL_NO_ERROR )
		{
			__debugbreak();
			throw exception(result);
		}
	}
}

#endif