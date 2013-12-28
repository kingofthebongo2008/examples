#ifndef __OS_WINDOWS_COM_POINTERS_H__
#define __OS_WINDOWS_COM_POINTERS_H__

#include <os/windows/com_ptr.h>

#include <windows.h>

inline void com_ptr_release( IUnknown* px )
{
	px->Release();
}

inline void com_ptr_add_ref( IUnknown* px )
{
	px->AddRef();
}

#endif

