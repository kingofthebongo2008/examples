#ifndef __OS_WINDOWS_OS_H__
#define __OS_WINDOWS_OS_H__

/* Include this file instead of including <windows.h> directly. */
#ifdef NOMINMAX
    #include <windows.h>
#else
    #define NOMINMAX
    #include <windows.h>
    #undef NOMINMAX
#endif

#endif