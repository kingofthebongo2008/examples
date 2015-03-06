#ifndef __ui_ui_h__
#define __ui_ui_h__

#if defined( UI_DLL_IMPORT )
    #define UI_DLL __declspec(dllimport)
#elif defined( UI_DLL_EXPORT )
    #define UI_DLL __declspec(dllexport)
#else
    #define UI_DLL_EXPORT
#endif


#endif
