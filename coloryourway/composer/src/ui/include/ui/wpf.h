#ifndef __ui_wpf_h__
#define __ui_wpf_h__

#include <windows.h>

#include <ui/ui.h>

namespace coloryourway
{
    namespace composer
    {
        namespace ui
        {

            UI_DLL HWND wpf_create_source(HWND parent);
            UI_DLL void wpf_destroy_source(HWND source);

        }
    }
}



#endif
