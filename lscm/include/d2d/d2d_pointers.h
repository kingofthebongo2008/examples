#ifndef __D2D_POINTERS_H__
#define __D2D_POINTERS_H__

#include <d2d1.h>

#include <os/windows/com_pointers.h>

namespace d2d
{
    typedef os::windows::com_ptr<ID2D1Factory>                ifactory_ptr;
    typedef os::windows::com_ptr<ID2D1RenderTarget>           irendertarget_ptr;
    typedef os::windows::com_ptr<ID2D1Brush>                  ibrush_ptr;
    typedef os::windows::com_ptr<ID2D1SolidColorBrush>        isolid_color_brush_ptr;
}

#endif

