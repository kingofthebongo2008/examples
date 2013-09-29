#ifndef __D2D_POINTERS_H__
#define __D2D_POINTERS_H__

#include <d2d1.h>
#include <d2d1_1.h>

#include <os/windows/com_pointers.h>

namespace d2d
{
    typedef os::windows::com_ptr<ID2D1Bitmap>                 ibitmap_ptr;
    typedef os::windows::com_ptr<ID2D1Brush>                  ibrush_ptr;
    typedef os::windows::com_ptr<ID2D1Factory>                ifactory_ptr;
    typedef os::windows::com_ptr<ID2D1HwndRenderTarget>       ihwndrendertarget_ptr;
    typedef os::windows::com_ptr<ID2D1RenderTarget>           irendertarget_ptr;
    typedef os::windows::com_ptr<ID2D1SolidColorBrush>        isolid_color_brush_ptr;

    typedef os::windows::com_ptr<ID2D1DeviceContext>          device_context_ptr;
    typedef os::windows::com_ptr<ID2D1Factory1>               ifactory1_ptr;

    
}

#endif

