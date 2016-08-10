#pragma once

#include <d2d1.h>
#include <d2d1_1.h>
#include <d2d1_2.h>
#include <d2d1_3.h>

#include <os/windows/com_pointers.h>

namespace d2d
{
    using  bitmap               = os::windows::com_ptr<ID2D1Bitmap>;
    using  brush                = os::windows::com_ptr<ID2D1Brush>;
    using  factory              = os::windows::com_ptr<ID2D1Factory>;
    using  hwndrendertarget     = os::windows::com_ptr<ID2D1HwndRenderTarget>;
    using  rendertarget         = os::windows::com_ptr<ID2D1RenderTarget>;
    using  solid_color_brush    = os::windows::com_ptr<ID2D1SolidColorBrush>;
    using  device_context       = os::windows::com_ptr<ID2D1DeviceContext>;
    using  factory1             = os::windows::com_ptr<ID2D1Factory1>;
}
