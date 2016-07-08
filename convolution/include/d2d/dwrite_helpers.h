#ifndef __dwrite_helpers_h__
#define __dwrite_helpers_h__

#include <cstdint>

#include <os/windows/com_error.h>
#include <os/windows/com_pointers.h>
#include <d2d/dwrite_pointers.h>

namespace dwrite
{
    inline ifactory_ptr create_dwrite_factory()
    {
        using namespace os::windows;

        ifactory_ptr result;
        throw_if_failed< com_exception> ( DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown**>(&result))) ;

        return result;
    }

    inline itextformat_ptr create_text_format(ifactory_ptr factory)
    {
        using namespace os::windows;

        itextformat_ptr result;

        throw_if_failed< com_exception > ( factory->CreateTextFormat(
        L"Courier",
        NULL,
        DWRITE_FONT_WEIGHT_REGULAR,
        DWRITE_FONT_STYLE_NORMAL,
        DWRITE_FONT_STRETCH_NORMAL,
        14.0f,
        L"en-us",
        &result
        ) );

        return result;
    }
}


#endif

