#pragma once

#include <cstdint>

#include <d2d/dwrite_pointers.h>
#include <d2d/dwrite_error.h>

namespace dwrite
{
    namespace helpers
    {
        inline factory create_dwrite_factory()
        {
            factory result;
            throw_if_failed(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown**>(&result)));
            return result;
        }

        inline textformat create_text_format(factory factory)
        {
            textformat result;

            throw_if_failed(factory->CreateTextFormat(
                L"Courier",
                NULL,
                DWRITE_FONT_WEIGHT_REGULAR,
                DWRITE_FONT_STYLE_NORMAL,
                DWRITE_FONT_STRETCH_NORMAL,
                14.0f,
                L"en-us",
                &result
            ));

            return result;
        }
    }
}

