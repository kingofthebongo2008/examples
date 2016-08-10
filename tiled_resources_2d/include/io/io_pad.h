#pragma once

#include <cstdint>
#include <windows.h>

#include <math/math_functions.h>
#include <util/util_bits.h>

#if defined(_PC)
    #include <io/platforms/pc/native_io_pad.h>
#endif

namespace io
{
    struct pad : public native_pad
    {
        using base = native_pad;

        pad_state update( const pad_state& o )
        {
            return base::update(o);
        }
    };
};


