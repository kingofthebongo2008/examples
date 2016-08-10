#pragma once

#include <os/windows/com_error.h>

namespace d2d
{
    class exception : public os::windows::com_exception
    {
    public:

        exception(const HRESULT hr) : os::windows::com_exception(hr)
        {

        }

    };

    inline void throw_if_failed(HRESULT r)
    {
        os::windows::throw_if_failed<exception>(r);
    }
}

