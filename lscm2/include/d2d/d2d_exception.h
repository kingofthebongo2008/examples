#ifndef __D2D_EXCEPTION_H__
#define __D2D_EXCEPTION_H__

#include <os/windows/com_error.h>

namespace d2d
{
    class exception : public os::windows::com_exception
    {
        public:
        exception ( const HRESULT hr ) : com_exception(hr)
        {

        }

    };
}


#endif