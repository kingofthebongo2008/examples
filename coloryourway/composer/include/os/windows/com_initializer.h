#ifndef __OS_WINDOWS_COM_INITIALIZER_H__
#define __OS_WINDOWS_COM_INITIALIZER_H__

#include <os/windows/com_error.h>

#include <util/util_noncopyable.h>

namespace os
{
    namespace windows
    {
        class com_initializer : private util::noncopyable
        {
            public:
            com_initializer()
            {
                HRESULT hr  = ::CoInitializeEx(nullptr, COINIT_MULTITHREADED);

                bool success = (hr == S_OK || hr == S_FALSE);

                if ( !success )
                {
                    throw com_exception(hr);
                }
            }


            ~com_initializer()
            {
                ::CoUninitialize();
            }

        };
    }
}



#endif
