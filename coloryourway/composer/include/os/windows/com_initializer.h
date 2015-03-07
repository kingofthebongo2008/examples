#ifndef __OS_WINDOWS_COM_INITIALIZER_H__
#define __OS_WINDOWS_COM_INITIALIZER_H__

#include <os/windows/com_error.h>

#include <util/util_noncopyable.h>

namespace os
{
    namespace windows
    {
        enum com_model
        {
            apartment_threaded = COINIT_APARTMENTTHREADED,
            multi_threaded = COINIT_MULTITHREADED
        };

        class com_initializer : private util::noncopyable
        {
            public:
                com_initializer() : com_initializer( multi_threaded )
            {

            }

            com_initializer( com_model m )
            {
                HRESULT hr  = ::CoInitializeEx(nullptr, m);

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
