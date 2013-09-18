#ifndef __OS_WINDOWS_ERROR_H__
#define __OS_WINDOWS_ERROR_H__

#include <exception>
#include <ole2.h>

namespace os
{
    namespace windows
    {
        class com_exception : public std::exception
        {
            public:
            com_exception ( const HRESULT hr ) : exception("com exception")
                , m_hr (hr)  
            {

            }

            private:
            const HRESULT m_hr;
            com_exception& operator=(const com_exception&);
        };


        template < typename exception > void throw_if_failed( HRESULT hr)
        {
            if (hr != S_OK)
            {
                throw exception(hr);
            }
        }
    }
}


#endif