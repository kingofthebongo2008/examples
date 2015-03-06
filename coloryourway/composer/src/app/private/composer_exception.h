#ifndef __elliptic_exception_h__
#define __elliptic_exception_h__

#include <cstdint>
#include <iostream>
#include <exception>

#include <cuda_runtime.h>

namespace coloryourway
{
    class exception : public std::exception
    {
    public:

        exception()
        {

        }

        const char * what() const override
        {
            return "elliptic_exception";
        }

    private:
    };
}

#endif
