#pragma once

#if defined(_PC)
#include <os/platforms/pc/platform_application.h>
#endif

namespace os
{
    class application : public platform_application
    {
    private:
        using base = platform_application;

    public:

        struct create_parameters : public platform_application::create_parameters
        {

        };

        application( const create_parameters& p ) : base(p)
        {

        }

    };
}

