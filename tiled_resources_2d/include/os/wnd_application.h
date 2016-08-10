#pragma once

#include <os/application.h>

#if defined(_PC)
#include <os/platforms/pc/platform_wnd_application.h>
#endif

namespace os
{
    class windowed_applicaion : public platform_windowed_applicaion
    {
        private:

        using  base = platform_windowed_applicaion;

        public:

        struct create_parameters : public platform_windowed_applicaion::create_parameters
        {


        };

        windowed_applicaion(const create_parameters& p) : base(p)
        {

        }

    };
}


