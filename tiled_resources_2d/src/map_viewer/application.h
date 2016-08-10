#pragma once

#if defined(_PC)
#include <platforms/pc/platform_application.h>
#endif

namespace app
{
    class application : public platform_application
    {
        private:
        using base = platform_application;

        public:

        struct create_parameters : public base::create_parameters
        {

        };

        application(const create_parameters& p) : base(p)
        {

        }
    };
}
