#pragma once

#if defined(_PC)
#include <gx/platforms/pc/gx_platform_default_application.h>
#endif

namespace gx
{
    class default_application : public platform_default_application
    {

    private:

        using base = platform_default_application;

    public:

        struct create_parameters : public platform_default_application::create_parameters
        {

        };

    public:
        default_application(const create_parameters & p) : base(p)
        {

        }
    };
}