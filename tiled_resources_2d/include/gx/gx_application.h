#pragma once

#include <algorithm>

#include <os/wnd_application.h>

#if defined(_PC)
#include <gx/platforms/pc/gx_platform_application.h>
#endif

namespace gx
{
    class application : public platform_application
    {
        public:

        struct create_parameters : platform_application::create_parameters
        {

        };

        private:

        using base = platform_application;

        public:
            application(const create_parameters& p ) :
            base( p )
        {

        }

        protected:

        virtual void on_render_frame() override
        {

        }

        virtual void on_post_render_frame() override
        {

        }

        private:


    };
}
