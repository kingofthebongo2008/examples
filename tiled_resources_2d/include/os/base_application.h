#pragma once

#include <cstdint>
#include <util/util_noncopyable.h>

namespace os
{
    class base_application : public util::noncopyable
    {
    public:

        virtual ~base_application()
        {

        }

        void quit()
        {
            on_quit();
        }

        std::int32_t run()
        {
            return on_run();
        }

        void render()
        {
            on_render();
        }

        void update()
        {
            on_update();
        }

    protected:

        virtual void on_quit() = 0;
        virtual void on_update() = 0;
        virtual void on_render() = 0;
        virtual std::int32_t on_run() = 0;

    };
}

