#include "precompiled.h"
#include <cstdint>

#include <gx/gx_default_application.h>
#include <gx/gx_view_port.h>

#include <sys/sys_profile_timer.h>

class sample_application : public gx::default_application
{
    typedef gx::default_application base;

public:

    sample_application(const wchar_t* window_title) : base(window_title)
        , m_elapsed_update_time(0.0)
    {

    }

protected:

    virtual void on_render_scene()
    {

    }

    void render_scene()
    {
        on_render_scene();
    }

    virtual void on_update_scene()
    {

    }

    void update_scene()
    {
        on_update_scene();
    }

    void on_update()
    {
        sys::profile_timer timer;

        update_scene();

        //Measure the update time and pass it to the render function
        m_elapsed_update_time = timer.milliseconds();
    }

    void on_render_frame()
    {
        sys::profile_timer timer;
        on_render_scene();
    }

    void on_resize(uint32_t width, uint32_t height)
    {
        base::on_resize(width, height);
        //Reset view port dimensions
        m_view_port.set_dimensions(width, height);
    }


protected:

    gx::view_port                           m_view_port;
    double                                  m_elapsed_update_time;
};

int32_t wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR    lpCmdLine, int       nCmdShow )
{
    auto app = new gx::application(L"qcrm");

    app->run();

    delete app;

    return 0;
}




