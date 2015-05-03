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
        m_wait_back_buffer = CreateEventEx(nullptr, FALSE, FALSE, EVENT_ALL_ACCESS);

        m_wait_back_buffer_fence = d3d12x::create_fence(this->m_context.m_device);
        m_frame_index = 1;
    }

    ~sample_application()
    {
        CloseHandle( m_wait_back_buffer );
    }

    void shutdown()
    {
        wait_gpu_for_back_buffer();
    }

protected:

    virtual void on_render_scene()
    {

    }

    void render_scene()
    {
        on_render_scene();
    }

    void wait_gpu_for_back_buffer()
    {
        const auto frame_index = m_frame_index;

        //signal for stop rendering
        os::windows::throw_if_failed<d3d12::exception>(m_wait_back_buffer_fence->Signal(frame_index));

        m_frame_index++;

        //wait for the gpu to finish frame
        if ( m_wait_back_buffer_fence->GetCompletedValue() < frame_index - 1 )
        {
            m_wait_back_buffer_fence->SetEventOnCompletion( frame_index - 1 , this->m_wait_back_buffer );
            WaitForSingleObject(m_wait_back_buffer, INFINITE);
        }
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

        wait_gpu_for_back_buffer();
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

    HANDLE                                  m_wait_back_buffer;
    d3d12::fence                            m_wait_back_buffer_fence;
    uint32_t                                m_frame_index;

    static const uint32_t                   m_gpu_buffered_frames = 3;

};

int32_t wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR    lpCmdLine, int       nCmdShow )
{
    auto app = new sample_application(L"qcrm");

    app->run();
    app->shutdown();
    
    delete app;

    return 0;
}




