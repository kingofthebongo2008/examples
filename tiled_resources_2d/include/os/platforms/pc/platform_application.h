#ifndef __OS_WINDOWS_WND_APPLICATION_H__
#define __OS_WINDOWS_WND_APPLICATION_H__

#include <cstdint>
#include <exception>

#include <Windows.h>

#include <os/base_application.h>

namespace os
{
	class platform_application : public base_application
    {
        public:

        struct create_parameters
        {
            HWND        m_hwnd;
            HINSTANCE   m_instance;
            HACCEL      m_accel_table;
            uint32_t    m_adapter_index;

            create_parameters() : m_hwnd(0), m_instance(0), m_accel_table(0), m_adapter_index(0)
            {

            }
        };

        platform_application( const create_parameters& p) : m_hwnd(p.m_hwnd), m_instance(p.m_instance), m_accel_table(p.m_accel_table), m_adapter_index(p.m_adapter_index)
        {

        }

        virtual ~platform_application()
        {

        }

        private:

        HWND        m_hwnd;
        HINSTANCE   m_instance;
        HACCEL      m_accel_table;
        uint32_t    m_adapter_index;

        protected:

        HWND get_window() const
        {
            return m_hwnd;
        }

        protected:

        virtual void on_update() = 0;
        virtual void on_render() = 0;

        void on_quit()
        {
            PostQuitMessage(0);
        }

        virtual std::int32_t on_run() 
        {
            HACCEL hAccelTable = m_accel_table;

            MSG msg = {};
                
            ::ShowWindow(m_hwnd, SW_SHOWDEFAULT);
            ::UpdateWindow(m_hwnd);

            while ( msg.message != WM_QUIT )
            {
                if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
                {
                    if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
                    {
                        TranslateMessage(&msg);
                        DispatchMessage(&msg);
                    }
                    else
                    {
                        update();
                        render();
                    }
                }
                else
                {
                        update();
                        render();
                }
            }

            return static_cast<std::int32_t> (msg.wParam);
        }
    };

}

#endif
