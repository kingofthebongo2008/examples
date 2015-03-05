#ifndef __OS_WINDOWS_WND_APPLICATION_H__
#define __OS_WINDOWS_WND_APPLICATION_H__

#include <cstdint>
#include <exception>

#include <Windows.h>

#include <util/util_noncopyable.h>

namespace os
{
    namespace windows
    {
	    class application : public util::noncopyable
        {
            public:
            application ( HWND hwnd, HINSTANCE instance, HACCEL accel_table ) : m_hwnd(hwnd), m_instance(instance), m_accel_table(accel_table)
            {

            }

            application ( HWND hwnd, HINSTANCE instance ) : m_hwnd(hwnd), m_instance(instance), m_accel_table(nullptr)
            {

            }

            void quit()
            {
                PostQuitMessage(0);
            }

            virtual ~application()
            {

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

            private:

            HWND        m_hwnd;
            HINSTANCE   m_instance;
            HACCEL      m_accel_table;

            protected:

            HWND get_window() const
            {
                return m_hwnd;
            }

            protected:
            virtual void on_update() = 0;
            virtual void on_render() = 0;

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

        inline HWND create_window( uint32_t width, uint32_t height, HINSTANCE instance, const wchar_t* window_name )
        {
            RECT r = { 0, 0, width, height };

            throw_if_failed<win32_exception> ( ::AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, false) );

            return CreateWindow( L"WindowClassCustom", window_name, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, 0, 0, instance, nullptr);
        }

        inline HWND create_window( HINSTANCE instance, const wchar_t* window_name )
        {
            return create_window( 1280, 720, instance, window_name );
        }

        inline ATOM create_window_class( HINSTANCE instance, WNDPROC proc )
        {
            WNDCLASSEX wcex = {};

            wcex.cbSize = sizeof(WNDCLASSEX);

            wcex.style          = CS_HREDRAW | CS_VREDRAW;
            wcex.lpfnWndProc    = proc;
            wcex.cbWndExtra     = 0;
            wcex.cbClsExtra     = 0;
            wcex.hInstance      = instance;
            wcex.hIcon          = LoadIcon( instance, MAKEINTRESOURCE(IDI_APPLICATION) );
            wcex.hCursor        = LoadCursor( NULL, IDC_ARROW );
            wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
            wcex.lpszMenuName   = nullptr;
            wcex.lpszClassName  = L"WindowClassCustom";
            wcex.hIconSm		= LoadIcon(instance, MAKEINTRESOURCE(IDI_APPLICATION));

                return RegisterClassEx(&wcex);
        }

        static inline LRESULT CALLBACK DefaultWindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

        class windowed_applicaion : public application
        {
            public:

            windowed_applicaion ( HINSTANCE instance, const wchar_t* window_name ) : application (  create_window( instance, window_name ), instance )
            {

            }
            void resize(uint32_t width, uint32_t height)
            {
                on_resize( width, height );
            }

            void activate()
            {
                on_activate();
            }

            void deactivate()
            {
                on_deactivate();
            }

            private:

            HWND create_window( HINSTANCE instance, const wchar_t* window_name )
            {
                auto wnd_class = create_window_class(instance, DefaultWindowProc );
                
                if (wnd_class)
                {
                    auto window = os::windows::create_window( instance, window_name );

                    if (window)
                    {
                        ::SetWindowLongPtr(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this) );
                        return window;
                    }
                    else
                    {
                        throw std::exception("Cannot create window");
                    }
                }
                else
                {
                    throw std::exception("Cannot create window class");
                }
            }

            protected:

            virtual void on_resize( uint32_t width, uint32_t height ) = 0;
            virtual void on_activate(){}
            virtual void on_deactivate() {}

        };

        static inline LRESULT CALLBACK DefaultWindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
        {
            auto wnd = reinterpret_cast< windowed_applicaion* >(::GetWindowLongPtr(hWnd, GWLP_USERDATA));

            switch (message)
            {
            case WM_ACTIVATE:
                if (wnd != nullptr )
                {
                    if ( LOWORD(wParam) != WA_INACTIVE )
                    {
                        wnd->activate();
                    }
                    else
                    {
                        wnd->deactivate();
                    }
                }
                break;
            case WM_PAINT:
                if (wnd !=0 )
                {
                    wnd->render();
                    ValidateRect( hWnd, NULL );
                }

                break;
            case WM_SIZE:
                {
                    if (wnd !=0 )
                    {
                        wnd->resize( LOWORD(lParam), HIWORD(lParam) );
                        wnd->render();
                    }
                }
                break;

            case WM_DESTROY:
                {
                    PostQuitMessage(0);
                }
                break;

            default:

                return DefWindowProc(hWnd, message, wParam, lParam);
            }

            return 0;
        }
    }
}

#endif
