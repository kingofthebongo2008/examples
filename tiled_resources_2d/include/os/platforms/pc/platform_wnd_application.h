#pragma once

#include <cstdint>
#include <exception>

#include <Windows.h>

#include <os/windows/com_error.h>


namespace os
{
    inline HWND create_window(uint32_t width, uint32_t height, HINSTANCE instance, const wchar_t* window_name)
    {
        RECT r = { 0U, 0U, (LONG)width, (LONG)height };

        windows::throw_if_failed<windows::win32_exception>(::AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, false));

        return CreateWindow(L"WindowClassCustom", window_name, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, 0, 0, instance, nullptr);
    }

    inline HWND create_window(HINSTANCE instance, const wchar_t* window_name)
    {
        return create_window(1600, 900, instance, window_name);
    }

    inline ATOM create_window_class(HINSTANCE instance, WNDPROC proc)
    {
        WNDCLASSEX wcex = {};

        wcex.cbSize = sizeof(WNDCLASSEX);

        wcex.style = CS_HREDRAW | CS_VREDRAW;
        wcex.lpfnWndProc = proc;
        wcex.cbWndExtra = 0;
        wcex.cbClsExtra = 0;
        wcex.hInstance = instance;
        //wcex.hIcon          = LoadIcon(  instance,  MAKEINTRESOURCE(IDI_APPLICATION) );
        wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
        wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        wcex.lpszMenuName = nullptr;
        wcex.lpszClassName = L"WindowClassCustom";
        //wcex.hIconSm		= LoadIcon(instance, MAKEINTRESOURCE(IDI_APPLICATION));

        return RegisterClassEx(&wcex);
    }

    class platform_windowed_applicaion : public application
    {

    public:

        struct create_parameters : public application::create_parameters
        {
            const wchar_t* m_window_name;

            create_parameters() : m_window_name(nullptr)
            {

            }
        };

    private:

        inline create_parameters create_window_params(const create_parameters& c)
        {
            create_parameters copy(c);
            copy.m_hwnd = create_window(c.m_instance, c.m_window_name);
            return copy;
        }

    public:

        platform_windowed_applicaion( const create_parameters& p ) : application( create_window_params(p) )
        {
            
        }

        void resize(uint32_t width, uint32_t height)
        {
            on_resize(width, height);
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

        HWND create_window(HINSTANCE instance, const wchar_t* window_name)
        {
            auto wnd_class = create_window_class(instance, DefaultWindowProc);

            if (wnd_class)
            {
                auto window = os::create_window(instance, window_name);

                if (window)
                {
                    ::SetWindowLongPtr(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));
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

        virtual void on_resize(uint32_t width, uint32_t height) = 0;
        virtual void on_activate() {}
        virtual void on_deactivate() {}

        static inline LRESULT CALLBACK DefaultWindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
        {

            auto wnd = reinterpret_cast<platform_windowed_applicaion*>(::GetWindowLongPtr(hWnd, GWLP_USERDATA));

            switch (message)
            {
                case WM_ACTIVATE:
                    if (wnd != nullptr)
                    {
                        if (LOWORD(wParam) != WA_INACTIVE)
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
                    if (wnd != 0)
                    {
                        ValidateRect(hWnd, NULL);
                    }

                    break;

                case WM_SIZE:
                {
                    if (wnd != 0)
                    {
                        wnd->resize(LOWORD(lParam), HIWORD(lParam));
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
    };
}

