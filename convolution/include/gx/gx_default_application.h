#ifndef __GX_DEFAULT_APPLICATION_H__
#define __GX_DEFAULT_APPLICATION_H__

#include <dinput.h>

#include <io/io_keyboard.h>
#include <io/io_mouse.h>

#include <gx/gx_application.h>

namespace gx
{
    class default_application : public gx::application
    {
        typedef gx::application base;

    public:

        default_application(const wchar_t* window_title) : base(window_title)
            , m_mouse_state(window_width(get_window()), window_height(get_window()))
        {
            using namespace os::windows;
            /*
            throw_if_failed< com_exception >(DirectInput8Create(GetModuleHandle(nullptr), DIRECTINPUT_VERSION, IID_IDirectInput8, reinterpret_cast<void**> (&m_direct_input), nullptr));

            throw_if_failed< com_exception >(m_direct_input->CreateDevice(GUID_SysKeyboard, &m_keyboard, nullptr));
            throw_if_failed< com_exception >(m_keyboard->SetDataFormat(&c_dfDIKeyboard));
            throw_if_failed< com_exception >(m_keyboard->SetCooperativeLevel(this->get_window(), DISCL_BACKGROUND | DISCL_NONEXCLUSIVE));

            throw_if_failed< com_exception >(m_direct_input->CreateDevice(GUID_SysMouse, &m_mouse, nullptr));
            throw_if_failed< com_exception >(m_mouse->SetDataFormat(&c_dfDIMouse2));
            throw_if_failed< com_exception >(m_mouse->SetCooperativeLevel(this->get_window(), DISCL_BACKGROUND | DISCL_NONEXCLUSIVE));
            */

            ShowCursor(false);
        }

    protected:

        io::keyboard_state get_keyboard_state() const
        {
            return m_keyboard_state;
        }

        io::mouse_state get_mouse_state() const
        {
            return m_mouse_state;
        }

    private:

        uint16_t window_width(HWND window) const
        {
            RECT r;
            ::GetClientRect(window, &r);
            return static_cast<uint16_t> (r.right - r.left);
        }

        uint16_t window_height(HWND window) const
        {
            RECT r;
            ::GetClientRect(window, &r);
            return static_cast<uint16_t> (r.bottom - r.top);
        }

    protected:

        void on_update()
        {
            using namespace os::windows;

            uint8_t data[256] = {};
            auto hr = m_keyboard->GetDeviceState(sizeof(data), &data[0]);

            //swap the shadow states
            m_keyboard_state.swap();

            if (hr == S_OK)
            {
                m_keyboard_state.set_state(data);
            }

            DIMOUSESTATE2 mouse_state = {};

            m_mouse_state.swap();

            hr = m_mouse->GetDeviceState(sizeof(mouse_state), &mouse_state);

            if (hr == S_OK)
            {
                m_mouse_state.set_state(mouse_state);
            }
        }

        void on_render_frame()
        {

        }

        void on_activate()
        {
            /*
            using namespace os::windows;

            auto hr = m_keyboard->Acquire();

            if (!(hr == S_OK || hr == S_FALSE))
            {
                throw_if_failed< com_exception >(hr);
            }

            hr = m_mouse->Acquire();
            if (!(hr == S_OK || hr == S_FALSE))
            {
                throw_if_failed< com_exception >(hr);
            }
            */
        }

        void on_deactivate() override
        {
            /*
            using namespace os::windows;
            throw_if_failed< com_exception >(m_keyboard->Unacquire());

            using namespace os::windows;
            throw_if_failed< com_exception >(m_mouse->Unacquire());
            */
        }

        void on_resize(uint32_t width, uint32_t height) override
        {
            base::on_resize(width, height);

            m_mouse_state.set_width(width);
            m_mouse_state.set_height(height);
        }

    private:

        os::windows::com_ptr<IDirectInput8>         m_direct_input;
        os::windows::com_ptr<IDirectInputDevice8>   m_keyboard;
        os::windows::com_ptr<IDirectInputDevice8>   m_mouse;

        io::keyboard_state                          m_keyboard_state;
        io::mouse_state                             m_mouse_state;
    };
}

#endif