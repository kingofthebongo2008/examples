#pragma once

#include <io/io_pad_state.h>
#include <Xinput.h>

namespace io
{
    namespace details
    {
        static const uint32_t invalid_user_index = 0xFFFFFFFF;

        inline uint32_t get_user_index()
        {
            XINPUT_STATE s = {};

            for (auto i = 0; i < 3; ++i)
            {
                if (XInputGetState(i, &s) == ERROR_SUCCESS )
                {
                    return i;
                }
            }

            return invalid_user_index;
        }
    }

    struct native_pad
    {
        uint32_t m_user_index = details::invalid_user_index;

        native_pad() : m_user_index(details::get_user_index() )
        {

        }

        
        uint8_t filter_trigger(uint8_t value)
        {
            if (value < XINPUT_GAMEPAD_TRIGGER_THRESHOLD)
            {
                value = 0;
            }

            return value;
        }

        int32_t abs_value(int32_t v)
        {
            if (v < 0) return -v;
            return v;
        }

        int16_t filter_thumb(int16_t min, int16_t value)
        {
            auto v = abs_value(value);

            if (v < min)
            {
                v = 0;
            }
            else
            {
                v = value;
            }

            return v;
        }

        float normalize_trigger(uint8_t value)
        {
            return value / 255.0f;
        }

        float normalize_thumb(int16_t value)
        {
            /*
            int32_t v0 = 2 * value + 1;
            return v0 / 255.0f;
            */
            return value / 32767.0f;
        }

        pad_state update(const pad_state& o)
        {
            if (m_user_index != details::invalid_user_index)
            {
                XINPUT_STATE s = {};

                if (XInputGetState(m_user_index, &s) == ERROR_SUCCESS)
                {
                    pad_state r(o);

                    r.swap();
                    r.set_mask(s.Gamepad.wButtons);

                    s.Gamepad.bLeftTrigger      = filter_trigger(s.Gamepad.bLeftTrigger);
                    s.Gamepad.bRightTrigger     = filter_trigger(s.Gamepad.bRightTrigger);

                    s.Gamepad.sThumbLX          = filter_thumb(XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE, s.Gamepad.sThumbLX);
                    s.Gamepad.sThumbLY          = filter_thumb(XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE, s.Gamepad.sThumbLY);

                    s.Gamepad.sThumbRX          = filter_thumb(XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE, s.Gamepad.sThumbRX);
                    s.Gamepad.sThumbRY          = filter_thumb(XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE, s.Gamepad.sThumbRY);

                    r.m_state.m_left_trigger    = normalize_trigger(s.Gamepad.bLeftTrigger);
                    r.m_state.m_right_trigger   = normalize_trigger(s.Gamepad.bRightTrigger);

                    r.m_state.m_thumb_left_x    = normalize_thumb(s.Gamepad.sThumbLX);
                    r.m_state.m_thumb_left_y    = normalize_thumb(s.Gamepad.sThumbLY);

                    r.m_state.m_thumb_right_x   = normalize_thumb(s.Gamepad.sThumbRX);
                    r.m_state.m_thumb_right_y   = normalize_thumb(s.Gamepad.sThumbRY);
                    
                    return r;
                }
                else
                {
                    return o;
                }
            }
            else
            {
                return o;
            }
        }
    };

}
