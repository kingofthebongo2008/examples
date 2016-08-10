#pragma once

#include <cstdint>
#include <windows.h>

#include <math/math_functions.h>
#include <util/util_bits.h>

namespace io
{
    struct pad_state
    {
        struct state
        {
            float    m_left_trigger             = 0.0f;
            float    m_right_trigger            = 0.0f;

            float    m_thumb_left_x             = 0.0f;
            float    m_thumb_left_y             = 0.0f;

            float    m_thumb_right_x            = 0.0f;
            float    m_thumb_right_y            = 0.0f;
            
            uint16_t m_buttons                  = 0;
        };

        public:

        enum mask : uint16_t
        {
            button_0 = 0,
            button_1 = 1,
            button_2 = 2,
            button_3 = 3,
            button_4 = 4,
            button_5 = 5,
            button_6 = 6,
            button_7 = 7,
            button_8 = 8,
            button_9 = 9,
            button_10 = 10,
            button_11 = 11,
            button_12 = 12,
            button_13 = 13,
            button_14 = 14,
            button_15 = 15
        };

        typedef uint16_t difference;

        void swap()
        {
            m_shadow_state = m_state;
        }

        difference get_difference() const
        {
            return m_state.m_buttons ^ m_shadow_state.m_buttons;
        }

        void set_state(uint32_t state)
        {
            m_state.m_buttons = static_cast<uint16_t>(state);
        }

        void set_mask(uint32_t mask)
        {
            m_state.m_buttons = m_state.m_buttons | static_cast<uint16_t> (mask);
        }

        void reset_mask(uint32_t mask)
        {
            m_state.m_buttons = m_state.m_buttons & (~(mask));
        }

        template <uint32_t button> void button_down()
        {
            m_state = util::bit_set<button, uint16_t>(m_state.m_buttons);
        }

        template <uint32_t button> void button_up()
        {
            m_state = util::bit_reset<button, uint16_t>(m_state.m_buttons);
        }

        template <uint32_t button> bool is_button_down() const
        {
            return util::bit_is_set<button, uint16_t>(m_state.m_buttons);
        }

        state   m_state;
        state   m_shadow_state;

    };
}
