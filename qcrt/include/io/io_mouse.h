#ifndef __IO_MOUSE_H__
#define __IO_MOUSE_H__

#include <cstdint>
#include <algorithm>

#include <dinput.h>

#include <util/util_bits.h>

namespace io
{
    class mouse_state
    {
        private:

        struct state
        {
            state() : 
            m_button_state(0)
            , m_dx(0)
            , m_dy(0)
            , m_x(0)
            , m_y(0)
            {
            }

            uint64_t    m_button_state;

            int16_t     m_dx;
            int16_t     m_dy;
        
            uint16_t    m_x;
            uint16_t    m_y;
        };

        public:

        enum mask : uint8_t
        {
            button0 = 0,
            button1 = 1,
            button2 = 2,
            button3 = 3,
            button4 = 4,
            button5 = 5,
            button6 = 6,
            button7 = 7
        };


        mouse_state(uint16_t width, uint16_t height) : 
            m_width(width)
            , m_height(height)
        {
            m_state.m_x = m_width / 2;
            m_state.m_y = m_height / 2;
        }

        void swap()
        {
            m_shadow_state = m_state;
        }

        void set_state( const DIMOUSESTATE2& mouse_state )
        {
            int32_t x = m_state.m_x;
            int32_t y = m_state.m_y;

            uint64_t sum = 0;

            for (auto i = 0; i < sizeof( mouse_state.rgbButtons) / sizeof( mouse_state.rgbButtons[0] ) ; ++i )
            {
                if (mouse_state.rgbButtons[i] & 0x080)
                {
                    sum += static_cast<uint64_t>(1) << static_cast<uint64_t>(i);
                }
            }

            m_state.m_button_state = sum;

            m_state.m_dx = static_cast<int16_t> (mouse_state.lX);
            m_state.m_dy = static_cast<int16_t> (mouse_state.lY);

            x += m_state.m_dx;
            y += m_state.m_dy;

            m_state.m_x = std::min ( static_cast<uint16_t> ( std::max ( 0, x )) , m_width ) ;
            m_state.m_y = std::min ( static_cast<uint16_t> ( std::max ( 0, y )), m_height ) ;
        }

        template <uint8_t button> void button_down()
        {
            m_state = util::bit_set<bit, uint64_t>(m_state.m_button_state);
        }

        template <uint8_t button> void button_up()
        {
            m_state = util::bit_reset<button, uint64_t>(m_state.m_button_state);
        }

        template <uint8_t button> bool is_button_down() const
        {
            return util::bit_is_set<button, uint64_t>(m_state.m_button_state);
        }

        template <uint8_t button> bool is_button_up() const
        {
            return !util::bit_is_set<button, uint64_t>(m_state.m_button_state);
        }

        template <uint8_t button> bool is_button_toggled() const
        {
            uint64_t difference = m_state.m_button_state ^ m_shadow_state.m_button_state;
            return util::bit_is_set<button, uint64_t>(difference);
        }

        void set_width ( uint16_t width )
        {
            m_width = width;
            m_state.m_x = std::min( m_state.m_x, m_width );
        }

        void set_height( uint16_t height )
        {
            m_height = height;
            m_state.m_y = std::min( m_state.m_y, m_height );
        }

        uint16_t get_x() const
        {
            return m_state.m_x;
        }

        uint16_t get_y() const
        {
            return m_state.m_y;
        }

        private:

        state   m_state;
        state   m_shadow_state;

        uint16_t m_width;
        uint16_t m_height;
    };

};


#endif