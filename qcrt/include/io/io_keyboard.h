#ifndef __IO_KEYBOARD_H__
#define __IO_KEYBOARD_H__

#include <cstdint>
#include <dinput.h>

#include <util/util_bits.h>

namespace io
{
    class keyboard_state
    {
        public:

        enum mask : uint8_t
        {
            key_escape              = DIK_ESCAPE,
            key_1                   = DIK_1,
            key_2                   = DIK_2,
            key_3                   = DIK_3,
            key_4                   = DIK_4,
            key_5                   = DIK_5,
            key_6                   = DIK_6,
            key_7                   = DIK_7,
            key_8                   = DIK_8,
            key_9                   = DIK_9,
            key_0                   = DIK_0,
            key_minus               = DIK_MINUS,            /* - on main keyboard */
            key_equals              = DIK_EQUALS,
            key_back                = DIK_BACK,             /* backspace */
            key_tab                 = DIK_TAB,
            key_q                   = DIK_Q,
            key_w                   = DIK_W,
            key_e                   = DIK_E,
            key_r                   = DIK_R,
            key_t                   = DIK_T,
            key_y                   = DIK_Y,
            key_u                   = DIK_U,
            key_i                   = DIK_I,
            key_o                   = DIK_O,
            key_p                   = DIK_P,
            key_lbracket            = DIK_LBRACKET,
            key_rbracket            = DIK_RBRACKET,
            key_return              = DIK_RETURN,           /* Enter on main keyboard */
            key_lcontrol            = DIK_LCONTROL,
            key_a                   = DIK_A,
            key_s                   =  DIK_S,
            key_d                   =  DIK_D,
            key_f                   =  DIK_F,
            key_g                   =  DIK_G,
            key_h                   =  DIK_H,
            key_j                   =  DIK_J,
            key_k                   =  DIK_K,
            key_l                   =  DIK_L,
            key_semicolon           =  DIK_SEMICOLON,
            key_apostrophe          =  DIK_APOSTROPHE,
            key_grave               =  DIK_GRAVE,            /* accent grave */
            key_lshift              =  DIK_LSHIFT,
            key_backslash           =  DIK_BACKSLASH,
            key_z                   =  DIK_Z,
            key_x                   =  DIK_X,
            key_c                   =  DIK_C,
            key_v                   =  DIK_V,
            key_b                   =  DIK_B,
            key_n                   =  DIK_N,
            key_m                   =  DIK_M,
            key_comma               =  DIK_COMMA,
            key_period              =  DIK_PERIOD,           /* . on main keyboard */
            key_slash               =  DIK_SLASH,            /* / on main keyboard */
            key_rshift              =  DIK_RSHIFT,
            key_multiply            =  DIK_MULTIPLY,         /* * on numeric keypad */
            key_lmenu               =  DIK_LMENU,            /* left Alt */
            key_space               =  DIK_SPACE,
            key_capital             =  DIK_CAPITAL,
            key_f1                  =  DIK_F1,
            key_f2                  =  DIK_F2,
            key_f3                  =  DIK_F3,
            key_f4                  =  DIK_F4,
            key_f5                  =  DIK_F5,
            key_f6                  =  DIK_F6,
            key_f7                  =  DIK_F7,
            key_f8                  =  DIK_F8,
            key_f9                  =  DIK_F9,
            key_f10                 =  DIK_F10,
            key_numlock             =  DIK_NUMLOCK,
            key_scroll              =  DIK_SCROLL,           /* Scroll Lock */
            key_numpad7             =  DIK_NUMPAD7,
            key_numpad8             =  DIK_NUMPAD8,
            key_numpad9             =  DIK_NUMPAD9,
            key_subtract            =  DIK_SUBTRACT,         /* - on numeric keypad */
            key_numpad4             =  DIK_NUMPAD4,
            key_numpad5             =  DIK_NUMPAD5,
            key_numpad6             =  DIK_NUMPAD6,
            key_add                 =  DIK_ADD,              /* + on numeric keypad */
            key_numpad1             =  DIK_NUMPAD1,
            key_numpad2             =  DIK_NUMPAD2,
            key_numpad3             =  DIK_NUMPAD3,
            key_numpad0             =  DIK_NUMPAD0,
            key_decimal             =  DIK_DECIMAL,          /* . on numeric keypad */
            key_oem_102             =  DIK_OEM_102,          /* <> or \| on RT 102-key keyboard (Non-U.S.) */
            key_f11                 =  DIK_F11,
            key_f12                 =  DIK_F12,
            key_f13                 =  DIK_F13,              /*                     (NEC PC98) */
            key_f14                 =  DIK_F14,              /*                     (NEC PC98) */
            key_f15                 =  DIK_F15,              /*                     (NEC PC98) */
            key_kana                =  DIK_KANA,             /* (Japanese keyboard)            */
            key_abnt_c1             =  DIK_ABNT_C1,          /* /? on Brazilian keyboard */
            key_convert             =  DIK_CONVERT,          /* (Japanese keyboard)            */
            key_noconvert           =  DIK_NOCONVERT,        /* (Japanese keyboard)            */
            key_yen                 =  DIK_YEN,              /* (Japanese keyboard)            */
            key_abnt_c2             =  DIK_ABNT_C2,          /* Numpad . on Brazilian keyboard */
            key_numpad_equals       =  DIK_NUMPADEQUALS,     /* = on numeric keypad (NEC PC98) */
            key_prev_track          =  DIK_PREVTRACK,        /* Previous Track (DIK_CIRCUMFLEX on Japanese keyboard) */
            key_at                  =  DIK_AT,               /*                     (NEC PC98) */
            key_colon               =  DIK_COLON,            /*                     (NEC PC98) */
            key_underline           =  DIK_UNDERLINE,        /*                     (NEC PC98) */
            key_kanji               =  DIK_KANJI,            /* (Japanese keyboard)            */
            key_stop                =  DIK_STOP,             /*                     (NEC PC98) */
            key_ax                  =  DIK_AX,               /*                     (Japan AX) */
            key_unlabled            =  DIK_UNLABELED,        /*                        (J3100) */
            key_next_track          =  DIK_NEXTTRACK,        /* Next Track */
            key_numpad_enter        =  DIK_NUMPADENTER,      /* Enter on numeric keypad */
            key_rcontrol            =  DIK_RCONTROL,
            key_mute                =  DIK_MUTE,             /* Mute */
            key_calculator          =  DIK_CALCULATOR,       /* Calculator */
            key_play_pause          =  DIK_PLAYPAUSE,        /* Play / Pause */
            key_media_stop          =  DIK_MEDIASTOP,        /* Media Stop */
            key_volume_down         =  DIK_VOLUMEDOWN,       /* Volume - */
            key_volume_up           =  DIK_VOLUMEUP,         /* Volume + */
            key_web_home            =  DIK_WEBHOME,          /* Web home */
            key_numpad_comma        =  DIK_NUMPADCOMMA,      /* , on numeric keypad (NEC PC98) */
            key_divide              =  DIK_DIVIDE,           /* / on numeric keypad */
            key_sysrq               =  DIK_SYSRQ,
            key_rmenu               =  DIK_RMENU,            /* right Alt */
            key_pause               =  DIK_PAUSE,            /* Pause */
            key_home                =  DIK_HOME,             /* Home on arrow keypad */
            key_up                  =  DIK_UP,               /* UpArrow on arrow keypad */
            key_prior               =  DIK_PRIOR,            /* PgUp on arrow keypad */
            key_left                =  DIK_LEFT,             /* LeftArrow on arrow keypad */
            key_right               =  DIK_RIGHT,            /* RightArrow on arrow keypad */
            key_end                 =  DIK_END,              /* End on arrow keypad */
            key_down                =  DIK_DOWN,             /* DownArrow on arrow keypad */
            key_next                =  DIK_NEXT,             /* PgDn on arrow keypad */
            key_insert              =  DIK_INSERT,           /* Insert on arrow keypad */
            key_delete              =  DIK_DELETE,           /* Delete on arrow keypad */
            key_lwin                =  DIK_LWIN,             /* Left Windows key */
            key_rwin                =  DIK_RWIN,             /* Right Windows key */
            key_apps                =  DIK_APPS,             /* AppMenu key */
            key_power               =  DIK_POWER,            /* System Power */
            key_sleep               =  DIK_SLEEP,            /* System Sleep */
            key_wake                =  DIK_WAKE,             /* System Wake */
            key_web_search          =  DIK_WEBSEARCH,        /* Web Search */
            key_web_favorities      =  DIK_WEBFAVORITES,     /* Web Favorites */
            key_web_refresh         =  DIK_WEBREFRESH,       /* Web Refresh */
            key_web_stop            =  DIK_WEBSTOP,          /* Web Stop */
            key_web_forward         =  DIK_WEBFORWARD,       /* Web Forward */
            key_web_back            =  DIK_WEBBACK,          /* Web Back */
            key_my_computer         =  DIK_MYCOMPUTER,       /* My Computer */
            key_mail                =  DIK_MAIL,             /* Mail */
            key_mediaselect         =  DIK_MEDIASELECT,      /* Media Select */

            /*
             *  Alternate names for keys, to facilitate transition from DOS.
             */
            key_backspace           = DIK_BACKSPACE,         /* backspace */
            key_numpad_star         = DIK_NUMPADSTAR,        /* * on numeric keypad */
            key_lalt                = DIK_LALT,              /* left Alt */
            key_capslock            = DIK_CAPSLOCK,          /* CapsLock */
            key_numpad_minus        = DIK_NUMPADMINUS,       /* - on numeric keypad */
            key_numpad_plus         = DIK_NUMPADPLUS,        /* + on numeric keypad */
            key_numpadperiod        = DIK_NUMPADPERIOD,      /* . on numeric keypad */
            key_numpadslash         = DIK_NUMPADSLASH,       /* / on numeric keypad */
            key_ralt                = DIK_RALT,              /* right Alt */
            key_uparrow             = DIK_UPARROW,           /* UpArrow on arrow keypad */
            key_pgup                = DIK_PGUP,              /* PgUp on arrow keypad */
            key_left_arrow          = DIK_LEFTARROW,         /* LeftArrow on arrow keypad */
            key_right_arrow         = DIK_RIGHTARROW,        /* RightArrow on arrow keypad */
            key_down_arrow          = DIK_DOWNARROW,         /* DownArrow on arrow keypad */
            key_pgdn                = DIK_PGDN,              /* PgDn on arrow keypad */

            key_circum_flex         = DIK_CIRCUMFLEX        /* Japanese keyboard */
        };


        keyboard_state()
        {
            m_states[0] = 0;
            m_states[1] = 0;
            m_states[2] = 0;
            m_states[3] = 0;

            m_shadow_states[0] = 0;
            m_shadow_states[1] = 0;
            m_shadow_states[2] = 0;
            m_shadow_states[3] = 0;
        }

        void swap()
        {
            m_shadow_states[0] = m_states[0];
            m_shadow_states[1] = m_states[1];
            m_shadow_states[2] = m_states[2];
            m_shadow_states[3] = m_states[3];
        }

        void set_state( uint8_t states[256] )
        {
            uint64_t sum = 0;

            for (auto i = 0, j = 0; i < 64; ++i, ++j)
            {
                auto bit = states[i] & 0x80;

                if (bit)
                {
                    sum += static_cast<uint64_t> ( 1 ) << static_cast<uint64_t> ( j );
                }
            }
            
            m_states[0] = sum;
            sum = 0;

            for (auto i = 64, j = 0; i < 128; ++i, ++j)
            {
                auto bit = states[i] & 0x80;

                if (bit)
                {
                    sum += static_cast<uint64_t> ( 1 ) << static_cast<uint64_t> ( j );
                }
            }

            m_states[1] = sum;
            sum = 0;

            for (auto i = 128, j = 0; i < 192; ++i, ++j)
            {
                auto bit = states[i] & 0x80;

                if (bit)
                {
                    sum += static_cast<uint64_t> ( 1 ) << static_cast<uint64_t> ( j );
                }
            }

            m_states[2] = sum;
            sum = 0;

            for (auto i = 192, j = 0; i < 256; ++i, ++j)
            {
                auto bit = states[i] & 0x80;

                if (bit)
                {
                    sum += static_cast<uint64_t> ( 1 ) << static_cast<uint64_t> ( j );
                }
            }

            m_states[3] = sum;

        }

        template <uint8_t button> void button_down()
        {
            uint32_t byte = button / 64;
            uint32_t bit  = button % 64;

            m_state = util::bit_set<bit, uint64_t>(m_states[byte]);
        }

        template <uint8_t button> void button_up()
        {
            uint32_t byte = button / 64;
            uint32_t bit  = button % 64;

            m_state = util::bit_reset<button, uint64_t>(m_states[byte]);
        }

        template <uint8_t button> bool is_button_down() const
        {
            uint32_t byte = button / 64;
            uint32_t bit  = button % 64;

            return util::bit_is_set<button, uint64_t>(m_states[byte]);
        }

        template <uint8_t button> bool is_button_up() const
        {
            uint32_t byte = button / 64;
            uint32_t bit  = button % 64;

            return !util::bit_is_set<button, uint64_t>(m_states[byte]);
        }

        template <uint8_t button> bool is_button_toggled() const
        {
            uint32_t byte = button / 64;
            uint32_t bit  = button % 64;

            uint64_t difference = m_states[byte] ^ m_shadow_states[byte];
            return util::bit_is_set<button, uint64_t>(difference);
        }

        private:

        uint64_t m_states[4];
        uint64_t m_shadow_states[4];
    };

};


#endif