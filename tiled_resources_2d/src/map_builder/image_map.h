#include <memory>
#include <string>

namespace map_builder
{
    struct image
    {
        struct color
        {
            uint8_t r;
            uint8_t g;
            uint8_t b;
            uint8_t a;

            static inline color white()
            {
                color r;
                r.r = 0xff;
                r.g = 0xff;
                r.b = 0xff;
                r.a = 0xff;
                return r;
            }

            static inline color red()
            {
                color r;
                r.r = 0xff;
                r.g = 0x0;
                r.b = 0x0;
                r.a = 0xff;
                return r;
            }

            static inline color green()
            {
                color r;
                r.r = 0x00;
                r.g = 0xff;
                r.b = 0x00;
                r.a = 0xff;
                return r;
            }

            static inline color blue()
            {
                color r;
                r.r = 0x00;
                r.g = 0x00;
                r.b = 0xff;
                r.a = 0xff;
                return r;
            }

        };

        uint16_t m_width  = 0;
        uint16_t m_height = 0;

        std::unique_ptr<uint8_t[]> m_data;

        image(uint16_t width, uint16_t height) : m_width(width), m_height(height), m_data ( new uint8_t[ width * height * sizeof(color)])
        {

        }
           
        color* data() const
        {
            return reinterpret_cast<color*>(m_data.get());
        }

        size_t size() const 
        {
            return m_width * m_height;
        }

        size_t bytes() const
        {
            return size() * sizeof(color);
        }
    };

    std::unique_ptr< image > make_image(uint16_t width, uint16_t height, image::color fill_color);

    void save_image( const image* i, const char* file_name );

    inline void save_image(const image* i, const std::string& file_name)
    {
        save_image(i, file_name.c_str());
    }
}
