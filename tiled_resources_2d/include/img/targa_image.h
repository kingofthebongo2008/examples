#pragma once

#include "targa.h"

#include <memory>
#include <string>

namespace img
{
    class tga
    {
        public:

        tga()
        {
            memset(&m_image, 0, sizeof(m_image));
        }

        explicit tga( const tga_image& i ) : m_image(i)
        {

        }

        ~tga()
        {
            tga_free_buffers( &m_image );
        }

        tga( tga&& o )
        {
            m_image = o.m_image;
            memset(&o.m_image, 0, sizeof(m_image));
        }

        tga& operator=(tga&& o)
        {
            m_image = o.m_image;
            memset(&o.m_image, 0, sizeof(m_image));
            return *this;
        }

        auto data() const
        {
            return m_image.image_data;
        }

        private:

        tga(const tga& o) = delete;
        tga& operator=(const tga&) = delete;

        tga_image m_image;
    };

    std::unique_ptr< tga > targa_load(const char* file_name);

    inline std::unique_ptr< tga > targa_load(const std::string& file_name)
    {
        return targa_load(file_name.c_str());
    }

    void targa_write_rgb(const char *file_name, uint8_t *image, const uint16_t width, const uint16_t height, const uint8_t depth = 32);
    void targa_write_rgb_rle(const char *file_name, uint8_t *image, const uint16_t width, const uint16_t height, const uint8_t depth = 32);
}
