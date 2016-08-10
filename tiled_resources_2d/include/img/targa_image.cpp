#include "precompiled.h"
#include "targa_image.h"

namespace img
{
    std::unique_ptr<tga> targa_load( const char* file_name )
    {
        tga_image img;

        auto r = tga_read(&img, file_name);
        if (r == TGA_NOERR)
        {
            return std::make_unique<tga>(img);
        }
        else
        {
            throw std::exception("error reading tga file");
        }
    }

    void targa_write_rgb(const char *file_name, uint8_t *image, const uint16_t width, const uint16_t height, const uint8_t depth)
    {
        auto r = tga_write_rgb(file_name, image, width, height, depth);
        if (r != TGA_NOERR)
        {
            throw std::exception("error writing tga file");
        }
    }

    void targa_write_rgb_rle(const char *file_name, uint8_t *image, const uint16_t width, const uint16_t height, const uint8_t depth)
    {
        auto r = tga_write_rgb_rle(file_name, image, width, height, depth);
        if (r != TGA_NOERR)
        {
            throw std::exception("error writing tga file");
        }
    }
}
