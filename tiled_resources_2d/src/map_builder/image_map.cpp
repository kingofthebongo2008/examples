#include "precompiled.h"
#include "image_map.h"
#include <img/targa_image.h>

namespace map_builder
{
    std::unique_ptr< image > make_image(uint16_t width, uint16_t height, image::color fill_color)
    {
        std::unique_ptr< image > r = std::make_unique<image>(width, height);

        auto s = r->size();
        auto d = r->data();

        for (auto i = 0U; i < s; ++i)
        {
            *d++ = fill_color;
        }

        return r;
    }

    void save_image(const image* i, const char* file_name)
    {
        img::targa_write_rgb_rle(file_name, &i->m_data[0], i->m_width, i->m_height );
    }
}
