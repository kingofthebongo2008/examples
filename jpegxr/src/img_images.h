#ifndef __img_images_h__
#define __img_images_h__

#include <cstdint>
#include <exception>
#include <memory>

#include <jxr/cuda_helper.h>

namespace example
{
    class image_3_channel
    {
        public:

        image_3_channel
            (
                std::shared_ptr< cuda::memory_buffer >  channel_0_buffer,
                std::shared_ptr< cuda::memory_buffer >  channel_1_buffer,
                std::shared_ptr< cuda::memory_buffer >  channel_2_buffer,

                uint32_t                                width,
                uint32_t                                height,
                uint32_t                                pitch
            ) :
              m_channel_0_buffer(channel_0_buffer)
            , m_channel_1_buffer(channel_1_buffer)
            , m_channel_2_buffer(channel_2_buffer)
            , m_width(width)
            , m_height(height)
            , m_pitch(pitch)
        {

        }


        std::shared_ptr< cuda::memory_buffer >  get_channel_0() const
        {
            return m_channel_0_buffer;
        }

        std::shared_ptr< cuda::memory_buffer >  get_channel_1() const
        {
            return m_channel_1_buffer;
        }

        std::shared_ptr< cuda::memory_buffer >  get_channel_2() const
        {
            return m_channel_2_buffer;
        }

        uint32_t    get_width() const
        {
            return m_width;
        }

        uint32_t    get_height() const
        {
            return m_height;
        }

        uint32_t    get_pitch() const
        {
            return m_pitch;
        }

        private:
        std::shared_ptr< cuda::memory_buffer >  m_channel_0_buffer;
        std::shared_ptr< cuda::memory_buffer >  m_channel_1_buffer;
        std::shared_ptr< cuda::memory_buffer >  m_channel_2_buffer;

        uint32_t                                m_width;
        uint32_t                                m_height;
        uint32_t                                m_pitch;

    };

    class ycocg_image : public image_3_channel
    {
        public:

        ycocg_image
            (
                std::shared_ptr< cuda::memory_buffer >  y_buffer,
                std::shared_ptr< cuda::memory_buffer >  co_buffer,
                std::shared_ptr< cuda::memory_buffer >  cg_buffer,

                uint32_t                                width,
                uint32_t                                height,
                uint32_t                                pitch
            ) :
        image_3_channel ( y_buffer, co_buffer, cg_buffer, width, height, pitch )
        {
        }

        std::shared_ptr< cuda::memory_buffer >  get_y() const
        {
            return get_channel_0();
        }

        std::shared_ptr< cuda::memory_buffer >  get_co() const
        {
            return get_channel_1();
        }

        std::shared_ptr< cuda::memory_buffer >  get_cg() const
        {
            return get_channel_2();
        }
    };

    class ycbcr_image : public image_3_channel
    {
        public:

        ycbcr_image
            (
                std::shared_ptr< cuda::memory_buffer >  y_buffer,
                std::shared_ptr< cuda::memory_buffer >  cb_buffer,
                std::shared_ptr< cuda::memory_buffer >  cr_buffer,

                uint32_t                                width,
                uint32_t                                height,
                uint32_t                                pitch
            ) :
        image_3_channel ( y_buffer, cb_buffer, cr_buffer, width, height, pitch )
        {
        }

        std::shared_ptr< cuda::memory_buffer >  get_y() const
        {
            return get_channel_0();
        }

        std::shared_ptr< cuda::memory_buffer >  get_cb() const
        {
            return get_channel_1();
        }

        std::shared_ptr< cuda::memory_buffer >  get_cr() const
        {
            return get_channel_2();
        }
    };
}

#endif
