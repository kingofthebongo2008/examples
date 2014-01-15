#ifndef __img_images_h__
#define __img_images_h__

#include <cstdint>
#include <exception>
#include <memory>

#include <jxr/cuda_helper.h>

namespace example
{
    class image_2d
    {
        public:

        image_2d
            (
                std::shared_ptr< cuda::memory_buffer >  channel_buffer,
                uint32_t                                width,
                uint32_t                                height,
                uint32_t                                pitch
            ) :
              m_channel_buffer(channel_buffer)
            , m_width(width)
            , m_height(height)
            , m_pitch(pitch)
        {

        }

        std::shared_ptr< cuda::memory_buffer >  get_data() const
        {
            return m_channel_buffer;
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

        operator std::shared_ptr< cuda::memory_buffer >() const
        {
            return get_data();
        }

        private:
        std::shared_ptr< cuda::memory_buffer >  m_channel_buffer;

        uint32_t                                m_width;
        uint32_t                                m_height;
        uint32_t                                m_pitch;
    };

    void* get_data( image_2d& image )
    {
        return image.get_data().get()->get();
    }

    void* get_data( std::shared_ptr<image_2d> image )
    {
        return image->get_data().get()->get();
    }

    template< typename arg1, typename arg2, typename arg3, typename arg4 > std::shared_ptr< image_2d> make_image_2d( arg1 a1, arg2 a2, arg3 a3, arg4 a4  )
    {
        return std::shared_ptr<image_2d> ( new image_2d( std::forward<arg1> (a1), std::forward<arg2> (a2), std::forward<arg3> (a3), std::forward<arg4> (a4) ) ) ;
    }


    template <int32_t channel_count>
    class composite_image
    {

    };

    template <>
    class composite_image<3>
    {
        public:

        composite_image
        (
            std::shared_ptr< image_2d >  channel_0_buffer,
            std::shared_ptr< image_2d >  channel_1_buffer,
            std::shared_ptr< image_2d >  channel_2_buffer
        )
        {
            m_channels[0] = channel_0_buffer;
            m_channels[1] = channel_1_buffer;
            m_channels[2] = channel_2_buffer;
        }

        std::shared_ptr< image_2d >  get_channel(uint32_t channel) const
        {
            return m_channels[channel];
        }

        private:
        std::shared_ptr< image_2d >  m_channels[3];
    };

    class ycocg_image : public composite_image<3>
    {
        typedef composite_image<3> base;

        public:

        ycocg_image
            (
                 std::shared_ptr< image_2d >  y_buffer,
                 std::shared_ptr< image_2d >  co_buffer,
                 std::shared_ptr< image_2d >  cg_buffer
            ) :
        base ( y_buffer, co_buffer, cg_buffer)
        {

        }
    };

    inline std::shared_ptr< image_2d >  get_y( const ycocg_image& image)
    {
        return image.get_channel(0);
    }

    inline std::shared_ptr< image_2d >  get_y( std::shared_ptr< ycocg_image> image)
    {
        return image->get_channel(0);
    }

    inline std::shared_ptr< image_2d >  get_co( const ycocg_image& image)
    {
        return image.get_channel(1);
    }

    inline std::shared_ptr< image_2d >  get_co( std::shared_ptr< ycocg_image> image)
    {
        return image->get_channel(1);
    }

    inline std::shared_ptr< image_2d >  get_cg(const ycocg_image& image)
    {
        return image.get_channel(2);
    }

    inline std::shared_ptr< image_2d >  get_cg(std::shared_ptr< ycocg_image> image)
    {
        return image->get_channel(2);
    }

    class ycbcr_image : public composite_image<3>
    {
        typedef composite_image<3> base;

        public:

        ycbcr_image
            (
                std::shared_ptr< image_2d >  y_buffer,
                std::shared_ptr< image_2d >  cb_buffer,
                std::shared_ptr< image_2d >  cr_buffer
            ) :
        base ( y_buffer, cb_buffer, cr_buffer)
        {
        }

    };

    inline std::shared_ptr< image_2d >  get_y( const ycbcr_image& image)
    {
        return image.get_channel(0);
    }

    inline std::shared_ptr< image_2d >  get_y( std::shared_ptr<ycbcr_image> image)
    {
        return image->get_channel(0);
    }

    inline std::shared_ptr< image_2d >  get_cb( const ycbcr_image& image)
    {
        return image.get_channel(1);
    }

    inline std::shared_ptr< image_2d >  get_cb( std::shared_ptr<ycbcr_image> image)
    {
        return image->get_channel(1);
    }

    inline std::shared_ptr< image_2d >  get_cr(const ycbcr_image& image)
    {
        return image.get_channel(2);
    }

    inline std::shared_ptr< image_2d >  get_cr(std::shared_ptr<ycbcr_image> image)
    {
        return image->get_channel(2);
    }
}

#endif
