#ifndef __img_loader_h__
#define __img_loader_h__

#include <cstdint>
#include <exception>
#include <memory>

#include <jxr/cuda_helper.h>

namespace example
{
    class image
    {
        typedef cuda::memory_buffer data_pointer;

        public:

        enum format : uint32_t
        {
            format_24bpp_rgb
        };

        private:

        typedef image this_type;


        image( const image& );
        image& operator=(const image&);

        format          m_format;
        uint32_t        m_pitch;
        uint16_t        m_width;
        uint16_t        m_height;
        data_pointer    m_data;        
        
        public:

        image ( format format, uint32_t pitch, uint32_t width, uint32_t height, data_pointer&& data ) :
          m_format(format)
        , m_pitch(pitch)
        , m_width( static_cast<uint16_t>(width) )
        , m_height( static_cast<uint16_t>(height) )
        , m_data( std::move(data) )
        {

        }

        image ( image&& rhs ) : 
              m_format( std::move( rhs.m_format ))
            , m_pitch( std::move( rhs.m_pitch) )
            , m_width( std::move( rhs.m_width) )
            , m_height( std::move( rhs.m_height) )
            , m_data( std::move(rhs.m_data) )
        {
        }

        image & operator=(image && rhs)
        {
            if (this != &rhs)
            {
                m_format = std::move(rhs.m_format);
                m_pitch  = std::move(rhs.m_pitch);
                m_width = std::move(rhs.m_width);
                m_height = std::move(rhs.m_height);
                m_data = std::move(rhs.m_data);
            }
            return *this;
        }

        const void* get() const
        {
            return m_data.get();
        }

        void*    get()
        {
            return m_data.get();
        }

        template <typename t> operator t*()
        {
            return reinterpret_cast<t*> (m_data.get());
        }

        template <typename t> operator const t*() const
        {
            return reinterpret_cast<t*> (m_data.get());
        }

        uint32_t get_width() const
        {
            return m_width;
        }

        uint32_t get_height() const
        {
            return m_height;
        }

        uint32_t get_pitch() const
        {
            return m_pitch;
        }

        uint32_t size() const
        {
            return m_pitch * m_height;
        }
    };

    std::unique_ptr< image > create_image ( const wchar_t* image_file_path );
}

#endif
