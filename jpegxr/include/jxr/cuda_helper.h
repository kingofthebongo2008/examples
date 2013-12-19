#ifndef __cuda_helper_h__
#define __cuda_helper_h__

#include <exception>

#include <cuda_runtime.h>

namespace cuda
{
    class exception : public std::exception
    {
        public:

        exception( cudaError_t error ) : m_error(error)
        {

        }

        const char * what() const override
        {
            return cudaGetErrorString(m_error);
        }

        private:

        cudaError_t m_error;
    };

    template < typename exception > void throw_if_failed( cudaError_t error )
    {
        if (error != cudaSuccess)
        {
            throw exception(error);
        }
    }

    void* malloc( std::size_t size )
    {
        void* r = nullptr;
        auto status = cudaMalloc( &r, size );
        if ( status == cudaSuccess )
        {
            return r;
        }
        else
        {
            return nullptr;
        }
    }

    inline void* allocate( std::size_t size, void* p )
    {
        auto r = malloc(size);
        if ( r == nullptr )
        {
            throw std::bad_alloc();
        }
        return r;
    }

    template <typename t> inline t* allocate(std::size_t size)
    {
        return reinterpret_cast<t*>(allocate(size, nullptr));
    }

    void  free( void* pointer )
    {
        cudaFree( pointer );
    }

    class memory_buffer
    {
        private:

        typedef memory_buffer   this_type;
        int*    m_value;

        void swap(memory_buffer & rhs)
        {
            int* tmp = m_value;
            m_value = rhs.m_value;
            rhs.m_value = tmp;
        }

        public:

        memory_buffer ( size_t size ) :
        m_value( allocate<int>(size) )
        {

        }

        memory_buffer ( memory_buffer&& rhs ) : m_value(rhs.m_value)
        {
            rhs.m_value = nullptr;
        }

        memory_buffer & operator=(memory_buffer && rhs)
        {
            this_type( static_cast< memory_buffer && >( rhs ) ).swap(*this);
            return *this;
        }


        ~memory_buffer()
        {
            free(m_value);
        }

        const void*    get() const
        {
            return m_value;
        }

        void*    get()
        {
            return m_value;
        }

        template <typename t> operator t*()
        {
            return reinterpret_cast<t*> (m_value);
        }

        template <typename t> operator const t*() const
        {
            return reinterpret_cast<t*> (m_value);
        }

        private:

        memory_buffer( const memory_buffer& );
        memory_buffer& operator=(const memory_buffer&);
    };
}

#endif