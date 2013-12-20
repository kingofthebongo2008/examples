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

    inline void* malloc( std::size_t size )
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

    inline void* malloc_pitch( std::size_t* pitch, size_t width, size_t height )
    {
        void* r = nullptr;
        auto status = cudaMallocPitch( &r, pitch, width, height );
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

    inline void* allocate_pitch( std::size_t* pitch, size_t width, size_t height, void* p )
    {
        auto r = malloc_pitch(pitch, width, height);
        if ( r == nullptr )
        {
            throw std::bad_alloc();
        }
        return r;
    }

    template <typename t> inline t* allocate_pitch( std::size_t* pitch, size_t width, size_t height)
    {
        return reinterpret_cast<t*>( allocate_pitch( pitch, width, height, nullptr) );
    }

    inline void  free( void* pointer )
    {
        cudaFree( pointer );
    }

    class default_cuda_allocator
    {
        public:

        void* allocate( std::size_t size )
        {
            return cuda::allocate<void*>( size );
        }

        void free ( void* pointer )
        {
            cuda::free ( pointer );
        }
    };

    template < class allocator >
    class memory_buffer_
    {
        public:

        typedef memory_buffer_<allocator>    this_type;

        private:

        typedef allocator                   allocator_type;
        void*                               m_value;
        allocator_type                      m_allocator;


        void swap(this_type & rhs)
        {
            std::swap ( m_allocator, rhs.m_allocator );
            std::swap ( m_value, rhs.m_value );
        }

        public:

        this_type ( size_t size , allocator_type alloc = allocator_type() ) :
        m_allocator( alloc )
        {
            m_value = m_allocator.allocate(size);
        }

        this_type( void* value ) :
        m_value ( reinterpret_cast<int*> ( value ) )
        {

        }

        this_type ( this_type&& rhs ) : m_value(rhs.m_value), m_allocator( std::move(rhs.m_allocator) )
        {
            rhs.m_value = nullptr;
        }

        this_type & operator=(this_type && rhs)
        {
            this_type( static_cast< this_type && >( rhs ) ).swap(*this);
            return *this;
        }


        ~memory_buffer_()
        {
            m_allocator.free( m_value );
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

        this_type( const this_type& );
        this_type& operator=(const this_type&);
    };

    class memory_buffer : public memory_buffer_<default_cuda_allocator>
    {
        typedef memory_buffer_<default_cuda_allocator>  base;
        typedef memory_buffer                           this_type;

        public:


        memory_buffer( size_t size, default_cuda_allocator alloc = default_cuda_allocator() ) :
        base ( size, alloc )
        {

        }

        memory_buffer( void* pointer ) :
        base ( pointer )
        {

        }

        this_type ( this_type&& rhs ) : base( std::move ( rhs ) )
        {

        }

        this_type & operator=(this_type && rhs)
        {
            base::operator=( std::move (rhs ) );
            return *this;
        }

        private:

        this_type( const this_type& );
        this_type& operator=(const this_type&);
    };

}

#endif