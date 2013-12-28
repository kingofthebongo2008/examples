#ifndef __MEM_ALLOC_ALIGNED_H__
#define __MEM_ALLOC_ALIGNED_H__

#include <memory>

#include <mem/mem_alloc.h>

namespace mem
{
    //default allocators guarantee 8 byte allocations, if you want more guarantees you can inherit from this class
    
    template <typename derived>
    class alloc_aligned 
    {
        public:

        //---------------------------------------------------------------------------------------
        void* operator new(std::size_t size)
        {
			size_t allocate_size = align( size, derived::alignment() );

            void* result = ::operator new( allocate_size ) ;

            if (result == nullptr)
            {
                throw std::bad_alloc();
            }
            
            return reinterpret_cast<void*> (  align( result, derived::alignment() ) );
        }

        //---------------------------------------------------------------------------------------
        void* operator new(std::size_t size, void* pointer)
        {
            size;
            return pointer;
        }

        //---------------------------------------------------------------------------------------
        void operator delete(void* pointer) throw()
        {
            ::operator delete(pointer);
        }
        //---------------------------------------------------------------------------------------
        void* operator new  (std::size_t size, const std::nothrow_t& t) throw()
        {
            size_t allocate_size = align( size, derived::alignment() );

            void* result = ::operator new( allocate_size , t) ;

            if (result )
            {
                return reinterpret_cast<void*> (  align( result, derived::alignment() ) );
            }
            else
            {
                return nullptr;
            }
        }

        void operator delete (void* pointer, const std::nothrow_t& t) throw()
        {
            ::operator delete(pointer, t);
        }

        //---------------------------------------------------------------------------------------
        void* operator new  [](std::size_t size)
        {
            size_t allocate_size = align( size, derived::alignment() );

            void* result = ::operator new[]( allocate_size ) ;

            if (result == nullptr)
            {
                throw std::bad_alloc();
            }

            return reinterpret_cast<void*> (  align( result, derived::alignment() ) );
        }

        void operator delete[](void* pointer) throw()
        {
            ::operator delete( pointer );
        }

        //---------------------------------------------------------------------------------------
        void* operator new  [](std::size_t size, const std::nothrow_t& t) throw()
        {
            size_t allocate_size = align( size, derived::alignment() );

            void* result = ::operator new[]( allocate_size , t) ;

            if (result )
            {
                return reinterpret_cast<void*> (  align( result, derived::alignment() ) );
            }
            else
            {
                return nullptr;
            }
        }

        void operator delete[](void* pointer, const std::nothrow_t& t) throw()
        {
            ::operator delete[]( pointer, t );
        }
    };
}


#endif
