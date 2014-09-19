#ifndef __MEM_STREAMFLOW_MEMORY_H__
#define __MEM_STREAMFLOW_MEMORY_H__

#include <mem/mem_streamflow.h>
#include <memory>

namespace streamflow
{
    //---------------------------------------------------------------------------------------
    inline void* operator new(std::size_t size)
    {
        void* result = mem::streamflow::get_heap(0)->allocate( size );

        if (result == nullptr)
        {
            throw std::bad_alloc();
        }

        return result;
    }

    inline void operator delete(void* pointer) throw()
    {
        if (pointer != nullptr)
        {
            mem::streamflow::get_heap(0)->free(pointer);
        }
    }
    //---------------------------------------------------------------------------------------
    inline void* operator new   (std::size_t size, const std::nothrow_t&) throw()
    {
        return mem::streamflow::get_heap(0)->allocate( size );
    }

    inline void operator delete (void* pointer, const std::nothrow_t&) throw()
    {
        if (pointer != nullptr)
        {
            mem::streamflow::get_heap(0)->free(pointer);
        }
    }
    //---------------------------------------------------------------------------------------
    inline void* operator new  [](std::size_t size)
    {
        void* result = mem::streamflow::get_heap(0)->allocate( size );

        if (result == nullptr)
        {
            throw std::bad_alloc();
        }

        return result;
    }

    inline void operator delete[](void* pointer) throw()
    {
        if (pointer != nullptr)
        {
            mem::streamflow::get_heap(0)->free(pointer);
        }
    }
    //---------------------------------------------------------------------------------------
    inline void* operator new  [](std::size_t size, const std::nothrow_t&) throw()
    {
        return mem::streamflow::get_heap(0)->allocate( size );
    }

    inline void operator delete[](void* pointer, const std::nothrow_t&) throw()
    {
        if (pointer != nullptr)
        {
            mem::streamflow::get_heap(0)->free(pointer);
        }
    }
    //---------------------------------------------------------------------------------------
}

#endif