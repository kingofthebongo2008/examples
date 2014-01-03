//
//  com_ptr.hpp
//
//  Copyright (c) 2001, 2002 Peter Dimov
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
//  See http://www.boost.org/libs/smart_ptr/com_ptr.html for documentation.
//

//move out of boost, because it is used specifically for com, added operator & to make easier calling com functions

#ifndef __OS_WINDOWS_COM_PTR_H__
#define __OS_WINDOWS_COM_PTR_H__

#include <cstdint>
#include <cstddef>

namespace mem
{
    //
    //  com_ptr
    //
    //  A smart pointer that uses intrusive reference counting.
    //
    //  Relies on unqualified calls to
    //  
    //      void com_ptr_add_ref(T * p);
    //      void com_ptr_release(T * p);
    //
    //          (p != 0)
    //
    //  The object is responsible for destroying itself.
    //

    template<class T> class intrusive_ptr
    {
    private:

        typedef intrusive_ptr this_type;

    public:

        typedef T element_type;

        intrusive_ptr(): px( 0 )
        {
        }

        intrusive_ptr( T * p, bool add_ref = true ): px( p )
        {
            if( px != 0 && add_ref ) intrusive_ptr_add_ref( px );
        }

        template<class U>
        intrusive_ptr( intrusive_ptr<U> const & rhs )
        : px( rhs.get() )
        {
            if( px != 0 ) intrusive_ptr_add_ref( px );
        }

        intrusive_ptr(intrusive_ptr const & rhs): px( rhs.px )
        {
            if( px != 0 ) intrusive_ptr_add_ref( px );
        }

        ~intrusive_ptr()
        {
            if( px != 0 ) intrusive_ptr_release( px );
        }

        template<class U> intrusive_ptr & operator=(intrusive_ptr<U> const & rhs)
        {
            this_type(rhs).swap(*this);
            return *this;
        }

        // Move support

        intrusive_ptr(intrusive_ptr && rhs): px( rhs.px )
        {
            rhs.px = 0;
        }

        intrusive_ptr & operator=(intrusive_ptr && rhs)
        {
            this_type( static_cast< intrusive_ptr && >( rhs ) ).swap(*this);
            return *this;
        }

        intrusive_ptr & operator=(intrusive_ptr const & rhs)
        {
            this_type(rhs).swap(*this);
            return *this;
        }

        intrusive_ptr & operator=(T * rhs)
        {
            this_type(rhs).swap(*this);
            return *this;
        }

        void reset()
        {
            this_type().swap( *this );
        }

        void reset( T * rhs )
        {
            this_type( rhs ).swap( *this );
        }

        T * get() const throw()
        {
            return px;
        }

        T & operator*() const throw()
        {
            return *px;
        }

        T * operator->() const throw()
        {
            return px;
        }
        
        // implicit conversion to "bool"
        typedef T * this_type::*unspecified_bool_type;

        operator unspecified_bool_type() const // never throws
        {
            return px == 0? 0: &this_type::px;
        }

        void swap(intrusive_ptr & rhs)
        {
            T * tmp = px;
            px = rhs.px;
            rhs.px = tmp;
        }

        private:
        T * px;
    };

    template<class T, class U> inline bool operator==(intrusive_ptr<T> const & a, intrusive_ptr<U> const & b)
    {
        return a.get() == b.get();
    }

    template<class T, class U> inline bool operator!=(intrusive_ptr<T> const & a, intrusive_ptr<U> const & b)
    {
        return a.get() != b.get();
    }

    template<class T, class U> inline bool operator==(intrusive_ptr<T> const & a, U * b)
    {
        return a.get() == b;
    }

    template<class T, class U> inline bool operator!=(intrusive_ptr<T> const & a, U * b)
    {
        return a.get() != b;
    }

    template<class T, class U> inline bool operator==(T * a, intrusive_ptr<U> const & b)
    {
        return a == b.get();
    }

    template<class T, class U> inline bool operator!=(T * a, intrusive_ptr<U> const & b)
    {
        return a != b.get();
    }

    template<class T> inline bool operator<(intrusive_ptr<T> const & a, intrusive_ptr<T> const & b)
    {
        return std::less<T *>()(a.get(), b.get());
    }

    template<class T> void swap(intrusive_ptr<T> & lhs, intrusive_ptr<T> & rhs)
    {
        lhs.swap(rhs);
    }

    // mem_fn support
    template<class T> T * get_pointer(intrusive_ptr<T> const & p)
    {
        return p.get();
    }
}
#endif
