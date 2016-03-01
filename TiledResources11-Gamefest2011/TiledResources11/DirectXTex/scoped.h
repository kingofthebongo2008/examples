//-------------------------------------------------------------------------------------
// scoped.h
//  
// Utility header with helper classes for exception-safe handling of resources
//
// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//-------------------------------------------------------------------------------------

#if defined(_MSC_VER) && (_MSC_VER > 1000)
#pragma once
#endif

#include <malloc.h>

//---------------------------------------------------------------------------------
template<class T> class ScopedArray
{
public:
    explicit ScopedArray( T *p = 0 ) : _pointer(p) {}
    ~ScopedArray()
    {
        delete [] _pointer;
        _pointer = nullptr;
    }

    bool IsNull() const { return (!_pointer); }

    T& operator[]( ptrdiff_t i ) const { return _pointer[i]; }

    void Reset(T *p = 0) { delete[] _pointer; _pointer = p; }

    T* Get() const { return _pointer; }

private:
    ScopedArray(const ScopedArray&);
    ScopedArray& operator=(const ScopedArray&);
        
    T* _pointer;
};


//---------------------------------------------------------------------------------
template<class T> class ScopedAlignedArray
{
public:
    explicit ScopedAlignedArray( T *p = 0 ) : _pointer(p) {}
    ~ScopedAlignedArray()
    {
        if ( _pointer )
        {
            _aligned_free( _pointer );
            _pointer = nullptr;
        }
    }

    bool IsNull() const { return (!_pointer); }

    T& operator[]( ptrdiff_t i ) const { return _pointer[i]; }

    void Reset(T *p = 0) { if (_pointer) { _aligned_free( _pointer ); } _pointer = p; }

    T* Get() const { return _pointer; }

private:
    ScopedAlignedArray(const ScopedAlignedArray&);
    ScopedAlignedArray& operator=(const ScopedAlignedArray&);
        
    T* _pointer;
};


//---------------------------------------------------------------------------------
class ScopedHandle
{
public:
    explicit ScopedHandle( HANDLE handle ) : _handle(handle) {}
    ~ScopedHandle()
    {
        if ( _handle != INVALID_HANDLE_VALUE )
        {
            CloseHandle( _handle );
            _handle = INVALID_HANDLE_VALUE;
        }
    }

    bool IsValid() const { return (_handle != INVALID_HANDLE_VALUE); }
    HANDLE Get() const { return _handle; }

private:
    HANDLE _handle;
};


//---------------------------------------------------------------------------------
template<class T> class ScopedObject
{
public:
    explicit ScopedObject( T *p = 0 ) : _pointer(p) {}
    ~ScopedObject()
    {
        if ( _pointer )
        {
            _pointer->Release();
            _pointer = nullptr;
        }
    }

    bool IsNull() const { return (!_pointer); }

    T& operator*() { return *_pointer; }
    T* operator->() { return _pointer; }
    T** operator&() { return &_pointer; }

    void Reset(T *p = 0) { if ( _pointer ) { _pointer->Release(); } _pointer = p; }

    T* Get() const { return _pointer; }

private:
    ScopedObject(const ScopedObject&);
    ScopedObject& operator=(const ScopedObject&);
        
    T* _pointer;
};
