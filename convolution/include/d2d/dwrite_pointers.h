#ifndef __DWRITE_POINTERS_H__
#define __DWRITE_POINTERS_H__

#include <dwrite.h>

#include <os/windows/com_pointers.h>

namespace dwrite
{
    typedef os::windows::com_ptr<IDWriteFactory>          ifactory_ptr;
    typedef os::windows::com_ptr<IDWriteTextFormat>       itextformat_ptr;
}


#endif

