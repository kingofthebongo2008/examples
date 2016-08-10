#pragma once

#include <dwrite.h>
#include <os/windows/com_pointers.h>

namespace dwrite
{
    using factory       =  os::windows::com_ptr<IDWriteFactory>;
    using textformat    =  os::windows::com_ptr<IDWriteTextFormat>;
}

