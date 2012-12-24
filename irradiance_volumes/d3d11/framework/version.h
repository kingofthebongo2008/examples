//-----------------------------------------------------------------------------
// File: Framework\Version.h
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------





#ifndef _VERSION_H_
#define _VERSION_H_

#define FW_MAJOR_VERSION 1
#define FW_MINOR_VERSION 0

#ifndef FRAMEWORK_VERSION
#error This sample did not defined FRAMEWORK_VERSION
#elif FRAMEWORK_VERSION > FW_MAJOR_VERSION
#error Sample expects a newer version of the framework. Please compile using framework included with this sample.
#elif FRAMEWORK_VERSION < FW_MAJOR_VERSION
#error Sample expects an older version of the framework. Please compile using original framework or download this sample again.
#endif

#endif // _VERSION_H_
