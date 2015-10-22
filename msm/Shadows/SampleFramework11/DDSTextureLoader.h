//--------------------------------------------------------------------------------------
// File: DDSTextureLoader.h
//
// Function for loading a DDS texture and creating a Direct3D 11 runtime resource for it
//
// Note these functions are useful as a light-weight runtime loader for DDS files. For
// a full-featured DDS file reader, writer, and texture processing pipeline see
// the 'Texconv' sample and the 'DirectXTex' library.
//
// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//
// http://go.microsoft.com/fwlink/?LinkId=248926
// http://go.microsoft.com/fwlink/?LinkId=248929
//--------------------------------------------------------------------------------------

#pragma once

#include "PCH.h"

namespace SampleFramework11
{

HRESULT CreateDDSTextureFromMemory( _In_ ID3D11Device* d3dDevice,
                                    _In_bytecount_(ddsDataSize) const uint8_t* ddsData,
                                    _In_ size_t ddsDataSize,
                                    _Out_opt_ ID3D11Resource** texture,
                                    _Out_opt_ ID3D11ShaderResourceView** textureView,
                                    _In_ size_t maxsize = 0
                                  );

HRESULT CreateDDSTextureFromFile( _In_ ID3D11Device* d3dDevice,
                                  _In_z_ const wchar_t* szFileName,
                                  _Out_opt_ ID3D11Resource** texture,
                                  _Out_opt_ ID3D11ShaderResourceView** textureView,
                                  _In_ size_t maxsize = 0
                                );

}