//--------------------------------------------------------------------------------------
// Util.h
//
// Provides a set of utility functions for the sample, including D3D11 resource creation
// helpers, and CPU core/HW thread counts.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once

#include <windows.h>
#include <d3d11.h>
#include <d3dx11.h>

VOID SetAdapterInfoForShaderCompilation( const WCHAR* strAdapterName );
ID3D11VertexShader* CompileVertexShader( ID3D11Device* pd3dDevice, const WCHAR* strFileName, const CHAR* strEntryPoint, ID3D10Blob** ppShaderBlob = NULL );
ID3D11PixelShader* CompilePixelShader( ID3D11Device* pd3dDevice, const WCHAR* strFileName, const CHAR* strEntryPoint );

ID3D11Buffer* CreateConstantBuffer( ID3D11Device* pd3dDevice, UINT SizeBytes, const VOID* pInitialData = NULL );
ID3D11Buffer* CreateVertexBuffer( ID3D11Device* pd3dDevice, UINT SizeBytes, const VOID* pInitialData = NULL );
ID3D11Buffer* CreateIndexBuffer( ID3D11Device* pd3dDevice, UINT SizeBytes, const VOID* pInitialData = NULL );

VOID GetCoreAndHWThreadCount( UINT* pCoreCount, UINT* pHWThreadCount );

