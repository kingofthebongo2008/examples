//--------------------------------------------------------------------------------------
// Util.cpp
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "SDKmisc.h"
#include "Util.h"

WCHAR g_strAdapterName[128] = L"";

VOID SetAdapterInfoForShaderCompilation( const WCHAR* strAdapterName )
{
    wcscpy_s( g_strAdapterName, strAdapterName );
    _wcslwr_s( g_strAdapterName );
}

BOOL AppendMacro( D3D10_SHADER_MACRO* pMacros, const UINT MacroCount, const CHAR* strName, const CHAR* strValue )
{
    UINT i = 0;
    while( i < MacroCount )
    {
        if( pMacros[i].Name == NULL )
        {
            pMacros[i].Name = strName;
            pMacros[i].Definition = strValue;
            return TRUE;
        }
        ++i;
    }
    return FALSE;
}

VOID AppendHardwareSpecificMacros( D3D10_SHADER_MACRO* pMacros, const UINT MacroCount, const WCHAR* strAdapterName )
{
    if( strAdapterName == NULL )
    {
        return;
    }

    if( wcsstr( strAdapterName, L"radeon" ) != NULL )
    {
        AppendMacro( pMacros, MacroCount, "RoundUV", "1" );
    }
}

HRESULT LoadAndCompileShaderFile( const WCHAR* strFile, const CHAR* strEntryPoint, const CHAR* strProfile, ID3D10Blob** ppBlob )
{
    if( ppBlob == NULL )
    {
        return E_INVALIDARG;
    }
    *ppBlob = NULL;

    WCHAR strFullFileName[MAX_PATH];
    HRESULT hr = DXUTFindDXSDKMediaFileCch( strFullFileName, MAX_PATH, strFile );
    if( FAILED(hr) )
    {
        OutputDebugStringA( "Shader compile error: file \"" );
        OutputDebugString( strFile );
        OutputDebugStringA( "\" not found.\n" );
        assert( FALSE );
        return hr;
    }

    // Compile the shaders
    DWORD dwShaderFlags = D3DCOMPILE_ENABLE_BACKWARDS_COMPATIBILITY;
#if defined( DEBUG ) || defined( _DEBUG )
    // Set the D3DCOMPILE_DEBUG flag to embed debug information in the shaders.
    // Setting this flag improves the shader debugging experience, but still allows 
    // the shaders to be optimized and to run exactly the way they will run in 
    // the release configuration of this program.
    dwShaderFlags |= D3DCOMPILE_DEBUG | D3DCOMPILE_OPTIMIZATION_LEVEL0 | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

    D3D10_SHADER_MACRO Macros[16] =
    {
        { strProfile, "1" },
        { NULL, NULL }
    };

    AppendHardwareSpecificMacros( Macros, ARRAYSIZE(Macros), g_strAdapterName );

    ID3DBlob* pShaderBuffer = NULL;
    ID3DBlob* pErrorBuffer = NULL;
    hr = D3DX11CompileFromFile( strFullFileName, Macros, NULL, strEntryPoint, strProfile, dwShaderFlags, 0, NULL, &pShaderBuffer, &pErrorBuffer, NULL );
    if( FAILED(hr) )
    {
        OutputDebugStringA( "Shader compile error: " );
        OutputDebugStringA( (const CHAR*)pErrorBuffer->GetBufferPointer() );
        OutputDebugStringA( "\n" );
        assert( FALSE );
        return hr;
    }

    *ppBlob = pShaderBuffer;
    return S_OK;
}

ID3D11VertexShader* CompileVertexShader( ID3D11Device* pd3dDevice, const WCHAR* strFileName, const CHAR* strEntryPoint, ID3D10Blob** ppShaderBlob )
{
    const CHAR* strProfile = "vs_4_0";
    if( pd3dDevice->GetFeatureLevel() >= D3D_FEATURE_LEVEL_10_1 )
    {
        strProfile = "vs_4_1";
    }

    ID3D11VertexShader* pShader = NULL;
    ID3D10Blob* pShaderBlob = NULL;
    HRESULT hr = LoadAndCompileShaderFile( strFileName, strEntryPoint, strProfile, &pShaderBlob );
    if( SUCCEEDED(hr) )
    {
        hr = pd3dDevice->CreateVertexShader( pShaderBlob->GetBufferPointer(), pShaderBlob->GetBufferSize(), NULL, &pShader );
    }

    if( ppShaderBlob != NULL )
    {
        *ppShaderBlob = pShaderBlob;
    }
    else if( pShaderBlob != NULL )
    {
        pShaderBlob->Release();
    }

    return pShader;
}

ID3D11PixelShader* CompilePixelShader( ID3D11Device* pd3dDevice, const WCHAR* strFileName, const CHAR* strEntryPoint )
{
    const CHAR* strProfile = "ps_4_0";
    if( pd3dDevice->GetFeatureLevel() >= D3D_FEATURE_LEVEL_10_1 )
    {
        strProfile = "ps_4_1";
    }

    ID3D11PixelShader* pShader = NULL;
    ID3D10Blob* pShaderBlob = NULL;
    HRESULT hr = LoadAndCompileShaderFile( strFileName, strEntryPoint, strProfile, &pShaderBlob );
    if( SUCCEEDED(hr) )
    {
        hr = pd3dDevice->CreatePixelShader( pShaderBlob->GetBufferPointer(), pShaderBlob->GetBufferSize(), NULL, &pShader );
    }

    if( pShaderBlob != NULL )
    {
        pShaderBlob->Release();
    }

    return pShader;
}

ID3D11Buffer* CreateBuffer( ID3D11Device* pd3dDevice, UINT SizeBytes, UINT BindFlags, const VOID* pInitialData )
{
    D3D11_BUFFER_DESC BufferDesc;
    ZeroMemory( &BufferDesc, sizeof(BufferDesc) );
    BufferDesc.BindFlags = BindFlags;
    BufferDesc.ByteWidth = SizeBytes;

    if( pInitialData != NULL )
    {
        BufferDesc.CPUAccessFlags = 0;
        BufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
    }
    else
    {
        BufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        BufferDesc.Usage = D3D11_USAGE_DYNAMIC;
    }

    D3D11_SUBRESOURCE_DATA InitData;
    ZeroMemory( &InitData, sizeof(InitData) );

    D3D11_SUBRESOURCE_DATA* pActiveInitData = NULL;
    if( pInitialData != NULL )
    {
        InitData.pSysMem = pInitialData;
        InitData.SysMemPitch = SizeBytes;
        pActiveInitData = &InitData;
    }

    ID3D11Buffer* pBuffer = NULL;
    HRESULT hr = pd3dDevice->CreateBuffer( &BufferDesc, pActiveInitData, &pBuffer );
    assert( SUCCEEDED(hr) && pBuffer != NULL ); hr;

    return pBuffer;
}

ID3D11Buffer* CreateConstantBuffer( ID3D11Device* pd3dDevice, UINT SizeBytes, const VOID* pInitialData )
{
    return CreateBuffer( pd3dDevice, SizeBytes, D3D11_BIND_CONSTANT_BUFFER, pInitialData );
}

ID3D11Buffer* CreateVertexBuffer( ID3D11Device* pd3dDevice, UINT SizeBytes, const VOID* pInitialData )
{
    return CreateBuffer( pd3dDevice, SizeBytes, D3D11_BIND_VERTEX_BUFFER, pInitialData );
}

ID3D11Buffer* CreateIndexBuffer( ID3D11Device* pd3dDevice, UINT SizeBytes, const VOID* pInitialData )
{
    return CreateBuffer( pd3dDevice, SizeBytes, D3D11_BIND_INDEX_BUFFER, pInitialData );
}

inline UINT CountBits( UINT Value )
{
    UINT BitCount = 0;
    while( Value != 0 )
    {
        Value &= ( Value - 1 );
        ++BitCount;
    }
    return BitCount;
}

VOID GetCoreAndHWThreadCount( UINT* pCoreCount, UINT* pHWThreadCount )
{
    if( pCoreCount != NULL )
    {
        *pCoreCount = 1;
    }
    if( pHWThreadCount != NULL )
    {
        *pHWThreadCount = 1;
    }

    SYSTEM_LOGICAL_PROCESSOR_INFORMATION Processors[64];
    ZeroMemory( Processors, sizeof(Processors) );

    DWORD BufferSize = sizeof(Processors);
    BOOL Result = GetLogicalProcessorInformation( Processors, &BufferSize );

    if( !Result )
    {
        return;
    }

    UINT ProcessorCount = 0;
    UINT HWThreadCount = 0;
    UINT InfoCount = ARRAYSIZE( Processors );
    InfoCount = min( InfoCount, BufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) );

    for( UINT i = 0; i < InfoCount; ++i )
    {
        if( Processors[i].Relationship == RelationProcessorCore )
        {
            ++ProcessorCount;
            HWThreadCount += CountBits( (UINT)Processors[i].ProcessorMask );
        }
    }

    if( pCoreCount != NULL )
    {
        *pCoreCount = ProcessorCount;
    }
    if( pHWThreadCount != NULL )
    {
        *pHWThreadCount = HWThreadCount;
    }
}
