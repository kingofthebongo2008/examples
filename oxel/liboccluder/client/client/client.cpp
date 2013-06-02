// client.cpp : Defines the entry point for the console application.
//
#include "precompiled.h"
#include <cstdint>
#include <exception>

#include <atlbase.h>
#include <atlcom.h>
#include <atlsafe.h>

#include <metahost.h>
#pragma comment(lib, "mscoree.lib")

#import "liboccluder.tlb"

namespace client
{
    class com_exception : public std::exception
    {

    };

    void v(HRESULT hr)
    {
        if (hr != S_OK)
        {
            throw com_exception();
        }
    }

    class ScopedRuntime
    {
        public:

        ScopedRuntime( CComPtr<ICLRRuntimeHost > runtime ) : m_Runtime(runtime)
        {
            v( m_Runtime->Start() );
        }

        ~ScopedRuntime()
        {
            v ( m_Runtime->Stop() );
        }

        CComPtr<ICLRRuntimeHost > m_Runtime;
    };

    class ScopedInitialize
    {
        public:
        ScopedInitialize()
        {
            v( CoInitializeEx(NULL, COINIT_MULTITHREADED) );
        }

        ~ScopedInitialize()
        {
            CoUninitialize();
        }
    };

    class HostControlImpl : public CComObjectRoot, public IHostControl 
    {
        BEGIN_COM_MAP(HostControlImpl)
            COM_INTERFACE_ENTRY(IHostControl)
        END_COM_MAP()

        public:

        ~HostControlImpl()
        {
        }

        STDMETHOD(GetHostManager)(  /* [in] */ REFIID riid,/* [out] */ void **ppObject)
        {
            ppObject = 0;
            return E_NOINTERFACE;
        }
        
        STDMETHOD(SetAppDomainManager)( /* [in] */ DWORD dwAppDomainID, /* [in] */ IUnknown *domainManager)
        {
            return domainManager->QueryInterface<liboccluder::IOccluderGeneration>(&metaHost);;
        }

        CComPtr<liboccluder::IOccluderGeneration>           metaHost;
    };

    class AtlModule : public CAtlExeModuleT<AtlModule>
    {

    };
}

using namespace client;

int _tmain(int argc, _TCHAR* argv[])
{
    ScopedInitialize comInitializer;
    AtlModule  atlModule;

    CComPtr<ICLRMetaHost>           metaHost;
    CComPtr<ICLRMetaHostPolicy>     metaHostPolicy;
    CComPtr<ICLRDebugging>          clrDebugging;

    v ( CLRCreateInstance (CLSID_CLRMetaHost, IID_ICLRMetaHost, (void**) &metaHost) );
    v ( CLRCreateInstance (CLSID_CLRMetaHostPolicy, IID_ICLRMetaHostPolicy, (void**) &metaHostPolicy) );
    v ( CLRCreateInstance (CLSID_CLRDebugging, IID_ICLRDebugging, (void**) &clrDebugging) );

    CComPtr<ICLRRuntimeInfo> runtimeInfo;

    v( metaHost->GetRuntime(_T("v4.0.30319"), IID_ICLRRuntimeInfo, (void**) &runtimeInfo ) ); 

    CComPtr<ICLRRuntimeHost > runtime;

    v( runtimeInfo->GetInterface( CLSID_CLRRuntimeHost, IID_ICLRRuntimeHost, (void**) &runtime ) );

    CComObject<HostControlImpl>* hostControl = 0; 

    v ( CComObject<HostControlImpl>::CreateInstance( &hostControl) );
    v( runtime->SetHostControl( hostControl ) ) ;

    HRESULT hr;
    DWORD result = 0;

    CComPtr<ICLRControl > clrControl;
    v( runtime->GetCLRControl(&clrControl ) );
    v( clrControl->SetAppDomainManagerType(_T("liboccluder, Version=1.0.0.0, Culture=neutral, PublicKeyToken=f60e4dbe816ac5fb, processorArchitecture=MSIL"),_T("liboccluder.AppDomainManager") ) );

    ScopedRuntime scopedRuntime(runtime);

    CComPtr<liboccluder::IOccluderGeneration>    generator(hostControl->metaHost);
    liboccluder::Vertex3 m[3];

    m[0].x = 0.0f;
    m[0].y = 1.0f;
    m[0].z = 20.0f;

    m[1].x = 1.0f;
    m[1].y = 2.0f;
    m[1].z = 30.0f;

    m[2].x = 1.0f;
    m[2].y = 2.0f;
    m[2].z = 30.0f;


    short sh[3];

    sh[0] = 0;
    sh[1] = 1;
    sh[2] = 2;

    liboccluder::Vertex3*  verticesOut = 0;
    std::int64_t           verticesOutSize = 0;
    short*                 indicesOut = 0 ;  
    std::int64_t           indicesOutSize = 0;
    
    generator->Compute(&m[0], 3, &sh[0], 3, &verticesOut, &verticesOutSize, &indicesOut, &indicesOutSize );

    CoTaskMemFree(verticesOut);
    CoTaskMemFree(indicesOut);
    
	return 0;
}

