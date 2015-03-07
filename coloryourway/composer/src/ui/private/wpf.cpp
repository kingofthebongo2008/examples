#include "precompiled.h"


#include <ui/wpf.h>


#include <cstdint>
#include <gcroot.h>

#include "wpfpage.h"

using namespace System;
using namespace System::Windows;
using namespace System::Windows::Documents;
using namespace System::Threading;
using namespace System::Windows::Controls;
using namespace System::Windows::Media;

void WPFButtonClicked(Object ^sender, MyPageEventArgs ^args);

void WPFButtonClicked(Object ^sender, MyPageEventArgs ^args)
{
    if (args->IsOK) //display data if OK button was clicked
    {
        
    }
    else
    {

    }
}



namespace coloryourway
{
    namespace composer
    {
        namespace ui
        {

            public ref class WPFPageHost
            {
            public:
                WPFPageHost();
                static WPFPage^ hostedPage;
                //initial property settings
                static System::Windows::Media::Brush^ initBackBrush;
                static System::Windows::Media::Brush^ initForeBrush;
                static System::Windows::Media::FontFamily^ initFontFamily;
                static System::Windows::FontStyle initFontStyle;
                static System::Windows::FontWeight initFontWeight;
                static double initFontSize;
            };

            WPFPageHost::WPFPageHost(){}



            UI_DLL HWND wpf_create_source(HWND parent)
            {
                RECT r;
                ::GetClientRect(parent, &r);

                auto width = 200;// r.right - r.left;
                auto height = 200;// r.bottom - r.top;
                System::Windows::Interop::HwndSourceParameters^ sourceParams = gcnew System::Windows::Interop::HwndSourceParameters( "hi" );
                sourceParams->PositionX = 500;// 5;
                sourceParams->PositionY = 500;// 5;
                sourceParams->Height = height;
                sourceParams->Width = width;
                sourceParams->ParentWindow = IntPtr(parent);
                sourceParams->WindowStyle = WS_VISIBLE | WS_OVERLAPPED; // style
                System::Windows::Interop::HwndSource^ source = gcnew System::Windows::Interop::HwndSource(*sourceParams);
                WPFPage ^myPage = gcnew WPFPage(width, height);
                //Assign a reference to the WPF page and a set of UI properties to a set of static properties in a class 
                //that is designed for that purpose.
                WPFPageHost::hostedPage = myPage;
                WPFPageHost::initBackBrush = myPage->Background;
                WPFPageHost::initFontFamily = myPage->DefaultFontFamily;
                WPFPageHost::initFontSize = myPage->DefaultFontSize;
                WPFPageHost::initFontStyle = myPage->DefaultFontStyle;
                WPFPageHost::initFontWeight = myPage->DefaultFontWeight;
                WPFPageHost::initForeBrush = myPage->DefaultForeBrush;
                myPage->OnButtonClicked += gcnew WPFPage::ButtonClickHandler(WPFButtonClicked);
                source->RootVisual = myPage;
                return (HWND)source->Handle.ToPointer();
            }

            UI_DLL void wpf_destroy_source(HWND source)
            {

            }

        }
    }
}
