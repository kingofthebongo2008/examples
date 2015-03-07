#include "precompiled.h"

#include <assert.h>
#include <fstream>
#include <string>

#include "composer_application.h"


#include <os/windows/com_initializer.h>

int32_t APIENTRY _tWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR    lpCmdLine, int       nCmdShow)
{
    os::windows::com_initializer com ( os::windows::apartment_threaded) ;
    using namespace coloryourway::composer;
    auto app = new sample_application(L"Composer");
    
    auto result = app->run();

    delete app;

    return 0;
}




