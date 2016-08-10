#include "precompiled.h"

#include <application.h>

#include <memory>


int32_t wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR    lpCmdLine, int       nCmdShow )
{
    hInstance;
    hPrevInstance;
    lpCmdLine;
    nCmdShow;

    auto parameters = app::application::create_parameters();

    parameters.m_adapter_index = 0;

    std::unique_ptr<app::application> app(new app::application(parameters));

    app->run();

    return 0;
}

