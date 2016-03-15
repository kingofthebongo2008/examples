#pragma once

#include "DeviceResources.h"

namespace TiledResources
{
    class Extras
    {
    public:
        Extras(const std::shared_ptr<DeviceResources>& deviceResources);
        void RenderAtmosphere();
    private:
        std::shared_ptr<DeviceResources> m_deviceResources;
    };
}
