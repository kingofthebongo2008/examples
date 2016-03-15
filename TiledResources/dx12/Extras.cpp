#include "pch.h"
#include "DirectXHelper.h"
#include "SampleSettings.h"
#include "DeviceResources2.h"
#include "Extras.h"

using namespace TiledResources;

using namespace concurrency;
using namespace DirectX;
using namespace Windows::Foundation;

Extras::Extras(const std::shared_ptr<DeviceResources>& deviceResources) :
    m_deviceResources(deviceResources)
{


}

void Extras::RenderAtmosphere()
{

}
