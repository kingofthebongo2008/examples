#include "precompiled.h"
#include <cstdint>

#include <gx/gx_default_application.h>
#include <gx/gx_view_port.h>

#include <sys/sys_profile_timer.h>
#include "d3dx12.h"

class sample_application : public gx::default_application
{
    typedef gx::default_application base;

public:

    sample_application(const wchar_t* window_title) : base(window_title)
        , m_elapsed_update_time(0.0)
        , m_rtv_heap( m_context.m_device.get(), 3 )
        , m_rtv_cpu_heap( m_rtv_heap.create_cpu_heap() )
    {
        using namespace os::windows;
        m_wait_back_buffer = CreateEventEx(nullptr, FALSE, FALSE, EVENT_ALL_ACCESS);

        auto device = this->m_context.m_device.get();

        m_wait_back_buffer_fence = d3d12x::create_fence( device );
        m_frame_index = 1;

        m_command_allocator = d3d12x::create_command_allocator(device, D3D12_COMMAND_LIST_TYPE_DIRECT );
        m_command_list = d3d12x::create_graphics_command_list(device, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_command_allocator, nullptr);
        m_command_queue = m_context.m_direct_command_queue;

        throw_if_failed<d3d12::exception>(m_command_list->Close());

        throw_if_failed<d3d12::exception>(m_context.m_swap_chain->GetBuffer(0, __uuidof(ID3D12Resource), (void**)&m_render_target));

        device->CreateRenderTargetView(m_render_target.get(), nullptr, m_rtv_cpu_heap(0) );

    }

    ~sample_application()
    {
        CloseHandle( m_wait_back_buffer );
    }

    void shutdown()
    {
        wait_gpu_for_back_buffer();
    }

protected:

    void resize_swap_chain(uint32_t width, uint32_t height)
    {
        using namespace d3d12;
        using namespace os::windows;

        DXGI_SWAP_CHAIN_DESC desc = {};

        //disable dxgi errors
        width = std::max(width, (uint32_t)(8));
        height = std::max(height, (uint32_t)(8));

        const uint32_t node_creation_mask[3] = { 0, 0, 0 };

        IUnknown* const queue[] = { m_command_queue.get() };

        throw_if_failed<exception>(m_context.m_swap_chain->GetDesc(&desc));
        throw_if_failed<exception>(m_context.m_swap_chain->ResizeBuffers1(desc.BufferCount, width, height, desc.BufferDesc.Format, desc.Flags, 0, queue ));
    }


    virtual void on_resize(uint32_t width, uint32_t height) override
    {
        
    }

    virtual void on_render_scene()
    {
        using namespace os::windows;

        throw_if_failed<d3d12::exception>(m_command_allocator->Reset() );
        throw_if_failed<d3d12::exception>(m_command_list->Reset(m_command_allocator, nullptr ));
        

        D3D12_RESOURCE_BARRIER b = {};
        b.Transition.pResource = m_render_target;
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        b.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

        

        
        //m_command_list->ResourceBarrier(1, &b);
        m_command_list->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_render_target, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET) );

        throw_if_failed<d3d12::exception>(m_command_list->Close());
        //m_command_list->OMSetRenderTargets(1, &m_rtv_cpu_heap(0), true, nullptr);


        float clear_color[] = { 0.0f, 0.2f, 0.4f, 1.0f };

        //m_command_list->ClearRenderTargetView(m_rtv_cpu_heap(0), clear_color, 0, nullptr);



        b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        b.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;

        //m_command_list->ResourceBarrier(1, &b);
        m_command_list->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_render_target, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT ));

        throw_if_failed<d3d12::exception>(m_command_list->Close());


    }

    void render_scene()
    {
        on_render_scene();
    }

    void wait_gpu_for_back_buffer()
    {
        const auto frame_index = m_frame_index;

        //signal for stop rendering
        os::windows::throw_if_failed<d3d12::exception>(m_wait_back_buffer_fence->Signal(frame_index));

        m_frame_index++;

        //wait for the gpu to finish frame
        if ( m_wait_back_buffer_fence->GetCompletedValue() < frame_index - 1 )
        {
            m_wait_back_buffer_fence->SetEventOnCompletion( frame_index - 1 , this->m_wait_back_buffer );
            WaitForSingleObject(m_wait_back_buffer, INFINITE);
        }
    }

    virtual void on_update_scene()
    {

    }

    void update_scene()
    {
        on_update_scene();
    }

    void on_update() override
    {
        sys::profile_timer timer;

        update_scene();

        //Measure the update time and pass it to the render function
        m_elapsed_update_time = timer.milliseconds();
    }

    void on_render_frame() override
    {
        sys::profile_timer timer;
        on_render_scene();
    }

    void on_post_render_frame() override
    {
        wait_gpu_for_back_buffer();
    }

    void on_resize(uint32_t width, uint32_t height)
    {

        base::on_resize(width, height);

        resize_swap_chain(width, height);

        //Reset view port dimensions
        m_view_port.set_dimensions(width, height);
    }

protected:

    gx::view_port                           m_view_port;
    double                                  m_elapsed_update_time;

    HANDLE                                  m_wait_back_buffer;
    d3d12::fence                            m_wait_back_buffer_fence;
    uint32_t                                m_frame_index;

    
    d3d12::command_allocator                m_command_allocator;
    d3d12::graphics_command_list            m_command_list;
    d3d12::command_queue                    m_command_queue;

    d3d12::resource                         m_render_target;

    d3d12x::descriptor_heap< D3D12_DESCRIPTOR_HEAP_TYPE_RTV> m_rtv_heap;
    d3d12x::cpu_descriptor_heap                              m_rtv_cpu_heap;

    uint32_t                                m_index_last_swap_buffer = 0;
    uint32_t                                m_swap_buffer_count = 3;
};

int32_t wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR    lpCmdLine, int       nCmdShow )
{
    auto app = new sample_application(L"qcrm");

    app->run();
    app->shutdown();
    
    delete app;

    return 0;
}




