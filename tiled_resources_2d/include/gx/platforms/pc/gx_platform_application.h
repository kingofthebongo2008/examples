#pragma once

#include <algorithm>

#include <d3d11/d3d11.h>

#include <os/wnd_application.h>

namespace gx
{
    class platform_application : public os::windowed_applicaion
    {
        private:

        using base = os::windowed_applicaion;

        public:

        struct create_parameters : public os::windowed_applicaion::create_parameters
        {
            create_parameters()
            {
                m_instance = ::GetModuleHandle(nullptr);
            }
        };

        platform_application( const create_parameters& p ) : 
        base( p )
        , m_context( d3d11::create_system_context ( get_window(), p.m_adapter_index ) )
        {

        }

        protected:

        d3d11::system_context   m_context;

        void    render_frame()
        {
            on_render_frame();
        }

        void post_render_frame()
        {
            on_post_render_frame();
        }

        virtual void on_render_frame()
        {

        }

        virtual void on_post_render_frame()
        {

        }

        void resize_swap_chain( uint32_t width, uint32_t height)
        {
            using namespace d3d11;
            using namespace os::windows;

            DXGI_SWAP_CHAIN_DESC desc = {};

            //disable dxgi errors
            width = std::max(width, (uint32_t)(8));
            height = std::max(height, (uint32_t)(8));

            d3d11::throw_if_failed(m_context.m_swap_chain->GetDesc(&desc));
            throw_if_failed<exception>(m_context.m_swap_chain->ResizeBuffers(desc.BufferCount, width, height, desc.BufferDesc.Format, desc.Flags ));
        }

        protected:

        virtual void on_update()
        {

        }

        virtual void on_resize( uint32_t width, uint32_t height )
        {
            resize_swap_chain(width, height);
        }
    };
}
