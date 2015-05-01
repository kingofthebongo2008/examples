#ifndef __GX_APPLICATION_H__
#define __GX_APPLICATION_H__

#include <algorithm>

#include <d3d11/d3d11_error.h>
#include <d3d11/d3d11_system.h>

#include <os/windows/wnd_application.h>

namespace gx
{
    class application : public os::windows::windowed_applicaion
    {
        private:
        typedef os::windows::windowed_applicaion base;

        public:
            application( HINSTANCE instance, const wchar_t* window_title ) : 
            base( instance, window_title )
            , m_context( d3d11::create_system_context ( get_window() ) )
            , m_occluded_by_another_window(false)
        {

        }

            application( const wchar_t* window_title  ) : 
            base( ::GetModuleHandle( nullptr ), window_title )
            , m_context( d3d11::create_system_context ( get_window() ) )
            , m_occluded_by_another_window(false)
        {

        }

        private:

        d3d11::system_context   m_context;
        bool                    m_occluded_by_another_window;

        protected:

        void    render_frame()
        {
            on_render_frame();
        }

        virtual void on_render_frame()
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

            throw_if_failed<exception>(m_context.m_swap_chain->GetDesc(&desc));
            throw_if_failed<exception>(m_context.m_swap_chain->ResizeBuffers(desc.BufferCount, width, height,  desc.BufferDesc.Format, desc.Flags));
        }

        private:

        virtual void on_render()
        {
            if (m_occluded_by_another_window)
            {
                HRESULT hr = m_context.m_swap_chain->Present(0, DXGI_PRESENT_TEST );

                if ( hr == S_OK)
                {
                    m_occluded_by_another_window = false;
                }

                if (hr != DXGI_STATUS_OCCLUDED)
                {
                    os::windows::throw_if_failed<d3d11::exception>(hr);
                }
            }
            else
            {
                render_frame();

                HRESULT hr = m_context.m_swap_chain->Present(0,0);

                if (hr == DXGI_STATUS_OCCLUDED)
                {
                    m_occluded_by_another_window = true;
                }
                else
                {
                    os::windows::throw_if_failed<d3d11::exception>(hr);
                }
            }
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

#endif