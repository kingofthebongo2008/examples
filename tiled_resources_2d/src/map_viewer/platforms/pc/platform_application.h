#pragma once

#include <base_application.h>

namespace app
{
    class platform_application : public base_application
    {
        private:

        using base = base_application;

        public:

        struct create_parameters : public base::create_parameters
        {


        };

        platform_application( const create_parameters& p ) : base( p )
        , m_occluded_by_another_window(false)
        {

        }

        private:
        bool                    m_occluded_by_another_window;

        
        protected:

        void on_render() override
        {
                render_frame();
                
                present();

                post_render_frame();
        }

        void on_post_render_frame() override
        {

        }

        void present()
        {
            if (m_occluded_by_another_window)
            {
                HRESULT hr = m_context.m_swap_chain->Present(0, DXGI_PRESENT_TEST);

                if (hr == S_OK)
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

                HRESULT hr = m_context.m_swap_chain->Present(0, 0);

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
       
        /*
        void on_resize(uint32_t w, uint32_t h) override
        {
            using namespace d3d11;
            using namespace os::windows;

            DXGI_SWAP_CHAIN_DESC desc = {};

            //disable dxgi errors
            w = std::max(w, (uint32_t)(8));
            h = std::max(h, (uint32_t)(8));

            throw_if_failed<exception>(m_context.m_swap_chain->GetDesc(&desc));
            throw_if_failed<exception>(m_context.m_swap_chain->ResizeBuffers(desc.BufferCount, w, h, desc.BufferDesc.Format, desc.Flags));
        }
        */
    };
}
