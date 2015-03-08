#ifndef __composer_renderable_h__
#define __composer_renderable_h__

#include <util/util_noncopyable.h>

#include "composer_render_context.h"

namespace coloryourway
{
    namespace composer
    {
        class renderable : private util::noncopyable
        {
            public:

            virtual ~renderable()
            {

            }

            void draw( render_context& c )
            {
                on_draw( c );
            }

            private:
                virtual void on_draw( render_context& c ) = 0;
        };
    }
}


#endif
