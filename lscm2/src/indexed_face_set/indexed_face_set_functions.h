#pragma once

#include "indexed_face_set_mesh.h"

namespace lscm
{
    namespace indexed_face_set
    {
        struct vector
        {
            float x;
            float y;
            float z;
            float w;
        };
                
        inline vector( vertex a, vertex b )
        {
            vector r = { b.x - a.x, b.y - a.y, b.z - a.z, b.w - a.w };
            return r;
        }

        inline float area( vertex a, const vertex b, const vertex c)
        {
            
        }
    };
}







