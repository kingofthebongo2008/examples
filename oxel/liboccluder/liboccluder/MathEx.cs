using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Oxel
{
    public static class MathEx
    {
        public static int Clamp(int t, int min, int max)
        {
            if (t < min)
                return min;
            if (t > max)
                return max;
            return t;
        }

        public static float Clamp(float t, float min, float max)
        {
            if (t < min)
                return min;
            if (t > max)
                return max;
            return t;
        }
    }
}
