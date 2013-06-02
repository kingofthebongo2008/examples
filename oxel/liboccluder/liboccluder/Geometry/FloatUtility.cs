using System;
using OpenTK;

namespace Oxel
{
    /*!
    **
    ** Copyright (c) 2009 by John W. Ratcliff mailto:jratcliffscarab@gmail.com
    **
    ** The MIT license:
    **
    ** Permission is hereby granted, FREE of charge, to any person obtaining a copy
    ** of this software and associated documentation files (the "Software"), to deal
    ** in the Software without restriction, including without limitation the rights
    ** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    ** copies of the Software, and to permit persons to whom the Software is furnished
    ** to do so, subject to the following conditions:
    **
    ** The above copyright notice and this permission notice shall be included in all
    ** copies or substantial portions of the Software.

    ** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    ** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    ** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    ** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    ** WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    ** CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    */
    public class FloatUtility
    {
        // Reference, from Stan Melax in Game Gems I
        //  Quaternion q;
        //  vector3 c = CrossProduct(v0,v1);
        //  REAL   d = DotProduct(v0,v1);
        //  REAL   s = (REAL)sqrt((1+d)*2);
        //  q.x = c.x / s;
        //  q.y = c.y / s;
        //  q.z = c.z / s;
        //  q.w = s /2.0f;
        //  return q;
        public static Quaternion RotationArc(Vector3 v0, Vector3 v1)
        {
            Vector3 cross = Vector3.Cross(v0, v1);
            float d = Vector3.Dot(v0, v1);
            float s = (float)Math.Sqrt((1.0f + d) * 2.0f);
            float recip = 1.0f / s;

            Quaternion quat = new Quaternion();
            quat.X = cross.X * recip;
            quat.Y = cross.Y * recip;
            quat.Z = cross.Z * recip;
            quat.W = s * 0.5f;
            return quat;
        }
    }
}
