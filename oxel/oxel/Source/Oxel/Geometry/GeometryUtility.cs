using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenTK;

namespace Oxel
{
    public class GeometryUtility
    {        
        public static Frustum ExtractFrustum(Matrix4 viewProjMatrix)
        {
            Frustum frustum = new Frustum();

            Plane left = new Plane();
            left.A = viewProjMatrix.M14 + viewProjMatrix.M11;
            left.B = viewProjMatrix.M24 + viewProjMatrix.M21;
            left.C = viewProjMatrix.M34 + viewProjMatrix.M31;
            left.D = viewProjMatrix.M44 + viewProjMatrix.M41;
            left.Normalize();
            frustum.Left = left;

            Plane right = new Plane();
            right.A = viewProjMatrix.M14 - viewProjMatrix.M11;
            right.B = viewProjMatrix.M24 - viewProjMatrix.M21;
            right.C = viewProjMatrix.M34 - viewProjMatrix.M31;
            right.D = viewProjMatrix.M44 - viewProjMatrix.M41;
            right.Normalize();
            frustum.Right = right;

            Plane top = new Plane();
            top.A = viewProjMatrix.M14 - viewProjMatrix.M12;
            top.B = viewProjMatrix.M24 - viewProjMatrix.M22;
            top.C = viewProjMatrix.M34 - viewProjMatrix.M32;
            top.D = viewProjMatrix.M44 - viewProjMatrix.M42;
            top.Normalize();
            frustum.Top = top;

            Plane bottom = new Plane();
            bottom.A = viewProjMatrix.M14 + viewProjMatrix.M12;
            bottom.B = viewProjMatrix.M24 + viewProjMatrix.M22;
            bottom.C = viewProjMatrix.M34 + viewProjMatrix.M32;
            bottom.D = viewProjMatrix.M44 + viewProjMatrix.M42;
            bottom.Normalize();
            frustum.Bottom = bottom;

            Plane near = new Plane();
            near.A = viewProjMatrix.M13;
            near.B = viewProjMatrix.M23;
            near.C = viewProjMatrix.M33;
            near.D = viewProjMatrix.M43;
            near.Normalize();
            frustum.Near = near;

            Plane far = new Plane();
            far.A = viewProjMatrix.M14 - viewProjMatrix.M13;
            far.B = viewProjMatrix.M24 - viewProjMatrix.M23;
            far.C = viewProjMatrix.M34 - viewProjMatrix.M33;
            far.D = viewProjMatrix.M44 - viewProjMatrix.M43;
            far.Normalize();
            frustum.Far = far;

            return frustum;
        }
    }
}
