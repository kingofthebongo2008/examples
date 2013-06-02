using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenTK;

namespace Oxel
{
    public static class Vector3Ex
    {
        public static bool AlmostEquals(Vector3 a, Vector3 b, float epsilon = 0.00001f)
        {
            return AlmostEquals(ref a, ref b, epsilon);
        }

        public static bool AlmostEquals(ref Vector3 a, ref Vector3 b, float epsilon = 0.00001f)
        {
            return (a.X >= (b.X - epsilon) && a.X <= (b.X + epsilon)) &&
                   (a.Y >= (b.Y - epsilon) && a.Y <= (b.Y + epsilon)) &&
                   (a.Z >= (b.Z - epsilon) && a.Z <= (b.Z + epsilon));
        }

        public static Vector3 GetRight(Vector3 Normal)
        {
            // A single normal is not enough to determine this problem -- you also need a "right" for the coordinate system.
            // You can calculate it as such:
            if (Math.Abs(Normal.X) > Math.Abs(Normal.Y))
                return Vector3.UnitY;
            return Vector3.UnitX;
        }

        public static void GetBasisVectors(Vector3 Normal, Vector3 Right, out Vector3 U, out Vector3 V)
        {
            // Once you have a normal and a "right," you can calculate an orthonormal basis:
            V = Vector3.Normalize(Vector3.Cross(Right, Normal));
            U = Vector3.Cross(Normal, V);
        }

        public static Vector2 Calc2DPoint(Vector3 p3d, Vector3 U, Vector3 V)
        {
            // And once you have your orthonormal basis, you can simply classify each point by dot products:
            return new Vector2(Vector3.Dot(p3d, U), Vector3.Dot(p3d, V));
        }

        public static bool IsPointInsideLineSegment(Vector3 line0, Vector3 line1, Vector3 point, float epsilon)
        {
            if (AreCollinear(line0, line1, line0, point, epsilon))
            {
                if (Vector3Ex.AlmostEquals(ref line0, ref point, epsilon) || Vector3Ex.AlmostEquals(ref line1, ref point, epsilon))
                    return false;

                // arccos(-1) = PI, this will test if the lines point toward eachother.
                float dot = (float)Vector3.Dot(Vector3.Normalize(point - line0), Vector3.Normalize(point - line1));
                return (dot >= (-1 + -epsilon) && dot <= (-1 + epsilon));
            }

            return false;
        }

        public static bool AreCollinear(Vector3 v0, Vector3 v1, Vector3 ov0, Vector3 ov1, float epsilon)
        {
            float dot = (float)Math.Abs(Vector3.Dot(Vector3.Normalize(v1 - v0), Vector3.Normalize(ov1 - ov0)));
            return (dot >= (1 + -epsilon) && dot <= (1 + epsilon));
        }
    }
}
