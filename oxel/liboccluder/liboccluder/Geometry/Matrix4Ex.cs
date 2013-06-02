using OpenTK;

namespace Oxel
{
    public static class Matrix4Ex
    {
        public static void CreateFromQuaternion(ref Quaternion q, ref Matrix4 m)
        {
            m = Matrix4.Identity;

            float X = q.X;
            float Y = q.Y;
            float Z = q.Z;
            float W = q.W;

            float xx = X * X;
            float xy = X * Y;
            float xz = X * Z;
            float xw = X * W;
            float yy = Y * Y;
            float yz = Y * Z;
            float yw = Y * W;
            float zz = Z * Z;
            float zw = Z * W;

            m.M11 = 1 - 2 * (yy + zz);
            m.M21 = 2 * (xy - zw);
            m.M31 = 2 * (xz + yw);
            m.M12 = 2 * (xy + zw);
            m.M22 = 1 - 2 * (xx + zz);
            m.M32 = 2 * (yz - xw);
            m.M13 = 2 * (xz - yw);
            m.M23 = 2 * (yz + xw);
            m.M33 = 1 - 2 * (xx + yy);
        }

        /// <summary>
        /// Build a rotation matrix from the specified quaternion.
        /// </summary>
        /// <param name="q">Quaternion to translate.</param>
        /// <returns>A matrix instance.</returns>
        public static Matrix4 CreateFromQuaternion(ref Quaternion q)
        {
            Matrix4 result = Matrix4.Identity;

            float X = q.X;
            float Y = q.Y;
            float Z = q.Z;
            float W = q.W;

            float xx = X * X;
            float xy = X * Y;
            float xz = X * Z;
            float xw = X * W;
            float yy = Y * Y;
            float yz = Y * Z;
            float yw = Y * W;
            float zz = Z * Z;
            float zw = Z * W;

            result.M11 = 1 - 2 * (yy + zz);
            result.M21 = 2 * (xy - zw);
            result.M31 = 2 * (xz + yw);
            result.M12 = 2 * (xy + zw);
            result.M22 = 1 - 2 * (xx + zz);
            result.M32 = 2 * (yz - xw);
            result.M13 = 2 * (xz - yw);
            result.M23 = 2 * (yz + xw);
            result.M33 = 1 - 2 * (xx + yy);
            return result;
        }
    }
}