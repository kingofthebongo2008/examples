using System;
using OpenTK;

namespace Oxel
{
    public static class QuaternionEx
    {
        public static void CreateFromMatrix(ref Matrix4 m, ref Quaternion q)
        {
            float trace = 1 + m.M11 + m.M22 + m.M33;
            float S = 0;
            float X = 0;
            float Y = 0;
            float Z = 0;
            float W = 0;

            if (trace > 0.0000001)
            {
                S = (float)Math.Sqrt(trace) * 2;
                X = (m.M23 - m.M32) / S;
                Y = (m.M31 - m.M13) / S;
                Z = (m.M12 - m.M21) / S;
                W = 0.25f * S;
            }
            else
            {
                if (m.M11 > m.M22 && m.M11 > m.M33)
                {
                    // Column 0: 
                    S = (float)Math.Sqrt(1.0 + m.M11 - m.M22 - m.M33) * 2;
                    X = 0.25f * S;
                    Y = (m.M12 + m.M21) / S;
                    Z = (m.M31 + m.M13) / S;
                    W = (m.M23 - m.M32) / S;
                }
                else if (m.M22 > m.M33)
                {
                    // Column 1: 
                    S = (float)Math.Sqrt(1.0 + m.M22 - m.M11 - m.M33) * 2;
                    X = (m.M12 + m.M21) / S;
                    Y = 0.25f * S;
                    Z = (m.M23 + m.M32) / S;
                    W = (m.M31 - m.M13) / S;
                }
                else
                {
                    // Column 2:
                    S = (float)Math.Sqrt(1.0 + m.M33 - m.M11 - m.M22) * 2;
                    X = (m.M31 + m.M13) / S;
                    Y = (m.M23 + m.M32) / S;
                    Z = 0.25f * S;
                    W = (m.M12 - m.M21) / S;
                }
            }
            q = new Quaternion(X, Y, Z, W);
        }

        /// <summary>
        /// Build a quaternion from the specified rotation matrix.
        /// </summary>
        /// <param name="m">Matrix to translate.</param>
        /// <returns>A quaternion</returns>
        public static Quaternion CreateFromMatrix(ref Matrix4 m)
        {
            Quaternion q;

            float trace = 1 + m.M11 + m.M22 + m.M33;
            float S = 0;
            float X = 0;
            float Y = 0;
            float Z = 0;
            float W = 0;

            if (trace > 0.0000001)
            {
                S = (float)Math.Sqrt(trace) * 2;
                X = (m.M23 - m.M32) / S;
                Y = (m.M31 - m.M13) / S;
                Z = (m.M12 - m.M21) / S;
                W = 0.25f * S;
            }
            else
            {
                if (m.M11 > m.M22 && m.M11 > m.M33)
                {
                    // Column 0: 
                    S = (float)Math.Sqrt(1.0 + m.M11 - m.M22 - m.M33) * 2;
                    X = 0.25f * S;
                    Y = (m.M12 + m.M21) / S;
                    Z = (m.M31 + m.M13) / S;
                    W = (m.M23 - m.M32) / S;
                }
                else if (m.M22 > m.M33)
                {
                    // Column 1: 
                    S = (float)Math.Sqrt(1.0 + m.M22 - m.M11 - m.M33) * 2;
                    X = (m.M12 + m.M21) / S;
                    Y = 0.25f * S;
                    Z = (m.M23 + m.M32) / S;
                    W = (m.M31 - m.M13) / S;
                }
                else
                {
                    // Column 2:
                    S = (float)Math.Sqrt(1.0 + m.M33 - m.M11 - m.M22) * 2;
                    X = (m.M31 + m.M13) / S;
                    Y = (m.M23 + m.M32) / S;
                    Z = 0.25f * S;
                    W = (m.M12 - m.M21) / S;
                }
            }
            q = new Quaternion(X, Y, Z, W);
            return q;
        }
    }
}