using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Oxel
{
    [DebuggerDisplay("{X},{Y},{Z}")]
    [StructLayout(LayoutKind.Sequential)]
    public struct Vector3i
    {
        public int X;
        public int Y;
        public int Z;

        public Vector3i(int x, int y, int z)
        {
            X = x; Y = y; Z = z;
        }

        public double Length
        {
            get { return Math.Sqrt(X * X + Y * Y + Z * Z); }
        }

        public double LengthSquared
        {
            get { return X * X + Y * Y + Z * Z; }
        }

        public override bool Equals(object obj)
        {
            if (obj is Vector3i)
            {
                Vector3i other = (Vector3i)obj;
                return this.X == other.X && this.Y == other.Y && this.Z == other.Z;
            }
            return base.Equals(obj);
        }

        public static bool operator ==(Vector3i a, Vector3i b)
        {
            return a.X == b.X && a.Y == b.Y && a.Z == b.Z;
        }

        public static bool operator !=(Vector3i a, Vector3i b)
        {
            return !(a.X == b.X && a.Y == b.Y && a.Z == b.Z);
        }

        public static Vector3i operator -(Vector3i a)
        {
            return new Vector3i(-a.X, -a.Y, -a.Z);
        }

        public static Vector3i operator -(Vector3i a, Vector3i b)
        {
            return new Vector3i(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
        }

        public static Vector3i operator +(Vector3i a, Vector3i b)
        {
            return new Vector3i(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
        }

        public override int GetHashCode()
        {
            return X ^ Y ^ Z;
        }

        public override string ToString()
        {
            return ("Vector3i (" + X + "," + Y + "," + Z + ")");
        }
    }
}
