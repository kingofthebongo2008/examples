using System;
using System.Diagnostics;
using OpenTK;

namespace Oxel
{
    [DebuggerDisplay("({MinX} {MinY} {MinZ}) ({MaxX} {MaxY} {MaxZ})")]
    public class AABBf
    {
        public float MinX = float.MaxValue;
        public float MaxX = float.MinValue;

        public float MinY = float.MaxValue;
        public float MaxY = float.MinValue;

        public float MinZ = float.MaxValue;
        public float MaxZ = float.MinValue;

        public AABBf()
        {
        }

        public AABBf(float minX, float minY, float minZ, float maxX, float maxY, float maxZ)
        {
            MinX = minX;
            MaxX = maxX;
            
            MinY = minY;
            MaxY = maxY;
            
            MinZ = minZ;
            MaxZ = maxZ;
        }

        public void Clear()
        {
            MinX = float.MaxValue;
            MaxX = float.MinValue;

            MinY = float.MaxValue;
            MaxY = float.MinValue;

            MinZ = float.MaxValue;
            MaxZ = float.MinValue;
        }

        public void Add(Vector3 point)
        {
            MinX = Math.Min(MinX, point.X);
            MaxX = Math.Max(MaxX, point.X);
            
            MinY = Math.Min(MinY, point.Y);
            MaxY = Math.Max(MaxY, point.Y);
            
            MinZ = Math.Min(MinZ, point.Z);
            MaxZ = Math.Max(MaxZ, point.Z);
        }

        public bool Contains(Vector3 point)
        {
            return (this.MaxX >= point.X) && (this.MinX <= point.X) &&
                   (this.MaxY >= point.Y) && (this.MinY <= point.Y) &&
                   (this.MaxZ >= point.Z) && (this.MinZ <= point.Z);
        }

        public bool Intersects(AABBf other)
        {
            return
                (MaxX > other.MinX && MaxX < other.MaxX) ||
                (MinX > other.MinX && MinX < other.MaxX) ||
                (MaxY > other.MinY && MaxY < other.MaxY) ||
                (MinY > other.MinY && MinY < other.MaxY) ||
                (MaxZ > other.MinZ && MaxZ < other.MaxZ) ||
                (MinZ > other.MinZ && MinZ < other.MaxZ);
        }

        public bool IsOutside(AABBf other)
        {
            return (MaxX - other.MinX) < 0 || (MinX - other.MaxX) > 0 ||
                   (MaxY - other.MinY) < 0 || (MinY - other.MaxY) > 0 ||
                   (MaxZ - other.MinZ) < 0 || (MinZ - other.MaxZ) > 0;
        }

        public AABBf Clone()
        {
            AABBf clone = new AABBf();
            clone.MaxX = this.MaxX;
            clone.MaxY = this.MaxY;
            clone.MaxZ = this.MaxZ;
            clone.MinX = this.MinX;
            clone.MinY = this.MinY;
            clone.MinZ = this.MinZ;

            return clone;
        }

        /// <summary>
        /// Finds the closest point on the box, if the point is inside the box, returns the point.
        /// </summary>
        /// <param name="point"></param>
        /// <returns></returns>
        public Vector3 ClosestPoint(Vector3 point)
        {
            Vector3 closestPoint = new Vector3();
            closestPoint.X = (point.X < MinX) ? MinX : (point.X > MaxX) ? MaxX : point.X;
            closestPoint.Y = (point.Y < MinY) ? MinY : (point.Y > MaxY) ? MaxY : point.Y;
            closestPoint.Z = (point.Z < MinZ) ? MinZ : (point.Z > MaxZ) ? MaxZ : point.Z;
            return closestPoint;
        }

        public Vector3 ClosestPointOnSurface(Vector3 point)
        {
            float minXDist = Math.Min(Math.Abs(point.X - MinX), Math.Abs(point.X - MaxX));
            float minYDist = Math.Min(Math.Abs(point.Y - MinY), Math.Abs(point.Y - MaxY));
            float minZDist = Math.Min(Math.Abs(point.Z - MinZ), Math.Abs(point.Z - MaxZ));
            if (minXDist <= minYDist && minXDist <= minZDist)
            {
                Vector3 closestPoint = new Vector3();
                closestPoint.X = (point.X < MinX) ? MinX : (point.X > MaxX) ? MaxX : (Math.Abs(point.X - MinX) < Math.Abs(point.X - MaxX) ? MinX : MaxX);
                closestPoint.Y = (point.Y < MinY) ? MinY : (point.Y > MaxY) ? MaxY : point.Y;
                closestPoint.Z = (point.Z < MinZ) ? MinZ : (point.Z > MaxZ) ? MaxZ : point.Z;
                return closestPoint;
            }
            else if (minYDist <= minXDist && minYDist <= minZDist)
            {
                Vector3 closestPoint = new Vector3();
                closestPoint.X = (point.X < MinX) ? MinX : (point.X > MaxX) ? MaxX : point.X;
                closestPoint.Y = (point.Y < MinY) ? MinY : (point.Y > MaxY) ? MaxY : (Math.Abs(point.Y - MinY) < Math.Abs(point.Y - MaxY) ? MinY : MaxY);
                closestPoint.Z = (point.Z < MinZ) ? MinZ : (point.Z > MaxZ) ? MaxZ : point.Z;
                return closestPoint;
            }
            else // if (minZDist <= minXDist && minZDist <= minYDist)
            {
                Vector3 closestPoint = new Vector3();
                closestPoint.X = (point.X < MinX) ? MinX : (point.X > MaxX) ? MaxX : point.X;
                closestPoint.Y = (point.Y < MinY) ? MinY : (point.Y > MaxY) ? MaxY : point.Y;
                closestPoint.Z = (point.Z < MinZ) ? MinZ : (point.Z > MaxZ) ? MaxZ : (Math.Abs(point.Z - MinZ) < Math.Abs(point.Z - MaxZ) ? MinZ : MaxZ);
                return closestPoint;
            }
        }
    }
}
