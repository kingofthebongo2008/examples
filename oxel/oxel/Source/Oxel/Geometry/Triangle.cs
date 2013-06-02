using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using OpenTK;
using System.Diagnostics;

namespace Oxel
{
    [DebuggerDisplay("{Plane}")]
    [StructLayout(LayoutKind.Sequential)]
    public class Triangle : IComparable<Triangle>
    {
        public Vector3 v0, v1, v2;

        public Triangle()
        {
        }

        public Triangle(Vector3 v0, Vector3 v1, Vector3 v2)
        {
            this.v0 = v0;
            this.v1 = v1;
            this.v2 = v2;
        }

        public Vector3 this[int index]
        {
            get
            {
                switch(index)
                {
                    case 0: return v0;
                    case 1: return v1;
                    case 2: return v2;
                    default: throw new InvalidOperationException();
                }
            }
            set
            {
                switch(index)
                {
                    case 0: v0 = value; break;
                    case 1: v1 = value; break;
                    case 2: v2 = value; break;
                    default: throw new InvalidOperationException();
                }
            }
        }

        public Vector3 NormalClockwise()
        {
            Vector3 v01 = v1 - v0;
            Vector3 v02 = v2 - v0;

            Vector3 normal = Vector3.Cross(v02, v01);
            normal.Normalize();
            normal.Normalize();

            return normal;
        }

        public Vector3 NormalCounterClockwise()
        {
            Vector3 v01 = v1 - v0;
            Vector3 v02 = v2 - v0;

            Vector3 normal = Vector3.Cross(v01, v02);
            normal.Normalize();
            normal.Normalize();

            return normal;
        }

        public Vector3 Center
        {
            get
            {
                double x = (v0.X + v1.X + v2.X) / 3.0;
                double y = (v0.Y + v1.Y + v2.Y) / 3.0;
                double z = (v0.Z + v1.Z + v2.Z) / 3.0;
                return new Vector3((float)x, (float)y, (float)z);
            }
        }

        public Plane Plane
        {
            get
            {
                // ax + by + cz + d = 0
                Vector3 normal = NormalClockwise();
                float d = -(v0.X * normal.X + v0.Y * normal.Y + v0.Z * normal.Z);

                return new Plane(normal, d);
            }
        }

        public bool HasColinearPoints(float epsilon)
        {
            if (Vector3Ex.AreCollinear(v0, v1, v1, v2, epsilon))
                return true;

            if (Vector3Ex.AreCollinear(v0, v1, v0, v2, epsilon))
                return true;

            return false;
        }

        public bool IsCoplanar(Triangle other)
        {
            //TODO add epsilon perhaps.
            return Plane == other.Plane;
        }

        private static bool IsTJunctionEdge(Vector3 line0, Vector3 line1, Vector3 oline0, Vector3 oline1, float epsilon, out Vector3 junctionPoint)
        {
            junctionPoint = Vector3.Zero;

            bool other0Inside = Vector3Ex.IsPointInsideLineSegment(line0, line1, oline0, epsilon);
            bool other1Inside = Vector3Ex.IsPointInsideLineSegment(line0, line1, oline1, epsilon);

            if (other0Inside && !other1Inside)
            {
                junctionPoint = oline0;
                return true;
            }
            else if (!other0Inside && other1Inside)
            {
                junctionPoint = oline1;
                return true;
            }

            return false;
        }

        public bool HasTJunction(Triangle other, float epsilon, out int edge, out Vector3 junctionPoint)
        {
            edge = 0;

            if (IsTJunctionEdge(v0, v1, other.v0, other.v1, epsilon, out junctionPoint))
                return !AreTwoVerticesShared(other, epsilon);
            if (IsTJunctionEdge(v0, v1, other.v1, other.v2, epsilon, out junctionPoint))
                return !AreTwoVerticesShared(other, epsilon);
            if (IsTJunctionEdge(v0, v1, other.v2, other.v0, epsilon, out junctionPoint))
                return !AreTwoVerticesShared(other, epsilon);

            edge = 1;

            if (IsTJunctionEdge(v1, v2, other.v0, other.v1, epsilon, out junctionPoint))
                return !AreTwoVerticesShared(other, epsilon);
            if (IsTJunctionEdge(v1, v2, other.v1, other.v2, epsilon, out junctionPoint))
                return !AreTwoVerticesShared(other, epsilon);
            if (IsTJunctionEdge(v1, v2, other.v2, other.v0, epsilon, out junctionPoint))
                return !AreTwoVerticesShared(other, epsilon);

            edge = 2;

            if (IsTJunctionEdge(v2, v0, other.v0, other.v1, epsilon, out junctionPoint))
                return !AreTwoVerticesShared(other, epsilon);
            if (IsTJunctionEdge(v2, v0, other.v1, other.v2, epsilon, out junctionPoint))
                return !AreTwoVerticesShared(other, epsilon);
            if (IsTJunctionEdge(v2, v0, other.v2, other.v0, epsilon, out junctionPoint))
                return !AreTwoVerticesShared(other, epsilon);

            return false;
        }

        public bool AreTwoVerticesShared(Triangle other, float epsilon)
        {
            int sharedVerticies = 0;

            sharedVerticies += Vector3Ex.AlmostEquals(ref v0, ref other.v0) ? 1 : 0;
            sharedVerticies += Vector3Ex.AlmostEquals(ref v0, ref other.v1) ? 1 : 0;
            sharedVerticies += Vector3Ex.AlmostEquals(ref v0, ref other.v2) ? 1 : 0;

            sharedVerticies += Vector3Ex.AlmostEquals(ref v1, ref other.v0) ? 1 : 0;
            sharedVerticies += Vector3Ex.AlmostEquals(ref v1, ref other.v1) ? 1 : 0;
            sharedVerticies += Vector3Ex.AlmostEquals(ref v1, ref other.v2) ? 1 : 0;

            sharedVerticies += Vector3Ex.AlmostEquals(ref v2, ref other.v0) ? 1 : 0;
            sharedVerticies += Vector3Ex.AlmostEquals(ref v2, ref other.v1) ? 1 : 0;
            sharedVerticies += Vector3Ex.AlmostEquals(ref v2, ref other.v2) ? 1 : 0;

            return sharedVerticies >= 2;
        }

        public int CompareTo(Triangle other)
        {
            return (int)Math.Round(other.Center.Length - Center.Length, MidpointRounding.AwayFromZero);
        }

        public static Triangle[] ToTriangleArray(int[] indicies, Vector4[] vertices)
        {
            Triangle[] triangles = new Triangle[indicies.Length / 3];
            int triIndex = 0;
            for (int i = 0; i < indicies.Length; i += 3)
            {
                Triangle t = new Triangle();
                t.v0 = vertices[indicies[i]].Xyz;
                t.v1 = vertices[indicies[i + 1]].Xyz;
                t.v2 = vertices[indicies[i + 2]].Xyz;

                triangles[triIndex] = t;
                triIndex++;
            }

            return triangles;
        }

        public static void FromTriangleArray(IEnumerable<Triangle> triangles, out int[] indicies, out Vector4[] vertices)
        {
            List<int> indexList = new List<int>();
            List<Vector4> vertexList = new List<Vector4>();

            foreach(Triangle t in triangles)
            {
                {
                    Vector4 v = new Vector4(t.v0, 1);
                    int index = vertexList.IndexOf(v);
                    if (index == -1)
                    {
                        index = vertexList.Count;
                        vertexList.Add(v);
                    }
                    indexList.Add(index);
                }
                {
                    Vector4 v = new Vector4(t.v1, 1);
                    int index = vertexList.IndexOf(v);
                    if (index == -1)
                    {
                        index = vertexList.Count;
                        vertexList.Add(v);
                    }
                    indexList.Add(index);
                }
                {
                    Vector4 v = new Vector4(t.v2, 1);
                    int index = vertexList.IndexOf(v);
                    if (index == -1)
                    {
                        index = vertexList.Count;
                        vertexList.Add(v);
                    }
                    indexList.Add(index);
                }
            }

            indicies = indexList.ToArray();
            vertices = vertexList.ToArray();
        }

        public static AABBf CreateBoundingBox(IEnumerable<Triangle> triangles)
        {
            AABBf box = new AABBf();
            foreach (Triangle t in triangles)
            {
                box.Add(t.v0);
                box.Add(t.v1);
                box.Add(t.v2);
            }

            return box;
        }
    }
}
