using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenTK;
using System.Diagnostics;

namespace Oxel
{
    [DebuggerDisplay("{v0} : {v1}")]
    public class Edge
    {
        //TODO: need a variable length epsilon depending on size of float.  Maybe change to doubles for doing high precision work with models.
        const float DefaultEpsilon = 0.0001f;

        public Vector3 v0;
        public Vector3 v1;

        public Edge(Vector3 v0, Vector3 v1)
        {
            this.v0 = v0;
            this.v1 = v1;
        }

        public override int GetHashCode()
        {
            return v0.GetHashCode() ^ v1.GetHashCode();
        }

        public override bool Equals(object obj)
        {
            if (!(obj is Edge))
                return false;

            return Equals((Edge)obj, DefaultEpsilon);
        }

        public bool Equals(Edge edge, float epsilon)
        {
            if (Vector3Ex.AlmostEquals(ref v0, ref edge.v0, epsilon))
                return Vector3Ex.AlmostEquals(ref v1, ref edge.v1, epsilon);
            else if (Vector3Ex.AlmostEquals(ref v0, ref edge.v1, epsilon))
                return Vector3Ex.AlmostEquals(ref v1, ref edge.v0, epsilon);

            return false;
        }

        public bool IsAnchoredAt(Vector3 v, float epsilon)
        {
            return Vector3Ex.AlmostEquals(ref v0, ref v, epsilon) ||
                Vector3Ex.AlmostEquals(ref v0, ref v, epsilon);
        }

        public bool IsConnected(Edge edge, float epsilon)
        {
            return Vector3Ex.AlmostEquals(ref v0, ref edge.v0, epsilon) ||
                Vector3Ex.AlmostEquals(ref v0, ref edge.v1, epsilon) ||
                Vector3Ex.AlmostEquals(ref v1, ref edge.v0, epsilon) ||
                Vector3Ex.AlmostEquals(ref v1, ref edge.v1, epsilon);
        }

        public bool IsConnectedAndCollinear(Edge edge, float epsilon)
        {
            return Vector3Ex.AreCollinear(v0, v1, edge.v0, edge.v1, epsilon);
        }

        public bool MergeConnectedAndCollinear(Edge edge, float epsilon, out Edge merged, out Vector3 mergedPoint)
        {
            merged = null;

            if (Vector3Ex.AreCollinear(v0, v1, edge.v0, edge.v1, epsilon))
            {
                if (Vector3Ex.AlmostEquals(ref v0, ref edge.v1, epsilon))
                {
                    merged = new Edge(edge.v0, v1);
                    mergedPoint = edge.v1;
                    return true;
                }
                else if (Vector3Ex.AlmostEquals(ref v1, ref edge.v0, epsilon))
                {
                    merged = new Edge(v0, edge.v1);
                    mergedPoint = edge.v0;
                    return true;
                }
            }

            mergedPoint = Vector3.Zero;
            return false;
        }
    }
}
