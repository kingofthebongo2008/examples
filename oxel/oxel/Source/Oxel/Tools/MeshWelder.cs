using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenTK;

namespace Oxel
{
    public class MeshWelder
    {
        public static void Weld(int precision, Vector3[] vertices, Tri[] tris)
        {
            int p = (int)Math.Pow((double)10, (double)precision);

            Dictionary<Vector3, int> pointHash = new Dictionary<Vector3, int>();
            Dictionary<int, int> remapList = new Dictionary<int, int>();
            for (int i = 0; i < vertices.Length; i++)
            {
                Vector3 vertex = vertices[i];
                vertex.X = (float)((int)(vertex.X * p));
                vertex.Y = (float)((int)(vertex.Y * p));
                vertex.Z = (float)((int)(vertex.Z * p));

                int existing_index;
                if (!pointHash.TryGetValue(vertex, out existing_index))
                {
                    pointHash.Add(vertex, i);
                }
                else
                {
                    remapList.Add(i, existing_index);
                }
            }

            foreach (Tri tri in tris)
            {
                int existing_index;
                if (remapList.TryGetValue(tri.P1.Vertex, out existing_index))
                {
                    tri.P1.Vertex = existing_index;
                }
                if (remapList.TryGetValue(tri.P2.Vertex, out existing_index))
                {
                    tri.P2.Vertex = existing_index;
                }
                if (remapList.TryGetValue(tri.P3.Vertex, out existing_index))
                {
                    tri.P3.Vertex = existing_index;
                }
            }
        }

        public static void Weld(int precision, Vector4[] vertices, int[] indicies, out Vector4[] newVertices, out int[] newIndicies)
        {
            List<Vector4> tempNewVertices = new List<Vector4>();
            List<int> tempNewIndicies = new List<int>();

            int p = (int)Math.Pow((double)10, (double)precision);

            Dictionary<Vector4, int> pointHash = new Dictionary<Vector4, int>();
            Dictionary<int, int> remapList = new Dictionary<int, int>();
            for (int i = 0; i < vertices.Length; i++)
            {
                Vector4 vertex = vertices[i];
                int iX = (int)(vertex.X);
                int iY = (int)(vertex.Y);
                int iZ = (int)(vertex.Z);

                vertex.X = iX + ((float)((vertex.X - iX) * p) / p);
                vertex.Y = iY + ((float)((vertex.Y - iY) * p) / p);
                vertex.Z = iZ + ((float)((vertex.Z - iZ) * p) / p);

                int existing_index;
                if (!pointHash.TryGetValue(vertex, out existing_index))
                {
                    pointHash.Add(vertex, tempNewVertices.Count);
                    tempNewVertices.Add(vertex);
                }
                else
                {
                    remapList.Add(i, existing_index);
                }
            }

            for (int i = 0; i < indicies.Length; i++)
            {
                int existing_index;
                if (remapList.TryGetValue(indicies[i], out existing_index))
                {
                    tempNewIndicies.Add(existing_index);
                }
                else
                {
                    tempNewIndicies.Add(indicies[i]);
                }
            }

            newVertices = tempNewVertices.ToArray();
            newIndicies = tempNewIndicies.ToArray();
        }
    }
}
