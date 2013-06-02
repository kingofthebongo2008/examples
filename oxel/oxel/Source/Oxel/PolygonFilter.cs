using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenTK;

namespace Oxel
{
    public static class PolygonFilter
    {
        public static Mesh Filter(VoxelizationInput input, Mesh mesh)
        {
            // Remove the top and bottom polygons
            if (input.RemoveTop || input.RemoveBottom)
            {
                Vector3 upAxis, downAxis;
                if (input.UpAxis == UpAxis.Y)
                {
                    upAxis = Vector3.UnitY;
                    downAxis = -Vector3.UnitY;
                }
                else // if (input.UpAxis == UpAxis.Z)
                {
                    upAxis = Vector3.UnitZ;
                    downAxis = -Vector3.UnitZ;
                }

                List<Triangle> filteredTrianlges = new List<Triangle>();

                Triangle[] triangles = Triangle.ToTriangleArray(mesh.Indicies, mesh.Vertices);
                foreach (Triangle t in triangles)
                {
                    Plane p = t.Plane;
                    Vector3 normal = t.NormalCounterClockwise();

                    // Remove all top polygon
                    if (input.RemoveTop)
                    {
                        if (Vector3Ex.AlmostEquals(ref normal, ref upAxis))
                            continue;
                    }

                    // Remove all bottom polygons that are with-in one voxel of the mesh bounds.
                    if (input.RemoveBottom)
                    {
                        if (Vector3Ex.AlmostEquals(ref normal, ref downAxis))
                        {
                            Vector3 closestPoint = input.Octree.MeshBounds.ClosestPointOnSurface(t.Center);
                            float distance = (t.Center - closestPoint).Length;
                            if (distance <= input.Octree.SmallestVoxelSideLength)
                                continue;
                        }
                    }

                    filteredTrianlges.Add(t);
                }

                Triangle.FromTriangleArray(filteredTrianlges, out mesh.Indicies, out mesh.Vertices);
            }

            return mesh;
        }
    }
}
