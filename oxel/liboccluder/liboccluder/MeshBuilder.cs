using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using OpenTK;

namespace Oxel
{
    public class MeshBuilder
    {
        public static void CreateListsFromMeshes(
            Dictionary<Vector4, int> vertexLookup,
            int[] vertexIndexLookup,
            Dictionary<CSGNode, CSGMesh> meshes,
            Vector4[] vertices, out int vertexCount,
            int[] polyIndices, out int polyIndexCount,
            int[] lineIndices, out int lineIndexCount)
        {
            vertexCount = 0;
            polyIndexCount = 0;
            lineIndexCount = 0;
            vertexLookup.Clear();

            foreach (var item in meshes)
            {
                var node = item.Key;
                var mesh = item.Value;
                var meshVertices = mesh.Vertices;
                var offset = node.Translation;
                for (int i = 0; i < meshVertices.Count; i++)
                {
                    var meshVertex = meshVertices[i];
                    int index;
                    var vertex = new Vector4(meshVertex.X + offset.X, meshVertex.Y + offset.Y, meshVertex.Z + offset.Z, 1);
                    if (!vertexLookup.TryGetValue(vertex, out index))
                    {
                        index = vertexCount;
                        vertexLookup.Add(vertex, index);
                        vertices[vertexCount] = vertex;
                        vertexCount++;
                    }
                    vertexIndexLookup[i] = index;
                }

                var polygons = mesh.Polygons;
                var edges = mesh.Edges;
                var planes = mesh.Planes;
                foreach (var polygon in polygons)
                {
                    if (!polygon.Visible ||
                        polygon.FirstIndex == -1)
                        continue;

                    var iterator = edges[polygon.FirstIndex];
                    if (iterator == null)
                        continue;

                    int first = vertexIndexLookup[iterator.VertexIndex];
                    iterator = edges[iterator.NextIndex];
                    int second = vertexIndexLookup[iterator.VertexIndex];

                    var twin = edges[iterator.TwinIndex];
                    var twinPolygonIndex = twin.PolygonIndex;
                    var twinPolygon = polygons[twinPolygonIndex];
                    var twinPlane = planes[twinPolygon.PlaneIndex];
                    var curPolygonIndex = iterator.PolygonIndex;
                    var curPolygon = polygons[curPolygonIndex];
                    var curPlane = planes[curPolygon.PlaneIndex];

                    if (!twinPolygon.Visible ||
                        !curPlane.Equals(twinPlane))
                    {
                        if (lineIndexCount <= lineIndices.Length - 2)
                        {
                            lineIndices[lineIndexCount] = first; lineIndexCount++;
                            lineIndices[lineIndexCount] = second; lineIndexCount++;
                        }
                    }

                    var previous = second;
                    var polygonFirst = edges[polygon.FirstIndex];
                    while (iterator != polygonFirst)
                    {
                        iterator = edges[iterator.NextIndex];

                        curPolygonIndex = iterator.PolygonIndex;
                        curPolygon = polygons[curPolygonIndex];
                        curPlane = planes[curPolygon.PlaneIndex];

                        twin = edges[iterator.TwinIndex];
                        twinPolygonIndex = twin.PolygonIndex;
                        twinPolygon = polygons[twinPolygonIndex];
                        twinPlane = planes[twinPolygon.PlaneIndex];

                        int third = vertexIndexLookup[iterator.VertexIndex];

                        if (!curPlane.Equals(twinPlane))
                        {
                            if (lineIndexCount <= lineIndices.Length - 2)
                            {
                                lineIndices[lineIndexCount] = previous; lineIndexCount++;
                                lineIndices[lineIndexCount] = third; lineIndexCount++;
                            }
                        }
                        else
                        {
                            if (!twinPolygon.Visible)
                            {
                                if (lineIndexCount <= lineIndices.Length - 2)
                                {
                                    lineIndices[lineIndexCount] = previous; lineIndexCount++;
                                    lineIndices[lineIndexCount] = third; lineIndexCount++;
                                }
                            }
                        }

                        previous = third;
                        if (polyIndexCount < polyIndices.Length - 3)
                        {
                            // Remove polygons where two of the vertices are the same vertex
                            if (first == second || first == third || second == third)
                            {
                                // CHANGE NDarnell Had to add a check to see if there were any triangles 
                                // that referenced the indices more than once.
                                //Debug.WriteLine("Found false triangle loop");
                            }
                            // Remove degenerate polygons (i.e. polygons that are infinitely thin).
                            else if (new Triangle(vertices[first].Xyz, vertices[second].Xyz, vertices[third].Xyz).HasColinearPoints(0.0001f))
                            {
                                // Ignore 'false triangles', the ones that are made of points that are colinear that the
                                // CSG system produced but that arn't normal / nessesary in the final version of the mesh.
                                //Debug.WriteLine("Found colinear triangle");
                            }
                            else
                            {
                                polyIndices[polyIndexCount] = first; polyIndexCount++;
                                polyIndices[polyIndexCount] = second; polyIndexCount++;
                                polyIndices[polyIndexCount] = third; polyIndexCount++;
                            }
                        }

                        second = third;
                    }
                }
            }
        }

        public static Mesh BuildMesh(List<AABBf> boxList)
        {
            Mesh mesh = new Mesh();

            if (boxList.Count == 0)
            {
                mesh.Vertices = new Vector4[0];
                mesh.Indicies = new int[0];
                return mesh;
            }

            List<CSGNode> nodes = new List<CSGNode>();
            for (int i = 0; i < boxList.Count; i++)
            {
                AABBf box = boxList[i];
                CSGNode node = new CSGNode(new Plane[] {
                    new Plane(0, -1, 0, 0),
                    new Plane(-1, 0, 0, 0),
                    new Plane(0, 0, -1, 0),
                    new Plane(0, 0, 1, box.MaxZ - box.MinZ),
                    new Plane(1, 0, 0, box.MaxX - box.MinX),
                    new Plane(0, 1, 0, box.MaxY - box.MinY)
                });
                node.Translation = new Vector3(box.MinX, box.MinY, box.MinZ);

                nodes.Add(node);
            }

            return BuildMesh(nodes);
        }

        public static Mesh BuildMesh(List<AABBi> boxList)
        {
            Mesh mesh = new Mesh();

            if (boxList.Count == 0)
            {
                mesh.Vertices = new Vector4[0];
                mesh.Indicies = new int[0];
                return mesh;
            }

            List<CSGNode> nodes = new List<CSGNode>();
            for (int i = 0; i < boxList.Count; i++)
            {
                AABBi box = boxList[i];
                CSGNode node = new CSGNode(new Plane[] {
                    new Plane(0, -1, 0, 0),
                    new Plane(-1, 0, 0, 0),
                    new Plane(0, 0, -1, 0),
                    new Plane(0, 0, 1, box.MaxZ - box.MinZ),
                    new Plane(1, 0, 0, box.MaxX - box.MinX),
                    new Plane(0, 1, 0, box.MaxY - box.MinY)
                });
                node.Translation = new Vector3(box.MinX, box.MinY, box.MinZ);

                nodes.Add(node);
            }

            return BuildMesh(nodes);
        }

        public static Mesh BuildMesh(List<CSGNode> nodes)
        {
            Mesh mesh = new Mesh();

            CSGTree tree = new CSGTree();
            tree.RootNode = CreateTree(nodes);

            var updateNodes = CSGUtility.FindChildBrushes(tree);
            var modifiedMeshes = CSGCategorization.ProcessCSGNodes(tree.RootNode, updateNodes);

            Dictionary<CSGNode, CSGMesh> validMeshes = new Dictionary<CSGNode, CSGMesh>();
            foreach (var item in modifiedMeshes)
                validMeshes[item.Key] = item.Value;

            Vector4[] vertices = new Vector4[65535];
            int[] polyIndices = new int[65535 * 4];
            int[] lineIndices = new int[65535 * 3];

            Dictionary<Vector4, int> vertexLookup = new Dictionary<Vector4, int>();
            int[] vertexIndexLookup = new int[65535];

            int vertexCount;
            int polyIndexCount;
            int lineIndexCount;
            CreateListsFromMeshes(vertexLookup, vertexIndexLookup, validMeshes, vertices, out vertexCount, polyIndices, out polyIndexCount, lineIndices, out lineIndexCount);

            Vector4[] verts = new Vector4[vertexCount];
            int[] indicies = new int[polyIndexCount];
            Array.Copy(vertices, verts, verts.Length);
            Array.Copy(polyIndices, indicies, indicies.Length);

            mesh.Vertices = verts;
            mesh.Indicies = indicies;

            return mesh;
        }

        public static Mesh BuildMesh(AABBf aabb, Vector3 delta_p, List<AABBi> boxList)
        {
            Mesh mesh = BuildMesh(boxList);

            Vector4 aabb_min = new Vector4(aabb.MinX, aabb.MinY, aabb.MinZ, 0);
            Vector4 delta_p4 = new Vector4(delta_p, 1);
            for (int i = 0; i < mesh.Vertices.Length; i++)
            {
                mesh.Vertices[i] = aabb_min + Vector4.Multiply(mesh.Vertices[i], delta_p4);
            }

            return mesh;
        }

        public static CSGNode CreateTree(List<CSGNode> nodes)
        {
            if (nodes.Count == 1)
            {
                return nodes[0];
            }
            else if (nodes.Count == 2)
            {
                return new CSGNode(CSGNodeType.Addition, nodes[0], nodes[1]);
            }

            CSGNode node = new CSGNode(CSGNodeType.Addition);
            node.Left = CreateTree(nodes.GetRange(0, nodes.Count / 2));
            node.Right = CreateTree(nodes.GetRange(nodes.Count / 2, (int)Math.Ceiling(nodes.Count / 2.0f)));
            return node;
        }
    }
}
