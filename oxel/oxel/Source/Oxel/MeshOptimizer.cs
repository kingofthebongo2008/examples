//#define DEBUGGING_TJUNCTIONS
#define MERGE_COLINEAR
#define CHECK_FOR_SHARED_WHEN_MERGING

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using OpenTK;

namespace Oxel
{
    public static class MeshOptimizer
    {
        //TODO: need a variable length epsilon depending on size of float.  Maybe change to doubles for doing high precision work with models.
        const float VERTEX_EPSILON = 0.0001f;

        public static Mesh Retriangulate(VoxelizationInput input, Mesh mesh, out List<List<Edge>> debugLoops)
        {
            debugLoops = new List<List<Edge>>();

            try
            {
                Triangle[] triangles = Triangle.ToTriangleArray(mesh.Indicies, mesh.Vertices);

                Debug.WriteLine("Sorting triangles based on plane.");

                List<PolygonTriangles> planeLookup = SortTrianglesIntoPlanes(triangles);

                Debug.WriteLine("Triangles sorted by planes: Done.");

                List<Vector4> processedVerticies = new List<Vector4>();
                List<int> processedIndicies = new List<int>();

                foreach (PolygonTriangles polygon in planeLookup)
                {
#if DEBUGGING_TJUNCTIONS
                    if (!Vector3Ex.AlmostEquals(polygon.Plane.Normal, Vector3.UnitY))
                        continue;
#endif

                    // Before we can begin margining edges we need to ensure that all shared collinear edges
                    // have a mate, one thing that can prevent this is the presence of T-Junctions, so first
                    // we need to fix them by splitting polygons that produce them.
                    polygon.Triangles = FindAndFixTJuntions(input, polygon.Triangles);

                    // We build a list of distinctive edges because a distinctive edge is either part of the inner loop
                    // or the outer loop of a group of polygons, because if two polygons share an edge that are co-planer
                    // that edge can't be on the inside or the outside of the plane.
                    List<Edge> distinctEdges = FindDistinctiveEdges(input, polygon.Triangles);
                    debugLoops.Add(distinctEdges.ToList());

                    // Merge the edges together that are collinear.
                    List<List<Edge>> outterLoops = FindOuterLoops(distinctEdges);
                    //allLoops.AddRange(outterLoops);

                    Vector3 right = Vector3Ex.GetRight(polygon.Plane.Normal);
                    Vector3 U, V;
                    Vector3Ex.GetBasisVectors(polygon.Plane.Normal, right, out U, out V);

                    Vector3 planeOrigin = polygon.Plane.PointOnPlane;

                    foreach (List<Edge> outterEdgeLoop in outterLoops)
                    {
                        List<Vector2> outerLoopTransformed = new List<Vector2>();
                        for (int i = 0; i < outterEdgeLoop.Count; i++)
                        {
                            Edge e = outterEdgeLoop[i];

                            Vector2 uv = Vector3Ex.Calc2DPoint(e.v0, U, V);

                            outerLoopTransformed.Add(uv);
                        }

                        //Debug.Assert(outerLoopTransformed.Count > 2);
                        if (outerLoopTransformed.Count > 2)
                        {
                            List<Poly2Tri.PolygonPoint> polyPoints = new List<Poly2Tri.PolygonPoint>();
                            foreach(var pt in outerLoopTransformed)
                            {
                                polyPoints.Add(new Poly2Tri.PolygonPoint(pt.X, pt.Y));
                            }
                            Poly2Tri.Polygon p = new Poly2Tri.Polygon(polyPoints);
                            Poly2Tri.P2T.Triangulate(Poly2Tri.TriangulationAlgorithm.DTSweep, p);

                            int currentVerticies = processedVerticies.Count;
                            foreach(var dt in p.Triangles)
                            {
                                processedIndicies.Add((currentVerticies + 0));
                                processedIndicies.Add((currentVerticies + 1));
                                processedIndicies.Add((currentVerticies + 2));

                                Vector3 pt0 = (((float)dt.Points._0.X * U) + ((float)dt.Points._0.Y * V)) - planeOrigin;
                                Vector3 pt1 = (((float)dt.Points._1.X * U) + ((float)dt.Points._1.Y * V)) - planeOrigin;
                                Vector3 pt2 = (((float)dt.Points._2.X * U) + ((float)dt.Points._2.Y * V)) - planeOrigin;
                                processedVerticies.Add(new Vector4(pt0, 1));
                                processedVerticies.Add(new Vector4(pt1, 1));
                                processedVerticies.Add(new Vector4(pt2, 1));

                                currentVerticies += 3;
                            }
                        }
                        else
                        {
                            Debug.WriteLine("outer loop problem!");
                        }
                    }

                    Debug.WriteLine("Distinct Edges " + processedIndicies.Count / 3);
                }

                Mesh retriangulatedMesh = new Mesh();
                retriangulatedMesh.Vertices = processedVerticies.ToArray();
                retriangulatedMesh.Indicies = processedIndicies.ToArray();

                return retriangulatedMesh;
            }
            catch (Exception ex)
            {
                return null;
            }
        }

        private static List<PolygonTriangles> SortTrianglesIntoPlanes(Triangle[] triangles)
        {
            Dictionary<Plane, List<Triangle>> planeLookup = new Dictionary<Plane, List<Triangle>>();

            foreach (Triangle t in triangles)
            {
                List<Triangle> triList;
                if (!planeLookup.TryGetValue(t.Plane, out triList))
                {
                    triList = new List<Triangle>();
                    planeLookup.Add(t.Plane, triList);
                }
                triList.Add(t);
            }

            List<PolygonTriangles> polygonTriangles = new List<PolygonTriangles>();
            foreach (var entry in planeLookup)
            {
                PolygonTriangles polygon = new PolygonTriangles();
                polygon.Plane = entry.Key;
                polygon.Triangles = entry.Value;

                polygonTriangles.Add(polygon);
            }
            //polygonTriangles.Sort();

            return polygonTriangles;
        }

        private static List<Triangle> FindAndFixTJuntions(VoxelizationInput input, List<Triangle> triangles)
        {
            // When one triangle shares two vertices with the edge of another triangle, but does not also 
            // share the edge with the triangle, there is a T-Junction present.

            FixStart:

            int counter = 0;
            foreach (Triangle t in triangles)
            {
                foreach (Triangle other in triangles)
                {
                    if (t == other)
                        continue;

                    Vector3 junctionPoint;
                    int edge;
                    if (t.HasTJunction(other, VERTEX_EPSILON, out edge, out junctionPoint))
                    {
                        Debug.WriteLine(counter + " FOUND T-JUNCTION!");
                        counter++;

                        Triangle t0 = null, t1 = null;
                        switch(edge)
                        {
                            case 0: // Edge 0 --- 1
                                t0 = new Triangle(junctionPoint, t.v2, t.v0);
                                t1 = new Triangle(junctionPoint, t.v1, t.v2);
                                break;
                            case 1: // Edge 1 --- 2
                                t0 = new Triangle(junctionPoint, t.v0, t.v1);
                                t1 = new Triangle(junctionPoint, t.v2, t.v0);
                                break;
                            case 2: // Edge 2 --- 0
                                t0 = new Triangle(junctionPoint, t.v1, t.v2);
                                t1 = new Triangle(junctionPoint, t.v0, t.v1);
                                break;
                            default:
                                throw new NotImplementedException();
                        }

                        int index = triangles.IndexOf(t);
                        triangles.Remove(t);
                        triangles.Insert(index, t0);
                        triangles.Insert(index, t1);

                        goto FixStart;
                    }
                }
            }

            return triangles;
        }

        private static List<Edge> FindDistinctiveEdges(VoxelizationInput input, List<Triangle> triangles)
        {
            List<Edge> edges = new List<Edge>();
            foreach (Triangle t in triangles)
            {
                // TODO NDarnell Winding Order
                edges.Add(new Edge(t.v0, t.v1));
                edges.Add(new Edge(t.v1, t.v2));
                edges.Add(new Edge(t.v2, t.v0));
            }

            List<Edge> distinctEdges =
                (from e in edges
                 where edges.FindAll(ed => e.Equals(ed, VERTEX_EPSILON)).Count == 1
                 select e).ToList();

            return distinctEdges;
        }

        private static List<List<Edge>> FindOuterLoops(List<Edge> edges)
        {
            List<List<Edge>> loops = new List<List<Edge>>();

            while (edges.Count != 0)
            {
                List<Edge> loop = new List<Edge>();

                loop.Add(edges[edges.Count - 1]);
                edges.RemoveAt(edges.Count - 1);

                while (true)
                {
                    Edge first = loop[0];
                    Edge last = loop[loop.Count - 1];

                    bool foundMatch = false;
                    for (int i = 0; i < edges.Count; i++)
                    {
                        Edge edge = edges[i];

                        if (Vector3Ex.AlmostEquals(ref first.v0, ref edge.v1, VERTEX_EPSILON))
                        {
                            loop.Insert(0, edge);
                            edges.RemoveAt(i);
                            foundMatch = true;
                            i--;

                            first = edge;
                        }
                        else if (Vector3Ex.AlmostEquals(ref last.v1, ref edge.v0, VERTEX_EPSILON))
                        {
                            loop.Add(edge);
                            edges.RemoveAt(i);
                            foundMatch = true;
                            i--;

                            last = edge;
                        }
                    }

                    if (foundMatch == false)
                    {
                        List<Edge> mergedLoop = new List<Edge>(loop);

#if MERGE_COLINEAR
                        for (int i = 0; i < mergedLoop.Count; i++)
                        {
                            Edge e = mergedLoop[i];
                            Edge ne = (i < mergedLoop.Count - 1) ? mergedLoop[i + 1] : mergedLoop[0];

                            Edge merged;
                            Vector3 mergedPoint;
                            if (e.MergeConnectedAndCollinear(ne, VERTEX_EPSILON, out merged, out mergedPoint))
                            {
#if CHECK_FOR_SHARED_WHEN_MERGING
                                // If we've found a potential merge-able pair we need to ensure that the point we're
                                // merging on isn't depended on by other edges in the loop.
                                if ((i+1) < mergedLoop.Count - 1)
                                {
                                    int edgeSharingMergePoint = mergedLoop.FindIndex(i, 
                                        findEdge => findEdge.IsAnchoredAt(mergedPoint, VERTEX_EPSILON));

                                    if (edgeSharingMergePoint != -1)
                                        continue;
                                }
#endif

                                mergedLoop.Insert(i, merged);
                                mergedLoop.Remove(e);
                                mergedLoop.Remove(ne);
                                i--;
                            }
                        }
#endif

                        loop = mergedLoop;

                        break;
                    }
                }

                loops.Add(loop);
            }

            return loops;
        }

        [DebuggerDisplay("{Plane} : Triangles: {Triangles.Count}")]
        public class PolygonTriangles : IComparable<PolygonTriangles>
        {
            public Plane Plane;
            public List<Triangle> Triangles;

            public int CompareTo(PolygonTriangles other)
            {
                return this.Plane.CompareTo(other.Plane);
            }
        }
    }
}
