using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using OpenTK;
using OpenTK.Graphics.OpenGL;

namespace Oxel
{
    public enum CellStatus
    {
        Inside,
        Outside,
        Intersecting,
        IntersectingBounds,
        Unknown
    }

    [DebuggerDisplay("Cell ({Children.Count}, {Status})")]
    public class VoxelizingOctreeCell
    {
        public VoxelizingOctree Tree;
        public List<VoxelizingOctreeCell> Children = new List<VoxelizingOctreeCell>();
        public List<Triangle> Triangles = new List<Triangle>();
        public AABBf Bounds;
        public AABBi VoxelBounds;
        public Vector3 Center;
        public float Length;
        public VoxelizingOctreeCell Root;
        public VoxelizingOctreeCell Parent;
        public CellStatus Status = CellStatus.Unknown;
        public int[] CellStatusAccumulation = new int[2];
        public int Level;

        public VoxelizingOctreeCell(VoxelizingOctree tree, VoxelizingOctreeCell root, Vector3 center, float length, int level)
        {
            Tree = tree;
            Root = root;
            Center = center;
            Length = length;
            Level = level;

            float half_length = length / 2.0f;

            Bounds = new AABBf();
            Bounds.MinX = center.X - half_length;
            Bounds.MinY = center.Y - half_length;
            Bounds.MinZ = center.Z - half_length;
            Bounds.MaxX = center.X + half_length;
            Bounds.MaxY = center.Y + half_length;
            Bounds.MaxZ = center.Z + half_length;

            VoxelBounds = new AABBi(
                (int)Math.Round((Bounds.MinX - tree.VoxelBounds.MinX) / tree.SmallestVoxelSideLength, MidpointRounding.AwayFromZero),
                (int)Math.Round((Bounds.MinY - tree.VoxelBounds.MinY) / tree.SmallestVoxelSideLength, MidpointRounding.AwayFromZero),
                (int)Math.Round((Bounds.MinZ - tree.VoxelBounds.MinZ) / tree.SmallestVoxelSideLength, MidpointRounding.AwayFromZero),
                (int)Math.Round((Bounds.MaxX - tree.VoxelBounds.MinX) / tree.SmallestVoxelSideLength, MidpointRounding.AwayFromZero),
                (int)Math.Round((Bounds.MaxY - tree.VoxelBounds.MinY) / tree.SmallestVoxelSideLength, MidpointRounding.AwayFromZero),
                (int)Math.Round((Bounds.MaxZ - tree.VoxelBounds.MinZ) / tree.SmallestVoxelSideLength, MidpointRounding.AwayFromZero));
        }

        public bool IsLeaf
        {
            get { return Children.Count == 0; }
        }

        public bool AccumulateStatus(CellStatus status)
        {
            Debug.Assert(status == CellStatus.Inside || status == CellStatus.Outside);

            CellStatusAccumulation[(int)status]++;

            if (CellStatusAccumulation[(int)status] >= Tree.CellStatusAccumulationConfirmationThreshold)
            {
                this.Status = status;
                return true;
            }

            return false;
        }

        public bool Contains(ref Triangle triangle)
        {
            return Bounds.Contains(triangle.v0) && Bounds.Contains(triangle.v1) && Bounds.Contains(triangle.v2);
        }

        public bool Intersects(ref Triangle triangle)
        {
            var boxhalfsize = new Vector3(Length / 2.0f, Length / 2.0f, Length / 2.0f);
            var boxcenter = new Vector3(Bounds.MinX + boxhalfsize.X, Bounds.MinY + boxhalfsize.Y, Bounds.MinZ + boxhalfsize.Z);

            //return VoxelizerCPU.IsTriangleCollidingWithVoxel(ref triangle, ref deltap, ref minpt);
            return VoxelizingOctree.triBoxOverlap(ref boxcenter, ref boxhalfsize, ref triangle);
        }

        public bool IntersectsMeshBounds()
        {
            return Tree.MeshBounds.Intersects(Bounds);
        }

        public bool IsOutsideMeshBounds()
        {
            return Tree.MeshBounds.IsOutside(Bounds);
        }

        public void EncloseTriangles(VoxelizingOctreeCell parent)
        {
            for (int i = 0; i < parent.Triangles.Count; i++)
            {
                Triangle t = parent.Triangles[i];
                if (Contains(ref t))
                {
                    Triangles.Add(t);
                    Parent.Triangles.RemoveAt(i);
                    i--;
                }
            }

            if (parent.IsIntersecting)
            {
                TestTriangleIntersection();
            }
        }

        public void TestTriangleIntersection()
        {
            VoxelizingOctreeCell p = this;

            while (p != null)
            {
                for (int i = 0; i < p.Triangles.Count; i++)
                {
                    Triangle t = p.Triangles[i];
                    if (Intersects(ref t))
                    {
                        Status = CellStatus.Intersecting;
                        return;
                    }
                }

                p = p.Parent;
            }

            // If we're not intersecting any triangles make sure we're also not intersecting the mesh bounds.
            if (IntersectsMeshBounds())
            {
                Status = CellStatus.IntersectingBounds;
            }
        }

        public void RecursiveSubdivide(int level)
        {
            if (level <= 0)
                return;

            if (Subdivide())
            {
                for (int i = 0; i < Children.Count; i++)
                {
                    Children[i].RecursiveSubdivide(level - 1);
                }
            }
        }

        public bool Subdivide()
        {
            float quarter_length = Length / 4.0f;

            int stop = 8;
            for (int x = -1; x <= 1; x += 2)
            {
                for (int y = -1; y <= 1; y += 2)
                {
                    for (int z = -1; z <= 1; z += 2)
                    {
                        VoxelizingOctreeCell newCell = new VoxelizingOctreeCell(Tree, Root, 
                            Center + new Vector3(x * quarter_length, y * quarter_length, z * quarter_length),
                            quarter_length * 2.0f,
                            Level + 1
                        );

                        newCell.Parent = this;
                        newCell.EncloseTriangles(this);

                        if (newCell.IsOutsideMeshBounds())
                            newCell.Status = CellStatus.Outside;

                        if (!newCell.IsIntersecting)
                            stop--;

                        Children.Add(newCell);
                    }
                }
            }

            if (stop == 0)
            {
                //Debug.Assert(!IsIntersecting);
                if (IsIntersecting)
                {
                    //Debugger.Break();
                }

                Children.Clear();
            }

            return stop != 0;
        }

        public void Draw(int level, CellStatus status, Vector4 color, float width)
        {
            if (status == Status && level == 0)
            {
                GL.LineWidth(width);
                GL.Color4(color);

                GL.Begin(BeginMode.Lines);
                // Top
                GL.Vertex3(Bounds.MaxX, Bounds.MaxY, Bounds.MaxZ);
                GL.Vertex3(Bounds.MaxX, Bounds.MaxY, Bounds.MinZ);
                GL.Vertex3(Bounds.MaxX, Bounds.MaxY, Bounds.MinZ);
                GL.Vertex3(Bounds.MinX, Bounds.MaxY, Bounds.MinZ);
                GL.Vertex3(Bounds.MinX, Bounds.MaxY, Bounds.MinZ);
                GL.Vertex3(Bounds.MinX, Bounds.MaxY, Bounds.MaxZ);
                GL.Vertex3(Bounds.MinX, Bounds.MaxY, Bounds.MaxZ);
                GL.Vertex3(Bounds.MaxX, Bounds.MaxY, Bounds.MaxZ);
                // Bottom
                GL.Vertex3(Bounds.MaxX, Bounds.MinY, Bounds.MaxZ);
                GL.Vertex3(Bounds.MaxX, Bounds.MinY, Bounds.MinZ);
                GL.Vertex3(Bounds.MaxX, Bounds.MinY, Bounds.MinZ);
                GL.Vertex3(Bounds.MinX, Bounds.MinY, Bounds.MinZ);
                GL.Vertex3(Bounds.MinX, Bounds.MinY, Bounds.MinZ);
                GL.Vertex3(Bounds.MinX, Bounds.MinY, Bounds.MaxZ);
                GL.Vertex3(Bounds.MinX, Bounds.MinY, Bounds.MaxZ);
                GL.Vertex3(Bounds.MaxX, Bounds.MinY, Bounds.MaxZ);
                // Sides
                GL.Vertex3(Bounds.MaxX, Bounds.MaxY, Bounds.MaxZ);
                GL.Vertex3(Bounds.MaxX, Bounds.MinY, Bounds.MaxZ);
                GL.Vertex3(Bounds.MinX, Bounds.MaxY, Bounds.MaxZ);
                GL.Vertex3(Bounds.MinX, Bounds.MinY, Bounds.MaxZ);
                GL.Vertex3(Bounds.MinX, Bounds.MaxY, Bounds.MinZ);
                GL.Vertex3(Bounds.MinX, Bounds.MinY, Bounds.MinZ);
                GL.Vertex3(Bounds.MaxX, Bounds.MaxY, Bounds.MinZ);
                GL.Vertex3(Bounds.MaxX, Bounds.MinY, Bounds.MinZ);
                GL.End();
            }

            foreach (var cell in Children)
            {
                cell.Draw(level - 1, status, color, width);
            }
        }

        public void Find(List<VoxelizingOctreeCell> cellList, CellStatus status)
        {
            if (status == Status)
            {
                cellList.Add(this);
            }

            foreach (var cell in Children)
            {
                cell.Find(cellList, status);
            }
        }

        public void AccumulateChildren(List<List<VoxelizingOctreeCell>> cellList, int level)
        {
            cellList[level].Add(this);

            foreach (var cell in Children)
            {
                cell.AccumulateChildren(cellList, level + 1);
            }
        }

        public bool IsIntersecting
        {
            get { return Status == CellStatus.Intersecting || Status == CellStatus.IntersectingBounds; }
        }
    }

    public class VoxelizingOctree
    {
        public readonly int CellStatusAccumulationConfirmationThreshold;

        private int m_maxLevels;
        private VoxelizingOctreeCell m_root;

        public AABBf MeshBounds;
        public AABBf VoxelBounds;
        public float SideLength;
        public Vector3i VoxelSize;
        public Vector3i WorldVoxelOffset;
        public double SmallestVoxelSideLength;

        public VoxelizingOctree(int maxLevels)
        {
            if (maxLevels < 1)
                throw new ArgumentException("maxLevels must be >= 1.");

            CellStatusAccumulationConfirmationThreshold = 1 << (maxLevels - 1);

            m_maxLevels = maxLevels;
        }

        public VoxelizingOctreeCell Root
        {
            get { return m_root; }
        }

        public int MaxLevels
        {
            get { return m_maxLevels; }
        }

        public void AccumulateChildren(out List<List<VoxelizingOctreeCell>> cellList)
        {
            cellList = new List<List<VoxelizingOctreeCell>>();
            for (int i = 0; i < MaxLevels; i++)
                cellList.Add(new List<VoxelizingOctreeCell>());

            Root.AccumulateChildren(cellList, 0);
        }

        public bool GenerateOctree(MeshData mesh)
        {
            if (mesh == null)
                return false;

            // Create a list of triangles from the list of faces in the model
            List<Triangle> triangles = new List<Triangle>();
            for (int i = 0; i < mesh.Tris.Length; i++)
            {
                Tri face = mesh.Tris[i];

                Triangle tri = new Triangle();
                tri.v0 = mesh.Vertices[face.P1.Vertex];
                tri.v1 = mesh.Vertices[face.P2.Vertex];
                tri.v2 = mesh.Vertices[face.P3.Vertex];

                triangles.Add(tri);
            }

            // Determine the axis-aligned bounding box for the triangles
            Vector3 center;
            CreateUniformBoundingBox(triangles, out MeshBounds, out VoxelBounds, out center, out SideLength);

            {
                SmallestVoxelSideLength = SideLength;
                for (int i = 1; i < m_maxLevels; i++)
                    SmallestVoxelSideLength *= 0.5;

                VoxelSize = new Vector3i();
                VoxelSize.X = (Int32)Math.Pow(2, m_maxLevels);
                VoxelSize.Y = (Int32)Math.Pow(2, m_maxLevels);
                VoxelSize.Z = (Int32)Math.Pow(2, m_maxLevels);
            }

            m_root = new VoxelizingOctreeCell(this, null, center, SideLength, 0);
            m_root.Root = m_root;
            m_root.Triangles = new List<Triangle>(triangles);
            m_root.Status = CellStatus.IntersectingBounds;
            m_root.RecursiveSubdivide(m_maxLevels - 1);

            WorldVoxelOffset = new Vector3i(
                0 - Root.VoxelBounds.MinX,
                0 - Root.VoxelBounds.MinY,
                0 - Root.VoxelBounds.MinZ);

            return true;
        }

        private void CreateUniformBoundingBox(List<Triangle> triangles, out AABBf originalBounds, out AABBf voxelBounds, out Vector3 center, out float length)
        {
            originalBounds = Triangle.CreateBoundingBox(triangles);
            Vector3 size = new Vector3(originalBounds.MaxX - originalBounds.MinX, originalBounds.MaxY - originalBounds.MinY, originalBounds.MaxZ - originalBounds.MinZ);
            float maxSize = Math.Max(size.X, Math.Max(size.Y, size.Z));

            center = new Vector3(
                originalBounds.MinX + (size.X / 2.0f),
                originalBounds.MinY + (size.Y / 2.0f),
                originalBounds.MinZ + (size.Z / 2.0f));

            length = maxSize;

            voxelBounds = new AABBf();
            voxelBounds.MinX = center.X - (length * 0.5f);
            voxelBounds.MinY = center.Y - (length * 0.5f);
            voxelBounds.MinZ = center.Z - (length * 0.5f);
            voxelBounds.MaxX = center.X + (length * 0.5f);
            voxelBounds.MaxY = center.Y + (length * 0.5f);
            voxelBounds.MaxZ = center.Z + (length * 0.5f);
        }

        public void Draw(int level, CellStatus status, Vector4 color, float width)
        {
            m_root.Draw(level, status, color, width);
        }

        public List<VoxelizingOctreeCell> Find(CellStatus status)
        {
            List<VoxelizingOctreeCell> cellList = new List<VoxelizingOctreeCell>();
            m_root.Find(cellList, status);
            return cellList;
        }






























































        public static void FINDMINMAX(float x0, float x1, float x2, out float min, out float max)
        {
            min = max = x0;
            if (x1 < min) min = x1;
            if (x1 > max) max = x1;
            if (x2 < min) min = x2;
            if (x2 > max) max = x2;
        }

        public static bool planeBoxOverlap(Vector3 normal, float d, Vector3 maxbox)
        {
            Vector3 vmin = new Vector3();
            vmin.X = (normal.X > 0.0f) ? -maxbox.X : maxbox.X;
            vmin.Y = (normal.Y > 0.0f) ? -maxbox.Y : maxbox.Y;
            vmin.Z = (normal.Z > 0.0f) ? -maxbox.Z : maxbox.Z;

            Vector3 vmax = new Vector3();
            vmax.X = (normal.X > 0.0f) ? maxbox.X : -maxbox.X;
            vmax.Y = (normal.Y > 0.0f) ? maxbox.Y : -maxbox.Y;
            vmax.Z = (normal.Z > 0.0f) ? maxbox.Z : -maxbox.Z;

            if (Vector3.Dot(normal, vmin) + d > 0.0f) return false;
            if (Vector3.Dot(normal, vmax) + d >= 0.0f) return true;

            return false;
        }


        /*======================== X-tests ========================*/
        public static bool AXISTEST_X01(float a, float b, float fa, float fb, ref Vector3 v0, ref Vector3 v2, ref Vector3 boxhalfsize)
        {
            float min = 0, max = 0;
            float p0 = a * v0.Y - b * v0.Z;
            float p2 = a * v2.Y - b * v2.Z;
            if (p0 < p2) { min = p0; max = p2; } else { min = p2; max = p0; }
            float rad = fa * boxhalfsize.Y + fb * boxhalfsize.Z;
            if (min > rad || max < -rad)
                return true;
            return false;
        }

        public static bool AXISTEST_X2(float a, float b, float fa, float fb, ref Vector3 v0, ref Vector3 v1, ref Vector3 boxhalfsize)
        {
            float min = 0, max = 0;
            float p0 = a * v0.Y - b * v0.Z;
            float p1 = a * v1.Y - b * v1.Z;
            if (p0 < p1) { min = p0; max = p1; } else { min = p1; max = p0; }
            float rad = fa * boxhalfsize.Y + fb * boxhalfsize.Z;
            if (min > rad || max < -rad)
                return true;
            return false;
        }

        /*======================== Y-tests ========================*/
        public static bool AXISTEST_Y02(float a, float b, float fa, float fb, ref Vector3 v0, ref Vector3 v2, ref Vector3 boxhalfsize)
        {
            float min = 0, max = 0;
            float p0 = -a * v0.X + b * v0.Z;
            float p2 = -a * v2.X + b * v2.Z;
            if (p0 < p2) { min = p0; max = p2; } else { min = p2; max = p0; }
            float rad = fa * boxhalfsize.X + fb * boxhalfsize.Z;
            if (min > rad || max < -rad)
                return true;
            return false;
        }

        public static bool AXISTEST_Y1(float a, float b, float fa, float fb, ref Vector3 v0, ref Vector3 v1, ref Vector3 boxhalfsize)
        {
            float min = 0, max = 0;
            float p0 = -a * v0.X + b * v0.Z;
            float p1 = -a * v1.X + b * v1.Z;
            if (p0 < p1) { min = p0; max = p1; } else { min = p1; max = p0; }
            float rad = fa * boxhalfsize.X + fb * boxhalfsize.Z;
            if (min > rad || max < -rad)
                return true;
            return false;
        }

        /*======================== Z-tests ========================*/

        public static bool AXISTEST_Z12(float a, float b, float fa, float fb, ref Vector3 v1, ref Vector3 v2, ref Vector3 boxhalfsize)
        {
            float min = 0, max = 0;
            float p1 = a * v1.X - b * v1.Y;
            float p2 = a * v2.X - b * v2.Y;
            if (p2 < p1) { min = p2; max = p1; } else { min = p1; max = p2; }
            float rad = fa * boxhalfsize.X + fb * boxhalfsize.Y;
            if (min > rad || max < -rad)
                return true;
            return false;
        }

        public static bool AXISTEST_Z0(float a, float b, float fa, float fb, ref Vector3 v0, ref Vector3 v1, ref Vector3 boxhalfsize)
        {
            float min = 0, max = 0;
            float p0 = a * v0.X - b * v0.Y;
            float p1 = a * v1.X - b * v1.Y;
            if (p0 < p1) { min = p0; max = p1; } else { min = p1; max = p0; }
            float rad = fa * boxhalfsize.X + fb * boxhalfsize.Y;
            if (min > rad || max < -rad) return true;
            return false;
        }

        public static bool triBoxOverlap(ref Vector3 boxcenter, ref Vector3 boxhalfsize, ref Triangle tri)
        {
            /*    use separating axis theorem to test overlap between triangle and box */
            /*    need to test for overlap in these directions: */
            /*    1) the {x,y,z}-directions (actually, since we use the AABB of the triangle */
            /*       we do not even need to test these) */
            /*    2) normal of the triangle */
            /*    3) crossproduct(edge from tri, {x,y,z}-directin) */
            /*       this gives 3x3=9 more tests */

            /* This is the fastest branch on Sun */
            /* move everything so that the boxcenter is in (0,0,0) */
            Vector3 v0 = tri.v0 - boxcenter;
            Vector3 v1 = tri.v1 - boxcenter;
            Vector3 v2 = tri.v2 - boxcenter;

            /* compute triangle edges */
            Vector3 e0 = v1 - v0;      /* tri edge 0 */
            Vector3 e1 = v2 - v1;      /* tri edge 1 */
            Vector3 e2 = v0 - v2;      /* tri edge 2 */

            /* Bullet 3:  */
            /*  test the 9 tests first (this was faster) */
            float fex = Math.Abs(e0.X);
            float fey = Math.Abs(e0.Y);
            float fez = Math.Abs(e0.Z);
            if (AXISTEST_X01(e0.Z, e0.Y, fez, fey, ref v0, ref v2, ref boxhalfsize))
                return false;
            if (AXISTEST_Y02(e0.Z, e0.X, fez, fex, ref v0, ref v2, ref boxhalfsize))
                return false;
            if (AXISTEST_Z12(e0.Y, e0.X, fey, fex, ref v1, ref v2, ref boxhalfsize))
                return false;

            fex = Math.Abs(e1.X);
            fey = Math.Abs(e1.Y);
            fez = Math.Abs(e1.Z);
            if (AXISTEST_X01(e1.Z, e1.Y, fez, fey, ref v0, ref v2, ref boxhalfsize))
                return false;
            if (AXISTEST_Y02(e1.Z, e1.X, fez, fex, ref v0, ref v2, ref boxhalfsize))
                return false;
            if (AXISTEST_Z0(e1.Y, e1.X, fey, fex, ref v0, ref v1, ref boxhalfsize))
                return false;

            fex = Math.Abs(e2.X);
            fey = Math.Abs(e2.Y);
            fez = Math.Abs(e2.Z);
            if (AXISTEST_X2(e2.Z, e2.Y, fez, fey, ref v0, ref v1, ref boxhalfsize))
                return false;
            if (AXISTEST_Y1(e2.Z, e2.X, fez, fex, ref v0, ref v1, ref boxhalfsize))
                return false;
            if (AXISTEST_Z12(e2.Y, e2.X, fey, fex, ref v1, ref v2, ref boxhalfsize))
                return false;

            /* Bullet 1: */
            /*  first test overlap in the {x,y,z}-directions */
            /*  find min, max of the triangle each direction, and test for overlap in */
            /*  that direction -- this is equivalent to testing a minimal AABB around */
            /*  the triangle against the AABB */

            /* test in X-direction */
            float min, max;
            FINDMINMAX(v0.X, v1.X, v2.X, out min, out max);
            if (min > boxhalfsize.X || max < -boxhalfsize.X)
                return false;

            /* test in Y-direction */
            FINDMINMAX(v0.Y, v1.Y, v2.Y, out min, out max);
            if (min > boxhalfsize.Y || max < -boxhalfsize.Y)
                return false;

            /* test in Z-direction */
            FINDMINMAX(v0.Z, v1.Z, v2.Z, out min, out max);
            if (min > boxhalfsize.Z || max < -boxhalfsize.Z)
                return false;

            /* Bullet 2: */
            /*  test if the box intersects the plane of the triangle */
            /*  compute plane equation of triangle: normal*x+d=0 */
            Vector3 normal;
            Vector3.Cross(ref e0, ref e1, out normal);
            float d;
            Vector3.Dot(ref normal, ref v0, out d);  /* plane eq: normal.x+d=0 */
            d = -d;
            if (!planeBoxOverlap(normal, d, boxhalfsize))
                return false;

            return true;   /* box and triangle overlaps */
        }
    }
}
