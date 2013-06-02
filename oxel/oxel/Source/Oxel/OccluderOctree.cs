using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using Oxel.OpenGL;

namespace Oxel
{
    class VolumeAccumulator
    {
        public float OutsideTotal;
        public float InsideTotal;
        public float IntersectingTotal;
    }

    struct RGB
    {
        public byte r;
        public byte g;
        public byte b;
    }

    class RedBlueTuple
    {
        public int R_Total;
        public int B_Total;
    }

    public class OccluderOctree : IOccluderGenerator
    {
        public VoxelizationInput input;

        public VoxelizationOutput Generate(VoxelizationInput input, Action<VoxelizationProgress> progress)
        {
            this.input = input;
            VoxelizationOutput output = new VoxelizationOutput();
            output.Octree = input.Octree;

            List<List<VoxelizingOctreeCell>> cellList = new List<List<VoxelizingOctreeCell>>();
            input.Octree.AccumulateChildren(out cellList);

            VolumeAccumulator volume = new VolumeAccumulator();
            VolumeAccumulator[] volumeAtLevel = new VolumeAccumulator[input.Octree.MaxLevels];
            for (int i = 0; i < input.Octree.MaxLevels; i++)
            {
                List<VoxelizingOctreeCell> childernAtDepth = cellList[i];

                VolumeAccumulator levelVolumeTotal = new VolumeAccumulator();

                Parallel.For(0, childernAtDepth.Count, () => new VolumeAccumulator(), (n, loop, partial) =>
                {
                    VoxelizingOctreeCell cell = childernAtDepth[n];
                    float sideLength = cell.Length;

                    switch (cell.Status)
                    {
                        case CellStatus.Inside:
                            partial.InsideTotal += (sideLength * sideLength * sideLength);
                            break;
                        case CellStatus.Outside:
                            partial.OutsideTotal += (sideLength * sideLength * sideLength);
                            break;
                        case CellStatus.Intersecting:
                        case CellStatus.IntersectingBounds:
                            if (cell.IsLeaf)
                                partial.IntersectingTotal += (sideLength * sideLength * sideLength);
                            break;
                    }

                    return partial;
                },
                partial =>
                {
                    lock (levelVolumeTotal)
                    {
                        levelVolumeTotal.InsideTotal += partial.InsideTotal;
                        levelVolumeTotal.OutsideTotal += partial.OutsideTotal;
                        levelVolumeTotal.IntersectingTotal += partial.IntersectingTotal;
                    }
                });

                volume.InsideTotal += levelVolumeTotal.InsideTotal;
                volume.OutsideTotal += levelVolumeTotal.OutsideTotal;
                volume.IntersectingTotal += levelVolumeTotal.IntersectingTotal;

                volumeAtLevel[i] = levelVolumeTotal;
            }


            Debug.WriteLine("Percentage of inner volume at each octree level");
            for (int i = 0; i < input.Octree.MaxLevels; i++)
            {
                Debug.WriteLine("Level {0}: Inner Volume {1}%", i, (volumeAtLevel[i].InsideTotal / volume.InsideTotal) * 100);
            }

            // A good check to perform is to compare the ratio of intersecting volume leaf nodes to the total volume 
            // we've determined is inside.  A tool could use this ratio to automatically determine a good octree level
            // by iterative optimization.  If a mesh for example fails to get at least a 1 : 0.5 ratio of intersecting:inner 
            // volume ratio it's a good bet that the octree does not subdivide enough levels in order to find enough inner volume
            // to meet our occlusion needs.  If further subdivision up to some maximum, lets say 8 fails to ever meet this ratio
            // one could say the mesh is not a good candidate for automating occluder generation.
            Debug.WriteLine("");
            float intersecting_inside_ratio = volume.InsideTotal / volume.IntersectingTotal;
            Debug.WriteLine("Intersecting : Inner = 1:{0}", intersecting_inside_ratio);
            Debug.WriteLine("Inner / (Inner + Intersecting) = {0}", volume.InsideTotal / (volume.InsideTotal + volume.IntersectingTotal));

            const float MINIMUM_INTERSECTING_TO_INSIDE_RATIO = 0.25f;

            AABBf meshBounds = input.Octree.MeshBounds;
            double dX = meshBounds.MaxX - meshBounds.MinX;
            double dY = meshBounds.MaxY - meshBounds.MinY;
            double dZ = meshBounds.MaxZ - meshBounds.MinZ;

            double reduction = 0.5;
            for (int i = 0; i <= input.Octree.MaxLevels * 2; i++)
                reduction *= 0.5;

            dX = dX * reduction;
            dY = dY * reduction;
            dZ = dZ * reduction;

            if (intersecting_inside_ratio > MINIMUM_INTERSECTING_TO_INSIDE_RATIO)
            {
                List<AABBi> innerBounds = new List<AABBi>();
                float innerVolumeGathered = 0.0f;
                for (int i = 0; i < input.Octree.MaxLevels; i++)
                {
                    for (int n = 0; n < cellList[i].Count; n++)
                    {
                        if (cellList[i][n].Status == CellStatus.Inside)
                        {
                            AABBf bound = cellList[i][n].Bounds;

                            AABBi bi = new AABBi();
                            bi.MaxX = (int)Math.Round(((double)bound.MaxX - (double)meshBounds.MinX) / dX, MidpointRounding.AwayFromZero);
                            bi.MaxY = (int)Math.Round(((double)bound.MaxY - (double)meshBounds.MinY) / dY, MidpointRounding.AwayFromZero);
                            bi.MaxZ = (int)Math.Round(((double)bound.MaxZ - (double)meshBounds.MinZ) / dZ, MidpointRounding.AwayFromZero);
                            bi.MinX = (int)Math.Round(((double)bound.MinX - (double)meshBounds.MinX) / dX, MidpointRounding.AwayFromZero);
                            bi.MinY = (int)Math.Round(((double)bound.MinY - (double)meshBounds.MinY) / dY, MidpointRounding.AwayFromZero);
                            bi.MinZ = (int)Math.Round(((double)bound.MinZ - (double)meshBounds.MinZ) / dZ, MidpointRounding.AwayFromZero);
                            innerBounds.Add(bi);
                        }
                    }

                    innerVolumeGathered += volumeAtLevel[i].InsideTotal / volume.InsideTotal;
                    if (innerVolumeGathered > input.MinimumVolume)
                    {
                        break;
                    }
                }

                Debug.WriteLine("Enough inner volume found {0}%", innerVolumeGathered * 100.0f);

                Mesh mesh = MeshBuilder.BuildMesh(innerBounds);

                for (int i = 0; i < mesh.Vertices.Length; i++)
                {
                    mesh.Vertices[i].X = (float)(((double)meshBounds.MinX) + (mesh.Vertices[i].X * dX));
                    mesh.Vertices[i].Y = (float)(((double)meshBounds.MinY) + (mesh.Vertices[i].Y * dY));
                    mesh.Vertices[i].Z = (float)(((double)meshBounds.MinZ) + (mesh.Vertices[i].Z * dZ));
                }

                if (input.Retriangulate)
                {
                    Mesh triangulatedMesh = MeshOptimizer.Retriangulate(input, mesh, out output.DebugLines);
                    if (triangulatedMesh != null)
                        mesh = triangulatedMesh;
                }

                mesh = PolygonFilter.Filter(input, mesh);
                
                output.OccluderMesh = new RenderableMesh(mesh, true);
            }
            else
            {
                Debug.WriteLine("Not enough inner volume found to continue.");
            }

            return output;
        }
    }
}