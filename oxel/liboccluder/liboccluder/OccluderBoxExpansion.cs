using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Cloo;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;

namespace Oxel
{
    public partial class OccluderBoxExpansion : IOccluderGenerator
    {
        struct Occluder
        {
            public AABBi Bounds;
            public double DeltaOcclusion;
        }

        public virtual VoxelizationOutput Generate(VoxelizationInput input, Action<VoxelizationProgress> progress)
        {
            VoxelizationProgress vp = new VoxelizationProgress();

            DateTime start = DateTime.Now;

            vp.Status = "Building voxel field from octree";
            progress(vp);

            VoxelField voxelField = new VoxelField(input.Octree);

            Byte fillByte = 2;
            float oldPercent = 1.0f;
            float newPercent = 1.0f;

            List<Occluder> occluders = new List<Occluder>();

            vp.Status = "Calculating original mesh silhouette coverage";
            progress(vp);

            SilhouetteOcclusionValidator sov = new SilhouetteOcclusionValidator(1024, 1024);

            long groundSideCoverage, groundTopCoverage;
            sov.ComputeCoverage(input.OriginalMesh, input.Octree.MeshBounds, out groundSideCoverage, out groundTopCoverage);
            long totalCoverage = groundSideCoverage + groundTopCoverage;
            if (totalCoverage == 0)
                totalCoverage = 1;

            vp.Status = "Fitting boxes into mesh...";
            progress(vp);

            long oldOcclusion = 0;

            do
            {
                Vector3i densestVoxel = FindHighestDensityVoxel(voxelField);

                AABBi occluderBounds;
                if (input.Type == OcclusionType.BoxExpansion)
                {
                    occluderBounds = ExpandAndFillBox(voxelField, ref densestVoxel, fillByte);
                }
                //else if (input.Type == OcclusionType.SimulatedAnnealing)
                //{
                //    occluderBounds = SimulatedAnnealingFill(input, sov, voxelField, ref densestVoxel, fillByte, occluders);
                //}
                else if (input.Type == OcclusionType.BruteForce)
                {
                    occluderBounds = BruteForceFill(input, sov, voxelField, densestVoxel, fillByte, occluders);
                }
                else
                {
                    throw new Exception("Unknown occluder generation type!");
                }

                List<AABBi> relevantOccluders = GetRelevantOccluders(input, occluders);
                relevantOccluders.Add(occluderBounds);

                long newOcclusion = MeasureOccluderOcclusion(sov, input, relevantOccluders);

                Occluder occluder = new Occluder();
                occluder.Bounds = occluderBounds;
                occluder.DeltaOcclusion = (newOcclusion - oldOcclusion) / (double)totalCoverage;

                occluders.Add(occluder);

                if (occluder.DeltaOcclusion > input.MinimumOcclusion)
                    oldOcclusion = newOcclusion;

                Debug.WriteLine("Coverage " + occluder.DeltaOcclusion);
                Debug.WriteLine("Bounds (" + occluder.Bounds.MinX + "x" + occluder.Bounds.MaxX + " " + occluder.Bounds.MinY + "x" + occluder.Bounds.MaxY + " " + occluder.Bounds.MinZ + "x" + occluder.Bounds.MaxZ + ")");

                oldPercent = newPercent;
                newPercent = MeasureUnboxedVoxels(voxelField);

                Debug.WriteLine("(" + densestVoxel.X + "," + densestVoxel.Y + "," + densestVoxel.Z + ")\tCoverage=" + ((1 - newPercent) * 100) + "%\tDelta=" + ((oldPercent - newPercent) * 100) + "%");

                vp.Progress = Math.Min(((1 - newPercent) / input.MinimumVolume), 1.0f);
                vp.SilhouetteCoverage = oldOcclusion / (double)totalCoverage;
                vp.VolumeCoverage = 1 - newPercent;
                vp.Status = String.Format("Occlusion Progress : {0:0.##}%", (100 * vp.Progress));

                progress(vp);

            } while (newPercent > (1 - input.MinimumVolume));

            Mesh mesh = BuildMeshFromBoxes(input, GetRelevantOccluders(input, occluders));

            VoxelizationOutput output = new VoxelizationOutput();

            if (input.Retriangulate)
            {
                vp.Status = "Retriangulating occluder mesh";
                progress(vp);

                Mesh triangulatedMesh = MeshOptimizer.Retriangulate(input, mesh, out output.DebugLines);
                if (triangulatedMesh != null)
                    mesh = triangulatedMesh;
            }

            vp.Status = "Filtering polygons";
            progress(vp);

            mesh = PolygonFilter.Filter(input, mesh);

            vp.Status = "Generating final occlusion mesh";
            progress(vp);

            // Prepare the output
            output.Octree = input.Octree;
            output.TimeTaken = DateTime.Now - start;
            output.VolumeCoverage = 1 - newPercent;
            output.SilhouetteCoverage = oldOcclusion / (double)totalCoverage;
            output.OccluderMesh = new RenderableMesh(mesh, true);

            vp.Status = "Cleanup...";
            progress(vp);

            sov.Dispose();

            return output;
        }

        List<AABBi> GetRelevantOccluders(VoxelizationInput input, List<Occluder> occluders)
        {
            var occluderBounds =
                from occluder in occluders
                where occluder.DeltaOcclusion > input.MinimumOcclusion
                orderby occluder.DeltaOcclusion descending
                select occluder.Bounds;

            return occluderBounds.ToList();
        }

        long MeasureOccluderOcclusion(SilhouetteOcclusionValidator sov, VoxelizationInput input, List<AABBi> occluderBounds)
        {
            Mesh mesh = BuildMeshFromBoxes(input, occluderBounds);
            RenderableMesh renderable = new RenderableMesh(mesh, true);

            long sideCoverage, topCoverage;
            sov.ComputeCoverage(renderable, input.Octree.MeshBounds, out sideCoverage, out topCoverage);

            renderable.Dispose();

            return sideCoverage + topCoverage;
        }

        Mesh BuildMeshFromBoxes(VoxelizationInput input, List<AABBi> boxes)
        {
            // Build the mesh, this will also remove all false triangle loops and collinear point triangles.
            Vector3 deltaP = new Vector3((float)input.Octree.SmallestVoxelSideLength, (float)input.Octree.SmallestVoxelSideLength, (float)input.Octree.SmallestVoxelSideLength);

            return MeshBuilder.BuildMesh(input.Octree.VoxelBounds, deltaP, boxes);
        }

        AABBi BruteForceFill(VoxelizationInput input, SilhouetteOcclusionValidator sov, VoxelField voxelField, Vector3i densestVoxel, byte fillByte, List<Occluder> currentOccluders)
        {
            Object syncroot = new Object();
            Int64 largestVolume = 1;
            AABBi largestOccluder = new AABBi(densestVoxel.X, densestVoxel.Y, densestVoxel.Z, densestVoxel.X + 1, densestVoxel.Y + 1, densestVoxel.Z + 1);

            int MaxTopOccluders = 2000;
            List<AABBi> bestOccluders = new List<AABBi>(MaxTopOccluders);

            Parallel.For(densestVoxel.Z + 1, voxelField.VoxelSize.Z, max_z =>
            {
                for (Int32 min_z = densestVoxel.Z; min_z >= 0; --min_z)
                {
                    for (Int32 max_y = densestVoxel.Y + 1; max_y < voxelField.VoxelSize.Y; ++max_y)
                    {
                        for (Int32 min_y = densestVoxel.Y; min_y >= 0; --min_y)
                        {
                            for (Int32 max_x = densestVoxel.X + 1; max_x < voxelField.VoxelSize.X; ++max_x)
                            {
                                for (Int32 min_x = densestVoxel.X; min_x >= 0; --min_x)
                                {
                                    Int32 dx = max_x - min_x;
                                    Int32 dy = max_y - min_y;
                                    Int32 dz = max_z - min_z;
                                    Int64 volume = dx * dy * dz;

                                    if (TestRangeForFreeSpace(voxelField, new AABBi(min_x, min_y, min_z, max_x, max_y, max_z)))
                                    {
                                        lock (syncroot)
                                        {
                                            if (volume > largestVolume)
                                            {
                                                largestVolume = volume;
                                                largestOccluder = new AABBi(min_x, min_y, min_z, max_x, max_y, max_z);
                                                if (bestOccluders.Count >= MaxTopOccluders)
                                                    bestOccluders.RemoveAt(MaxTopOccluders - 1);
                                                bestOccluders.Insert(0, largestOccluder);
                                            }
                                        }
                                    }
                                    else
                                    {
                                        // if we can't expand outward any further there's no point in checking more.
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                Debug.WriteLine("Checked " + max_z);
            });

            List<AABBi> relevantOccluders = GetRelevantOccluders(input, currentOccluders);

            long bestCoverage = 0;
            AABBi bestCoverageVolume = largestOccluder;
            foreach (AABBi occluder in bestOccluders)
            {
                List<AABBi> tempOccluders = relevantOccluders.ToList();
                tempOccluders.Add(occluder);
                long coverage = MeasureOccluderOcclusion(sov, input, tempOccluders);
                if (coverage > bestCoverage)
                {
                    bestCoverage = coverage;
                    bestCoverageVolume = occluder;
                }
            }

            FillRange(voxelField, bestCoverageVolume, fillByte);

            return bestCoverageVolume;
        }
        
        void ComputeNext(Random random, AABBi current, AABBi next, VoxelField volume, ref Vector3i densestVoxel, long delta, double temperature)
        {
            current.Clone(next);

            do
            {
                double probability = random.NextDouble();
                if (probability < Math.Exp(-delta / temperature))
                    next.MinX = densestVoxel.X + random.Next(0 - densestVoxel.X, 0);
                probability = random.NextDouble();
                if (probability < Math.Exp(-delta / temperature))
                    next.MinY = densestVoxel.Y + random.Next(0 - densestVoxel.Y, 0);
                probability = random.NextDouble();
                if (probability < Math.Exp(-delta / temperature))
                    next.MinZ = densestVoxel.Z + random.Next(0 - densestVoxel.Z, 0);

                probability = random.NextDouble();
                if (probability < Math.Exp(-delta / temperature))
                    next.MaxX = densestVoxel.X + random.Next(0, volume.VoxelSize.X - densestVoxel.X) + 1;
                probability = random.NextDouble();
                if (probability < Math.Exp(-delta / temperature))
                    next.MaxY = densestVoxel.Y + random.Next(0, volume.VoxelSize.Y - densestVoxel.Y) + 1;
                probability = random.NextDouble();
                if (probability < Math.Exp(-delta / temperature))
                    next.MaxZ = densestVoxel.Z + random.Next(0, volume.VoxelSize.Z - densestVoxel.Z) + 1;

            } while (!TestRangeForFreeSpace(volume, 
                new Vector3i(next.MinX, next.MinY, next.MinZ),
                new Vector3i(next.MaxX, next.MaxY, next.MaxZ)));
        }

        AABBi SimulatedAnnealingFill(VoxelizationInput input, SilhouetteOcclusionValidator sov, VoxelField volume, ref Vector3i densestVoxel, byte fillByte, List<Occluder> currentOccluders)
        {
            AABBi current = new AABBi(densestVoxel.X, densestVoxel.Y, densestVoxel.Z, densestVoxel.X + 1, densestVoxel.Y + 1, densestVoxel.Z + 1);
            AABBi next = new AABBi(0, 0, 0, 0, 0, 0);

            int iteration = -1;

            List<AABBi> relevantOccluders = GetRelevantOccluders(input, currentOccluders);

            List<AABBi> occluders = relevantOccluders.ToList();
            occluders.Add(current);
            long coverage = MeasureOccluderOcclusion(sov, input, occluders);

            double coolignAlpha = 0.999;
            double temperature = 400.0;
            double epsilon = 0.001;

            Random random = new Random(1337);

            int maxItterations = 1000;

            long delta = 0;
            while (temperature > epsilon && iteration < maxItterations)
            {
                iteration++;

                ComputeNext(random, current, next, volume, ref densestVoxel, delta, temperature);

                occluders = relevantOccluders.ToList();
                occluders.Add(next);
                delta = MeasureOccluderOcclusion(sov, input, occluders) - coverage;

                if (delta < 0)
                {
                    next.Clone(current);
                    coverage = delta + coverage;
                }
                else
                {
                    double probability = random.NextDouble();

                    if (probability < Math.Exp(-delta / temperature))
                    {
                        next.Clone(current);
                        coverage = delta + coverage;
                    }
                }

                temperature *= coolignAlpha;

                if (iteration % 400 == 0)
                    Console.WriteLine(coverage);
            }

            FillRange(volume,
                new Vector3i(current.MinX, current.MinY, current.MinZ),
                new Vector3i(current.MaxX, current.MaxY, current.MaxZ),
                fillByte);

            return current;
        }
    }
}
