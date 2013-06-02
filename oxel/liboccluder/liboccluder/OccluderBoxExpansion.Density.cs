using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Oxel
{
    public partial class OccluderBoxExpansion
    {
        protected static Vector3i FindHighestDensityVoxel(VoxelField volume)
        {
            float denestDistance = float.MinValue;
            Vector3i densestVoxel = new Vector3i(0, 0, 0);

            Object syncroot = new Object();

            Parallel.For(0, volume.VoxelSize.Z, z =>
            {
                for (Int32 y = 0; y < volume.VoxelSize.Y; ++y)
                {
                    for (Int32 x = 0; x < volume.VoxelSize.X; ++x)
                    {
                        byte value = volume.GetVoxel(x, y, z);

                        // Ignore empty voxels and already boxed voxels
                        if (value != 1)
                            continue;

                        float closestExtDist = FindShortestDistanceToAbnormalVoxel(volume, x, y, z);

                        if (closestExtDist > denestDistance)
                        {
                            lock (syncroot)
                            {
                                if (closestExtDist > denestDistance)
                                {
                                    denestDistance = closestExtDist;
                                    densestVoxel = new Vector3i(x, y, z);
                                }
                            }
                        }
                    }
                }
            });

            return densestVoxel;
        }

        protected static float FindShortestDistanceToAbnormalVoxel(VoxelField volume, int x, int y, int z)
        {
            int pX, nX, pY, nY, pZ, nZ;
            pX = nX = pY = nY = pZ = nZ = 0;

            Vector3i voxel = new Vector3i(x, y, z);

            do
            {
                // +Z Axis
                float distance = FindDistanceToAbnormalVoxelInRange(volume, voxel,
                    x - nX, y - nY, z + (pZ + 1),
                    x + pX, y + pY, z + (pZ + 1));
                if (distance != float.MaxValue)
                    return distance;
                pZ++;

                // -Z Axis
                distance = FindDistanceToAbnormalVoxelInRange(volume, voxel,
                    x - nX, y - nY, z - (nZ + 1),
                    x + pX, y + pY, z - (nZ + 1));
                if (distance != float.MaxValue)
                    return distance;
                nZ++;

                // +Y Axis
                distance = FindDistanceToAbnormalVoxelInRange(volume, voxel,
                    x - nX, y + (pY + 1), z - nZ,
                    x + pX, y + (pY + 1), z + pZ);
                if (distance != float.MaxValue)
                    return distance;
                pY++;

                // -Y Axis
                distance = FindDistanceToAbnormalVoxelInRange(volume, voxel,
                    x - nX, y - (nY + 1), z - nZ,
                    x + pX, y - (nY + 1), z + pZ);
                if (distance != float.MaxValue)
                    return distance;
                nY++;

                // +X Axis
                distance = FindDistanceToAbnormalVoxelInRange(volume, voxel,
                    x + (pX + 1), y - nY, z - nZ,
                    x + (pX + 1), y + pY, z + pZ);
                if (distance != float.MaxValue)
                    return distance;
                pX++;

                // -X Axis
                distance = FindDistanceToAbnormalVoxelInRange(volume, voxel,
                    x - (nX + 1), y - nY, z - nZ,
                    x - (nX + 1), y + pY, z + pZ);
                if (distance != float.MaxValue)
                    return distance;
                nX++;

            } while (true);
        }

        protected static float FindDistanceToAbnormalVoxelInRange(
            VoxelField volume,
            Vector3i voxel,
            int startX, int startY, int startZ,
            int endX, int endY, int endZ)
        {
            // 1. Ignore directions that take us outside the bounds of the voxel volume.
            if (startX < 0 || startY < 0 || startZ < 0)
                return float.MaxValue;

            float closestValueDist = float.MaxValue;

            // 2. Find the value along the axis
            for (Int32 z = startZ; z <= endZ; ++z)
            {
                for (Int32 y = startY; y <= endY; ++y)
                {
                    for (Int32 x = startX; x <= endX; ++x)
                    {
                        byte value = volume.GetVoxel(x, y, z);
                        if (value != 1)
                        {
                            float dist = (float)(voxel - new Vector3i(x, y, z)).LengthSquared;
                            if (dist < closestValueDist)
                            {
                                closestValueDist = dist;
                            }
                        }
                    }
                }
            }

            return closestValueDist;
        }
    }
}
