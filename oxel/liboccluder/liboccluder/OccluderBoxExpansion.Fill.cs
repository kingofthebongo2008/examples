using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Oxel
{
    public partial class OccluderBoxExpansion
    {
        protected static bool TestRangeForFreeSpace(VoxelField volume, AABBi box)
        {
            return TestRangeForFreeSpace(volume, new Vector3i(box.MinX, box.MinY, box.MinZ), new Vector3i(box.MaxX - 1, box.MaxY - 1, box.MaxZ - 1));
        }

        protected static bool TestRangeForFreeSpace(VoxelField volume, Vector3i start, Vector3i end)
        {
            // 1. Ignore directions that take us outside the bounds of the voxel volume.
            if (start.X < 0 || start.Y < 0 || start.Z < 0)
                return false;

            // 2. Ensure that we can safely expand into the area, expand as long as the voxel isn't empty.
            //    Note: it is ok to expand into a space already occupied.
            for (Int32 z = start.Z; z <= end.Z; ++z)
            {
                for (Int32 y = start.Y; y <= end.Y; ++y)
                {
                    for (Int32 x = start.X; x <= end.X; ++x)
                    {
                        byte value = volume.GetVoxel(x, y, z);
                        // If a voxel has a '1' then it's unused volume space, if it is '0' it is empty, 
                        // and if anything else, it has a box in it already.
                        if (value != 1)
                            return false;
                    }
                }
            }

            return true;
        }

        protected static bool FillRange(VoxelField volume, AABBi box, Byte fillByte)
        {
            return FillRange(volume, new Vector3i(box.MinX, box.MinY, box.MinZ), new Vector3i(box.MaxX - 1, box.MaxY - 1, box.MaxZ - 1), fillByte);
        }

        protected static bool FillRange(VoxelField volume, Vector3i start, Vector3i end, Byte fillByte)
        {
            // 1. Ensure that we can safely expand into the area.
            if (!TestRangeForFreeSpace(volume, start, end))
                return false;

            // 2. Fill area
            for (Int32 z = start.Z; z <= end.Z; ++z)
            {
                for (Int32 y = start.Y; y <= end.Y; ++y)
                {
                    for (Int32 x = start.X; x <= end.X; ++x)
                    {
                        volume.SetVoxel(x, y, z, fillByte);
                    }
                }
            }

            return true;
        }

        protected static AABBi ExpandAndFillBox(VoxelField volume, ref Vector3i originVoxel, Byte fillByte)
        {
            int pX, nX, pY, nY, pZ, nZ;
            pX = nX = pY = nY = pZ = nZ = 0;

            volume.SetVoxel(originVoxel.X, originVoxel.Y, originVoxel.Z, fillByte);

            bool boxGrew = false;
            do
            {
                // +Z Axis
                bool pZGrew = FillRange(volume,
                    new Vector3i(originVoxel.X - nX, originVoxel.Y - nY, originVoxel.Z + (pZ + 1)),
                    new Vector3i(originVoxel.X + pX, originVoxel.Y + pY, originVoxel.Z + (pZ + 1)), fillByte);
                if (pZGrew)
                    pZ++;

                // -Z Axis
                bool nZGrew = FillRange(volume,
                    new Vector3i(originVoxel.X - nX, originVoxel.Y - nY, originVoxel.Z - (nZ + 1)),
                    new Vector3i(originVoxel.X + pX, originVoxel.Y + pY, originVoxel.Z - (nZ + 1)), fillByte);
                if (nZGrew)
                    nZ++;

                // +Y Axis
                bool pYGrew = FillRange(volume,
                    new Vector3i(originVoxel.X - nX, originVoxel.Y + (pY + 1), originVoxel.Z - nZ),
                    new Vector3i(originVoxel.X + pX, originVoxel.Y + (pY + 1), originVoxel.Z + pZ), fillByte);
                if (pYGrew)
                    pY++;

                // -Y Axis
                bool nYGrew = FillRange(volume,
                    new Vector3i(originVoxel.X - nX, originVoxel.Y - (nY + 1), originVoxel.Z - nZ),
                    new Vector3i(originVoxel.X + pX, originVoxel.Y - (nY + 1), originVoxel.Z + pZ), fillByte);
                if (nYGrew)
                    nY++;

                // +X Axis
                bool pXGrew = FillRange(volume,
                    new Vector3i(originVoxel.X + (pX + 1), originVoxel.Y - nY, originVoxel.Z - nZ),
                    new Vector3i(originVoxel.X + (pX + 1), originVoxel.Y + pY, originVoxel.Z + pZ), fillByte);
                if (pXGrew)
                    pX++;

                // -X Axis
                bool nXGrew = FillRange(volume,
                    new Vector3i(originVoxel.X - (nX + 1), originVoxel.Y - nY, originVoxel.Z - nZ),
                    new Vector3i(originVoxel.X - (nX + 1), originVoxel.Y + pY, originVoxel.Z + pZ), fillByte);
                if (nXGrew)
                    nX++;

                boxGrew = (pZGrew || nZGrew || pYGrew || nYGrew || pXGrew || nXGrew);
            } while (boxGrew);

            AABBi box = new AABBi();
            box.MinX = originVoxel.X - nX;
            box.MinY = originVoxel.Y - nY;
            box.MinZ = originVoxel.Z - nZ;

            box.MaxX = originVoxel.X + pX + 1;
            box.MaxY = originVoxel.Y + pY + 1;
            box.MaxZ = originVoxel.Z + pZ + 1;

            return box;
        }

        protected static float MeasureUnboxedVoxels(VoxelField volume)
        {
            int unboxedVoxels = 0;
            int totalvoxels = 0;

            for (Int32 x = 0; x < volume.VoxelSize.X; ++x)
            {
                for (Int32 y = 0; y < volume.VoxelSize.Y; ++y)
                {
                    for (Int32 z = 0; z < volume.VoxelSize.Z; ++z)
                    {
                        byte value = volume.GetVoxel(x, y, z);
                        if (value == 1)
                            unboxedVoxels++;
                        if (value != 0)
                            totalvoxels++;
                    }
                }
            }

            return unboxedVoxels / (float)totalvoxels;
        }
    }
}
