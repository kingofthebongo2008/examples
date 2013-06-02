using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace Oxel
{
    public class VoxelField
    {
        public Vector3i VoxelSize;

        public Vector3i CellSize;
        public VoxelCell[][][] Cells;

        public VoxelField(VoxelizingOctree tree)
        {
            CellSize = new Vector3i(
                (int)Math.Ceiling((tree.VoxelSize.X + 1) / (double)VoxelCell.SizeX),
                (int)Math.Ceiling((tree.VoxelSize.Y + 1) / (double)VoxelCell.SizeY),
                (int)Math.Ceiling((tree.VoxelSize.Z + 1) / (double)VoxelCell.SizeZ));
            VoxelSize = tree.VoxelSize;

            // Initialize the cell tables, but don't actually fill in the arrays with voxel cell instances
            // until we know those cells will be filled.
            Cells = new VoxelCell[CellSize.X][][];
            for (int x = 0; x < Cells.Length; x++)
            {
                Cells[x] = new VoxelCell[CellSize.Y][];

                for (int y = 0; y < Cells[x].Length; y++)
                {
                    Cells[x][y] = new VoxelCell[CellSize.Z];
                }
            }

            CreateSolidVolume(tree, tree.Root, tree.VoxelSize);
        }

        public byte GetVoxel(int x, int y, int z)
        {
            VoxelCell cell = Cells[x / VoxelCell.SizeX][y / VoxelCell.SizeY][z / VoxelCell.SizeZ];
            if (cell == null)
                return 0;

            return cell.GetAt(x % VoxelCell.SizeX, y % VoxelCell.SizeY, z % VoxelCell.SizeZ);
        }

        public void SetVoxel(int x, int y, int z, byte value)
        {
            VoxelCell cell = Cells[x / VoxelCell.SizeX][y / VoxelCell.SizeY][z / VoxelCell.SizeZ];
            if (cell == null)
            {
                cell = new VoxelCell();
                Cells[x / VoxelCell.SizeX][y / VoxelCell.SizeY][z / VoxelCell.SizeZ] = cell;
            }

            cell.SetAt(x % VoxelCell.SizeX, y % VoxelCell.SizeY, z % VoxelCell.SizeZ, value);
        }

        private void CreateSolidVolume(VoxelizingOctree tree, VoxelizingOctreeCell cell, Vector3i volumeLength)
        {
            if (cell.Status == CellStatus.Inside)
            {
                int min_x = cell.VoxelBounds.MinX + tree.WorldVoxelOffset.X;
                int min_y = cell.VoxelBounds.MinY + tree.WorldVoxelOffset.Y;
                int min_z = cell.VoxelBounds.MinZ + tree.WorldVoxelOffset.Z;
                int max_x = cell.VoxelBounds.MaxX + tree.WorldVoxelOffset.X;
                int max_y = cell.VoxelBounds.MaxY + tree.WorldVoxelOffset.Y;
                int max_z = cell.VoxelBounds.MaxZ + tree.WorldVoxelOffset.Z;

                for (int x = min_x; x < max_x; x++)
                {
                    for (int y = min_y; y < max_y; y++)
                    {
                        for (int z = min_z; z < max_z; z++)
                        {
                            SetVoxel(x, y, z, 1);
                        }
                    }
                }
            }

            foreach (var child in cell.Children)
            {
                CreateSolidVolume(tree, child, volumeLength);
            }
        }
    }

    public class VoxelCell
    {
        public const int SizeX = 32;
        public const int SizeY = 32;
        public const int SizeZ = 32;

        byte[] m_voxels;

        public VoxelCell()
        {
            m_voxels = new byte[SizeX * SizeY * SizeZ];
        }

        public byte GetAt(int x, int y, int z)
        {
            Int32 index = x + (y * SizeX) + (z * SizeX * SizeY);
            return m_voxels[index];
        }

        public void SetAt(int x, int y, int z, byte value)
        {
            Int32 index = x + (y * SizeX) + (z * SizeX * SizeY);
            m_voxels[index] = value;
        }
    }
}
