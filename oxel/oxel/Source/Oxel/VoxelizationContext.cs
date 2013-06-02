using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenTK;

namespace Oxel
{
    public class VoxelizationContext : IDisposable
    {
        public MeshData CurrentMesh;
        public string CurrentMeshFile;

        public int[] OriginalIndex;
        public Vector4[] Vectors;

        public RenderableMesh OriginalMesh;
        public RenderableMesh OccluderMesh;

        public VoxelizingOctree Octree;

        public VoxelizationOutput VoxelizationOutput;

        public VoxelizationContext()
        {
        }

        ~VoxelizationContext()
        {
            Dispose();
        }

        public void Dispose()
        {
            GC.SuppressFinalize(this);

            if (OriginalMesh != null)
            {
                OriginalMesh.Dispose();
                OriginalMesh = null;
            }

            if (OccluderMesh != null)
            {
                OccluderMesh.Dispose();
                OccluderMesh = null;
            }
        }
    }
}
