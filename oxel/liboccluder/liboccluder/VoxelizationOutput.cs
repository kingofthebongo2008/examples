using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Oxel
{
    public class VoxelizationOutput
    {
        public VoxelizingOctree Octree;
        public RenderableMesh OccluderMesh;
        public TimeSpan TimeTaken;
        public float VolumeCoverage;
        public double SilhouetteCoverage;

        // Diagnostics
        public List<List<Edge>> DebugLines;
    }
}
