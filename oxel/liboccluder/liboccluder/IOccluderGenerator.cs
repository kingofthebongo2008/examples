using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Oxel
{
    public interface IOccluderGenerator
    {
        VoxelizationOutput Generate(VoxelizationInput input, Action<VoxelizationProgress> progress);
    }
}
