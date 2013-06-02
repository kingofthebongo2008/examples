using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace Oxel
{
    public interface IImporter
    {
        MeshData Load(string filename);
    }
}