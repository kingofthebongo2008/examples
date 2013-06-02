﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenTK;

namespace Oxel
{
    public interface IExporter
    {
        void Save(string path, Vector4[] vertices, int[] indices);
    }
}
