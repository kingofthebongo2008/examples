using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenTK;

namespace Oxel
{
    public class Mesh
    {
        public Vector4[] Vertices;
        public int[] Indicies;

        public Mesh()
        {
        }

        public Mesh(Vector4[] vertices, int[] indicies)
        {
            Vertices = vertices;
            Indicies = indicies;
        }
    }
}
