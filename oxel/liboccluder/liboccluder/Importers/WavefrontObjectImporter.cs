using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using OpenTK;

namespace Oxel
{
    public class WavefrontObjectImporter : IImporter
    {
        bool m_autoweld;

        public WavefrontObjectImporter(bool autoweld)
        {
            m_autoweld = autoweld;
        }

        public MeshData Load(string filename)
        {
            using (FileStream s = File.Open(filename, FileMode.Open))
            {
                return LoadStream(s);
            }
        }

        MeshData LoadStream(Stream stream)
        {
            StreamReader reader = new StreamReader(stream);
            List<Vector3> points = new List<Vector3>();
            List<Vector3> normals = new List<Vector3>();
            List<Vector2> texCoords = new List<Vector2>();
            List<Tri> tris = new List<Tri>();
            string line;
            char[] splitChars = { ' ' };
            while ((line = reader.ReadLine()) != null)
            {
                line = line.Trim(splitChars);
                line = line.Replace("  ", " ");

                string[] parameters = line.Split(splitChars);

                switch (parameters[0])
                {
                    case "p":
                        // MeshPoint
                        break;

                    case "v":
                        // Vertex
                        float x = (float)float.Parse(parameters[1], CultureInfo.InvariantCulture.NumberFormat);
                        float y = (float)float.Parse(parameters[2], CultureInfo.InvariantCulture.NumberFormat);
                        float z = (float)float.Parse(parameters[3], CultureInfo.InvariantCulture.NumberFormat);
                        points.Add(new Vector3(x, y, z));
                        break;

                    case "vt":
                        // TexCoord
                        float u = (float)float.Parse(parameters[1], CultureInfo.InvariantCulture.NumberFormat);
                        float v = (float)float.Parse(parameters[2], CultureInfo.InvariantCulture.NumberFormat);
                        texCoords.Add(new Vector2(u, v));
                        break;

                    case "vn":
                        // Normal
                        float nx = (float)float.Parse(parameters[1], CultureInfo.InvariantCulture.NumberFormat);
                        float ny = (float)float.Parse(parameters[2], CultureInfo.InvariantCulture.NumberFormat);
                        float nz = (float)float.Parse(parameters[3], CultureInfo.InvariantCulture.NumberFormat);
                        normals.Add(new Vector3(nx, ny, nz));
                        break;

                    case "f":
                        // Face
                        tris.AddRange(parseFace(points.Count, parameters));
                        break;
                }
            }

            Vector3[] p = points.ToArray();
            Vector2[] tc = texCoords.ToArray();
            Vector3[] n = normals.ToArray();
            Tri[] f = tris.ToArray();


            //map_d is the alpha texture, if it is provided there's an alpha layer, and thus we need to ignore it.


            // Perform welding...
            if (m_autoweld)
            {
                MeshWelder.Weld(4, p, f);
            }

            // If there are no specified texcoords or normals, we add a dummy one.
            // That way the Points will have something to refer to.
            if (tc.Length == 0)
            {
                tc = new Vector2[1];
                tc[0] = new Vector2(0, 0);
            }

            if (n.Length == 0)
            {
                n = new Vector3[1];
                n[0] = new Vector3(1, 0, 0);
            }

            return new MeshData(p, n, tc, f);
        }

        static Tri[] parseFace(int vertexCount, string[] indices)
        {
            MeshPoint[] p = new MeshPoint[indices.Length - 1];
            for (int i = 0; i < p.Length; i++)
            {
                p[i] = parsePoint(vertexCount, indices[i + 1]);
            }
            return Triangulate(p);
            //return new Face(p);
        }

        // Takes an array of points and returns an array of triangles.
        // The points form an arbitrary polygon.
        static Tri[] Triangulate(MeshPoint[] ps)
        {
            List<Tri> ts = new List<Tri>();
            if (ps.Length < 3)
            {
                throw new Exception("Invalid shape!  Must have >2 points");
            }

            MeshPoint lastButOne = ps[1];
            MeshPoint lastButTwo = ps[0];
            for (int i = 2; i < ps.Length; i++)
            {
                Tri t = new Tri(lastButTwo, lastButOne, ps[i]);
                lastButOne = ps[i];
                lastButTwo = ps[i - 1];
                ts.Add(t);
            }
            return ts.ToArray();
        }

        static MeshPoint parsePoint(int vertexCount, string s)
        {
            char[] splitChars = { '/' };
            string[] parameters = s.Split(splitChars);
            int vert, tex, norm;
            vert = tex = norm = 0;
            vert = int.Parse(parameters[0]);
            // Texcoords and normals are optional in .obj files
            if (parameters[1] != "")
                tex = int.Parse(parameters[1]);
            if (parameters[2] != "")
                norm = int.Parse(parameters[2]);

            // Make index base-0 but only for positive indicies
            if (vert > 0)
                vert -= 1;
            else
                vert += vertexCount;
            
            if (tex > 0)
                tex -= 1;

            if (norm > 0)
                norm -= 1;

            return new MeshPoint(vert, norm, tex);
        }
    }
}