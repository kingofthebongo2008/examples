using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenTK.Graphics.OpenGL;
using OpenTK;

namespace Oxel
{
    public class RenderableMesh
    {
        uint m_handleVB;
        uint m_handleIB;
        int m_indexLength;
        bool m_cached;
        Mesh m_mesh;

        public int[] Indicies
        {
            get { return m_mesh.Indicies; }
        }

        public Vector4[] Vertices
        {
            get { return m_mesh.Vertices; }
        }

        public int Triangles
        {
            get { return m_indexLength / 3; }
        }

        public bool Cached
        {
            get { return m_cached; }
        }

        public RenderableMesh(Mesh mesh, bool cached)
        {
            m_cached = cached;

            if (m_cached)
            {
                m_mesh = mesh;
            }

            m_indexLength = m_mesh.Indicies.Length;

            // Initialize buffers for original bush
            GL.GenBuffers(1, out m_handleVB);
            GL.BindBuffer(BufferTarget.ArrayBuffer, m_handleVB);
            GL.BufferData(BufferTarget.ArrayBuffer, (IntPtr)(m_mesh.Vertices.Length * sizeof(float) * 4), m_mesh.Vertices, BufferUsageHint.StaticDraw);

            GL.GenBuffers(1, out m_handleIB);
            GL.BindBuffer(BufferTarget.ElementArrayBuffer, m_handleIB);
            GL.BufferData(BufferTarget.ElementArrayBuffer, (IntPtr)(m_mesh.Indicies.Length * sizeof(int)), m_mesh.Indicies, BufferUsageHint.StaticDraw);
        }

        public void Dispose()
        {
            GL.DeleteBuffers(1, ref m_handleVB);
            GL.DeleteBuffers(1, ref m_handleIB);
        }

        public void Render(Material effect)
        {
            if (effect.IsAlphaBlending)
            {
                GL.Enable(EnableCap.Blend);
                GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);

                GL.Disable(EnableCap.DepthTest);
            }

            GL.DepthFunc(DepthFunction.Less);
            GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);

            if (effect.ShaderHandle > 0)
            {
                GL.UseProgram(effect.ShaderHandle);

                effect.LoadShaderConstants();
            }

            GL.FrontFace(FrontFaceDirection.Ccw);
            GL.BindBuffer(BufferTarget.ArrayBuffer, m_handleVB);
            GL.BindBuffer(BufferTarget.ElementArrayBuffer, m_handleIB);
            GL.VertexPointer(4, VertexPointerType.Float, sizeof(float) * 4, 0);
            GL.IndexPointer(IndexPointerType.Int, sizeof(int), 0);
            GL.DrawElements(BeginMode.Triangles, m_indexLength, DrawElementsType.UnsignedInt, 0);

            if (effect.ShaderHandle > 0)
                GL.UseProgram(0);

            if (effect.IsAlphaBlending)
            {
                GL.Disable(EnableCap.Blend);
            }

            if (effect.ShowLines)
            {
                GL.Color4(1.0f, 1.0f, 1.0f, 1.0f);
                GL.DepthFunc(DepthFunction.Lequal);
                GL.LineWidth(2.0f);

                GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Line);
                GL.DrawElements(BeginMode.Triangles, m_indexLength, DrawElementsType.UnsignedInt, 0);

                GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
                GL.DepthFunc(DepthFunction.Less);
            }
        }
    }

    public class Material
    {
        public int ShaderHandle;
        public bool ShowLines;
        public bool IsAlphaBlending;

        private Dictionary<string, object> m_constants;

        public Material()
        {
            m_constants = new Dictionary<string, object>();
        }

        public void SetVector4(string constantName, Vector4 value)
        {
            m_constants[constantName] = value;
        }

        public void LoadShaderConstants()
        {
            foreach (var constant in m_constants)
            {
                if (constant.Value is Vector4)
                {
                    int constantIndex = GL.GetUniformLocation(ShaderHandle, constant.Key);
                    if (constantIndex != -1)
                        GL.Uniform4(constantIndex, (Vector4)constant.Value);
                }
            }
        }
    }
}
