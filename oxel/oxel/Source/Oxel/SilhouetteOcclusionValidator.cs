using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using Oxel.OpenGL;

namespace Oxel
{
    public class SilhouetteOcclusionValidator : IDisposable
    {
        int m_pixelWidth;
        int m_pixelHeight;

        int m_fboHandle;
        int m_colorTextureHandle;
        int m_renderbufferHandle;

        int[] m_occlusionQueries;

        public SilhouetteOcclusionValidator(int pixelWidth, int pixelHeight)
        {
            m_pixelWidth = pixelWidth;
            m_pixelHeight = pixelHeight;

            CreateBuffers();
        }

        private void CreateBuffers()
        {
            // Create color texture
            GL.GenTextures(1, out m_colorTextureHandle);
            GL.BindTexture(TextureTarget.Texture2D, m_colorTextureHandle);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Clamp);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Clamp);
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba8, m_pixelWidth, m_pixelHeight, 0, PixelFormat.Rgba, PixelType.UnsignedByte, IntPtr.Zero);

            //TODO: test for GL Error here (might be unsupported format)
            
            // prevent feedback, reading and writing to the same image is a bad idea
            GL.BindTexture(TextureTarget.Texture2D, 0);

            // Create depth Renderbuffer
            GL.GenRenderbuffers(1, out m_renderbufferHandle);
            GL.Ext.BindRenderbuffer(RenderbufferTarget.RenderbufferExt, m_renderbufferHandle);
            GL.Ext.RenderbufferStorage(RenderbufferTarget.RenderbufferExt, (RenderbufferStorage)All.Depth24Stencil8, m_pixelWidth, m_pixelHeight);

            //TODO: test for GL Error here (might be unsupported format)

            // Create a FBO and attach the textures
            GL.Ext.GenFramebuffers(1, out m_fboHandle);
            GL.Ext.BindFramebuffer(FramebufferTarget.FramebufferExt, m_fboHandle);
            GL.Ext.FramebufferTexture2D(FramebufferTarget.FramebufferExt, FramebufferAttachment.ColorAttachment0Ext, TextureTarget.Texture2D, m_colorTextureHandle, 0);
            GL.Ext.FramebufferRenderbuffer(FramebufferTarget.FramebufferExt, FramebufferAttachment.DepthAttachmentExt, RenderbufferTarget.RenderbufferExt, m_renderbufferHandle);
            GL.Ext.FramebufferRenderbuffer(FramebufferTarget.FramebufferExt, FramebufferAttachment.StencilAttachmentExt, RenderbufferTarget.RenderbufferExt, m_renderbufferHandle);

            GLEx.CheckFboStatus();

            // return to visible framebuffer
            GL.Ext.BindFramebuffer(FramebufferTarget.FramebufferExt, 0);

            // Allocate the occlusion query we're going to use for 
            m_occlusionQueries = new int[64];
            GL.GenQueries(m_occlusionQueries.Length, m_occlusionQueries);
        }

        public void Dispose()
        {
            GL.DeleteTextures(1, ref m_colorTextureHandle);
            GL.DeleteRenderbuffers(1, ref m_renderbufferHandle);
            GL.Ext.DeleteFramebuffers(1, ref m_fboHandle);

            GL.DeleteQueries(m_occlusionQueries.Length, m_occlusionQueries);
        }

        public void ComputeCoverage(RenderableMesh mesh, AABBf meshBounds, out long sideCoverage, out long topCoverage)
        {
            float longestSide = Math.Max(Math.Max(meshBounds.MaxX - meshBounds.MinX, meshBounds.MaxY - meshBounds.MinY), meshBounds.MaxZ - meshBounds.MinZ);
            float farPlane = longestSide * 2.0f;

            float x = (meshBounds.MinX + meshBounds.MaxX) / 2.0f;
            float y = meshBounds.MinY;
            float z = (meshBounds.MinZ + meshBounds.MaxZ) / 2.0f;
            Vector3 origin = new Vector3(x, y, z);
            Vector3 up = Vector3.UnitY;
            Vector3 look = -Vector3.UnitX;

            for (int i = 0; i < m_occlusionQueries.Length; i++)
            {
                Matrix4 orbit = Matrix4.CreateRotationY(MathHelper.DegreesToRadians((365 / 64.0f) * i));

                Matrix4 projection = Matrix4.CreatePerspectiveFieldOfView(MathHelper.DegreesToRadians(90), 1.0f, 0.1f, farPlane);
                Matrix4 view = Matrix4.LookAt(origin + Vector3.Transform(new Vector3(longestSide, 0, 0), orbit), origin, Vector3.TransformNormal(up, orbit));

                RenderView(view, projection, mesh, m_occlusionQueries[i]);

                //Bitmap bmp = GLEx.BitmapColorBuffer(m_pixelWidth, m_pixelHeight);
                //bmp.Save("C:\\test_" + i + ".bmp");
            }

            // Gather all the occlusion queries we performed
            long[] m_occlusionQueryResults = new long[64];
            for (int i = 0; i < m_occlusionQueries.Length; i++)
            {
                int ready = 0;
                while (ready == 0)
                {
                    GL.GetQueryObject(m_occlusionQueries[i], GetQueryObjectParam.QueryResultAvailable, out ready);
                }

                GL.GetQueryObject(m_occlusionQueries[i], GetQueryObjectParam.QueryResult, out m_occlusionQueryResults[i]);
            }

            // Reset the current frame buffer.
            GL.Ext.BindFramebuffer(FramebufferTarget.FramebufferExt, 0);

            long totalSidePixels = 0;
            long totalTopPixels = 0;
            for (int i = 0; i < m_occlusionQueries.Length; i++)
            {
                totalSidePixels += m_occlusionQueryResults[i];
            }

            sideCoverage = totalSidePixels;
            topCoverage = totalTopPixels;
        }

        private void RenderView(Matrix4 worldViewMatrix, Matrix4 projMatrix, RenderableMesh mesh, int occlusionQueryHandle)
        {
            // Setup VBO state
            GL.EnableClientState(ArrayCap.VertexArray);
            GL.EnableClientState(ArrayCap.IndexArray);

            GL.Enable(EnableCap.StencilTest);
            GL.StencilFunc(StencilFunction.Always, 1, 1);
            GL.StencilOp(StencilOp.Keep, StencilOp.Keep, StencilOp.Replace);

            GL.PushAttrib(AttribMask.ViewportBit);
            GL.Viewport(0, 0, m_pixelWidth, m_pixelHeight);

            GL.Ext.BindFramebuffer(FramebufferTarget.FramebufferExt, m_fboHandle);

            // since there's only 1 Color buffer attached this is not explicitly required
            GL.DrawBuffer((DrawBufferMode)FramebufferAttachment.ColorAttachment0Ext);
            GL.Ext.FramebufferTexture2D(FramebufferTarget.FramebufferExt, FramebufferAttachment.ColorAttachment0Ext, TextureTarget.Texture2D, m_colorTextureHandle, 0);
            GL.Ext.FramebufferRenderbuffer(FramebufferTarget.FramebufferExt, FramebufferAttachment.DepthAttachmentExt, RenderbufferTarget.RenderbufferExt, m_renderbufferHandle);
            GL.Ext.FramebufferRenderbuffer(FramebufferTarget.FramebufferExt, FramebufferAttachment.StencilAttachmentExt, RenderbufferTarget.RenderbufferExt, m_renderbufferHandle);

            GL.ClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit | ClearBufferMask.StencilBufferBit);

            GL.PushMatrix();

            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadMatrix(ref worldViewMatrix);
            GL.MatrixMode(MatrixMode.Projection);
            GL.LoadMatrix(ref projMatrix);

            GL.Enable(EnableCap.DepthTest);
            GL.Disable(EnableCap.CullFace);

            var effect = new Material();
            effect.ShaderHandle = 0;
            effect.ShowLines = false;

            GL.Color4(1.0f, 1.0f, 1.0f, 1.0f);
            mesh.Render(effect);

            GL.PopMatrix();

            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            // Render a quad with stencil test turned on, this will give me an accurate number of pixels rendered for
            // the object.

            GL.StencilFunc(StencilFunction.Equal, 1, 1);
            GL.StencilOp(StencilOp.Keep, StencilOp.Keep, StencilOp.Keep);

            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadIdentity();
            GL.MatrixMode(MatrixMode.Projection);
            GL.LoadIdentity();

            GL.Ortho(0, 1, 1, 0, -1, 1);

            GL.Disable(EnableCap.DepthTest);

            GL.BeginQuery(QueryTarget.SamplesPassed, occlusionQueryHandle);

            GL.Color4(1.0f, 1.0f, 1.0f, 1.0f);
            GL.Begin(BeginMode.Quads);
            GL.TexCoord2(0f, 1f); GL.Vertex2(0f, 0f);
            GL.TexCoord2(1f, 1f); GL.Vertex2(1f, 0f);
            GL.TexCoord2(1f, 0f); GL.Vertex2(1f, 1f);
            GL.TexCoord2(0f, 0f); GL.Vertex2(0f, 1f);
            GL.End();

            GL.EndQuery(QueryTarget.SamplesPassed);

            GL.Enable(EnableCap.DepthTest);
            GL.Disable(EnableCap.StencilTest);

            GL.PopAttrib();
        }
    }
}
