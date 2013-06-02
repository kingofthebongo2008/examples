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
    public class RobustVoxelizer : IDisposable
    {
        public int CubemapWidth;
        public int CubemapHeight;

        uint[] CubemapFboHandle;
        uint[] CubemapColorTexture;
        uint[] CubemapDepthRenderbuffer;

        Vector3[] cubemapDirections;
        Vector3[] cubemapUps;
        Matrix4[] cubemapProjections;
        RGB[][] cubemapColorBuffers;
        float[][] cubemapDepthBuffers;

        int shader_frontback;

        VoxelizationInput m_input;

        public RobustVoxelizer(int cubemapWidth, int cubemapHeight)
        {
            CubemapWidth = cubemapWidth;
            CubemapHeight = cubemapHeight;

            CreateShaders();
            CreateCubemapBuffers();
        }

        public void Dispose()
        {
            // Destroy shaders
            GL.DeleteProgram(shader_frontback);

            // Destroy cubemap
            GL.DeleteTextures(6, CubemapColorTexture);
            GL.DeleteRenderbuffers(6, CubemapDepthRenderbuffer);
            GL.Ext.DeleteFramebuffers(6, CubemapFboHandle);
        }

        private void CreateShaders()
        {
            String fragSource = Properties.Resources.ps_FrontBack;
            String vertSource = Properties.Resources.vs_FrontBack;
            shader_frontback = GLEx.CreateShaderProgramStrings(vertSource, fragSource);
        }

        private void CreateCubemapBuffers()
        {
            CubemapFboHandle = new uint[6];
            CubemapColorTexture = new uint[6];
            CubemapDepthRenderbuffer = new uint[6];

            cubemapDirections = new Vector3[6];
            cubemapDirections[0] = new Vector3(0.0f, 0.0f, 1.0f);
            cubemapDirections[1] = new Vector3(0.0f, 0.0f, -1.0f);
            cubemapDirections[2] = new Vector3(0.0f, 1.0f, 0.0f);
            cubemapDirections[3] = new Vector3(0.0f, -1.0f, 0.0f);
            cubemapDirections[4] = new Vector3(1.0f, 0.0f, 0.0f);
            cubemapDirections[5] = new Vector3(-1.0f, 0.0f, 0.0f);
            cubemapUps = new Vector3[6];
            cubemapUps[0] = Vector3.UnitY;
            cubemapUps[1] = Vector3.UnitY;
            cubemapUps[2] = Vector3.UnitZ;
            cubemapUps[3] = Vector3.UnitZ;
            cubemapUps[4] = Vector3.UnitY;
            cubemapUps[5] = Vector3.UnitY;

            cubemapColorBuffers = new RGB[6][];
            for (int i = 0; i < 6; i++)
                cubemapColorBuffers[i] = new RGB[CubemapWidth * CubemapHeight];

            cubemapDepthBuffers = new float[6][];
            for (int i = 0; i < 6; i++)
                cubemapDepthBuffers[i] = new float[CubemapWidth * CubemapHeight];

            cubemapProjections = new Matrix4[6];

            // Create Color Texture
            GL.GenTextures(6, CubemapColorTexture);
            for (int i = 0; i < CubemapColorTexture.Length; i++)
            {
                GL.BindTexture(TextureTarget.Texture2D, CubemapColorTexture[i]);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Clamp);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Clamp);
                GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba8, CubemapWidth, CubemapHeight, 0, PixelFormat.Rgba, PixelType.UnsignedByte, IntPtr.Zero);
            }

            // test for GL Error here (might be unsupported format)

            GL.BindTexture(TextureTarget.Texture2D, 0); // prevent feedback, reading and writing to the same image is a bad idea

            // Create Depth Renderbuffer
            GL.Ext.GenRenderbuffers(6, CubemapDepthRenderbuffer);
            for (int i = 0; i < CubemapDepthRenderbuffer.Length; i++)
            {
                GL.Ext.BindRenderbuffer(RenderbufferTarget.RenderbufferExt, CubemapDepthRenderbuffer[i]);
                GL.Ext.RenderbufferStorage(RenderbufferTarget.RenderbufferExt, (RenderbufferStorage)All.DepthComponent32, CubemapWidth, CubemapHeight);
            }

            // test for GL Error here (might be unsupported format)

            // Create a FBO and attach the textures
            GL.Ext.GenFramebuffers(6, CubemapFboHandle);
            for (int i = 0; i < CubemapFboHandle.Length; i++)
            {
                GL.Ext.BindFramebuffer(FramebufferTarget.FramebufferExt, CubemapFboHandle[i]);
                GL.Ext.FramebufferTexture2D(FramebufferTarget.FramebufferExt, FramebufferAttachment.ColorAttachment0Ext, TextureTarget.Texture2D, CubemapColorTexture[i], 0);
                GL.Ext.FramebufferRenderbuffer(FramebufferTarget.FramebufferExt, FramebufferAttachment.DepthAttachmentExt, RenderbufferTarget.RenderbufferExt, CubemapDepthRenderbuffer[i]);
            }

            GLEx.CheckFboStatus();

            GL.Ext.BindFramebuffer(FramebufferTarget.FramebufferExt, 0); // return to visible framebuffer
        }

        public VoxelizationOutput Voxelize(VoxelizationInput input, Action<VoxelizationProgress> progress)
        {
            // Setup VBO state
            GL.EnableClientState(ArrayCap.VertexArray);
            GL.EnableClientState(ArrayCap.IndexArray);

            m_input = input;

            VoxelizationOutput output = new VoxelizationOutput();
            output.Octree = input.Octree;

            VoxelizationProgress vp = new VoxelizationProgress();
            vp.Status = "Voxelizing mesh with " + input.Octree.MaxLevels + " subdivision levels";
            progress(vp);

            GL.PushAttrib(AttribMask.AllAttribBits);
            for (int i = 0; i <= input.Octree.MaxLevels; i++)
            {
                vp.Progress = (i / (input.Octree.MaxLevels + 1.0f));
                vp.Status = "Voxelizing octree level " + i;
                progress(vp);
                RecursiveSolveStatus(input.Octree.Root, i);
            }
            GL.PopAttrib();

            vp.Progress = 1;
            vp.Status = "Done voxelizing mesh";
            progress(vp);

            return output;
        }

        private bool RecursiveSolveStatus(VoxelizingOctreeCell cell, int maxDepth)
        {
            if (maxDepth < 0)
                return false;

            if (cell.IsLeaf && cell.IsIntersecting)
                return false;

            switch (cell.Status)
            {
                case CellStatus.Unknown:
                    SolveStatus(cell);
                    return true;
                case CellStatus.Intersecting:
                case CellStatus.IntersectingBounds:
                    for (int i = 0; i < cell.Children.Count; i++)
                    {
                        RecursiveSolveStatus(cell.Children[i], maxDepth - 1);
                    }
                    return true;
            }

            return false;
        }

        private void SolveStatus(VoxelizingOctreeCell cell)
        {
            const int MIN_INSIDE_FACES = 4;
            const float MIN_INSIDE_PERCENTAGE = 0.03f;

            int cubemap_sides_seeing_inside = 0;

            for (int i = 0; i < 6; i++)
            {
                RenderCubeSide(i, cell, cubemapDirections[i], cubemapUps[i]);
                ReadDepthBuffer(i);
                float backfacePercentage = CalculateBackfacePercentage(cubemapColorBuffers[i], CubemapWidth, CubemapHeight);

                if (backfacePercentage > MIN_INSIDE_PERCENTAGE)
                    cubemap_sides_seeing_inside++;
            }

            // This is a seed cell so go ahead and mark the status we believe it to be.
            if (cubemap_sides_seeing_inside >= MIN_INSIDE_FACES || cubemap_sides_seeing_inside == 0)
            {
                if (cubemap_sides_seeing_inside >= MIN_INSIDE_FACES)
                    cell.Status = CellStatus.Inside;
                else // cubemap_sides_seeing_inside == 0
                    cell.Status = CellStatus.Outside;

                RecursivePropagateStatus(cell, cell.Status, m_input.Octree.Root);
            }
            else
            {
                // Unable to solve status exactly.
            }

            // Restore the standard back buffer.
            //GL.Ext.FramebufferTexture2D(FramebufferTarget.FramebufferExt, FramebufferAttachment.ColorAttachment0Ext, TextureTarget.Texture2D, 0, 0);
            //GL.Ext.FramebufferRenderbuffer(FramebufferTarget.FramebufferExt, FramebufferAttachment.DepthAttachmentExt, RenderbufferTarget.RenderbufferExt, 0);

            GL.Ext.BindFramebuffer(FramebufferTarget.FramebufferExt, 0);
        }

        private float CalculateBackfacePercentage(RGB[] buffer, int width, int height)
        {
            GL.ReadPixels<RGB>(0, 0, width, height, PixelFormat.Rgb, PixelType.UnsignedByte, buffer);

            int r_total = 0;
            int b_total = 0;

            Parallel.For(0, buffer.Length, () => new RedBlueTuple { R_Total = 0, B_Total = 0 }, (i, loop, partial) =>
            {
                partial.R_Total += buffer[i].r;
                partial.B_Total += buffer[i].b;
                return partial;
            },
            partial =>
            {
                Interlocked.Add(ref r_total, partial.R_Total);
                Interlocked.Add(ref b_total, partial.B_Total);
            });

            return r_total / (float)(r_total + b_total);
        }

        private void RecursivePropagateStatus(VoxelizingOctreeCell seed, CellStatus status, VoxelizingOctreeCell current)
        {
            if (current.IsLeaf && current.IsIntersecting)
                return;

            switch (current.Status)
            {
                case CellStatus.Unknown:
                    // Only propogate status to unknown cells
                    PropagateStatus(seed, status, current);
                    return;
                case CellStatus.Intersecting:
                case CellStatus.IntersectingBounds:
                    for (int i = current.Children.Count - 1; i >= 0; i--)
                    {
                        RecursivePropagateStatus(seed, status, current.Children[i]);
                    }
                    return;
            }
        }

        private void RenderCubeSide(int cubemapIndex, VoxelizingOctreeCell cell, Vector3 look, Vector3 up)
        {
            AABBf root_bounds = cell.Root.Bounds;
            float root_length = root_bounds.MaxX - root_bounds.MinX;
            float s_div2 = (cell.Bounds.MaxX - cell.Bounds.MinX) * 0.5f;

            GL.MatrixMode(MatrixMode.Projection);
            Matrix4 perspective = Matrix4.CreatePerspectiveFieldOfView(MathHelper.DegreesToRadians(90.0f), 1.0f, s_div2, root_length);
            GL.LoadMatrix(ref perspective);

            Matrix4 modelView = Matrix4.LookAt(cell.Center, cell.Center + look, up);
            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadMatrix(ref modelView);

            //cubemapProjections[cubemapIndex] = perspective * modelView;
            cubemapProjections[cubemapIndex] = modelView * perspective;

            GL.PushMatrix();

            GL.Ext.BindFramebuffer(FramebufferTarget.FramebufferExt, CubemapFboHandle[cubemapIndex]);

            // since there's only 1 Color buffer attached this is not explicitly required
            GL.DrawBuffer((DrawBufferMode)FramebufferAttachment.ColorAttachment0Ext);

            GL.ClearColor(0.0f, 0.0f, 0.0f, 0f);
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            GL.Ext.FramebufferTexture2D(FramebufferTarget.FramebufferExt, FramebufferAttachment.ColorAttachment0Ext, TextureTarget.Texture2D, CubemapColorTexture[cubemapIndex], 0);
            GL.Ext.FramebufferRenderbuffer(FramebufferTarget.FramebufferExt, FramebufferAttachment.DepthAttachmentExt, RenderbufferTarget.RenderbufferExt, CubemapDepthRenderbuffer[cubemapIndex]);

            GL.PushAttrib(AttribMask.ViewportBit);
            GL.Viewport(0, 0, CubemapWidth, CubemapHeight);

            DrawTwoSidedOriginalMesh();

            GL.PopAttrib();
            GL.PopMatrix();
        }

        private void DrawTwoSidedOriginalMesh()
        {
            GL.Enable(EnableCap.DepthTest);
            GL.Disable(EnableCap.CullFace);

            if (m_input.OriginalMesh != null)
            {
                var material = new Material();
                material.ShaderHandle = shader_frontback;
                material.ShowLines = false;
                material.SetVector4("Front", new Vector4(0, 0, 1, 1));
                material.SetVector4("Back", new Vector4(1, 0, 0, 1));
                m_input.OriginalMesh.Render(material);
            }
        }

        private void ReadDepthBuffer(int cubemapIndex)
        {
            GL.ReadPixels<float>(0, 0, CubemapWidth, CubemapHeight, PixelFormat.DepthComponent, PixelType.Float, cubemapDepthBuffers[cubemapIndex]);
        }

        private void PropagateStatus(VoxelizingOctreeCell seed, CellStatus status, VoxelizingOctreeCell current)
        {
            Vector4[] cell_bounds_verts = new Vector4[8];

            AABBf bounds = current.Bounds;
            cell_bounds_verts[0] = new Vector4(bounds.MaxX, bounds.MaxY, bounds.MaxZ, 1);
            cell_bounds_verts[1] = new Vector4(bounds.MaxX, bounds.MaxY, bounds.MinZ, 1);
            cell_bounds_verts[2] = new Vector4(bounds.MinX, bounds.MaxY, bounds.MinZ, 1);
            cell_bounds_verts[3] = new Vector4(bounds.MinX, bounds.MaxY, bounds.MaxZ, 1);
            cell_bounds_verts[4] = new Vector4(bounds.MaxX, bounds.MinY, bounds.MaxZ, 1);
            cell_bounds_verts[5] = new Vector4(bounds.MaxX, bounds.MinY, bounds.MinZ, 1);
            cell_bounds_verts[6] = new Vector4(bounds.MinX, bounds.MinY, bounds.MinZ, 1);
            cell_bounds_verts[7] = new Vector4(bounds.MinX, bounds.MinY, bounds.MaxZ, 1);

            // Loop over the number of cube faces
            for (int i = 0; i < 6; i++)
            {
                // Loop over the 8 vertices that make up the corners of a voxel cell.  If any of the corners is visible
                // the voxel will have the status accumulated into it.
                for (int n = 0; n < 8; n++)
                {
                    Vector4 result;
                    Vector4.Transform(ref cell_bounds_verts[n], ref cubemapProjections[i], out result);

                    float x = result.X / result.W;
                    float y = result.Y / result.W;
                    float z = result.Z / result.W;

                    if (x >= -1.0f && x <= 1.0f && y >= -1.0f && y <= 1.0f)
                    {
                        int depthBufferX = (int)(((x + 1.0f) / 2.0f) * (CubemapWidth - 1));
                        int depthBufferY = (int)(((y + 1.0f) / 2.0f) * (CubemapHeight - 1));

                        float sampledDepth = cubemapDepthBuffers[i][depthBufferY * CubemapWidth + depthBufferX];
                        float ndc_sampledDepth = ((sampledDepth * 2.0f) - 1.0f);

                        if (z > -1.0f && z < ndc_sampledDepth)
                        {
                            // Accumulate the status on the cell must overcome threshold to be confirmed as inside or outside.
                            // If enough other voxels propagate the same status to the cell, it becomes a cell of that type.
                            if (current.AccumulateStatus(status))
                                return;
                        }
                    }
                }
            }
        }
    }
}
