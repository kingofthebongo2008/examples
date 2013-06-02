using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;
using Oxel.OpenGL;
using QuickFont;

namespace Oxel
{
    // TODO Future improvement - 
    // If the cubemap cells are read-back and then we divide each face up into 4 quadrants, we can make intelligent statements,
    // like to be inside you must have red visible in greater than 50% of the quadrants.  This will make it possible to instantly
    // cut off cells that fall outside of a piece of geometry, where the box is 50% inside the mesh and 50% outside the mesh.
    // 
    // Further more, perhaps we can use this knowledge to further subdivide the mesh?
    public partial class MainWindow : Form
    {
        bool showUnknownCells = true;
        bool showInsideCells = true;
        bool showOutsideCells = false;
        bool showIntersectingCells = false;
        bool showIntersectingBoundsCells = false;
        bool showDebugLines = true;
        bool m_showOriginalMesh = true;
        bool m_showOccluderMesh = true;
        bool m_showOccluderLines = true;
        bool m_showMeshLines = true;
        bool m_alphaBlendingMode = false;

        int m_visualizedMaxLevel;

        int shader_frontback;

        Operations m_operations;

        FpsCamera m_camera = new FpsCamera();

        Stopwatch renderTimer = new Stopwatch();

        bool glLoaded;

        QFont codeText;

        GLControl m_gl;

        public MainWindow(VoxelizationInput input)
        {
            InitializeComponent();

            LinkLabel label = new LinkLabel();
            label.Text = "Bug/Feature?";
            label.BackColor = Color.Transparent;
            label.LinkColor = Color.Blue;
            label.ActiveLinkColor = Color.Blue;
            label.DisabledLinkColor = Color.Blue;
            label.VisitedLinkColor = Color.Blue;
            label.LinkClicked += (s, e) =>
            {
                Process.Start("mailto:NickDarnell@gmail.com?subject=[Oxel] Bug/Feature");
            };
            ToolStripControlHost host = new ToolStripControlHost(label);
            host.Alignment = ToolStripItemAlignment.Right;
            m_menu.SuspendLayout();
            m_menu.Items.Add(host);
            m_menu.ResumeLayout(true);

            m_gl = new GLControl(new GraphicsMode(32, 24, 8));
            m_gl.BackColor = System.Drawing.Color.Black;
            m_gl.Dock = System.Windows.Forms.DockStyle.Fill;
            m_gl.Location = new System.Drawing.Point(0, 0);
            m_gl.Name = "m_gl";
            m_gl.Size = new System.Drawing.Size(716, 516);
            m_gl.TabIndex = 2;
            m_gl.VSync = false;
            m_gl.Load += new System.EventHandler(this.m_gl_Load);
            m_gl.Paint += new System.Windows.Forms.PaintEventHandler(this.m_gl_Paint);
            m_gl.KeyPress += new System.Windows.Forms.KeyPressEventHandler(this.m_gl_KeyPress);
            m_gl.MouseDown += new System.Windows.Forms.MouseEventHandler(this.m_gl_MouseDown);
            m_gl.MouseMove += new System.Windows.Forms.MouseEventHandler(this.m_gl_MouseMove);
            m_gl.MouseUp += new System.Windows.Forms.MouseEventHandler(this.m_gl_MouseUp);
            m_gl.MouseWheel += new MouseEventHandler(m_gl_MouseWheel);
            m_gl.Resize += new System.EventHandler(this.m_gl_Resize);

            this.splitContainer1.Panel1.Controls.Add(this.m_gl);

            m_operations = new Operations();
            m_operations.Initialize(input);

            m_propertyGrid.SelectedObject = m_operations.Input;
            m_operations.Input.PropertyChanged += new PropertyChangedEventHandler(vp_PropertyChanged);
        }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);

            m_gl.MakeCurrent();
        }

        protected override void OnClosing(CancelEventArgs e)
        {
            string settingPath = Path.Combine(Path.GetDirectoryName(Application.ExecutablePath), "Oxel.Settings.xml");
            VoxelizationInput.Save(settingPath, m_operations.Input);

            if (m_operations != null)
            {
                m_operations.Dispose();
                m_operations = null;
            }

            base.OnClosing(e);
        }

        private void openToolStripMenuItem_Click(object sender, EventArgs e)
        {
            OpenFileDialog of = new OpenFileDialog();
            of.Filter = "Wavefront Object (*.obj)|*.obj;";

            if (of.ShowDialog(this) == DialogResult.OK)
            {
                Open(of.FileName);
            }
        }

        private void InitOpenGLResources()
        {
            String fragSource = Properties.Resources.ps_FrontBack;
            String vertSource = Properties.Resources.vs_FrontBack;
            shader_frontback = GLEx.CreateShaderProgramStrings(vertSource, fragSource);

            GL.ClearColor(0.0f, 0.0f, 0.0f, 0f);
            GL.Enable(EnableCap.DepthTest);

            // Setup VBO state
            GL.EnableClientState(ArrayCap.VertexArray);
            GL.EnableClientState(ArrayCap.IndexArray);

            try
            {
                codeText = new QFont("consolab.ttf", Properties.Resources.consolab, 10, FontStyle.Bold);
            }
            catch (System.Exception ex)
            {
                Debug.WriteLine(ex.ToString());
            }

            GL.BindTexture(TextureTarget.Texture2D, 0);
        }

        void OnRenderFrame(FrameEventArgs e)
        {
            GL.MatrixMode(MatrixMode.Projection);
            Matrix4 p = Matrix4.CreatePerspectiveFieldOfView(MathHelper.PiOver4, m_gl.Width / (float)m_gl.Height, m_camera.NearPlane, m_camera.FarPlane);
            GL.LoadMatrix(ref p);

            GL.MatrixMode(MatrixMode.Modelview);

            Matrix4 transformation = Matrix4.Mult(Matrix4.CreateRotationY(m_camera.cameraYaw),
                                                  Matrix4.CreateRotationX(m_camera.cameraPitch));
            Matrix4 translation = Matrix4.CreateTranslation(-m_camera.cameraPosition);
            Matrix4 world = transformation * translation;
            GL.LoadMatrix(ref transformation);
            GL.MultMatrix(ref translation);

            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            GL.PushMatrix();

            DrawScene();

            GL.PopMatrix();

            m_gl.SwapBuffers();
        }

        private void DrawOctree(VoxelizingOctree o)
        {
            GL.Enable(EnableCap.DepthTest);

            if (showIntersectingCells)
                o.Root.Draw(o.MaxLevels - 1, CellStatus.Intersecting, new Vector4(0.25f, 1.0f, 0.25f, 1.0f), 1.0f);

            if (showIntersectingBoundsCells)
                o.Root.Draw(o.MaxLevels - 1, CellStatus.IntersectingBounds, new Vector4(1.0f, 1.0f, 0.0f, 1.0f), 1.0f);

            for (int i = 0; i < m_visualizedMaxLevel; i++)
            {
                if (showUnknownCells)
                    o.Root.Draw(i, CellStatus.Unknown, new Vector4(1.0f, 1.0f, 1.0f, 1.0f), 1.0f);
                if (showOutsideCells)
                    o.Root.Draw(i, CellStatus.Outside, new Vector4(1.0f, 0.25f, 0.25f, 1.0f), 1.0f);
                if (showInsideCells)
                    o.Root.Draw(i, CellStatus.Inside, new Vector4(0.25f, 0.25f, 1.0f, 1.0f), 1.0f);
            }
        }

        private void DrawScene()
        {
            GL.ClearColor(0.0f, 0.0f, 0.0f, 0f);
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            GL.Enable(EnableCap.DepthTest);
            GL.Disable(EnableCap.CullFace);

            if (m_operations.Context == null)
                return;

            if (m_showOriginalMesh)
            {
                if (m_operations.Context.OriginalMesh != null)
                {
                    var material = new Material();
                    material.ShaderHandle = shader_frontback;
                    material.ShowLines = m_showMeshLines;
                    if (m_alphaBlendingMode)
                    {
                        material.IsAlphaBlending = true;
                        material.SetVector4("Front", new Vector4(1, 0, 0, 0.5f));
                        material.SetVector4("Back", new Vector4(1, 0, 0, 0.5f));
                    }
                    else
                    {
                        material.SetVector4("Front", new Vector4(0, 0, 1, 1));
                        material.SetVector4("Back", new Vector4(1, 0, 0, 1));
                    }
                    m_operations.Context.OriginalMesh.Render(material);
                }
            }

            if (m_showOccluderMesh)
            {
                if (m_operations.Context.OccluderMesh != null)
                {
                    var material = new Material();
                    material.ShaderHandle = shader_frontback;
                    material.ShowLines = m_showOccluderLines;
                    if (m_alphaBlendingMode)
                    {
                        material.IsAlphaBlending = true;
                        material.SetVector4("Front", new Vector4(0, 1, 0, 0.5f));
                        material.SetVector4("Back", new Vector4(0, 1, 0, 0.5f));
                    }
                    else
                    {
                        material.SetVector4("Front", new Vector4(0, 0, 1, 1));
                        material.SetVector4("Back", new Vector4(1, 0, 0, 1));
                    }
                    m_operations.Context.OccluderMesh.Render(material);
                }
            }

            DrawOctree(m_operations.Context.Octree);

            DrawDebug();

            DrawOverlay(m_operations.Context.OriginalMesh, m_operations.Context.OccluderMesh);
        }

        private void DrawDebug()
        {
            if (!showDebugLines)
                return;

            if (m_operations.Context.VoxelizationOutput == null)
                return;

            List<List<Edge>> debugLines = m_operations.Context.VoxelizationOutput.DebugLines;
            if (debugLines == null)
                return;

            GL.DepthFunc(DepthFunction.Lequal);
            GL.LineWidth(5.0f);

            GL.Begin(BeginMode.Lines);
            foreach (List<Edge> loop in debugLines)
            {
                //Random r = new Random(1024);

                for (int i = 0; i < loop.Count; i++)
                {
                    //GL.Color4((float)r.NextDouble(), (float)r.NextDouble(), (float)r.NextDouble(), 1.0f);
                    GL.Color4(0, 1, 0, 1.0f);

                    GL.Vertex3(loop[i].v0);
                    GL.Vertex3(loop[i].v1);
                }
            }
            GL.End();
        }

        void OnUpdateFrame(FrameEventArgs e)
        {
            m_camera.OnUpdateFrame(e);
        }

        private void m_gl_Paint(object sender, PaintEventArgs e)
        {
            if (!glLoaded)
                return;

            double elapsedSeconds = renderTimer.Elapsed.TotalSeconds;
            renderTimer.Restart();

            if (elapsedSeconds == 0)
                return;

            if (m_gl.Context.IsCurrent)
            {
                OnUpdateFrame(new FrameEventArgs(elapsedSeconds));
                OnRenderFrame(new FrameEventArgs(elapsedSeconds));
            }
        }

        void Application_Idle(object sender, EventArgs e)
        {
            if (User32.IsApplicationActivate())
            {
                while (m_gl.IsIdle)
                {
                    m_gl.Invalidate();
                }
            }
        }

        private void m_gl_Load(object sender, EventArgs e)
        {
            glLoaded = true;
            Application.Idle += Application_Idle;

            InitOpenGLResources();
        }

        void m_gl_MouseUp(object sender, System.Windows.Forms.MouseEventArgs e)
        {
            m_camera.OnMouseUp(m_gl, e);
        }

        void m_gl_MouseDown(object sender, System.Windows.Forms.MouseEventArgs e)
        {
            m_camera.OnMouseDown(m_gl, e);
        }

        void m_gl_Resize(object sender, EventArgs e)
        {
            if (m_gl.Width == 0 || m_gl.Height == 0)
                return;

            GL.Viewport(0, 0, m_gl.Width, m_gl.Height);
        }

        private void m_gl_MouseMove(object sender, System.Windows.Forms.MouseEventArgs e)
        {
            m_camera.OnMouseMove(m_gl, e);
        }

        void m_gl_MouseWheel(object sender, System.Windows.Forms.MouseEventArgs e)
        {
            m_camera.OnMouseWheel(m_gl, e);
        }

        private void m_gl_KeyPress(object sender, System.Windows.Forms.KeyPressEventArgs e)
        {
            if (m_operations.Context == null)
                return;

            VoxelizingOctree o = m_operations.Context.Octree;

            switch (e.KeyChar)
            {
                case '-':
                case '_':
                    m_visualizedMaxLevel = MathEx.Clamp(m_visualizedMaxLevel - 1, 0, o.MaxLevels);
                    break;
                case '=':
                case '+':
                    m_visualizedMaxLevel = MathEx.Clamp(m_visualizedMaxLevel + 1, 0, o.MaxLevels);
                    break;
                case 'u':
                    m_showUnknownVoxelMenu.Checked = !m_showUnknownVoxelMenu.Checked;
                    break;
                case 'i':
                    m_showInternalVoxelMenu.Checked = !m_showInternalVoxelMenu.Checked;
                    break;
                case 'o':
                    m_showExternalVoxelMenu.Checked = !m_showExternalVoxelMenu.Checked;
                    break;
                case 'm':
                    m_showOriginalMeshMenu.Checked = !m_showOriginalMeshMenu.Checked;
                    break;
            }
        }

        private bool Open(string meshFile)
        {
            if (!File.Exists(meshFile))
                return false;

            m_operations.Open(meshFile, m_operations.Input.WindingOrder);

            m_visualizedMaxLevel = m_operations.Context.Octree.MaxLevels;
            m_showOriginalMesh = true;

            m_camera.cameraYaw = (float)(Math.PI * 0.5);

            m_camera.cameraPosition = m_operations.Context.Octree.Root.Center - new Vector3(-(m_operations.Context.Octree.MeshBounds.MinX - m_operations.Context.Octree.MeshBounds.MaxX), 0, 0);
            m_camera.Speed = (m_operations.Context.Octree.Root.Bounds.MaxX - m_operations.Context.Octree.Root.Bounds.MinX) / 10.0f;
            m_camera.FastSpeed = m_camera.Speed * 10.0f;

            m_camera.FarPlane = (m_operations.Context.Octree.Root.Bounds.MaxX - m_operations.Context.Octree.Root.Bounds.MinX) * 4;
            m_camera.NearPlane = m_camera.FarPlane * m_camera.NearFarRatio;

            GL.BindBuffer(BufferTarget.ArrayBuffer, 0);
            GL.BindBuffer(BufferTarget.ElementArrayBuffer, 0);

            return true;
        }

        void vp_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            if (m_operations.Context == null)
                return;

            switch (e.PropertyName)
            {
                case "OctreeLevels":
                    m_operations.Context.Octree = new VoxelizingOctree(m_operations.Input.OctreeLevels);
                    m_operations.Context.Octree.GenerateOctree(m_operations.Context.CurrentMesh);

                    m_visualizedMaxLevel = m_operations.Context.Octree.MaxLevels;
                    break;
                case "WindingOrder":
                    m_operations.Open(m_operations.Context.CurrentMeshFile, m_operations.Input.WindingOrder);
                    break;
            }
        }

        private void voxelizeToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Voxelize();
        }

        private void Voxelize()
        {
            m_gl.Context.MakeCurrent(null);

            ProgressDialog dialog = new ProgressDialog();

            m_operations.GenerateOccluder(dialog.UpdateProgress, new Action(() => {

                dialog.Invoke(new Action(() =>
                {
                    dialog.CanClose = true;
                    dialog.Close();
                }));
            }));

            dialog.ShowDialog(this);

            m_showOriginalMesh = false;

            m_gl.Refresh();
            m_gl.MakeCurrent();
        }

        private void DrawOverlay(RenderableMesh orignalMesh, RenderableMesh occluderMesh)
        {
            GL.PushAttrib(AttribMask.AllAttribBits);

            GL.Color4(1.0f, 1.0f, 1.0f, 1.0f);
            GL.Enable(EnableCap.Texture2D);
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);

            GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
            GL.DepthFunc(DepthFunction.Lequal);

            QFont.Begin();

            float yOffset = 5;

            if (orignalMesh != null)
                PrintCode(String.Format("Orignal Triangles   : {0,5}", orignalMesh.Triangles), ref yOffset);
            if (occluderMesh != null)
                PrintCode(String.Format("Occluder Triangles  : {0,5}", occluderMesh.Triangles), ref yOffset);

            PrintCode(String.Format("Occluder Levels     : {0,5}", m_visualizedMaxLevel), ref yOffset);
            if (m_operations.Context.VoxelizationOutput != null)
            {
                PrintCode(String.Format("Volume Coverage     : {0,5:0.##}%", (100 * m_operations.Context.VoxelizationOutput.VolumeCoverage)), ref yOffset);
                PrintCode(String.Format("Silhouette Coverage : {0,5:0.##}%", (100 * m_operations.Context.VoxelizationOutput.SilhouetteCoverage)), ref yOffset);
                PrintCode(String.Format("Time Taken          : {0,5:0.##} seconds", m_operations.Context.VoxelizationOutput.TimeTaken.TotalSeconds), ref yOffset);
            }

            QFont.End();

            GL.PopAttrib();
        }

        private void PrintCode(string code, ref float yOffset)
        {
            GL.PushMatrix();
            yOffset += 5;
            GL.Translate(10f, yOffset, 0f);
            codeText.Print(code, new RectangleF(0, 0, Width, 1f), QFontAlignment.Left);
            yOffset += codeText.Measure(code, new RectangleF(0, 0, Width, 1f), QFontAlignment.Left).Height;
            GL.PopMatrix();
        }

        private void exitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void aboutToolStripMenuItem_Click(object sender, EventArgs e)
        {
            AboutDialog aboutDialog = new AboutDialog();
            aboutDialog.ShowDialog(this);
        }

        private void saveToolStripMenuItem_Click(object sender, EventArgs e)
        {
            SaveFileDialog saveDialog = new SaveFileDialog();
            saveDialog.AddExtension = true;
            saveDialog.Filter = "Wavefront Object (*.obj)|*.obj;";

            if (saveDialog.ShowDialog(this) != DialogResult.OK)
                return;

            m_operations.Save(saveDialog.FileName);
        }

        private void ShowInternalVoxels(object sender, EventArgs e)
        {
            showInsideCells = !showInsideCells;
        }

        private void ShowExternalVoxels(object sender, EventArgs e)
        {
            showOutsideCells = !showOutsideCells;
        }

        private void ShowUnknownVoxels(object sender, EventArgs e)
        {
            showUnknownCells = !showUnknownCells;
        }

        private void ShowIntersectingVoxels(object sender, EventArgs e)
        {
            showIntersectingCells = !showIntersectingCells;
        }

        private void ShowIntersectingBoundsVoxels(object sender, EventArgs e)
        {
            showIntersectingBoundsCells = !showIntersectingBoundsCells;
        }

        private void ShowDebugLines(object sender, EventArgs e)
        {
            showDebugLines = !showDebugLines;
        }

        private void ShowOriginalMesh(object sender, EventArgs e)
        {
            m_showOriginalMesh = !m_showOriginalMesh;
        }

        private void ShowOccluderMesh(object sender, EventArgs e)
        {
            m_showOccluderMesh = !m_showOccluderMesh;
        }

        private void occluderLinesToolStripMenuItem_Click(object sender, EventArgs e)
        {
            m_showOccluderLines = !m_showOccluderLines;
        }

        private void meshLinesToolStripMenuItem_Click(object sender, EventArgs e)
        {
            m_showMeshLines = !m_showMeshLines;
        }

        private void alphaBlendingModeToolStripMenuItem_Click(object sender, EventArgs e)
        {
            m_alphaBlendingMode = !m_alphaBlendingMode;
        }

        private void viewToolStripMenuItem_DropDownOpening(object sender, EventArgs e)
        {
            m_showInternalVoxelMenu.Checked = showInsideCells;
            m_showExternalVoxelMenu.Checked = showOutsideCells;
            m_showUnknownVoxelMenu.Checked = showUnknownCells;
            m_showIntersectingVoxelMenu.Checked = showIntersectingCells;
            m_showIntersectingBoundsVoxelMenu.Checked = showIntersectingBoundsCells;
            m_debugLinesToolStripMenuItem.Checked = showDebugLines;
            m_showOriginalMeshMenu.Checked = m_showOriginalMesh;
            m_showOccluderMeshMenu.Checked = m_showOccluderMesh;
            occluderLinesToolStripMenuItem.Checked = m_showOccluderLines;
            meshLinesToolStripMenuItem.Checked = m_showMeshLines;
            alphaBlendingModeToolStripMenuItem.Checked = m_alphaBlendingMode;
        }
    }
}