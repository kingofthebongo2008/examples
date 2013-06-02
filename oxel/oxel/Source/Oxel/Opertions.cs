using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using OpenTK;
using OpenTK.Graphics;

namespace Oxel
{
    public class Operations : IDisposable
    {
        public VoxelizationInput Input;
        public VoxelizationContext Context;

        public Operations()
        {
        }

        ~Operations()
        {
            Dispose();
        }

        public void Dispose()
        {
            GC.SuppressFinalize(this);

            if (Context != null)
                Context.Dispose();
        }

        public void Initialize(VoxelizationInput input)
        {
            Input = input;
        }

        public bool Open(string meshFile, WindingOrder order)
        {
            if (Context != null)
            {
                Context.Dispose();
                Context = null;
            }

            Context = new VoxelizationContext();

            try
            {
                IImporter importer = IOFactory.ImporterFactory(meshFile);
                Context.CurrentMeshFile = meshFile;
                Context.CurrentMesh = importer.Load(meshFile);
            }
            catch (IOException)
            {
                Logger.DisplayError("Unable to open the file: " + meshFile);
                return false;
            }

            Context.Octree = new VoxelizingOctree(Input.OctreeLevels);
            Context.Octree.GenerateOctree(Context.CurrentMesh);

            Vector4[] orignalVertices = new Vector4[Context.CurrentMesh.Vertices.Length];
            for (int i = 0; i < orignalVertices.Length; i++)
            {
                orignalVertices[i] = new Vector4(Context.CurrentMesh.Vertices[i], 1);
            }

            int[] originalIndicies = new int[Context.CurrentMesh.Tris.Length * 3];

            if (order == WindingOrder.CounterClockwise)
            {
                for (int i = 0; i < Context.CurrentMesh.Tris.Length; i++)
                {
                    Tri t = Context.CurrentMesh.Tris[i];
                    int index = i * 3;
                    originalIndicies[index + 0] = t.P1.Vertex;
                    originalIndicies[index + 1] = t.P2.Vertex;
                    originalIndicies[index + 2] = t.P3.Vertex;
                }
            }
            else if (order == WindingOrder.Clockwise)
            {
                for (int i = 0; i < Context.CurrentMesh.Tris.Length; i++)
                {
                    Tri t = Context.CurrentMesh.Tris[i];
                    int index = i * 3;
                    originalIndicies[index + 0] = t.P1.Vertex;
                    originalIndicies[index + 2] = t.P2.Vertex;
                    originalIndicies[index + 1] = t.P3.Vertex;
                }
            }

            if (Context.OriginalMesh != null)
            {
                Context.OriginalMesh.Dispose();
                Context.OriginalMesh = null;
            }

            Mesh mesh = new Mesh();
            mesh.Indicies = originalIndicies;
            mesh.Vertices = orignalVertices;

            Context.OriginalMesh = new RenderableMesh(mesh, true);

            return true;
        }

        public WaitHandle GenerateOccluder(Action<VoxelizationProgress> progress, Action done)
        {
            ManualResetEvent waitHandle = new ManualResetEvent(false);

            if (Context == null || Context.OriginalMesh == null)
            {
                Logger.DisplayError("Please Open a mesh first.");
                waitHandle.Set();
                return waitHandle;
            }

            Input.Octree = Context.Octree;
            Input.OriginalMesh = Context.OriginalMesh;

            VoxelizationInput input = Input.Clone();
            VoxelizationOutput output = null;

            IOccluderGenerator occluder;
            switch (input.Type)
            {
                case OcclusionType.Octree:
                    occluder = new OccluderOctree();
                    break;
                case OcclusionType.BoxExpansion:
                    occluder = new OccluderBoxExpansion();
                    break;
                //case OcclusionType.SimulatedAnnealing:
                //    occluder = new OccluderBoxExpansion();
                //    break;
                case OcclusionType.BruteForce:
                    occluder = new OccluderBoxExpansion();
                    break;
                default:
                    throw new Exception("Unknown occluder type.");
            }

            Thread thread = new Thread(() =>
            {
                INativeWindow window = new OpenTK.NativeWindow();
                IGraphicsContext gl = new GraphicsContext(new GraphicsMode(32, 24, 8), window.WindowInfo);
                gl.MakeCurrent(window.WindowInfo);

                while (window.Exists)
                {
                    window.ProcessEvents();

                    try
                    {
                        RobustVoxelizer voxelizer = new RobustVoxelizer(512, 512);
                        output = voxelizer.Voxelize(input, progress);
                        voxelizer.Dispose();

                        output = occluder.Generate(input, progress);
                    }
                    catch (System.Exception ex)
                    {
                        Debug.WriteLine(ex.ToString());
                    }

                    window.Close();
                    break;
                }

                gl.MakeCurrent(null);

                if (Context.OccluderMesh != null)
                    Context.OccluderMesh.Dispose();

                Context.Octree = output.Octree;
                Context.OccluderMesh = output.OccluderMesh;
                Context.VoxelizationOutput = output;

                waitHandle.Set();

                done();
            });

            thread.IsBackground = true;
            thread.SetApartmentState(ApartmentState.STA);
            thread.Start();

            return waitHandle;
        }

        public bool Save(string fileName)
        {
            if (Context == null || Context.OccluderMesh == null)
            {
                Logger.DisplayError("No occluder found!  Please Open a mesh and then (Build > Voxelize) to generate an occluder before saving.");
                return false;
            }

            string fileExt = Path.GetExtension(fileName);

            try
            {
                IExporter exporter = IOFactory.ExporterFactory(fileName);
                exporter.Save(fileName, Context.OccluderMesh.Vertices, Context.OccluderMesh.Indicies);
            }
            catch (System.Exception)
            {
                Logger.DisplayError("Unable to save the file");
                return false;
            }

            return true;
        }
    }
}
