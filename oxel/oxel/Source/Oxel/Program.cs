using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using System.Threading;
using System.IO;

namespace Oxel
{
    class Program
    {
        public class ProgramOptions : VoxelizationInput
        {
            [CommandLineParser.Name("c", "Command line mode")]
            public bool UseCommandLine;

            [CommandLineParser.Required]
            public string InputMesh;

            [CommandLineParser.Required]
            public string OutputMesh;
        }

        [STAThread]
        public static void Main(string[] args)
        {
            AppDomain.CurrentDomain.AssemblyResolve += OnAssemblyResolve;

            Run(args);
        }

        [DllImport("kernel32.dll")]
        static extern bool AttachConsole(int processId);
        const int PARENT_PROCESS_ID = -1;

        public static void Run(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            ProgramOptions options = new ProgramOptions();

            string settingPath = Path.Combine(Path.GetDirectoryName(Application.ExecutablePath), "Oxel.Settings.xml");
            VoxelizationInput input = VoxelizationInput.Load(settingPath);
            if (input == null)
                input = new VoxelizationInput();

            if (args.Contains("-c"))
            {
                // Make sure user can see console output
                AttachConsole(PARENT_PROCESS_ID);

                input.Clone(options);

                if (!CommandLineParser.Parse<ProgramOptions>(args, ref options))
                    return;

                options.Clone(input);
            }
            else
            {
                CommandLineParser.Parse<VoxelizationInput>(args, ref input);
            }

            if (options.UseCommandLine)
            {
                Logger.IsCommandLine = true;

                Operations operations = new Operations();
                operations.Initialize(input);
                operations.Open(options.InputMesh, input.WindingOrder);
                WaitHandle waitHandle = operations.GenerateOccluder((VoxelizationProgress vp) => {
                    string coverage =
                        String.Format("Volume Coverage     : {0,5:0.##}%", (100 * vp.VolumeCoverage)) + "    " +
                        String.Format("Silhouette Coverage : {0,5:0.##}%", (100 * vp.SilhouetteCoverage));

                    if (!String.IsNullOrEmpty(vp.Status))
                        Console.WriteLine(vp.Status + "\r\n");

                    Console.WriteLine(coverage);
                }, new Action(() => { }));
                waitHandle.WaitOne();
                operations.Save(options.OutputMesh);
            }
            else
            {
                using (MainWindow window = new MainWindow(input))
                {
                    window.ShowDialog();
                }
            }
        }

        /// <summary>
        /// This loads the embedded assemblies when .Net fails to find them on disk.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="args"></param>
        /// <returns></returns>
        static Assembly OnAssemblyResolve(object sender, ResolveEventArgs args)
        {
            String resourceName = "Oxel.Assemblies." + new AssemblyName(args.Name).Name + ".dll";

            using (var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resourceName))
            {
                if (stream != null)
                {
                    Byte[] assemblyData = new Byte[stream.Length];
                    stream.Read(assemblyData, 0, assemblyData.Length);
                    return Assembly.Load(assemblyData);
                }
            }

            return null;
        }
    }
}