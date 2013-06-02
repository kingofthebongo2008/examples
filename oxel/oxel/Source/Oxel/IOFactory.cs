using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Oxel
{
    public class IOFactory
    {
        public static IImporter ImporterFactory(string filename)
        {
            if (filename.EndsWith(".obj")) return new WavefrontObjectImporter(false);

            throw new ArgumentException("Invalid filename");
        }

        public static IExporter ExporterFactory(string filename)
        {
            if (filename.EndsWith(".obj")) return new WavefrontObjectExporter();

            throw new ArgumentException("Invalid filename");
        }
    }
}