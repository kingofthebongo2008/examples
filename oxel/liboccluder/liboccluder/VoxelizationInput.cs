using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml.Serialization;

namespace Oxel
{
    public enum UpAxis
    {
        Y,
        Z
    }

    public enum WindingOrder
    {
        Clockwise,
        CounterClockwise
    }

    public enum OcclusionType
    {
        [Description("Takes the fewest number of octree levels necessary to cover the required volume.")]
        Octree,
        [Description("Expands boxes until the volume coverage has been met.")]
        BoxExpansion,
        //[Description("Runs an optimization problem on the inner volume to find the optimal occluder volumes.")]
        //SimulatedAnnealing,
        [Description("Performs the box expansion algorithm using a brute force search on volume space.")]
        BruteForce,
    }

    public class VoxelizationInput : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;

        [CommandLineParser.Name("VoxelLevel")]
        public int m_voxelLevel;

        [CommandLineParser.Name("MinVolume")]
        public float m_minVolume;

        [CommandLineParser.Name("MinOcclusion")]
        public float m_minOcclusion;

        [CommandLineParser.Name("Type")]
        public OcclusionType m_type;

        [CommandLineParser.Name("Retriangulate")]
        public bool m_retriangulate;

        [CommandLineParser.Name("RemoveTop")]
        public bool m_removeTop;

        [CommandLineParser.Name("RemoveBottom")]
        public bool m_removeBottom;

        [CommandLineParser.Name("UpAxis")]
        public UpAxis m_upAxis;

        [CommandLineParser.Name("WindingOrder")]
        public WindingOrder m_windingOrder;

        [XmlIgnore]
        [CommandLineParser.Ignore]
        public RenderableMesh OriginalMesh;

        [XmlIgnore]
        [CommandLineParser.Ignore]
        public VoxelizingOctree Octree;

        public VoxelizationInput()
        {
            m_voxelLevel = 6;
            m_minVolume = 0.65f;
            m_minOcclusion = 0.03f;
            m_type = OcclusionType.BoxExpansion;
            m_retriangulate = true;
            m_removeTop = false;
            m_removeBottom = false;
            m_upAxis = UpAxis.Y;
            m_windingOrder = WindingOrder.CounterClockwise;
        }

        [Category("Input")]
        [DisplayName("Octree Levels")]
        [Description("The number of subdivisions the octree will perform.")]
        public int OctreeLevels
        {
            get { return m_voxelLevel; }
            set
            {
                if (m_voxelLevel != value)
                {
                    m_voxelLevel = value;
                    if (PropertyChanged != null)
                        PropertyChanged(this, new PropertyChangedEventArgs("OctreeLevels"));
                }
            }
        }

        [Category("Input")]
        [Description("The process that will be used to generate the occluder.")]
        public OcclusionType Type
        {
            get { return m_type; }
            set
            {
                if (m_type != value)
                {
                    m_type = value;
                    if (PropertyChanged != null)
                        PropertyChanged(this, new PropertyChangedEventArgs("Type"));
                }
            }
        }

        [Category("Input")]
        [DisplayName("Up Axis")]
        [Description("The up axis of the original mesh.")]
        public UpAxis UpAxis
        {
            get { return m_upAxis; }
            set
            {
                if (m_upAxis != value)
                {
                    m_upAxis = value;
                    if (PropertyChanged != null)
                        PropertyChanged(this, new PropertyChangedEventArgs("UpAxis"));
                }
            }
        }

        [Category("Input")]
        [DisplayName("Winding Order")]
        [Description("The winding order of the triangles.")]
        public WindingOrder WindingOrder
        {
            get { return m_windingOrder; }
            set
            {
                if (m_windingOrder != value)
                {
                    m_windingOrder = value;
                    if (PropertyChanged != null)
                        PropertyChanged(this, new PropertyChangedEventArgs("WindingOrder"));
                }
            }
        }

        [Category("Filter")]
        [DisplayName("Minimum Occlusion")]
        [Description("The minimum occlusion that each new volume must have when tested against the silhouette of the original mesh.")]
        [TypeConverter(typeof(PercentageConverter))]
        public float MinimumOcclusion
        {
            get { return m_minOcclusion; }
            set
            {
                if (m_minOcclusion != value)
                {
                    m_minOcclusion = value;
                    if (PropertyChanged != null)
                        PropertyChanged(this, new PropertyChangedEventArgs("MinimumOcclusion"));
                }
            }
        }

        [Category("Post Process")]
        [Description("Retriangulates the mesh to attempt to reduce triangle count on the final occluder.")]
        public bool Retriangulate
        {
            get { return m_retriangulate; }
            set
            {
                if (m_retriangulate != value)
                {
                    m_retriangulate = value;
                    if (PropertyChanged != null)
                        PropertyChanged(this, new PropertyChangedEventArgs("Retriangulate"));
                }
            }
        }

        [Category("Post Process")]
        [DisplayName("Remove Top")]
        [Description("Remove all top polygons.")]
        public bool RemoveTop
        {
            get { return m_removeTop; }
            set
            {
                if (m_removeTop != value)
                {
                    m_removeTop = value;
                    if (PropertyChanged != null)
                        PropertyChanged(this, new PropertyChangedEventArgs("RemoveTop"));
                }
            }
        }

        [Category("Post Process")]
        [DisplayName("Remove Bottom")]
        [Description("Remove all bottom polygons that are with-in one voxel of the mesh bounds.")]
        public bool RemoveBottom
        {
            get { return m_removeBottom; }
            set
            {
                if (m_removeBottom != value)
                {
                    m_removeBottom = value;
                    if (PropertyChanged != null)
                        PropertyChanged(this, new PropertyChangedEventArgs("RemoveBottom"));
                }
            }
        }

        [Category("Stop Condition")]
        [DisplayName("Minimum Volume")]
        [Description("The minimum amount of volume that must at least have been attempted to have been filled before stopping.")]
        [TypeConverter(typeof(PercentageConverter))]
        public float MinimumVolume
        {
            get { return m_minVolume; }
            set
            {
                if (m_minVolume != value)
                {
                    m_minVolume = value;
                    if (PropertyChanged != null)
                        PropertyChanged(this, new PropertyChangedEventArgs("MinimumVolume"));
                }
            }
        }


        internal VoxelizationInput Clone()
        {
            VoxelizationInput input = new VoxelizationInput();
            return Clone(input);
        }

        internal VoxelizationInput Clone(VoxelizationInput input)
        {
            input.m_voxelLevel = m_voxelLevel;
            input.m_minVolume = m_minVolume;
            input.m_type = m_type;
            input.m_retriangulate = m_retriangulate;
            input.m_minOcclusion = m_minOcclusion;
            input.m_removeTop = m_removeTop;
            input.m_removeBottom = m_removeBottom;
            input.m_upAxis = m_upAxis;
            input.OriginalMesh = OriginalMesh;
            input.Octree = Octree;

            return input;
        }

        public static VoxelizationInput Load(string file)
        {
            try
            {
                if (!File.Exists(file))
                    return null;

                VoxelizationInput settings = null;
                XmlSerializer mySerializer = new XmlSerializer(typeof(VoxelizationInput));
                using (FileStream myFileStream = new FileStream(file, FileMode.Open))
                {
                    settings = (VoxelizationInput)mySerializer.Deserialize(myFileStream);
                }

                return settings;
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                return null;
            }
        }

        public static bool Save(string file, VoxelizationInput settings)
        {
            try
            {
                XmlSerializer x = new XmlSerializer(typeof(VoxelizationInput));
                using (StreamWriter myWriter = new StreamWriter(file))
                {
                    x.Serialize(myWriter, settings);
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                return false;
            }

            return true;
        }
    }
}
