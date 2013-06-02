using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Policy;
using System.Text;

using Oxel;
using OpenTK;

namespace liboccluder
{
    [ComVisible(true), Guid("79788E82-7DFC-446D-B6FA-5112B8BAA328")]
    [StructLayout(LayoutKind.Sequential)]
    public struct Vertex3
    {
        public float x;
        public float y;
        public float z;
    };

    class CreateMeshData
    {
        public static MeshData Create(Vertex3[] verticesIn, short[] indices)
        {
            Vector3[] vertices = new Vector3[verticesIn.Length];
            Vector3[] normals = new Vector3[verticesIn.Length];
            Vector2[] uv = new Vector2[verticesIn.Length];
            Tri[] tri   = new Tri[ indices.Length / 3 ];

            for (uint i = 0; i < verticesIn.Length; ++i)
            {
                vertices[i].X = verticesIn[i].x;
                vertices[i].Y = verticesIn[i].y;
                vertices[i].Z = verticesIn[i].z;

                normals[i].X = 0.0f;
                normals[i].Y = 0.0f;
                normals[i].Z = 1.0f;

                uv[i].X = 0;
                uv[i].Y = 0;
            }

            for (int i = 0; i < indices.Length / 3 ; i += 3 )
            {
                MeshPoint p1 = new MeshPoint( 3 * i + 0, 3 * i + 0, 3 * i + 0 );
                MeshPoint p2 = new MeshPoint( 3 * i + 1, 3 * i + 1, 3 * i + 1);
                MeshPoint p3 = new MeshPoint( 3 * i + 2, 3 * i + 2, 3 * i + 2);

                tri[i] = new Tri( p1, p2, p3);
            }

            return new MeshData( vertices, normals, uv, tri);
        }
    };

    
    [ComVisible(true), Guid("1F38B050-DA3A-4237-9F40-8CA192F90D77"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IOccluderGeneration
    {
        /// <summary>
        ///		Write a message
        /// </summary>
        /// <param name="message">message to write</param>
        void Write([MarshalAs(UnmanagedType.BStr)]string message);

        void Write2( [MarshalAs(UnmanagedType.LPArray, SizeParamIndex=1 ) ] Vertex3[] vertices, long size);

        void Compute( [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1 ) ] Vertex3[] verticesIn, long verticesInSize,
                      [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 3)] short[] inidicesIn, long indicesSize,
                      [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 5)] out Vertex3[] verticesOut, out long verticesOutSize,
                      [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 7)] out short[] inidicesOut, out long indicesOutSize);
    }
    
    [GuidAttribute("91E80A13-9EF7-48E8-8F71-D72F70561E4B"), ComVisible(true) ]
    public class AppDomainManager : System.AppDomainManager, IOccluderGeneration
    {
        public override void InitializeNewDomain(AppDomainSetup appDomainInfo)
        {
            // let the unmanaged host know about us
            InitializationFlags = AppDomainManagerInitializationOptions.RegisterWithHost;
        }

        public override AppDomain CreateDomain(string friendlyName, Evidence securityInfo, AppDomainSetup appDomainInfo)
        {
            var appDomain = base.CreateDomain(friendlyName, securityInfo, appDomainInfo);
            return appDomain;
        }
        
        void IOccluderGeneration.Write(string message)
        {
            Console.WriteLine(message);
        }

        void IOccluderGeneration.Write2( Vertex3[] vertices, long size )
        {
            for (uint i = 0; i < size; ++i )
            {
                Console.Write(vertices[i].x);
                Console.Write(" ");
                Console.Write(vertices[i].y);
                Console.Write(" ");
                Console.WriteLine(vertices[i].z);
            }
        }

        void IOccluderGeneration.Compute
            (
                      [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1 ) ] Vertex3[] verticesIn, long verticesInSize,
                      [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 3)] short[] indicesIn, long indicesSize,
                      [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 5)] out Vertex3[] verticesOut, out long verticesOutSize,
                      [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 7)] out short[] inidicesOut, out long indicesOutSize
            )
        {
            MeshData meshData = CreateMeshData.Create(verticesIn, indicesIn);
           
            verticesOut = new Vertex3[3];
            verticesOutSize = 3;
            verticesOut[0].x = 1.0f;
            verticesOut[0].y = 1.0f;
            verticesOut[0].z = 1.0f;

            verticesOut[1].x = 2.0f;
            verticesOut[1].y = 2.0f;
            verticesOut[1].z = 2.0f;

            verticesOut[2].x = 3.0f;
            verticesOut[2].y = 3.0f;
            verticesOut[2].z = 3.0f;

            inidicesOut = new short[3];
            indicesOutSize = 3;
            inidicesOut[0] = 3;
            inidicesOut[1] = 2;
            inidicesOut[2] = 1;
        }
    };

   
}
