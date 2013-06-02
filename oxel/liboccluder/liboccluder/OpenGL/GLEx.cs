using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;

namespace Oxel.OpenGL
{
    public static class GLEx
    {
        public static bool CheckFboStatus()
        {
            switch (GL.Ext.CheckFramebufferStatus(FramebufferTarget.FramebufferExt))
            {
                case FramebufferErrorCode.FramebufferCompleteExt:
                    {
                        Trace.WriteLine("FBO: The framebuffer is complete and valid for rendering.");
                        return true;
                    }
                case FramebufferErrorCode.FramebufferIncompleteAttachmentExt:
                    {
                        Trace.WriteLine("FBO: One or more attachment points are not framebuffer attachment complete. This could mean there’s no texture attached or the format isn’t renderable. For color textures this means the base format must be RGB or RGBA and for depth textures it must be a DEPTH_COMPONENT format. Other causes of this error are that the width or height is zero or the z-offset is out of range in case of render to volume.");
                        break;
                    }
                case FramebufferErrorCode.FramebufferIncompleteMissingAttachmentExt:
                    {
                        Trace.WriteLine("FBO: There are no attachments.");
                        break;
                    }
                /* case  FramebufferErrorCode.GL_FRAMEBUFFER_INCOMPLETE_DUPLICATE_ATTACHMENT_EXT: 
                     {
                         Trace.WriteLine("FBO: An object has been attached to more than one attachment point.");
                         break;
                     }*/
                case FramebufferErrorCode.FramebufferIncompleteDimensionsExt:
                    {
                        Trace.WriteLine("FBO: Attachments are of different size. All attachments must have the same width and height.");
                        break;
                    }
                case FramebufferErrorCode.FramebufferIncompleteFormatsExt:
                    {
                        Trace.WriteLine("FBO: The color attachments have different format. All color attachments must have the same format.");
                        break;
                    }
                case FramebufferErrorCode.FramebufferIncompleteDrawBufferExt:
                    {
                        Trace.WriteLine("FBO: An attachment point referenced by GL.DrawBuffers() doesn’t have an attachment.");
                        break;
                    }
                case FramebufferErrorCode.FramebufferIncompleteReadBufferExt:
                    {
                        Trace.WriteLine("FBO: The attachment point referenced by GL.ReadBuffers() doesn’t have an attachment.");
                        break;
                    }
                case FramebufferErrorCode.FramebufferUnsupportedExt:
                    {
                        Trace.WriteLine("FBO: This particular FBO configuration is not supported by the implementation.");
                        break;
                    }
                default:
                    {
                        Trace.WriteLine("FBO: Status unknown. (yes, this is really bad.)");
                        break;
                    }
            }
            return false;
        }

        public static int CreateShaderProgramFiles(string vsFile, string psFile)
        {
            String fragSource = (psFile != null) ? File.ReadAllText(psFile) : null;
            String vertSource = (vsFile != null) ? File.ReadAllText(vsFile) : null;
            return CreateShaderProgramStrings(vertSource, fragSource);
        }

        public static int CreateShaderProgramStrings(string vertSource, string fragSource)
        {
            // create a shader object.
            int shader = GL.CreateProgram();

            int vert = 0, frag = 0;
            // create shader objects for all three types.
            if (vertSource != null)
                vert = GL.CreateShader(ShaderType.VertexShader);
            if (fragSource != null)
                frag = GL.CreateShader(ShaderType.FragmentShader);

            // compile shaders.
            if (vert != 0)
            {
                CompileShader(vert, vertSource);
                GL.AttachShader(shader, vert);
            }
            if (frag != 0)
            {
                CompileShader(frag, fragSource);
                GL.AttachShader(shader, frag);
            }

            GL.LinkProgram(shader);

            // output link info log.
            string info;
            GL.GetProgramInfoLog(shader, out info);
            Debug.WriteLine(info);

            //// Set the input type of the primitives we are going to feed the geometry shader, this should be the same as
            //// the primitive type given to GL.Begin. If the types do not match a GL error will occur (todo: verify GL_INVALID_ENUM, on glBegin)
            //GL.ProgramParameter(shader, AssemblyProgramParameterArb.GeometryInputType, (int)All.Lines);
            //// Set the output type of the geometry shader. Because we input Lines we will output LineStrip(s).
            //GL.ProgramParameter(shader, AssemblyProgramParameterArb.GeometryOutputType, (int)All.LineStrip);

            //// We must tell the shader program how much vertices the geometry shader will output (at most).
            //// The simple way is to query the maximum and use that.
            //int tmp;
            //// Get the maximum amount of vertices into tmp.
            //GL.GetInteger((GetPName)ExtGeometryShader4.MaxGeometryOutputVerticesExt, out tmp);
            //// And feed amount that to the shader program. (0x0400 on a HD3850, with catalyst 9.8)
            //GL.ProgramParameter(shader, AssemblyProgramParameterArb.GeometryVerticesOut, tmp);

            // Clean up resources. Note the program object is not deleted.
            if (vert != 0)
                GL.DeleteShader(vert);
            if (frag != 0)
                GL.DeleteShader(frag);

            return shader;
        }

        public static void CompileShader(int shader, string source)
        {
            GL.ShaderSource(shader, source);
            GL.CompileShader(shader);

            string info;
            GL.GetShaderInfoLog(shader, out info);
            Debug.WriteLine(info);

            int compileResult;
            GL.GetShader(shader, ShaderParameter.CompileStatus, out compileResult);
            if (compileResult != 1)
            {
                Debug.WriteLine("Compile Error!");
                Debug.WriteLine(source);
            }
        }

        private static Bitmap BitmapDepthBuffer(float[] pixels, int width, int height)
        {
            Bitmap bmp = new Bitmap(width, height);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float depth = Math.Max(pixels[y * width + x] - 0.90f, 0.0f) / 0.10f;
                    int color = (int)(255.0f * depth);
                    bmp.SetPixel(x, (height - 1) - y, Color.FromArgb(color, color, color));
                }
            }

            return bmp;
        }

        public static Bitmap BitmapColorBuffer(int width, int height)
        {
            if (GraphicsContext.CurrentContext == null)
                throw new GraphicsContextMissingException();

            Bitmap bmp = new Bitmap(width, height);

            System.Drawing.Imaging.BitmapData data =
                bmp.LockBits(
                    new Rectangle(0, 0, width, height), 
                    System.Drawing.Imaging.ImageLockMode.WriteOnly,
                    System.Drawing.Imaging.PixelFormat.Format24bppRgb);

            GL.ReadPixels(0, 0, width, height, PixelFormat.Bgr, PixelType.UnsignedByte, data.Scan0);
            bmp.UnlockBits(data);

            bmp.RotateFlip(RotateFlipType.RotateNoneFlipY);
            return bmp;
        }
    }
}
