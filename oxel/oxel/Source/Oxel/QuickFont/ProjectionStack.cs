using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using OpenTK.Graphics;


namespace QuickFont
{
    class ProjectionStack
    {

        struct Viewport
        {
            public int X, Y, Width, Height;
            public Viewport(int X, int Y, int Width, int Height) { this.X = X; this.Y = Y; this.Width = Width; this.Height = Height; }
        }

        // Used to save the current state in Begin() and restore it in End()
        static Stack<Matrix4> projection_stack = new Stack<Matrix4>();
        static Stack<Matrix4> modelview_stack = new Stack<Matrix4>();
        static Stack<Matrix4> texture_stack = new Stack<Matrix4>();
        static Stack<Viewport> viewport_stack = new Stack<Viewport>();

        // Used as temporary storage when saving / restoring the current state.
        static Viewport viewport = new Viewport();
        static Matrix4 matrix = new Matrix4();


        public static void Begin()
        {

            GraphicsContext.Assert();

            // Save the state of everything we are going to modify:
            // the current matrix mode, viewport state and the projection, modelview and texture matrices.
            // All these will be restored in the TextPrinter.End() method.
            int current_matrix;
            GL.GetInteger(GetPName.MatrixMode, out current_matrix);

            GL.GetInteger(GetPName.Viewport, out viewport.X);
            viewport_stack.Push(viewport);

            GL.GetFloat(GetPName.ProjectionMatrix, out matrix.Row0.X);
            projection_stack.Push(matrix);
            GL.GetFloat(GetPName.ModelviewMatrix, out matrix.Row0.X);
            modelview_stack.Push(matrix);
            GL.GetFloat(GetPName.TextureMatrix, out matrix.Row0.X);
            texture_stack.Push(matrix);

            // Prepare to draw text. We want pixel perfect precision, so we setup a 2D mode,
            // with size equal to the window (in pixels). 
            // While we could also render text in 3D mode, it would be very hard to get
            // pixel-perfect precision.
            GL.MatrixMode(MatrixMode.Projection);
            GL.LoadIdentity();
            GL.Ortho(viewport.X, viewport.Width, viewport.Height, viewport.Y, -1.0, 1.0);

            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadIdentity();

            GL.MatrixMode(MatrixMode.Texture);
            GL.LoadIdentity();

            GL.MatrixMode((MatrixMode)current_matrix);
        }



        public static void End()
        {

            GraphicsContext.Assert();

            int current_matrix;
            GL.GetInteger(GetPName.MatrixMode, out current_matrix);

            viewport = viewport_stack.Pop();
            GL.Viewport(viewport.X, viewport.Y, viewport.Width, viewport.Height);

            GL.MatrixMode(MatrixMode.Texture);
            matrix = texture_stack.Pop();
            GL.LoadMatrix(ref matrix);

            GL.MatrixMode(MatrixMode.Modelview);
            matrix = modelview_stack.Pop();
            GL.LoadMatrix(ref matrix);

            GL.MatrixMode(MatrixMode.Projection);
            matrix = projection_stack.Pop();
            GL.LoadMatrix(ref matrix);

            GL.MatrixMode((MatrixMode)current_matrix);
        }




    }
}
