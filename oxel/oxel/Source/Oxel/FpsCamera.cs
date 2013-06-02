using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenTK;
using System.Drawing;

namespace Oxel
{
    public class FpsCamera
    {
        public float FastSpeed = 1000.0f;
        public float Speed = 10.0f;
        public float NearPlane = 1.0f;
        public float FarPlane = 10000.0f;
        public float NearFarRatio = 0.0001f;

        public float cameraYaw = 0;
        public float cameraPitch = 0;
        public Vector3 cameraPosition = Vector3.Zero;

        Point lastMousePosition = Point.Empty;

        bool cameraRotating = false;

        public void OnMouseUp(GLControl sender, System.Windows.Forms.MouseEventArgs e)
        {
            cameraRotating = false;
        }

        public void OnMouseDown(GLControl sender, System.Windows.Forms.MouseEventArgs e)
        {
            lastMousePosition = e.Location;
            cameraRotating = true;
            sender.Capture = true;
        }

        public void OnMouseWheel(GLControl sender, System.Windows.Forms.MouseEventArgs e)
        {
            float speed = (User32.IsKeyPushedDown(System.Windows.Forms.Keys.ShiftKey) ? 3.0f : 1.0f);
            TranslateCameraRelative(-Vector3.UnitZ * e.Delta * speed);
        }

        public void OnMouseMove(GLControl sender, System.Windows.Forms.MouseEventArgs e)
        {
            if (cameraRotating)
            {
                cameraYaw += (e.X - lastMousePosition.X) / 130.0f;
                cameraPitch += (e.Y - lastMousePosition.Y) / 130.0f;
                cameraPitch = MathEx.Clamp(cameraPitch, (float)-Math.PI / 2.0f + 0.01f, (float)Math.PI / 2.0f - 0.01f);
                lastMousePosition.X = e.X;
                lastMousePosition.Y = e.Y;
            }
        }

        public void OnUpdateFrame(FrameEventArgs e)
        {
            Matrix4 invTransformation = Matrix4.Mult(
                                            Matrix4.CreateRotationX(-cameraPitch),
                                            Matrix4.CreateRotationY(-cameraYaw));

            Vector3 source = Vector3.Zero;
            if (User32.IsKeyPushedDown(System.Windows.Forms.Keys.W))
                source = Vector3.Add(source, -Vector3.UnitZ);
            if (User32.IsKeyPushedDown(System.Windows.Forms.Keys.S))
                source = Vector3.Add(source, Vector3.UnitZ);
            if (User32.IsKeyPushedDown(System.Windows.Forms.Keys.A))
                source = Vector3.Add(source, -Vector3.UnitX);
            if (User32.IsKeyPushedDown(System.Windows.Forms.Keys.D))
                source = Vector3.Add(source, Vector3.UnitX);

            if (User32.IsKeyPushedDown(System.Windows.Forms.Keys.ShiftKey))
                Vector3.Multiply(ref source, (float)(FastSpeed * e.Time), out source);
            else
                Vector3.Multiply(ref source, (float)(Speed * e.Time), out source);

            Vector3 result;
            Vector3.TransformPosition(ref source, ref invTransformation, out result);
            cameraPosition = Vector3.Add(cameraPosition, result);
        }

        public void TranslateCameraRelative(Vector3 offset)
        {
            Matrix4 invTransformation = Matrix4.Mult(
                                Matrix4.CreateRotationX(-cameraPitch),
                                Matrix4.CreateRotationY(-cameraYaw));

            Vector3 source = offset;

            Vector3 result;
            Vector3.TransformPosition(ref source, ref invTransformation, out result);
            cameraPosition = Vector3.Add(cameraPosition, result);
        }
    }
}
