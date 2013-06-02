using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Drawing;

namespace Oxel
{
    public partial class ProgressBarEx : ProgressBar
    {
        public ProgressBarEx()
        {
           // this.SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.ResizeRedraw, true);
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);

            int percent = (int)(((double)Value / (double)Maximum) * 100);
            e.Graphics.DrawString(percent.ToString() + "%",
                new Font("Arial", (float)8.25, FontStyle.Regular), Brushes.Black,
                new PointF(Width / 2 - 10, Height / 2 - 7));
        }
    }
}
