using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace Oxel
{
    public partial class ProgressDialog : Form
    {
        public bool CanClose;

        public ProgressDialog()
        {
            InitializeComponent();
        }

        public void UpdateProgress(VoxelizationProgress vp)
        {
            if (InvokeRequired)
            {
                Invoke(new Action<VoxelizationProgress>(UpdateProgress), vp);
            }
            else
            {
                m_progressBar.Value = (int)(vp.Progress * 100);
                if (m_progressBar.Value > 0)
                {
                    m_progressBar.Value -= 1;
                    m_progressBar.Value += 1;
                }
                m_coverageProgress.Text = 
                    String.Format("Volume Coverage     : {0,5:0.##}%", (100 * vp.VolumeCoverage)) + "    " +
                    String.Format("Silhouette Coverage : {0,5:0.##}%", (100 * vp.SilhouetteCoverage));

                if (!String.IsNullOrEmpty(vp.Status))
                    m_progressText.AppendText(vp.Status + "\r\n");
            }
        }

        private void ProgressDialog_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (!CanClose)
            {
                if (e.CloseReason == CloseReason.UserClosing)
                    e.Cancel = true;
            }
        }
    }
}
