namespace Oxel
{
    partial class ProgressDialog
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(ProgressDialog));
            this.m_progressText = new System.Windows.Forms.TextBox();
            this.m_coverageProgress = new System.Windows.Forms.Label();
            this.m_progressBar = new Oxel.ProgressBarEx();
            this.SuspendLayout();
            // 
            // m_progressText
            // 
            this.m_progressText.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.m_progressText.Location = new System.Drawing.Point(12, 53);
            this.m_progressText.Multiline = true;
            this.m_progressText.Name = "m_progressText";
            this.m_progressText.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.m_progressText.Size = new System.Drawing.Size(518, 133);
            this.m_progressText.TabIndex = 1;
            // 
            // m_coverageProgress
            // 
            this.m_coverageProgress.AutoSize = true;
            this.m_coverageProgress.Location = new System.Drawing.Point(12, 37);
            this.m_coverageProgress.Name = "m_coverageProgress";
            this.m_coverageProgress.Size = new System.Drawing.Size(0, 13);
            this.m_coverageProgress.TabIndex = 2;
            // 
            // m_progressBar
            // 
            this.m_progressBar.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.m_progressBar.Location = new System.Drawing.Point(12, 12);
            this.m_progressBar.Margin = new System.Windows.Forms.Padding(6);
            this.m_progressBar.MarqueeAnimationSpeed = 0;
            this.m_progressBar.Name = "m_progressBar";
            this.m_progressBar.Size = new System.Drawing.Size(518, 19);
            this.m_progressBar.TabIndex = 0;
            this.m_progressBar.Value = 100;
            // 
            // ProgressDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(545, 201);
            this.Controls.Add(this.m_coverageProgress);
            this.Controls.Add(this.m_progressText);
            this.Controls.Add(this.m_progressBar);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "ProgressDialog";
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Occluder Generation Progress";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.ProgressDialog_FormClosing);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        ProgressBarEx m_progressBar;
        private System.Windows.Forms.TextBox m_progressText;
        private System.Windows.Forms.Label m_coverageProgress;
    }
}