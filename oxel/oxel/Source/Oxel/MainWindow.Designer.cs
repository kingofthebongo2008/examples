namespace Oxel
{
    partial class MainWindow
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainWindow));
            this.m_menu = new System.Windows.Forms.MenuStrip();
            this.fileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.openToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.saveToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem2 = new System.Windows.Forms.ToolStripSeparator();
            this.exitToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.viewToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.m_showInternalVoxelMenu = new System.Windows.Forms.ToolStripMenuItem();
            this.m_showExternalVoxelMenu = new System.Windows.Forms.ToolStripMenuItem();
            this.m_showUnknownVoxelMenu = new System.Windows.Forms.ToolStripMenuItem();
            this.m_showIntersectingVoxelMenu = new System.Windows.Forms.ToolStripMenuItem();
            this.m_showIntersectingBoundsVoxelMenu = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem1 = new System.Windows.Forms.ToolStripSeparator();
            this.occluderLinesToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.meshLinesToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem4 = new System.Windows.Forms.ToolStripSeparator();
            this.m_debugLinesToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem3 = new System.Windows.Forms.ToolStripSeparator();
            this.m_showOriginalMeshMenu = new System.Windows.Forms.ToolStripMenuItem();
            this.m_showOccluderMeshMenu = new System.Windows.Forms.ToolStripMenuItem();
            this.buildToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.voxelizeToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.helpToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.aboutToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.m_statusLabel = new System.Windows.Forms.ToolStripStatusLabel();
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.m_propertyGrid = new System.Windows.Forms.PropertyGrid();
            this.alphaBlendingModeToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem5 = new System.Windows.Forms.ToolStripSeparator();
            this.m_menu.SuspendLayout();
            this.statusStrip1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            this.SuspendLayout();
            // 
            // m_menu
            // 
            this.m_menu.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.fileToolStripMenuItem,
            this.viewToolStripMenuItem,
            this.buildToolStripMenuItem,
            this.helpToolStripMenuItem});
            this.m_menu.Location = new System.Drawing.Point(0, 0);
            this.m_menu.Name = "m_menu";
            this.m_menu.Size = new System.Drawing.Size(1008, 24);
            this.m_menu.TabIndex = 0;
            this.m_menu.Text = "menuStrip1";
            // 
            // fileToolStripMenuItem
            // 
            this.fileToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.openToolStripMenuItem,
            this.saveToolStripMenuItem,
            this.toolStripMenuItem2,
            this.exitToolStripMenuItem});
            this.fileToolStripMenuItem.Name = "fileToolStripMenuItem";
            this.fileToolStripMenuItem.Size = new System.Drawing.Size(37, 20);
            this.fileToolStripMenuItem.Text = "File";
            // 
            // openToolStripMenuItem
            // 
            this.openToolStripMenuItem.Name = "openToolStripMenuItem";
            this.openToolStripMenuItem.Size = new System.Drawing.Size(134, 22);
            this.openToolStripMenuItem.Text = "Open";
            this.openToolStripMenuItem.Click += new System.EventHandler(this.openToolStripMenuItem_Click);
            // 
            // saveToolStripMenuItem
            // 
            this.saveToolStripMenuItem.Name = "saveToolStripMenuItem";
            this.saveToolStripMenuItem.Size = new System.Drawing.Size(134, 22);
            this.saveToolStripMenuItem.Text = "Save";
            this.saveToolStripMenuItem.Click += new System.EventHandler(this.saveToolStripMenuItem_Click);
            // 
            // toolStripMenuItem2
            // 
            this.toolStripMenuItem2.Name = "toolStripMenuItem2";
            this.toolStripMenuItem2.Size = new System.Drawing.Size(131, 6);
            // 
            // exitToolStripMenuItem
            // 
            this.exitToolStripMenuItem.Name = "exitToolStripMenuItem";
            this.exitToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Alt | System.Windows.Forms.Keys.F4)));
            this.exitToolStripMenuItem.Size = new System.Drawing.Size(134, 22);
            this.exitToolStripMenuItem.Text = "Exit";
            this.exitToolStripMenuItem.Click += new System.EventHandler(this.exitToolStripMenuItem_Click);
            // 
            // viewToolStripMenuItem
            // 
            this.viewToolStripMenuItem.CheckOnClick = true;
            this.viewToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.m_showInternalVoxelMenu,
            this.m_showExternalVoxelMenu,
            this.m_showUnknownVoxelMenu,
            this.m_showIntersectingVoxelMenu,
            this.m_showIntersectingBoundsVoxelMenu,
            this.toolStripMenuItem1,
            this.occluderLinesToolStripMenuItem,
            this.meshLinesToolStripMenuItem,
            this.toolStripMenuItem4,
            this.m_debugLinesToolStripMenuItem,
            this.toolStripMenuItem3,
            this.m_showOriginalMeshMenu,
            this.m_showOccluderMeshMenu,
            this.toolStripMenuItem5,
            this.alphaBlendingModeToolStripMenuItem});
            this.viewToolStripMenuItem.Name = "viewToolStripMenuItem";
            this.viewToolStripMenuItem.Size = new System.Drawing.Size(44, 20);
            this.viewToolStripMenuItem.Text = "View";
            this.viewToolStripMenuItem.DropDownOpening += new System.EventHandler(this.viewToolStripMenuItem_DropDownOpening);
            // 
            // m_showInternalVoxelMenu
            // 
            this.m_showInternalVoxelMenu.Name = "m_showInternalVoxelMenu";
            this.m_showInternalVoxelMenu.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Alt | System.Windows.Forms.Keys.I)));
            this.m_showInternalVoxelMenu.Size = new System.Drawing.Size(252, 22);
            this.m_showInternalVoxelMenu.Text = "Internal Voxels";
            this.m_showInternalVoxelMenu.Click += new System.EventHandler(this.ShowInternalVoxels);
            // 
            // m_showExternalVoxelMenu
            // 
            this.m_showExternalVoxelMenu.CheckOnClick = true;
            this.m_showExternalVoxelMenu.Name = "m_showExternalVoxelMenu";
            this.m_showExternalVoxelMenu.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Alt | System.Windows.Forms.Keys.O)));
            this.m_showExternalVoxelMenu.Size = new System.Drawing.Size(252, 22);
            this.m_showExternalVoxelMenu.Text = "External Voxels";
            this.m_showExternalVoxelMenu.Click += new System.EventHandler(this.ShowExternalVoxels);
            // 
            // m_showUnknownVoxelMenu
            // 
            this.m_showUnknownVoxelMenu.Name = "m_showUnknownVoxelMenu";
            this.m_showUnknownVoxelMenu.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Alt | System.Windows.Forms.Keys.U)));
            this.m_showUnknownVoxelMenu.Size = new System.Drawing.Size(252, 22);
            this.m_showUnknownVoxelMenu.Text = "Unknown Voxels";
            this.m_showUnknownVoxelMenu.Click += new System.EventHandler(this.ShowUnknownVoxels);
            // 
            // m_showIntersectingVoxelMenu
            // 
            this.m_showIntersectingVoxelMenu.CheckOnClick = true;
            this.m_showIntersectingVoxelMenu.Name = "m_showIntersectingVoxelMenu";
            this.m_showIntersectingVoxelMenu.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Alt | System.Windows.Forms.Keys.N)));
            this.m_showIntersectingVoxelMenu.Size = new System.Drawing.Size(252, 22);
            this.m_showIntersectingVoxelMenu.Text = "Intersecting Voxels";
            this.m_showIntersectingVoxelMenu.Click += new System.EventHandler(this.ShowIntersectingVoxels);
            // 
            // m_showIntersectingBoundsVoxelMenu
            // 
            this.m_showIntersectingBoundsVoxelMenu.CheckOnClick = true;
            this.m_showIntersectingBoundsVoxelMenu.Name = "m_showIntersectingBoundsVoxelMenu";
            this.m_showIntersectingBoundsVoxelMenu.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Alt | System.Windows.Forms.Keys.B)));
            this.m_showIntersectingBoundsVoxelMenu.Size = new System.Drawing.Size(252, 22);
            this.m_showIntersectingBoundsVoxelMenu.Text = "Intersecting Bounds Voxels";
            this.m_showIntersectingBoundsVoxelMenu.Click += new System.EventHandler(this.ShowIntersectingBoundsVoxels);
            // 
            // toolStripMenuItem1
            // 
            this.toolStripMenuItem1.Name = "toolStripMenuItem1";
            this.toolStripMenuItem1.Size = new System.Drawing.Size(249, 6);
            // 
            // occluderLinesToolStripMenuItem
            // 
            this.occluderLinesToolStripMenuItem.Name = "occluderLinesToolStripMenuItem";
            this.occluderLinesToolStripMenuItem.Size = new System.Drawing.Size(252, 22);
            this.occluderLinesToolStripMenuItem.Text = "Occluder Lines";
            this.occluderLinesToolStripMenuItem.Click += new System.EventHandler(this.occluderLinesToolStripMenuItem_Click);
            // 
            // meshLinesToolStripMenuItem
            // 
            this.meshLinesToolStripMenuItem.Name = "meshLinesToolStripMenuItem";
            this.meshLinesToolStripMenuItem.Size = new System.Drawing.Size(252, 22);
            this.meshLinesToolStripMenuItem.Text = "Mesh Lines";
            this.meshLinesToolStripMenuItem.Click += new System.EventHandler(this.meshLinesToolStripMenuItem_Click);
            // 
            // toolStripMenuItem4
            // 
            this.toolStripMenuItem4.Name = "toolStripMenuItem4";
            this.toolStripMenuItem4.Size = new System.Drawing.Size(249, 6);
            // 
            // m_debugLinesToolStripMenuItem
            // 
            this.m_debugLinesToolStripMenuItem.Name = "m_debugLinesToolStripMenuItem";
            this.m_debugLinesToolStripMenuItem.Size = new System.Drawing.Size(252, 22);
            this.m_debugLinesToolStripMenuItem.Text = "Debug Lines";
            this.m_debugLinesToolStripMenuItem.Click += new System.EventHandler(this.ShowDebugLines);
            // 
            // toolStripMenuItem3
            // 
            this.toolStripMenuItem3.Name = "toolStripMenuItem3";
            this.toolStripMenuItem3.Size = new System.Drawing.Size(249, 6);
            // 
            // m_showOriginalMeshMenu
            // 
            this.m_showOriginalMeshMenu.Name = "m_showOriginalMeshMenu";
            this.m_showOriginalMeshMenu.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Alt | System.Windows.Forms.Keys.M)));
            this.m_showOriginalMeshMenu.Size = new System.Drawing.Size(252, 22);
            this.m_showOriginalMeshMenu.Text = "Original Mesh";
            this.m_showOriginalMeshMenu.Click += new System.EventHandler(this.ShowOriginalMesh);
            // 
            // m_showOccluderMeshMenu
            // 
            this.m_showOccluderMeshMenu.Name = "m_showOccluderMeshMenu";
            this.m_showOccluderMeshMenu.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Alt | System.Windows.Forms.Keys.C)));
            this.m_showOccluderMeshMenu.Size = new System.Drawing.Size(252, 22);
            this.m_showOccluderMeshMenu.Text = "Occluder Mesh";
            this.m_showOccluderMeshMenu.Click += new System.EventHandler(this.ShowOccluderMesh);
            // 
            // buildToolStripMenuItem
            // 
            this.buildToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.voxelizeToolStripMenuItem});
            this.buildToolStripMenuItem.Name = "buildToolStripMenuItem";
            this.buildToolStripMenuItem.Size = new System.Drawing.Size(46, 20);
            this.buildToolStripMenuItem.Text = "Build";
            // 
            // voxelizeToolStripMenuItem
            // 
            this.voxelizeToolStripMenuItem.Name = "voxelizeToolStripMenuItem";
            this.voxelizeToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Alt | System.Windows.Forms.Keys.V)));
            this.voxelizeToolStripMenuItem.Size = new System.Drawing.Size(189, 22);
            this.voxelizeToolStripMenuItem.Text = "Build Occluder";
            this.voxelizeToolStripMenuItem.Click += new System.EventHandler(this.voxelizeToolStripMenuItem_Click);
            // 
            // helpToolStripMenuItem
            // 
            this.helpToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.aboutToolStripMenuItem});
            this.helpToolStripMenuItem.Name = "helpToolStripMenuItem";
            this.helpToolStripMenuItem.Size = new System.Drawing.Size(44, 20);
            this.helpToolStripMenuItem.Text = "Help";
            // 
            // aboutToolStripMenuItem
            // 
            this.aboutToolStripMenuItem.Name = "aboutToolStripMenuItem";
            this.aboutToolStripMenuItem.Size = new System.Drawing.Size(107, 22);
            this.aboutToolStripMenuItem.Text = "About";
            this.aboutToolStripMenuItem.Click += new System.EventHandler(this.aboutToolStripMenuItem_Click);
            // 
            // statusStrip1
            // 
            this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.m_statusLabel});
            this.statusStrip1.Location = new System.Drawing.Point(0, 540);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.Size = new System.Drawing.Size(1008, 22);
            this.statusStrip1.TabIndex = 1;
            this.statusStrip1.Text = "statusStrip1";
            // 
            // m_statusLabel
            // 
            this.m_statusLabel.Name = "m_statusLabel";
            this.m_statusLabel.Size = new System.Drawing.Size(39, 17);
            this.m_statusLabel.Text = "Ready";
            // 
            // splitContainer1
            // 
            this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer1.FixedPanel = System.Windows.Forms.FixedPanel.Panel2;
            this.splitContainer1.Location = new System.Drawing.Point(0, 24);
            this.splitContainer1.Name = "splitContainer1";
            // 
            // splitContainer1.Panel2
            // 
            this.splitContainer1.Panel2.Controls.Add(this.m_propertyGrid);
            this.splitContainer1.Size = new System.Drawing.Size(1008, 516);
            this.splitContainer1.SplitterDistance = 716;
            this.splitContainer1.TabIndex = 1;
            // 
            // m_propertyGrid
            // 
            this.m_propertyGrid.Dock = System.Windows.Forms.DockStyle.Fill;
            this.m_propertyGrid.Location = new System.Drawing.Point(0, 0);
            this.m_propertyGrid.Name = "m_propertyGrid";
            this.m_propertyGrid.Size = new System.Drawing.Size(288, 516);
            this.m_propertyGrid.TabIndex = 0;
            this.m_propertyGrid.ToolbarVisible = false;
            // 
            // alphaBlendingModeToolStripMenuItem
            // 
            this.alphaBlendingModeToolStripMenuItem.Name = "alphaBlendingModeToolStripMenuItem";
            this.alphaBlendingModeToolStripMenuItem.Size = new System.Drawing.Size(252, 22);
            this.alphaBlendingModeToolStripMenuItem.Text = "Alpha Blending Mode";
            this.alphaBlendingModeToolStripMenuItem.Click += new System.EventHandler(this.alphaBlendingModeToolStripMenuItem_Click);
            // 
            // toolStripMenuItem5
            // 
            this.toolStripMenuItem5.Name = "toolStripMenuItem5";
            this.toolStripMenuItem5.Size = new System.Drawing.Size(249, 6);
            // 
            // MainWindow
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1008, 562);
            this.Controls.Add(this.splitContainer1);
            this.Controls.Add(this.statusStrip1);
            this.Controls.Add(this.m_menu);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MainMenuStrip = this.m_menu;
            this.Name = "MainWindow";
            this.Text = "Oxel";
            this.m_menu.ResumeLayout(false);
            this.m_menu.PerformLayout();
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            this.splitContainer1.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.MenuStrip m_menu;
        private System.Windows.Forms.ToolStripMenuItem fileToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem openToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem exitToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem helpToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem aboutToolStripMenuItem;
        private System.Windows.Forms.StatusStrip statusStrip1;
        private System.Windows.Forms.ToolStripMenuItem viewToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem m_showInternalVoxelMenu;
        private System.Windows.Forms.ToolStripMenuItem m_showExternalVoxelMenu;
        private System.Windows.Forms.ToolStripMenuItem m_showUnknownVoxelMenu;
        private System.Windows.Forms.ToolStripMenuItem m_showOriginalMeshMenu;
        private System.Windows.Forms.ToolStripMenuItem buildToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem voxelizeToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem m_showIntersectingVoxelMenu;
        private System.Windows.Forms.ToolStripMenuItem m_showIntersectingBoundsVoxelMenu;
        private System.Windows.Forms.ToolStripMenuItem m_showOccluderMeshMenu;
        private System.Windows.Forms.SplitContainer splitContainer1;
        private System.Windows.Forms.PropertyGrid m_propertyGrid;
        private System.Windows.Forms.ToolStripMenuItem saveToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem2;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem1;
        private System.Windows.Forms.ToolStripStatusLabel m_statusLabel;
        private System.Windows.Forms.ToolStripMenuItem m_debugLinesToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem3;
        private System.Windows.Forms.ToolStripMenuItem occluderLinesToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem meshLinesToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem4;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem5;
        private System.Windows.Forms.ToolStripMenuItem alphaBlendingModeToolStripMenuItem;
    }
}