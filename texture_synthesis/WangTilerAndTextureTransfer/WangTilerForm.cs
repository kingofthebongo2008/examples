using System;
using System.IO;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;

namespace WangTileCreator
{

	/// <summary>
	/// Form that controls the Image Quilting and Wang Tiling application.
	/// Robert Burke, rob@mle.media.mit.edu
	/// 10 Aug 2003
	/// 
	/// Caveat: this is all "Weekend Project" code.  So it's not necessarily the prettiest thing ever.
	/// Please accept my apologies in advance.
	/// </summary>
	public class WangTileForm : System.Windows.Forms.Form
	{

		public Bitmap StartingTextureMap = null;
		public bool StartingTextureMapValid { get { return StartingTextureMap != null; } }

		public Bitmap QuiltedTextureMap = null;
		public bool QuiltedTextureMapValid { get { return QuiltedTextureMap != null; } }

		public WangTile[] WangTiles = null;
		public bool WangTilesValid { get { return WangTiles != null; } }



		private System.Windows.Forms.MainMenu mainMenu1;
		private System.Windows.Forms.MenuItem menuItem1;
		private System.Windows.Forms.MenuItem menuItem2;
		private System.Windows.Forms.MenuItem menuItem3;
		private System.Windows.Forms.MenuItem menuItem4;
		private System.Windows.Forms.OpenFileDialog openFileDialog1;
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.Label labelWidthHeight;
		private System.Windows.Forms.PictureBox pictureBoxSourceBitmap;
		private System.Windows.Forms.GroupBox groupBox2;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.StatusBar statusBar1;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.Button button1;
		private System.Windows.Forms.Button button2;
		private System.Windows.Forms.NumericUpDown numericUpDownQuiltingWidth;
		private System.Windows.Forms.NumericUpDown numericUpDownQuiltingHeight;
		private System.Windows.Forms.NumericUpDown numericUpDownQuiltingBlockSize;
		private System.Windows.Forms.NumericUpDown numericUpDownQuiltingBlockOverlap;
		private System.Windows.Forms.PictureBox pictureBoxQuiltedBitmap;
		private System.Windows.Forms.TextBox textBoxDebugSpew;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.NumericUpDown numericUpDownNumCandidateQuiltBlocks;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.MenuItem menuItem5;
		private System.Windows.Forms.SaveFileDialog saveFileDialog1;
		private System.Windows.Forms.NumericUpDown numericUpDownSelectFromNBestCandidateQuiltErrorTolerance;
		private System.Windows.Forms.GroupBox groupBox3;
		private System.Windows.Forms.Button button3;
		private System.Windows.Forms.Label label8;
		private System.Windows.Forms.NumericUpDown numericUpDownMaxWangMatchError;
		private System.Windows.Forms.Label label9;
		private System.Windows.Forms.NumericUpDown numericUpDownMaxWangMatchAttempts;
		private System.Windows.Forms.Label label10;
		private System.Windows.Forms.NumericUpDown numericUpDownWangTileSize;
		private System.Windows.Forms.Button button4;
		private System.Windows.Forms.Button button5;
		private System.Windows.Forms.TextBox textBoxWangSpecs;
		private System.Windows.Forms.Label label11;
		private System.Windows.Forms.Button button6;
		private System.Windows.Forms.CheckBox checkBoxOnePixelOverlapBetweenTiles;
		private System.Windows.Forms.MenuItem menuItem7;
		private System.Windows.Forms.MenuItem menuItem8;
		private System.Windows.Forms.MenuItem menuItem9;
		private System.Windows.Forms.MenuItem menuItem10;
		private System.Windows.Forms.MenuItem menuItem11;
		private System.Windows.Forms.MenuItem menuItem12;
		private System.Windows.Forms.MenuItem menuItem6;
		private System.Windows.Forms.MenuItem menuItem13;
		private System.Windows.Forms.MenuItem menuItem14;
		private System.Windows.Forms.Button button7;
		//private System.ComponentModel.IContainer components;

		public WangTileForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			Util.SD = new Util.SpewDelegate(OnSpew);
			Util.RDD = new Util.RefreshDisplayDelegate(UpdateDisplay);
			Util.SBV = new Util.SetBitmapForViewingDelegate(SetBitmapForViewingWithWangTilerForm);
		}
		public void OnSpew(string s)
		{
			this.textBoxDebugSpew.Text = s + "\r\n"+ this.textBoxDebugSpew.Text;
			Application.DoEvents();
		}
		public void SetBitmapForViewingWithWangTilerForm(Bitmap b, int viewWindow)
		{
			if (viewWindow == 0)
			{
				this.pictureBoxQuiltedBitmap.Image = b;
			}

		}

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		protected override void Dispose( bool disposing )
		{
			base.Dispose( disposing );
		}

		#region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.mainMenu1 = new System.Windows.Forms.MainMenu();
			this.menuItem1 = new System.Windows.Forms.MenuItem();
			this.menuItem2 = new System.Windows.Forms.MenuItem();
			this.menuItem3 = new System.Windows.Forms.MenuItem();
			this.menuItem4 = new System.Windows.Forms.MenuItem();
			this.menuItem7 = new System.Windows.Forms.MenuItem();
			this.menuItem11 = new System.Windows.Forms.MenuItem();
			this.menuItem12 = new System.Windows.Forms.MenuItem();
			this.menuItem9 = new System.Windows.Forms.MenuItem();
			this.menuItem5 = new System.Windows.Forms.MenuItem();
			this.menuItem8 = new System.Windows.Forms.MenuItem();
			this.menuItem10 = new System.Windows.Forms.MenuItem();
			this.menuItem14 = new System.Windows.Forms.MenuItem();
			this.menuItem6 = new System.Windows.Forms.MenuItem();
			this.menuItem13 = new System.Windows.Forms.MenuItem();
			this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
			this.pictureBoxSourceBitmap = new System.Windows.Forms.PictureBox();
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.button7 = new System.Windows.Forms.Button();
			this.labelWidthHeight = new System.Windows.Forms.Label();
			this.groupBox2 = new System.Windows.Forms.GroupBox();
			this.button3 = new System.Windows.Forms.Button();
			this.label7 = new System.Windows.Forms.Label();
			this.numericUpDownSelectFromNBestCandidateQuiltErrorTolerance = new System.Windows.Forms.NumericUpDown();
			this.label6 = new System.Windows.Forms.Label();
			this.numericUpDownNumCandidateQuiltBlocks = new System.Windows.Forms.NumericUpDown();
			this.button2 = new System.Windows.Forms.Button();
			this.button1 = new System.Windows.Forms.Button();
			this.label5 = new System.Windows.Forms.Label();
			this.label4 = new System.Windows.Forms.Label();
			this.numericUpDownQuiltingBlockOverlap = new System.Windows.Forms.NumericUpDown();
			this.label3 = new System.Windows.Forms.Label();
			this.numericUpDownQuiltingBlockSize = new System.Windows.Forms.NumericUpDown();
			this.label2 = new System.Windows.Forms.Label();
			this.label1 = new System.Windows.Forms.Label();
			this.numericUpDownQuiltingHeight = new System.Windows.Forms.NumericUpDown();
			this.numericUpDownQuiltingWidth = new System.Windows.Forms.NumericUpDown();
			this.pictureBoxQuiltedBitmap = new System.Windows.Forms.PictureBox();
			this.statusBar1 = new System.Windows.Forms.StatusBar();
			this.textBoxDebugSpew = new System.Windows.Forms.TextBox();
			this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
			this.groupBox3 = new System.Windows.Forms.GroupBox();
			this.checkBoxOnePixelOverlapBetweenTiles = new System.Windows.Forms.CheckBox();
			this.button6 = new System.Windows.Forms.Button();
			this.label11 = new System.Windows.Forms.Label();
			this.textBoxWangSpecs = new System.Windows.Forms.TextBox();
			this.button5 = new System.Windows.Forms.Button();
			this.button4 = new System.Windows.Forms.Button();
			this.label8 = new System.Windows.Forms.Label();
			this.numericUpDownMaxWangMatchError = new System.Windows.Forms.NumericUpDown();
			this.label9 = new System.Windows.Forms.Label();
			this.numericUpDownMaxWangMatchAttempts = new System.Windows.Forms.NumericUpDown();
			this.label10 = new System.Windows.Forms.Label();
			this.numericUpDownWangTileSize = new System.Windows.Forms.NumericUpDown();
			this.groupBox1.SuspendLayout();
			this.groupBox2.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownSelectFromNBestCandidateQuiltErrorTolerance)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownNumCandidateQuiltBlocks)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownQuiltingBlockOverlap)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownQuiltingBlockSize)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownQuiltingHeight)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownQuiltingWidth)).BeginInit();
			this.groupBox3.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownMaxWangMatchError)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownMaxWangMatchAttempts)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownWangTileSize)).BeginInit();
			this.SuspendLayout();
			// 
			// mainMenu1
			// 
			this.mainMenu1.MenuItems.AddRange(new System.Windows.Forms.MenuItem[] {
																					  this.menuItem1,
																					  this.menuItem7,
																					  this.menuItem8});
			// 
			// menuItem1
			// 
			this.menuItem1.Index = 0;
			this.menuItem1.MenuItems.AddRange(new System.Windows.Forms.MenuItem[] {
																					  this.menuItem2,
																					  this.menuItem3,
																					  this.menuItem4});
			this.menuItem1.Text = "&File";
			// 
			// menuItem2
			// 
			this.menuItem2.Index = 0;
			this.menuItem2.Text = "&Open Source Texture...";
			this.menuItem2.Click += new System.EventHandler(this.menuItem2_Click);
			// 
			// menuItem3
			// 
			this.menuItem3.Index = 1;
			this.menuItem3.Text = "-";
			// 
			// menuItem4
			// 
			this.menuItem4.Index = 2;
			this.menuItem4.Text = "E&xit";
			this.menuItem4.Click += new System.EventHandler(this.menuItem4_Click);
			// 
			// menuItem7
			// 
			this.menuItem7.Index = 1;
			this.menuItem7.MenuItems.AddRange(new System.Windows.Forms.MenuItem[] {
																					  this.menuItem11,
																					  this.menuItem12,
																					  this.menuItem9,
																					  this.menuItem5});
			this.menuItem7.Text = "&Quilting";
			// 
			// menuItem11
			// 
			this.menuItem11.Index = 0;
			this.menuItem11.Text = "&Generate Quilt from Source Bitmap";
			this.menuItem11.Click += new System.EventHandler(this.button1_Click);
			// 
			// menuItem12
			// 
			this.menuItem12.Index = 1;
			this.menuItem12.Text = "-";
			// 
			// menuItem9
			// 
			this.menuItem9.Index = 2;
			this.menuItem9.Text = "&Load Quilt Bitmap...";
			this.menuItem9.Click += new System.EventHandler(this.button3_Click);
			// 
			// menuItem5
			// 
			this.menuItem5.Index = 3;
			this.menuItem5.Text = "&Save Quilt Bitmap...";
			this.menuItem5.Click += new System.EventHandler(this.menuItem5_Click);
			// 
			// menuItem8
			// 
			this.menuItem8.Index = 2;
			this.menuItem8.MenuItems.AddRange(new System.Windows.Forms.MenuItem[] {
																					  this.menuItem10,
																					  this.menuItem14,
																					  this.menuItem6,
																					  this.menuItem13});
			this.menuItem8.Text = "Wang Tiling";
			// 
			// menuItem10
			// 
			this.menuItem10.Index = 0;
			this.menuItem10.Text = "&Generate Wang Tiles";
			this.menuItem10.Click += new System.EventHandler(this.button4_Click);
			// 
			// menuItem14
			// 
			this.menuItem14.Index = 1;
			this.menuItem14.Text = "-";
			// 
			// menuItem6
			// 
			this.menuItem6.Index = 2;
			this.menuItem6.Text = "&Save Individual Tiles...";
			this.menuItem6.Click += new System.EventHandler(this.button5_Click);
			// 
			// menuItem13
			// 
			this.menuItem13.Index = 3;
			this.menuItem13.Text = "Sa&ve MegaTile...";
			this.menuItem13.Click += new System.EventHandler(this.button6_Click);
			// 
			// openFileDialog1
			// 
			this.openFileDialog1.Title = "Select a bitmap image to use as a texture map";
			// 
			// pictureBoxSourceBitmap
			// 
			this.pictureBoxSourceBitmap.Location = new System.Drawing.Point(16, 24);
			this.pictureBoxSourceBitmap.Name = "pictureBoxSourceBitmap";
			this.pictureBoxSourceBitmap.Size = new System.Drawing.Size(256, 256);
			this.pictureBoxSourceBitmap.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
			this.pictureBoxSourceBitmap.TabIndex = 0;
			this.pictureBoxSourceBitmap.TabStop = false;
			// 
			// groupBox1
			// 
			this.groupBox1.Controls.Add(this.button7);
			this.groupBox1.Controls.Add(this.labelWidthHeight);
			this.groupBox1.Controls.Add(this.pictureBoxSourceBitmap);
			this.groupBox1.Location = new System.Drawing.Point(24, 16);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.Size = new System.Drawing.Size(336, 344);
			this.groupBox1.TabIndex = 1;
			this.groupBox1.TabStop = false;
			this.groupBox1.Text = "Source bitmap";
			// 
			// button7
			// 
			this.button7.Location = new System.Drawing.Point(16, 312);
			this.button7.Name = "button7";
			this.button7.Size = new System.Drawing.Size(120, 23);
			this.button7.TabIndex = 12;
			this.button7.Text = "Load Source Bitmap";
			this.button7.Click += new System.EventHandler(this.menuItem2_Click);
			// 
			// labelWidthHeight
			// 
			this.labelWidthHeight.Location = new System.Drawing.Point(16, 288);
			this.labelWidthHeight.Name = "labelWidthHeight";
			this.labelWidthHeight.Size = new System.Drawing.Size(232, 23);
			this.labelWidthHeight.TabIndex = 1;
			this.labelWidthHeight.Text = "Load a valid source bitmap to begin.";
			// 
			// groupBox2
			// 
			this.groupBox2.Controls.Add(this.button3);
			this.groupBox2.Controls.Add(this.label7);
			this.groupBox2.Controls.Add(this.numericUpDownSelectFromNBestCandidateQuiltErrorTolerance);
			this.groupBox2.Controls.Add(this.label6);
			this.groupBox2.Controls.Add(this.numericUpDownNumCandidateQuiltBlocks);
			this.groupBox2.Controls.Add(this.button2);
			this.groupBox2.Controls.Add(this.button1);
			this.groupBox2.Controls.Add(this.label5);
			this.groupBox2.Controls.Add(this.label4);
			this.groupBox2.Controls.Add(this.numericUpDownQuiltingBlockOverlap);
			this.groupBox2.Controls.Add(this.label3);
			this.groupBox2.Controls.Add(this.numericUpDownQuiltingBlockSize);
			this.groupBox2.Controls.Add(this.label2);
			this.groupBox2.Controls.Add(this.label1);
			this.groupBox2.Controls.Add(this.numericUpDownQuiltingHeight);
			this.groupBox2.Controls.Add(this.numericUpDownQuiltingWidth);
			this.groupBox2.Controls.Add(this.pictureBoxQuiltedBitmap);
			this.groupBox2.Location = new System.Drawing.Point(384, 16);
			this.groupBox2.Name = "groupBox2";
			this.groupBox2.Size = new System.Drawing.Size(424, 472);
			this.groupBox2.TabIndex = 2;
			this.groupBox2.TabStop = false;
			this.groupBox2.Text = "Quilting Utility";
			// 
			// button3
			// 
			this.button3.Location = new System.Drawing.Point(160, 432);
			this.button3.Name = "button3";
			this.button3.Size = new System.Drawing.Size(120, 23);
			this.button3.TabIndex = 17;
			this.button3.Text = "Load Quilt";
			this.button3.Click += new System.EventHandler(this.button3_Click);
			// 
			// label7
			// 
			this.label7.Location = new System.Drawing.Point(32, 408);
			this.label7.Name = "label7";
			this.label7.Size = new System.Drawing.Size(104, 23);
			this.label7.TabIndex = 16;
			this.label7.Text = "Error Tolerance:";
			// 
			// numericUpDownSelectFromNBestCandidateQuiltErrorTolerance
			// 
			this.numericUpDownSelectFromNBestCandidateQuiltErrorTolerance.DecimalPlaces = 1;
			this.numericUpDownSelectFromNBestCandidateQuiltErrorTolerance.Location = new System.Drawing.Point(144, 408);
			this.numericUpDownSelectFromNBestCandidateQuiltErrorTolerance.Maximum = new System.Decimal(new int[] {
																													 10,
																													 0,
																													 0,
																													 0});
			this.numericUpDownSelectFromNBestCandidateQuiltErrorTolerance.Name = "numericUpDownSelectFromNBestCandidateQuiltErrorTolerance";
			this.numericUpDownSelectFromNBestCandidateQuiltErrorTolerance.Size = new System.Drawing.Size(64, 20);
			this.numericUpDownSelectFromNBestCandidateQuiltErrorTolerance.TabIndex = 15;
			this.numericUpDownSelectFromNBestCandidateQuiltErrorTolerance.Value = new System.Decimal(new int[] {
																												   1,
																												   0,
																												   0,
																												   65536});
			// 
			// label6
			// 
			this.label6.Location = new System.Drawing.Point(8, 384);
			this.label6.Name = "label6";
			this.label6.Size = new System.Drawing.Size(128, 23);
			this.label6.TabIndex = 14;
			this.label6.Text = "Num Candidate Blocks:";
			// 
			// numericUpDownNumCandidateQuiltBlocks
			// 
			this.numericUpDownNumCandidateQuiltBlocks.Location = new System.Drawing.Point(144, 384);
			this.numericUpDownNumCandidateQuiltBlocks.Maximum = new System.Decimal(new int[] {
																								 200,
																								 0,
																								 0,
																								 0});
			this.numericUpDownNumCandidateQuiltBlocks.Minimum = new System.Decimal(new int[] {
																								 1,
																								 0,
																								 0,
																								 0});
			this.numericUpDownNumCandidateQuiltBlocks.Name = "numericUpDownNumCandidateQuiltBlocks";
			this.numericUpDownNumCandidateQuiltBlocks.Size = new System.Drawing.Size(64, 20);
			this.numericUpDownNumCandidateQuiltBlocks.TabIndex = 13;
			this.numericUpDownNumCandidateQuiltBlocks.Value = new System.Decimal(new int[] {
																							   15,
																							   0,
																							   0,
																							   0});
			// 
			// button2
			// 
			this.button2.Location = new System.Drawing.Point(288, 432);
			this.button2.Name = "button2";
			this.button2.Size = new System.Drawing.Size(120, 23);
			this.button2.TabIndex = 12;
			this.button2.Text = "Save Quilt";
			this.button2.Click += new System.EventHandler(this.menuItem5_Click);
			// 
			// button1
			// 
			this.button1.Location = new System.Drawing.Point(24, 432);
			this.button1.Name = "button1";
			this.button1.Size = new System.Drawing.Size(120, 23);
			this.button1.TabIndex = 11;
			this.button1.Text = "Generate Quilt";
			this.button1.Click += new System.EventHandler(this.button1_Click);
			// 
			// label5
			// 
			this.label5.Location = new System.Drawing.Point(216, 344);
			this.label5.Name = "label5";
			this.label5.Size = new System.Drawing.Size(96, 23);
			this.label5.TabIndex = 10;
			this.label5.Text = "(~1/6 block size)";
			// 
			// label4
			// 
			this.label4.Location = new System.Drawing.Point(56, 344);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(80, 23);
			this.label4.TabIndex = 9;
			this.label4.Text = "Block Overlap:";
			// 
			// numericUpDownQuiltingBlockOverlap
			// 
			this.numericUpDownQuiltingBlockOverlap.Increment = new System.Decimal(new int[] {
																								8,
																								0,
																								0,
																								0});
			this.numericUpDownQuiltingBlockOverlap.Location = new System.Drawing.Point(144, 344);
			this.numericUpDownQuiltingBlockOverlap.Maximum = new System.Decimal(new int[] {
																							  65536,
																							  0,
																							  0,
																							  0});
			this.numericUpDownQuiltingBlockOverlap.Minimum = new System.Decimal(new int[] {
																							  4,
																							  0,
																							  0,
																							  0});
			this.numericUpDownQuiltingBlockOverlap.Name = "numericUpDownQuiltingBlockOverlap";
			this.numericUpDownQuiltingBlockOverlap.Size = new System.Drawing.Size(64, 20);
			this.numericUpDownQuiltingBlockOverlap.TabIndex = 8;
			this.numericUpDownQuiltingBlockOverlap.Value = new System.Decimal(new int[] {
																							10,
																							0,
																							0,
																							0});
			// 
			// label3
			// 
			this.label3.Location = new System.Drawing.Point(56, 320);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(80, 23);
			this.label3.TabIndex = 7;
			this.label3.Text = "Block Size:";
			// 
			// numericUpDownQuiltingBlockSize
			// 
			this.numericUpDownQuiltingBlockSize.Increment = new System.Decimal(new int[] {
																							 8,
																							 0,
																							 0,
																							 0});
			this.numericUpDownQuiltingBlockSize.Location = new System.Drawing.Point(144, 320);
			this.numericUpDownQuiltingBlockSize.Maximum = new System.Decimal(new int[] {
																						   65536,
																						   0,
																						   0,
																						   0});
			this.numericUpDownQuiltingBlockSize.Minimum = new System.Decimal(new int[] {
																						   8,
																						   0,
																						   0,
																						   0});
			this.numericUpDownQuiltingBlockSize.Name = "numericUpDownQuiltingBlockSize";
			this.numericUpDownQuiltingBlockSize.Size = new System.Drawing.Size(64, 20);
			this.numericUpDownQuiltingBlockSize.TabIndex = 6;
			this.numericUpDownQuiltingBlockSize.Value = new System.Decimal(new int[] {
																						 32,
																						 0,
																						 0,
																						 0});
			// 
			// label2
			// 
			this.label2.Location = new System.Drawing.Point(168, 288);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(56, 23);
			this.label2.TabIndex = 5;
			this.label2.Text = "Height:";
			// 
			// label1
			// 
			this.label1.Location = new System.Drawing.Point(8, 288);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(80, 23);
			this.label1.TabIndex = 4;
			this.label1.Text = "Quilting Width:";
			// 
			// numericUpDownQuiltingHeight
			// 
			this.numericUpDownQuiltingHeight.Increment = new System.Decimal(new int[] {
																						  256,
																						  0,
																						  0,
																						  0});
			this.numericUpDownQuiltingHeight.Location = new System.Drawing.Point(232, 288);
			this.numericUpDownQuiltingHeight.Maximum = new System.Decimal(new int[] {
																						65536,
																						0,
																						0,
																						0});
			this.numericUpDownQuiltingHeight.Minimum = new System.Decimal(new int[] {
																						256,
																						0,
																						0,
																						0});
			this.numericUpDownQuiltingHeight.Name = "numericUpDownQuiltingHeight";
			this.numericUpDownQuiltingHeight.Size = new System.Drawing.Size(64, 20);
			this.numericUpDownQuiltingHeight.TabIndex = 3;
			this.numericUpDownQuiltingHeight.Value = new System.Decimal(new int[] {
																					  512,
																					  0,
																					  0,
																					  0});
			// 
			// numericUpDownQuiltingWidth
			// 
			this.numericUpDownQuiltingWidth.Increment = new System.Decimal(new int[] {
																						 256,
																						 0,
																						 0,
																						 0});
			this.numericUpDownQuiltingWidth.Location = new System.Drawing.Point(96, 288);
			this.numericUpDownQuiltingWidth.Maximum = new System.Decimal(new int[] {
																					   65536,
																					   0,
																					   0,
																					   0});
			this.numericUpDownQuiltingWidth.Minimum = new System.Decimal(new int[] {
																					   256,
																					   0,
																					   0,
																					   0});
			this.numericUpDownQuiltingWidth.Name = "numericUpDownQuiltingWidth";
			this.numericUpDownQuiltingWidth.Size = new System.Drawing.Size(64, 20);
			this.numericUpDownQuiltingWidth.TabIndex = 2;
			this.numericUpDownQuiltingWidth.Value = new System.Decimal(new int[] {
																					 512,
																					 0,
																					 0,
																					 0});
			// 
			// pictureBoxQuiltedBitmap
			// 
			this.pictureBoxQuiltedBitmap.Location = new System.Drawing.Point(16, 24);
			this.pictureBoxQuiltedBitmap.Name = "pictureBoxQuiltedBitmap";
			this.pictureBoxQuiltedBitmap.Size = new System.Drawing.Size(368, 256);
			this.pictureBoxQuiltedBitmap.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
			this.pictureBoxQuiltedBitmap.TabIndex = 0;
			this.pictureBoxQuiltedBitmap.TabStop = false;
			this.pictureBoxQuiltedBitmap.Click += new System.EventHandler(this.pictureBoxQuiltedBitmap_Click);
			// 
			// statusBar1
			// 
			this.statusBar1.Location = new System.Drawing.Point(0, 595);
			this.statusBar1.Name = "statusBar1";
			this.statusBar1.Size = new System.Drawing.Size(1096, 22);
			this.statusBar1.TabIndex = 3;
			// 
			// textBoxDebugSpew
			// 
			this.textBoxDebugSpew.Location = new System.Drawing.Point(24, 504);
			this.textBoxDebugSpew.Multiline = true;
			this.textBoxDebugSpew.Name = "textBoxDebugSpew";
			this.textBoxDebugSpew.Size = new System.Drawing.Size(1000, 80);
			this.textBoxDebugSpew.TabIndex = 4;
			this.textBoxDebugSpew.Text = "Debug Spew";
			// 
			// groupBox3
			// 
			this.groupBox3.Controls.Add(this.checkBoxOnePixelOverlapBetweenTiles);
			this.groupBox3.Controls.Add(this.button6);
			this.groupBox3.Controls.Add(this.label11);
			this.groupBox3.Controls.Add(this.textBoxWangSpecs);
			this.groupBox3.Controls.Add(this.button5);
			this.groupBox3.Controls.Add(this.button4);
			this.groupBox3.Controls.Add(this.label8);
			this.groupBox3.Controls.Add(this.numericUpDownMaxWangMatchError);
			this.groupBox3.Controls.Add(this.label9);
			this.groupBox3.Controls.Add(this.numericUpDownMaxWangMatchAttempts);
			this.groupBox3.Controls.Add(this.label10);
			this.groupBox3.Controls.Add(this.numericUpDownWangTileSize);
			this.groupBox3.Location = new System.Drawing.Point(816, 16);
			this.groupBox3.Name = "groupBox3";
			this.groupBox3.Size = new System.Drawing.Size(248, 472);
			this.groupBox3.TabIndex = 5;
			this.groupBox3.TabStop = false;
			this.groupBox3.Text = "Wang Tiling";
			// 
			// checkBoxOnePixelOverlapBetweenTiles
			// 
			this.checkBoxOnePixelOverlapBetweenTiles.Checked = true;
			this.checkBoxOnePixelOverlapBetweenTiles.CheckState = System.Windows.Forms.CheckState.Checked;
			this.checkBoxOnePixelOverlapBetweenTiles.Location = new System.Drawing.Point(24, 128);
			this.checkBoxOnePixelOverlapBetweenTiles.Name = "checkBoxOnePixelOverlapBetweenTiles";
			this.checkBoxOnePixelOverlapBetweenTiles.Size = new System.Drawing.Size(192, 24);
			this.checkBoxOnePixelOverlapBetweenTiles.TabIndex = 30;
			this.checkBoxOnePixelOverlapBetweenTiles.Text = "One pixel overlap between tiles";
			// 
			// button6
			// 
			this.button6.Location = new System.Drawing.Point(136, 432);
			this.button6.Name = "button6";
			this.button6.Size = new System.Drawing.Size(96, 23);
			this.button6.TabIndex = 29;
			this.button6.Text = "Save MegaTile";
			this.button6.Click += new System.EventHandler(this.button6_Click);
			// 
			// label11
			// 
			this.label11.Location = new System.Drawing.Point(24, 160);
			this.label11.Name = "label11";
			this.label11.Size = new System.Drawing.Size(144, 32);
			this.label11.TabIndex = 28;
			this.label11.Text = "Wang Tile Colours (NESW):";
			// 
			// textBoxWangSpecs
			// 
			this.textBoxWangSpecs.Location = new System.Drawing.Point(24, 192);
			this.textBoxWangSpecs.Multiline = true;
			this.textBoxWangSpecs.Name = "textBoxWangSpecs";
			this.textBoxWangSpecs.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
			this.textBoxWangSpecs.Size = new System.Drawing.Size(176, 176);
			this.textBoxWangSpecs.TabIndex = 27;
			this.textBoxWangSpecs.Text = "0312\r\n1302\r\n0203\r\n1013\r\n0310\r\n1200\r\n0012\r\n0002\r\n1003\r\n0213\r\n1210\r\n1300";
			// 
			// button5
			// 
			this.button5.Location = new System.Drawing.Point(8, 432);
			this.button5.Name = "button5";
			this.button5.Size = new System.Drawing.Size(120, 23);
			this.button5.TabIndex = 26;
			this.button5.Text = "Save Individual Tiles";
			this.button5.Click += new System.EventHandler(this.button5_Click);
			// 
			// button4
			// 
			this.button4.Location = new System.Drawing.Point(8, 400);
			this.button4.Name = "button4";
			this.button4.Size = new System.Drawing.Size(120, 23);
			this.button4.TabIndex = 25;
			this.button4.Text = "Generate Wang Tiles";
			this.button4.Click += new System.EventHandler(this.button4_Click);
			// 
			// label8
			// 
			this.label8.Location = new System.Drawing.Point(8, 96);
			this.label8.Name = "label8";
			this.label8.Size = new System.Drawing.Size(120, 23);
			this.label8.TabIndex = 24;
			this.label8.Text = "Max Matching Error:";
			// 
			// numericUpDownMaxWangMatchError
			// 
			this.numericUpDownMaxWangMatchError.Increment = new System.Decimal(new int[] {
																							 1000,
																							 0,
																							 0,
																							 0});
			this.numericUpDownMaxWangMatchError.Location = new System.Drawing.Point(140, 96);
			this.numericUpDownMaxWangMatchError.Maximum = new System.Decimal(new int[] {
																						   100000,
																						   0,
																						   0,
																						   0});
			this.numericUpDownMaxWangMatchError.Name = "numericUpDownMaxWangMatchError";
			this.numericUpDownMaxWangMatchError.Size = new System.Drawing.Size(64, 20);
			this.numericUpDownMaxWangMatchError.TabIndex = 23;
			this.numericUpDownMaxWangMatchError.Value = new System.Decimal(new int[] {
																						 1000,
																						 0,
																						 0,
																						 0});
			// 
			// label9
			// 
			this.label9.Location = new System.Drawing.Point(4, 72);
			this.label9.Name = "label9";
			this.label9.Size = new System.Drawing.Size(128, 23);
			this.label9.TabIndex = 22;
			this.label9.Text = "Max Attempts:";
			// 
			// numericUpDownMaxWangMatchAttempts
			// 
			this.numericUpDownMaxWangMatchAttempts.Location = new System.Drawing.Point(140, 72);
			this.numericUpDownMaxWangMatchAttempts.Maximum = new System.Decimal(new int[] {
																							  200,
																							  0,
																							  0,
																							  0});
			this.numericUpDownMaxWangMatchAttempts.Minimum = new System.Decimal(new int[] {
																							  1,
																							  0,
																							  0,
																							  0});
			this.numericUpDownMaxWangMatchAttempts.Name = "numericUpDownMaxWangMatchAttempts";
			this.numericUpDownMaxWangMatchAttempts.Size = new System.Drawing.Size(64, 20);
			this.numericUpDownMaxWangMatchAttempts.TabIndex = 21;
			this.numericUpDownMaxWangMatchAttempts.Value = new System.Decimal(new int[] {
																							15,
																							0,
																							0,
																							0});
			// 
			// label10
			// 
			this.label10.Location = new System.Drawing.Point(32, 32);
			this.label10.Name = "label10";
			this.label10.Size = new System.Drawing.Size(96, 23);
			this.label10.TabIndex = 20;
			this.label10.Text = "Wang Tile Size:";
			// 
			// numericUpDownWangTileSize
			// 
			this.numericUpDownWangTileSize.Increment = new System.Decimal(new int[] {
																						32,
																						0,
																						0,
																						0});
			this.numericUpDownWangTileSize.Location = new System.Drawing.Point(140, 32);
			this.numericUpDownWangTileSize.Maximum = new System.Decimal(new int[] {
																					  65536,
																					  0,
																					  0,
																					  0});
			this.numericUpDownWangTileSize.Minimum = new System.Decimal(new int[] {
																					  32,
																					  0,
																					  0,
																					  0});
			this.numericUpDownWangTileSize.Name = "numericUpDownWangTileSize";
			this.numericUpDownWangTileSize.Size = new System.Drawing.Size(64, 20);
			this.numericUpDownWangTileSize.TabIndex = 19;
			this.numericUpDownWangTileSize.Value = new System.Decimal(new int[] {
																					256,
																					0,
																					0,
																					0});
			// 
			// WangTileForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(1096, 617);
			this.Controls.Add(this.groupBox3);
			this.Controls.Add(this.textBoxDebugSpew);
			this.Controls.Add(this.statusBar1);
			this.Controls.Add(this.groupBox2);
			this.Controls.Add(this.groupBox1);
			this.Menu = this.mainMenu1;
			this.Name = "WangTileForm";
			this.Text = "Texture Quilting and Wang Tiling Utility";
			this.groupBox1.ResumeLayout(false);
			this.groupBox2.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownSelectFromNBestCandidateQuiltErrorTolerance)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownNumCandidateQuiltBlocks)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownQuiltingBlockOverlap)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownQuiltingBlockSize)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownQuiltingHeight)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownQuiltingWidth)).EndInit();
			this.groupBox3.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownMaxWangMatchError)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownMaxWangMatchAttempts)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownWangTileSize)).EndInit();
			this.ResumeLayout(false);

		}
		#endregion

		public void UpdateDisplay()
		{
			if (StartingTextureMapValid)
			{
				this.pictureBoxSourceBitmap.Image = StartingTextureMap;
				this.labelWidthHeight.Text  = "Width: " + StartingTextureMap.Width 
					+ " Height: " + StartingTextureMap.Height;
			}
			else
			{
				this.pictureBoxSourceBitmap.Image = null;
				this.labelWidthHeight.Text  = "Load a valid source bitmap to begin.";
			}
			if (QuiltedTextureMapValid)
			{
				this.pictureBoxQuiltedBitmap.Image = QuiltedTextureMap;
			}

		}

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main() 
		{
			Application.Run(new WangTileForm());
		}

		private void menuItem4_Click(object sender, System.EventArgs e)
		{
			Application.Exit();
		}

		private void menuItem2_Click(object sender, System.EventArgs e)
		{
			if(openFileDialog1.ShowDialog() == DialogResult.OK)
			{
				try
				{
					StartingTextureMap = new Bitmap(openFileDialog1.FileName);
				}
				catch (Exception)
				{
					MessageBox.Show("Invalid texture map file (try a bitmap).");

				}
				UpdateDisplay();
			}

			
		}

		private void button1_Click(object sender, System.EventArgs e)
		{
			if (!StartingTextureMapValid) { MessageBox.Show("Please load a valid source texture map first."); return; } 
			QuiltedTextureMap = ImageQuilter.QuiltTextureMap(
				this.StartingTextureMap,
				(int)this.numericUpDownQuiltingWidth.Value,
				(int)this.numericUpDownQuiltingHeight.Value,
				(int)this.numericUpDownQuiltingBlockSize.Value,
				(int)this.numericUpDownQuiltingBlockOverlap.Value,
				(int)this.numericUpDownNumCandidateQuiltBlocks.Value,
				(int)this.numericUpDownSelectFromNBestCandidateQuiltErrorTolerance.Value
				);
			UpdateDisplay();
		}

		private void menuItem5_Click(object sender, System.EventArgs e)
		{
			if (QuiltedTextureMap != null)
			{
				if (saveFileDialog1.ShowDialog() == DialogResult.OK)
				{
					QuiltedTextureMap.Save(saveFileDialog1.FileName, System.Drawing.Imaging.ImageFormat.Bmp);
				}
			}
		}

		private void pictureBoxQuiltedBitmap_Click(object sender, System.EventArgs e)
		{
		
		}

		private void button3_Click(object sender, System.EventArgs e)
		{
			if(openFileDialog1.ShowDialog() == DialogResult.OK)
			{
				try
				{
					QuiltedTextureMap = new Bitmap(openFileDialog1.FileName);
				}
				catch (Exception)
				{
					MessageBox.Show("Invalid quilt texture file (try a bitmap).");
				}
				UpdateDisplay();
			}
		
		}

		private void button4_Click(object sender, System.EventArgs e)
		{
			if (!QuiltedTextureMapValid) 
			{
				MessageBox.Show("Please load a valid quilt file first.");
				return;
			}
			if (QuiltedTextureMap.Width <= (int)numericUpDownWangTileSize.Value)
			{
				MessageBox.Show("Your Wang Tiles must be smaller than your quilted image texture.");
				return;
			}
			WangTile[] wangTiles = WangTile.LoadDescriptionsFromFile(new StreamReader(new MemoryStream(System.Text.Encoding.ASCII.GetBytes(this.textBoxWangSpecs.Text))));
			WangTiles = WangTiler.WangTile(wangTiles, new Size((int)numericUpDownWangTileSize.Value, (int)numericUpDownWangTileSize.Value), 
				new Bitmap[]{QuiltedTextureMap}, (int)numericUpDownQuiltingBlockOverlap.Value, 
				(int)numericUpDownMaxWangMatchAttempts.Value, (int)numericUpDownMaxWangMatchError.Value,
				checkBoxOnePixelOverlapBetweenTiles.Checked);
			Console.WriteLine("yeah");

		}

		private void button5_Click(object sender, System.EventArgs e)
		{
			if (!WangTilesValid) return;
			if (saveFileDialog1.ShowDialog() == DialogResult.OK)
			{
				string fileName = saveFileDialog1.FileName;
				for (int i = 0; i < WangTiles.Length; i++)
				{
					WangTiles[i].WangBitmap.Save(fileName+i+".bmp", System.Drawing.Imaging.ImageFormat.Bmp);
				}
			}
		
		}

		private void button6_Click(object sender, System.EventArgs e)
		{
			if (!WangTilesValid) return;
			if (saveFileDialog1.ShowDialog() == DialogResult.OK)
			{
				Bitmap[] bitmaps = new Bitmap[WangTiles.Length];
				for (int i = 0; i < WangTiles.Length; i++)
				{
					bitmaps[i] = WangTiles[i].WangBitmap;
				}
				WangTiler.SaveMegaTile(bitmaps, saveFileDialog1.FileName+".bmp");
			}
		}


	}
}
