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
	/// Form that controls the Image Transfer application.
	/// Robert Burke, rob@mle.media.mit.edu
	/// 10 Aug 2003
	/// 
	/// Caveat: this is all "Weekend Project" code.  So it's not necessarily the prettiest thing ever.
	/// Please accept my apologies in advance.
	/// </summary>
	public class ImageTransferForm : System.Windows.Forms.Form
	{
		Bitmap SourceTexture = null;
		Bitmap SourceTextureCMap = null;
		Bitmap TargetTexture = null;
		Bitmap TargetTextureCMap = null;
		Bitmap PreviousIterationQuiltedTexture = null;
		Bitmap CurrentQuiltedTexture = null;



		private System.Windows.Forms.MainMenu mainMenu1;
		private System.Windows.Forms.MenuItem menuItem1;
		private System.Windows.Forms.MenuItem menuItem3;
		private System.Windows.Forms.MenuItem menuItem4;
		private System.Windows.Forms.OpenFileDialog openFileDialog1;
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.GroupBox groupBox2;
		private System.Windows.Forms.StatusBar statusBar1;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.Button button1;
		private System.Windows.Forms.Button button2;
		private System.Windows.Forms.NumericUpDown numericUpDownQuiltingBlockOverlap;
		private System.Windows.Forms.PictureBox pictureBoxQuiltedBitmap;
		private System.Windows.Forms.TextBox textBoxDebugSpew;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.NumericUpDown numericUpDownNumCandidateQuiltBlocks;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.MenuItem menuItem5;
		private System.Windows.Forms.SaveFileDialog saveFileDialog1;
		private System.Windows.Forms.NumericUpDown numericUpDownSelectFromNBestCandidateQuiltErrorTolerance;
		private System.Windows.Forms.Button button3;
		private System.Windows.Forms.PictureBox pictureBoxSourceTexture;
		private System.Windows.Forms.GroupBox groupBox4;
		private System.Windows.Forms.GroupBox groupBox5;
		private System.Windows.Forms.PictureBox pictureBoxTargetCorrespondenceMap;
		private System.Windows.Forms.GroupBox groupBox3;
		private System.Windows.Forms.PictureBox pictureBoxSourceTextureCorrespondenceMap;
		private System.Windows.Forms.Button button4;
		private System.Windows.Forms.Button button5;
		private System.Windows.Forms.Button button6;
		private System.Windows.Forms.GroupBox groupBox6;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.Label labelSmallestBlockSize;
		private System.Windows.Forms.NumericUpDown numericUpDownLargestQuiltingBlockSize;
		private System.Windows.Forms.GroupBox groupBox7;
		private System.Windows.Forms.PictureBox pictureBoxTargetTexture;
		private System.Windows.Forms.Button button7;
		private System.Windows.Forms.Button button8;
		private System.Windows.Forms.Button button9;
		private System.Windows.Forms.Label label9;
		private System.Windows.Forms.NumericUpDown numericUpDownNumIterations;
		private System.Windows.Forms.NumericUpDown numericUpDownBlockReductionPer;
		private System.Windows.Forms.PictureBox pictureBoxPreviousQuiltLevelCMap;
		private System.Windows.Forms.Button button10;
//		private System.ComponentModel.IContainer components;

		public ImageTransferForm()
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
			Util.SBV = new Util.SetBitmapForViewingDelegate(SetBitmapForViewingWithImageTransferForm);
		}
		public void OnSpew(string s)
		{
			this.textBoxDebugSpew.Text = s + "\r\n"+ this.textBoxDebugSpew.Text;
			Application.DoEvents();
		}
		public void SetBitmapForViewingWithImageTransferForm(Bitmap b, int viewWindow)
		{
			if (viewWindow == 0)
			{
				this.pictureBoxQuiltedBitmap.Image = b;
			}
			if (viewWindow == 1)
			{
				this.pictureBoxPreviousQuiltLevelCMap.Image = b;
			}
		}

		void DoTextureTransfer()
		{
			int numIterations = (int)this.numericUpDownNumIterations.Value;
			int blockSize = (int)this.numericUpDownLargestQuiltingBlockSize.Value;
			float blocksizeOverlapPercent = (float)this.numericUpDownQuiltingBlockOverlap.Value;
			int numCandidateQuiltBlocks = (int)this.numericUpDownNumCandidateQuiltBlocks.Value;
			float errorTolerance = (float)numericUpDownSelectFromNBestCandidateQuiltErrorTolerance.Value;


			float blocksizeReductionPerIter = (float)this.numericUpDownBlockReductionPer.Value;
			for (int iteration = 0; iteration < numIterations; iteration++)
			{
				int blockOverlap = (int)((float)blockSize * blocksizeOverlapPercent);
				if (blockOverlap < 3) blockOverlap = 3;
				float alpha = 0.8f * ((float)(iteration+1-1))/((float)(numIterations-1))+0.1f;
				Util.Spew("Iteration " + (iteration+1) + ": alpha " + alpha +", blockSize " + blockSize + " overlap " + blockOverlap);

				// do it
				this.PreviousIterationQuiltedTexture = ImageQuilter.TextureTransfer(
					this.SourceTexture, this.SourceTextureCMap,
					this.TargetTexture, this.TargetTextureCMap,
					this.PreviousIterationQuiltedTexture,
					blockSize, blockOverlap,
					numCandidateQuiltBlocks, errorTolerance,
					alpha
					);

				this.UpdateDisplay();

				blockSize = (int)((float)blockSize * (1f-blocksizeReductionPerIter));
				if (blockSize < 4) blockSize = 4;
			}
			this.CurrentQuiltedTexture = PreviousIterationQuiltedTexture;
			UpdateDisplay();
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
			this.menuItem5 = new System.Windows.Forms.MenuItem();
			this.menuItem3 = new System.Windows.Forms.MenuItem();
			this.menuItem4 = new System.Windows.Forms.MenuItem();
			this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
			this.pictureBoxSourceTexture = new System.Windows.Forms.PictureBox();
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.button7 = new System.Windows.Forms.Button();
			this.button8 = new System.Windows.Forms.Button();
			this.button9 = new System.Windows.Forms.Button();
			this.groupBox7 = new System.Windows.Forms.GroupBox();
			this.pictureBoxTargetTexture = new System.Windows.Forms.PictureBox();
			this.button6 = new System.Windows.Forms.Button();
			this.button5 = new System.Windows.Forms.Button();
			this.button4 = new System.Windows.Forms.Button();
			this.groupBox3 = new System.Windows.Forms.GroupBox();
			this.pictureBoxSourceTextureCorrespondenceMap = new System.Windows.Forms.PictureBox();
			this.groupBox5 = new System.Windows.Forms.GroupBox();
			this.pictureBoxTargetCorrespondenceMap = new System.Windows.Forms.PictureBox();
			this.groupBox4 = new System.Windows.Forms.GroupBox();
			this.groupBox6 = new System.Windows.Forms.GroupBox();
			this.pictureBoxPreviousQuiltLevelCMap = new System.Windows.Forms.PictureBox();
			this.button10 = new System.Windows.Forms.Button();
			this.groupBox2 = new System.Windows.Forms.GroupBox();
			this.label9 = new System.Windows.Forms.Label();
			this.numericUpDownBlockReductionPer = new System.Windows.Forms.NumericUpDown();
			this.labelSmallestBlockSize = new System.Windows.Forms.Label();
			this.label5 = new System.Windows.Forms.Label();
			this.label2 = new System.Windows.Forms.Label();
			this.numericUpDownNumIterations = new System.Windows.Forms.NumericUpDown();
			this.button3 = new System.Windows.Forms.Button();
			this.label7 = new System.Windows.Forms.Label();
			this.numericUpDownSelectFromNBestCandidateQuiltErrorTolerance = new System.Windows.Forms.NumericUpDown();
			this.label6 = new System.Windows.Forms.Label();
			this.numericUpDownNumCandidateQuiltBlocks = new System.Windows.Forms.NumericUpDown();
			this.button2 = new System.Windows.Forms.Button();
			this.button1 = new System.Windows.Forms.Button();
			this.label4 = new System.Windows.Forms.Label();
			this.numericUpDownQuiltingBlockOverlap = new System.Windows.Forms.NumericUpDown();
			this.label3 = new System.Windows.Forms.Label();
			this.numericUpDownLargestQuiltingBlockSize = new System.Windows.Forms.NumericUpDown();
			this.pictureBoxQuiltedBitmap = new System.Windows.Forms.PictureBox();
			this.statusBar1 = new System.Windows.Forms.StatusBar();
			this.textBoxDebugSpew = new System.Windows.Forms.TextBox();
			this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
			this.groupBox1.SuspendLayout();
			this.groupBox7.SuspendLayout();
			this.groupBox3.SuspendLayout();
			this.groupBox5.SuspendLayout();
			this.groupBox4.SuspendLayout();
			this.groupBox6.SuspendLayout();
			this.groupBox2.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownBlockReductionPer)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownNumIterations)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownSelectFromNBestCandidateQuiltErrorTolerance)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownNumCandidateQuiltBlocks)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownQuiltingBlockOverlap)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownLargestQuiltingBlockSize)).BeginInit();
			this.SuspendLayout();
			// 
			// mainMenu1
			// 
			this.mainMenu1.MenuItems.AddRange(new System.Windows.Forms.MenuItem[] {
																					  this.menuItem1});
			// 
			// menuItem1
			// 
			this.menuItem1.Index = 0;
			this.menuItem1.MenuItems.AddRange(new System.Windows.Forms.MenuItem[] {
																					  this.menuItem5,
																					  this.menuItem3,
																					  this.menuItem4});
			this.menuItem1.Text = "&File";
			// 
			// menuItem5
			// 
			this.menuItem5.Index = 0;
			this.menuItem5.Text = "Save &Quilted Bitmap...";
			this.menuItem5.Click += new System.EventHandler(this.menuItem5_Click);
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
			// openFileDialog1
			// 
			this.openFileDialog1.Title = "Select a bitmap image to use as a texture map";
			// 
			// pictureBoxSourceTexture
			// 
			this.pictureBoxSourceTexture.Location = new System.Drawing.Point(8, 16);
			this.pictureBoxSourceTexture.Name = "pictureBoxSourceTexture";
			this.pictureBoxSourceTexture.Size = new System.Drawing.Size(152, 128);
			this.pictureBoxSourceTexture.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
			this.pictureBoxSourceTexture.TabIndex = 0;
			this.pictureBoxSourceTexture.TabStop = false;
			// 
			// groupBox1
			// 
			this.groupBox1.Controls.Add(this.button7);
			this.groupBox1.Controls.Add(this.button8);
			this.groupBox1.Controls.Add(this.button9);
			this.groupBox1.Controls.Add(this.groupBox7);
			this.groupBox1.Controls.Add(this.button6);
			this.groupBox1.Controls.Add(this.button5);
			this.groupBox1.Controls.Add(this.button4);
			this.groupBox1.Controls.Add(this.groupBox3);
			this.groupBox1.Controls.Add(this.groupBox5);
			this.groupBox1.Controls.Add(this.groupBox4);
			this.groupBox1.Controls.Add(this.groupBox6);
			this.groupBox1.Controls.Add(this.button10);
			this.groupBox1.Location = new System.Drawing.Point(24, 16);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.Size = new System.Drawing.Size(504, 480);
			this.groupBox1.TabIndex = 1;
			this.groupBox1.TabStop = false;
			this.groupBox1.Text = "Source bitmaps";
			// 
			// button7
			// 
			this.button7.Location = new System.Drawing.Point(376, 288);
			this.button7.Name = "button7";
			this.button7.Size = new System.Drawing.Size(112, 40);
			this.button7.TabIndex = 22;
			this.button7.Text = "Set C Map To Target";
			this.button7.Click += new System.EventHandler(this.button7_Click);
			// 
			// button8
			// 
			this.button8.Location = new System.Drawing.Point(376, 240);
			this.button8.Name = "button8";
			this.button8.Size = new System.Drawing.Size(112, 40);
			this.button8.TabIndex = 21;
			this.button8.Text = "Load Target Corr\'ce Map";
			this.button8.Click += new System.EventHandler(this.button8_Click);
			// 
			// button9
			// 
			this.button9.Location = new System.Drawing.Point(376, 192);
			this.button9.Name = "button9";
			this.button9.Size = new System.Drawing.Size(112, 40);
			this.button9.TabIndex = 20;
			this.button9.Text = "Load Target";
			this.button9.Click += new System.EventHandler(this.button9_Click);
			// 
			// groupBox7
			// 
			this.groupBox7.Controls.Add(this.pictureBoxTargetTexture);
			this.groupBox7.Location = new System.Drawing.Point(16, 184);
			this.groupBox7.Name = "groupBox7";
			this.groupBox7.Size = new System.Drawing.Size(168, 152);
			this.groupBox7.TabIndex = 19;
			this.groupBox7.TabStop = false;
			this.groupBox7.Text = "Target";
			// 
			// pictureBoxTargetTexture
			// 
			this.pictureBoxTargetTexture.Location = new System.Drawing.Point(8, 24);
			this.pictureBoxTargetTexture.Name = "pictureBoxTargetTexture";
			this.pictureBoxTargetTexture.Size = new System.Drawing.Size(152, 120);
			this.pictureBoxTargetTexture.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
			this.pictureBoxTargetTexture.TabIndex = 0;
			this.pictureBoxTargetTexture.TabStop = false;
			// 
			// button6
			// 
			this.button6.Location = new System.Drawing.Point(376, 128);
			this.button6.Name = "button6";
			this.button6.Size = new System.Drawing.Size(112, 40);
			this.button6.TabIndex = 7;
			this.button6.Text = "Set C Map To Texture";
			this.button6.Click += new System.EventHandler(this.button6_Click_1);
			// 
			// button5
			// 
			this.button5.Location = new System.Drawing.Point(376, 80);
			this.button5.Name = "button5";
			this.button5.Size = new System.Drawing.Size(112, 40);
			this.button5.TabIndex = 6;
			this.button5.Text = "Load Texture Corr\'ce Map";
			this.button5.Click += new System.EventHandler(this.button5_Click_1);
			// 
			// button4
			// 
			this.button4.Location = new System.Drawing.Point(376, 32);
			this.button4.Name = "button4";
			this.button4.Size = new System.Drawing.Size(112, 40);
			this.button4.TabIndex = 5;
			this.button4.Text = "Load Texture";
			this.button4.Click += new System.EventHandler(this.button4_Click_1);
			// 
			// groupBox3
			// 
			this.groupBox3.Controls.Add(this.pictureBoxSourceTextureCorrespondenceMap);
			this.groupBox3.Location = new System.Drawing.Point(192, 24);
			this.groupBox3.Name = "groupBox3";
			this.groupBox3.Size = new System.Drawing.Size(168, 152);
			this.groupBox3.TabIndex = 4;
			this.groupBox3.TabStop = false;
			this.groupBox3.Text = "Texture Corresp\'nce Map";
			// 
			// pictureBoxSourceTextureCorrespondenceMap
			// 
			this.pictureBoxSourceTextureCorrespondenceMap.Location = new System.Drawing.Point(8, 16);
			this.pictureBoxSourceTextureCorrespondenceMap.Name = "pictureBoxSourceTextureCorrespondenceMap";
			this.pictureBoxSourceTextureCorrespondenceMap.Size = new System.Drawing.Size(152, 128);
			this.pictureBoxSourceTextureCorrespondenceMap.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
			this.pictureBoxSourceTextureCorrespondenceMap.TabIndex = 0;
			this.pictureBoxSourceTextureCorrespondenceMap.TabStop = false;
			// 
			// groupBox5
			// 
			this.groupBox5.Controls.Add(this.pictureBoxTargetCorrespondenceMap);
			this.groupBox5.Location = new System.Drawing.Point(192, 184);
			this.groupBox5.Name = "groupBox5";
			this.groupBox5.Size = new System.Drawing.Size(168, 152);
			this.groupBox5.TabIndex = 3;
			this.groupBox5.TabStop = false;
			this.groupBox5.Text = "Target Correspondence Map";
			// 
			// pictureBoxTargetCorrespondenceMap
			// 
			this.pictureBoxTargetCorrespondenceMap.Location = new System.Drawing.Point(8, 24);
			this.pictureBoxTargetCorrespondenceMap.Name = "pictureBoxTargetCorrespondenceMap";
			this.pictureBoxTargetCorrespondenceMap.Size = new System.Drawing.Size(144, 120);
			this.pictureBoxTargetCorrespondenceMap.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
			this.pictureBoxTargetCorrespondenceMap.TabIndex = 0;
			this.pictureBoxTargetCorrespondenceMap.TabStop = false;
			// 
			// groupBox4
			// 
			this.groupBox4.Controls.Add(this.pictureBoxSourceTexture);
			this.groupBox4.Location = new System.Drawing.Point(16, 24);
			this.groupBox4.Name = "groupBox4";
			this.groupBox4.Size = new System.Drawing.Size(168, 152);
			this.groupBox4.TabIndex = 2;
			this.groupBox4.TabStop = false;
			this.groupBox4.Text = "Texture";
			// 
			// groupBox6
			// 
			this.groupBox6.Controls.Add(this.pictureBoxPreviousQuiltLevelCMap);
			this.groupBox6.Location = new System.Drawing.Point(192, 344);
			this.groupBox6.Name = "groupBox6";
			this.groupBox6.Size = new System.Drawing.Size(168, 128);
			this.groupBox6.TabIndex = 18;
			this.groupBox6.TabStop = false;
			this.groupBox6.Text = "Previous Quilt Level";
			// 
			// pictureBoxPreviousQuiltLevelCMap
			// 
			this.pictureBoxPreviousQuiltLevelCMap.Location = new System.Drawing.Point(16, 24);
			this.pictureBoxPreviousQuiltLevelCMap.Name = "pictureBoxPreviousQuiltLevelCMap";
			this.pictureBoxPreviousQuiltLevelCMap.Size = new System.Drawing.Size(136, 96);
			this.pictureBoxPreviousQuiltLevelCMap.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
			this.pictureBoxPreviousQuiltLevelCMap.TabIndex = 0;
			this.pictureBoxPreviousQuiltLevelCMap.TabStop = false;
			// 
			// button10
			// 
			this.button10.Location = new System.Drawing.Point(376, 432);
			this.button10.Name = "button10";
			this.button10.Size = new System.Drawing.Size(112, 32);
			this.button10.TabIndex = 23;
			this.button10.Text = "Clear";
			this.button10.Click += new System.EventHandler(this.button10_Click);
			// 
			// groupBox2
			// 
			this.groupBox2.Controls.Add(this.label9);
			this.groupBox2.Controls.Add(this.numericUpDownBlockReductionPer);
			this.groupBox2.Controls.Add(this.labelSmallestBlockSize);
			this.groupBox2.Controls.Add(this.label5);
			this.groupBox2.Controls.Add(this.label2);
			this.groupBox2.Controls.Add(this.numericUpDownNumIterations);
			this.groupBox2.Controls.Add(this.button3);
			this.groupBox2.Controls.Add(this.label7);
			this.groupBox2.Controls.Add(this.numericUpDownSelectFromNBestCandidateQuiltErrorTolerance);
			this.groupBox2.Controls.Add(this.label6);
			this.groupBox2.Controls.Add(this.numericUpDownNumCandidateQuiltBlocks);
			this.groupBox2.Controls.Add(this.button2);
			this.groupBox2.Controls.Add(this.button1);
			this.groupBox2.Controls.Add(this.label4);
			this.groupBox2.Controls.Add(this.numericUpDownQuiltingBlockOverlap);
			this.groupBox2.Controls.Add(this.label3);
			this.groupBox2.Controls.Add(this.numericUpDownLargestQuiltingBlockSize);
			this.groupBox2.Controls.Add(this.pictureBoxQuiltedBitmap);
			this.groupBox2.Location = new System.Drawing.Point(552, 16);
			this.groupBox2.Name = "groupBox2";
			this.groupBox2.Size = new System.Drawing.Size(424, 472);
			this.groupBox2.TabIndex = 2;
			this.groupBox2.TabStop = false;
			this.groupBox2.Text = "Quilting Utility";
			// 
			// label9
			// 
			this.label9.Location = new System.Drawing.Point(216, 336);
			this.label9.Name = "label9";
			this.label9.Size = new System.Drawing.Size(128, 40);
			this.label9.TabIndex = 28;
			this.label9.Text = "Block Reduction % Per Iteration:";
			this.label9.TextAlign = System.Drawing.ContentAlignment.TopRight;
			// 
			// numericUpDownBlockReductionPer
			// 
			this.numericUpDownBlockReductionPer.DecimalPlaces = 3;
			this.numericUpDownBlockReductionPer.Increment = new System.Decimal(new int[] {
																							 1,
																							 0,
																							 0,
																							 65536});
			this.numericUpDownBlockReductionPer.Location = new System.Drawing.Point(352, 336);
			this.numericUpDownBlockReductionPer.Maximum = new System.Decimal(new int[] {
																						   1,
																						   0,
																						   0,
																						   0});
			this.numericUpDownBlockReductionPer.Name = "numericUpDownBlockReductionPer";
			this.numericUpDownBlockReductionPer.Size = new System.Drawing.Size(64, 20);
			this.numericUpDownBlockReductionPer.TabIndex = 27;
			this.numericUpDownBlockReductionPer.Value = new System.Decimal(new int[] {
																						 333,
																						 0,
																						 0,
																						 196608});
			this.numericUpDownBlockReductionPer.ValueChanged += new System.EventHandler(this.calcSmallBlockSize);
			// 
			// labelSmallestBlockSize
			// 
			this.labelSmallestBlockSize.Location = new System.Drawing.Point(128, 328);
			this.labelSmallestBlockSize.Name = "labelSmallestBlockSize";
			this.labelSmallestBlockSize.Size = new System.Drawing.Size(64, 23);
			this.labelSmallestBlockSize.TabIndex = 23;
			this.labelSmallestBlockSize.Text = "-";
			// 
			// label5
			// 
			this.label5.Location = new System.Drawing.Point(16, 328);
			this.label5.Name = "label5";
			this.label5.Size = new System.Drawing.Size(112, 23);
			this.label5.TabIndex = 22;
			this.label5.Text = "Smallest Block Size:";
			// 
			// label2
			// 
			this.label2.Location = new System.Drawing.Point(280, 312);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(64, 23);
			this.label2.TabIndex = 19;
			this.label2.Text = "Iterations:";
			// 
			// numericUpDownNumIterations
			// 
			this.numericUpDownNumIterations.Location = new System.Drawing.Point(352, 312);
			this.numericUpDownNumIterations.Maximum = new System.Decimal(new int[] {
																					   10,
																					   0,
																					   0,
																					   0});
			this.numericUpDownNumIterations.Minimum = new System.Decimal(new int[] {
																					   1,
																					   0,
																					   0,
																					   0});
			this.numericUpDownNumIterations.Name = "numericUpDownNumIterations";
			this.numericUpDownNumIterations.Size = new System.Drawing.Size(64, 20);
			this.numericUpDownNumIterations.TabIndex = 18;
			this.numericUpDownNumIterations.Value = new System.Decimal(new int[] {
																					 3,
																					 0,
																					 0,
																					 0});
			this.numericUpDownNumIterations.ValueChanged += new System.EventHandler(this.calcSmallBlockSize);
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
			// label4
			// 
			this.label4.Location = new System.Drawing.Point(24, 352);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(96, 23);
			this.label4.TabIndex = 9;
			this.label4.Text = "Block Overlap %:";
			// 
			// numericUpDownQuiltingBlockOverlap
			// 
			this.numericUpDownQuiltingBlockOverlap.DecimalPlaces = 2;
			this.numericUpDownQuiltingBlockOverlap.Increment = new System.Decimal(new int[] {
																								5,
																								0,
																								0,
																								131072});
			this.numericUpDownQuiltingBlockOverlap.Location = new System.Drawing.Point(128, 352);
			this.numericUpDownQuiltingBlockOverlap.Maximum = new System.Decimal(new int[] {
																							  1,
																							  0,
																							  0,
																							  0});
			this.numericUpDownQuiltingBlockOverlap.Name = "numericUpDownQuiltingBlockOverlap";
			this.numericUpDownQuiltingBlockOverlap.Size = new System.Drawing.Size(64, 20);
			this.numericUpDownQuiltingBlockOverlap.TabIndex = 8;
			this.numericUpDownQuiltingBlockOverlap.Value = new System.Decimal(new int[] {
																							17,
																							0,
																							0,
																							131072});
			// 
			// label3
			// 
			this.label3.Location = new System.Drawing.Point(16, 304);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(104, 23);
			this.label3.TabIndex = 7;
			this.label3.Text = "Largest Block Size:";
			// 
			// numericUpDownLargestQuiltingBlockSize
			// 
			this.numericUpDownLargestQuiltingBlockSize.Increment = new System.Decimal(new int[] {
																									8,
																									0,
																									0,
																									0});
			this.numericUpDownLargestQuiltingBlockSize.Location = new System.Drawing.Point(128, 304);
			this.numericUpDownLargestQuiltingBlockSize.Maximum = new System.Decimal(new int[] {
																								  65536,
																								  0,
																								  0,
																								  0});
			this.numericUpDownLargestQuiltingBlockSize.Minimum = new System.Decimal(new int[] {
																								  8,
																								  0,
																								  0,
																								  0});
			this.numericUpDownLargestQuiltingBlockSize.Name = "numericUpDownLargestQuiltingBlockSize";
			this.numericUpDownLargestQuiltingBlockSize.Size = new System.Drawing.Size(64, 20);
			this.numericUpDownLargestQuiltingBlockSize.TabIndex = 6;
			this.numericUpDownLargestQuiltingBlockSize.Value = new System.Decimal(new int[] {
																								24,
																								0,
																								0,
																								0});
			this.numericUpDownLargestQuiltingBlockSize.ValueChanged += new System.EventHandler(this.calcSmallBlockSize);
			// 
			// pictureBoxQuiltedBitmap
			// 
			this.pictureBoxQuiltedBitmap.Location = new System.Drawing.Point(16, 24);
			this.pictureBoxQuiltedBitmap.Name = "pictureBoxQuiltedBitmap";
			this.pictureBoxQuiltedBitmap.Size = new System.Drawing.Size(280, 256);
			this.pictureBoxQuiltedBitmap.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
			this.pictureBoxQuiltedBitmap.TabIndex = 0;
			this.pictureBoxQuiltedBitmap.TabStop = false;
			// 
			// statusBar1
			// 
			this.statusBar1.Location = new System.Drawing.Point(0, 595);
			this.statusBar1.Name = "statusBar1";
			this.statusBar1.Size = new System.Drawing.Size(1088, 22);
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
			// ImageTransferForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(1088, 617);
			this.Controls.Add(this.textBoxDebugSpew);
			this.Controls.Add(this.statusBar1);
			this.Controls.Add(this.groupBox2);
			this.Controls.Add(this.groupBox1);
			this.Menu = this.mainMenu1;
			this.Name = "ImageTransferForm";
			this.Text = "Image Quilting Texture Transfer";
			this.Load += new System.EventHandler(this.ImageTransferForm_Load);
			this.groupBox1.ResumeLayout(false);
			this.groupBox7.ResumeLayout(false);
			this.groupBox3.ResumeLayout(false);
			this.groupBox5.ResumeLayout(false);
			this.groupBox4.ResumeLayout(false);
			this.groupBox6.ResumeLayout(false);
			this.groupBox2.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownBlockReductionPer)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownNumIterations)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownSelectFromNBestCandidateQuiltErrorTolerance)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownNumCandidateQuiltBlocks)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownQuiltingBlockOverlap)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numericUpDownLargestQuiltingBlockSize)).EndInit();
			this.ResumeLayout(false);

		}
		#endregion

		public void UpdateDisplay()
		{
			this.pictureBoxTargetTexture.Image  = this.TargetTexture;
			this.pictureBoxTargetCorrespondenceMap.Image = this.TargetTextureCMap;
			this.pictureBoxSourceTexture.Image  = this.SourceTexture;
			this.pictureBoxSourceTextureCorrespondenceMap.Image = this.SourceTextureCMap;
			this.pictureBoxQuiltedBitmap.Image = this.CurrentQuiltedTexture;
			this.pictureBoxPreviousQuiltLevelCMap.Image = this.PreviousIterationQuiltedTexture;
		}

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main() 
		{
			Application.Run(new ImageTransferForm());
		}

		private void menuItem4_Click(object sender, System.EventArgs e)
		{
			Application.Exit();
		}

		private void button1_Click(object sender, System.EventArgs e)
		{
			// DO THE BUSINESS OF GENERATING THE QUILT
			DoTextureTransfer();
		}

		private void menuItem5_Click(object sender, System.EventArgs e)
		{
			if (this.CurrentQuiltedTexture != null)
			{
				if (saveFileDialog1.ShowDialog() == DialogResult.OK)
				{
					CurrentQuiltedTexture.Save(saveFileDialog1.FileName, System.Drawing.Imaging.ImageFormat.Bmp);
				}
			}
		}

		private void button3_Click(object sender, System.EventArgs e)
		{
			if(openFileDialog1.ShowDialog() == DialogResult.OK)
			{
				try
				{
					this.PreviousIterationQuiltedTexture = new Bitmap(openFileDialog1.FileName);
					this.CurrentQuiltedTexture = PreviousIterationQuiltedTexture;
				}
				catch (Exception)
				{
					MessageBox.Show("Invalid quilt texture file (try a bitmap).");
				}
				UpdateDisplay();
			}
		}

		private void calcSmallBlockSize(object sender, System.EventArgs e)
		{
			double d = ((double)this.numericUpDownLargestQuiltingBlockSize.Value * Math.Pow(1-(double)this.numericUpDownBlockReductionPer.Value, (double)this.numericUpDownNumIterations.Value-1));
			
			this.labelSmallestBlockSize.Text = ""+d.ToString("0");
		}

		private void ImageTransferForm_Load(object sender, System.EventArgs e)
		{
			calcSmallBlockSize(sender, e);
		}

		private Bitmap LoadATexture(PictureBox targetBox)
		{
			Bitmap temp = null;
			if(openFileDialog1.ShowDialog() == DialogResult.OK)
			{
				try
				{
					temp = new Bitmap(openFileDialog1.FileName);
				}
				catch (Exception)
				{
					MessageBox.Show("Invalid quilt texture file (try a bitmap).");
				}
			}
			return temp;
		}

		private void button4_Click_1(object sender, System.EventArgs e)
		{
			Bitmap temp = LoadATexture(this.pictureBoxSourceTexture);
			if (temp != null) 
			{
				SourceTexture = temp;
				if (SourceTextureCMap == null) SourceTextureCMap = SourceTexture;
			}
			UpdateDisplay();
		}

		private void button5_Click_1(object sender, System.EventArgs e)
		{
			Bitmap temp = LoadATexture(this.pictureBoxSourceTextureCorrespondenceMap);
			if (temp != null) SourceTextureCMap = temp;
			UpdateDisplay();
		}

		private void button6_Click_1(object sender, System.EventArgs e)
		{
			this.SourceTextureCMap = this.SourceTexture;
			UpdateDisplay();
		}

		private void button9_Click(object sender, System.EventArgs e)
		{
			Bitmap temp = LoadATexture(this.pictureBoxTargetTexture);
			if (temp != null) 
			{
				this.TargetTexture = temp;
				if (TargetTextureCMap == null) TargetTextureCMap = TargetTexture;
			}
			UpdateDisplay();
		}

		private void button8_Click(object sender, System.EventArgs e)
		{
			Bitmap temp = LoadATexture(this.pictureBoxTargetTexture);
			if (temp != null) this.TargetTextureCMap = temp;
			UpdateDisplay();
		}

		private void button7_Click(object sender, System.EventArgs e)
		{
			this.TargetTextureCMap = this.TargetTexture;
			UpdateDisplay();
		}

		private void button10_Click(object sender, System.EventArgs e)
		{
			this.PreviousIterationQuiltedTexture = null;
			UpdateDisplay();
		}




	}
}
