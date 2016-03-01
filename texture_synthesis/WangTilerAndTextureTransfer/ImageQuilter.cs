using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;
using System.Drawing.Imaging;
using System.IO;

namespace WangTileCreator
{
	/// <summary>
	/// Guts of the Image Quilting Code
	/// Robert Burke, rob@mle.media.mit.edu
	/// 10 Aug 2003
	/// 
	/// Caveat: this is all "Weekend Project" code.  So it's not necessarily the prettiest thing ever.
	/// Please accept my apologies in advance.
	/// </summary>
	public class ImageQuilter
	{
		#region Image Quilting - Efros and Freeman Siggraph 2001 implementation


		static Random r = new Random();

		/// <summary>
		/// Paste a section from one bitmap into the other
		/// </summary>
		public static void PasteSection(Bitmap from, Bitmap to, Rectangle fromRegion, Point toUpperLeftCorner)
		{
			for (int i = 0; i < fromRegion.Width; i++)
			{
				for (int j = 0; j < fromRegion.Height; j++)
				{
					to.SetPixel(toUpperLeftCorner.X+i, toUpperLeftCorner.Y+j, from.GetPixel(fromRegion.Left+i, fromRegion.Top+j));
				}
			}
		}

		/// <summary>
		/// Paste a section from one bitmap into another, using the two cut functions provided as seames.  The hotizcut determines the top seam,
		/// and the vertcut determines the seam on the left hand side.
		/// </summary>
		public static void PasteSectionRespectingCuts(Bitmap from, Bitmap to, Rectangle fromRegion, Point toUpperLeftCorner, int[]horizCut, int[] vertCut)
		{
			for (int i = 0; i < fromRegion.Width; i++)
			{
				for (int j = 0; j < fromRegion.Height; j++)
				{
					if (j >= horizCut[i] && i >= vertCut[j])
					{
						to.SetPixel(toUpperLeftCorner.X+i, toUpperLeftCorner.Y+j, from.GetPixel(fromRegion.Left+i, fromRegion.Top+j));
					}
				}
			}
		}

		/// <summary>
		/// Paste a section from one bitmap into another, using the two cut functions provided as seames.  The hotizcut determines the top seam,
		/// and the vertcut determines the seam on the left hand side.  Along the seam we blend the pixels from the two images together.
		/// </summary>
		public static void PasteSectionRespectingCutsAndBlendSeam(Bitmap from, Bitmap to, Rectangle fromRegion, Point toUpperLeftCorner, int[]horizCut, int[] vertCut)
		{
			for (int i = 0; i < fromRegion.Width; i++)
			{
				for (int j = 0; j < fromRegion.Height; j++)
				{
					if (j == horizCut[i] && i == vertCut[j])
					{
						to.SetPixel(toUpperLeftCorner.X+i, toUpperLeftCorner.Y+j,
							Util.BlendColors(to.GetPixel(toUpperLeftCorner.X+i, toUpperLeftCorner.Y+j), from.GetPixel(fromRegion.Left+i, fromRegion.Top+j))
							);
					}
					else if (j >= horizCut[i] && i >= vertCut[j])
					{
						to.SetPixel(toUpperLeftCorner.X+i, toUpperLeftCorner.Y+j, from.GetPixel(fromRegion.Left+i, fromRegion.Top+j));
					}
				}
			}
		}


		public static long GetPlacementError(Bitmap from, Bitmap to, Rectangle fromRegion, Point toUpperLeftCorner)
		{
			long totalError = 0;
			for (int i = 0; i < fromRegion.Width; i++)
			{
				for (int j = 0; j < fromRegion.Height; j++)
				{
					Color fromPixel = from.GetPixel(fromRegion.Left+i, fromRegion.Top+j);
					int toX = toUpperLeftCorner.X+i;
					int toY = toUpperLeftCorner.Y+j;
					if (toX >= 0 && toY >= 0)
					{
						Color toPixel = to.GetPixel(toX, toY);

						if (toPixel.A > 0)
						{
							totalError += Util.GetError(toPixel, fromPixel);
						}
					}
				}
			}

			return totalError;
		}

		public static long GetPlacementErrorForTextureTransfer(
			Bitmap sourceTexture, Bitmap sourceCMap,
			Bitmap targetTexture, Bitmap targetCMap,
			Bitmap previousIterationQuiltTexture,
			Bitmap  currentIterationQuiltTexture,
			Rectangle sourceRegion, Point quiltTextureUpperLeftCorner,
			float alpha)
		{
			long errorFromSourceTextureMatching = 0;
			long errorFromCorrespondenceMap = 0;
			long errorFromPreviousIteration = 0;
			int errorFromSourceTextureMatchingIndices = 0;
			int errorTotalIndicesCount = sourceRegion.Width * sourceRegion.Height;
			for (int i = 0; i < sourceRegion.Width; i++)
			{
				for (int j = 0; j < sourceRegion.Height; j++)
				{
					Color sourceTexturePixel = sourceTexture.GetPixel(sourceRegion.Left+i, sourceRegion.Top+j);
					Color sourceCMapPixel = sourceCMap.GetPixel(sourceRegion.Left+i, sourceRegion.Top+j);
					int toX = quiltTextureUpperLeftCorner.X+i;
					int toY = quiltTextureUpperLeftCorner.Y+j;

					if (toX >= 0 && toY >= 0)
					{
						Color currentQuiltTexturePixel = currentIterationQuiltTexture.GetPixel(toX, toY);
						Color targetCMapPixel = targetCMap.GetPixel(toX, toY);

						errorFromCorrespondenceMap += Util.GetError(sourceCMapPixel, targetCMapPixel);

						if (previousIterationQuiltTexture != null)
						{
							Color previousQuiltTexturePixel = previousIterationQuiltTexture.GetPixel(toX, toY);
							errorFromPreviousIteration += Util.GetError(sourceTexturePixel, previousQuiltTexturePixel);
						}

						if (currentQuiltTexturePixel.A > 0) 
						{
							errorFromSourceTextureMatching += Util.GetError(sourceTexturePixel, currentQuiltTexturePixel);
							errorFromSourceTextureMatchingIndices++;
						}
					}
				}
			}

			//if (previousIterationQuiltTexture != null) errorFromSourceTextureMatching *= 2; // To keep these weighted the same?
			//errorFromSourceTextureMatching = (long)((float)errorFromSourceTextureMatching * (float)errorTotalIndicesCount / (float)errorFromSourceTextureMatchingIndices);

			// alpha times the block overlap matching error, plus 1-alpha times the error from the correspondence map
			errorFromSourceTextureMatching = (long) ( ((float)errorFromSourceTextureMatching) * alpha  );
			errorFromCorrespondenceMap = (long) ( ((float)errorFromCorrespondenceMap) * (1.0f-alpha)  );
			errorFromPreviousIteration = (long) ( ((float)errorFromPreviousIteration) * (1.0f-alpha)  );

			return errorFromSourceTextureMatching+errorFromCorrespondenceMap+errorFromPreviousIteration;
		}



		#region Dynamic Programming GetMinErrorBoundaryCut functions

		/// <summary>
		/// Uses a dynamic programming algorithm to determine which VERTICAL path through the texture is optimal. 
		/// The indices returned are counted from the LEFT. So 0 means right along the left edge.
		/// </summary>
		/// <param name="from">The 'from' Bitmap</param>
		/// <param name="to">The 'to' Bitmap</param>
		/// <param name="fromRegion">The overlapping region.  Not the entire image being pasted in, just the overlapping region.</param>
		/// <param name="toUpperLeftCorner">The place it's being pasted to.</param>
		/// <returns></returns>
		public static int[] GetMinErrorBoundaryCutVertical(Bitmap from, Bitmap to, Rectangle fromRegion, Point toRegion)
		{
			long[,] dynamicProgrammingMatrix = new long[fromRegion.Height, fromRegion.Width]; // NOTE: This is in (row, col) format.
			// Start at the top and work your way down to the bottom

			long eij = 0;
			for (int j = 0; j < fromRegion.Width; j++)
			{
				dynamicProgrammingMatrix[0, j] = Util.GetError(from.GetPixel(fromRegion.Left + j, fromRegion.Top), to.GetPixel(toRegion.X+j, toRegion.Y) );
			}
			for (int i = 1; i < fromRegion.Height; i++)
			{
				// Pixel on the left
				eij = Util.GetError(from.GetPixel(fromRegion.Left + 0, fromRegion.Top + i), to.GetPixel(toRegion.X + 0, toRegion.Y + i) );
				dynamicProgrammingMatrix[i, 0]
					= eij + Math.Min(dynamicProgrammingMatrix[i-1, 0+1], dynamicProgrammingMatrix[i-1, 0]);
				// Pixel on the right
				eij = Util.GetError(from.GetPixel(fromRegion.Left + fromRegion.Width-1, fromRegion.Top + i), to.GetPixel(toRegion.X + fromRegion.Width-1, toRegion.Y + i) );
				dynamicProgrammingMatrix[i, fromRegion.Width-1]
					= eij + Math.Min(dynamicProgrammingMatrix[i-1, fromRegion.Width-2], dynamicProgrammingMatrix[i-1, fromRegion.Width-1]);

				// Pixels in the middle
				for (int j = 1; j < fromRegion.Width-1; j++)
				{
					eij = Util.GetError(from.GetPixel(fromRegion.Left + j, fromRegion.Top + i), to.GetPixel(toRegion.X + j, toRegion.Y + i) );
					dynamicProgrammingMatrix[i, j]
						= eij + Math.Min(Math.Min(dynamicProgrammingMatrix[i-1, j-1], dynamicProgrammingMatrix[i-1, j]), dynamicProgrammingMatrix[i-1, j+1]);
				}
			}

			// Trace back the ideal route
			int[] idealRoute = new int[fromRegion.Height];
			int currentIndex = 0;
			long min = Int64.MaxValue;
			for (int i = 0; i < fromRegion.Width; i++)
			{
				if (dynamicProgrammingMatrix[fromRegion.Height-1, i] < min) { min = dynamicProgrammingMatrix[fromRegion.Height-1, i]; currentIndex = i; }
			}
			idealRoute[fromRegion.Height-1] = currentIndex;
			for (int i = fromRegion.Height-2; i >= 0; i--)
			{
				if (currentIndex == 0)
				{
					if (dynamicProgrammingMatrix[i, currentIndex] > dynamicProgrammingMatrix[i, currentIndex+1]) currentIndex++;
				}
				else if (currentIndex == fromRegion.Width-1)
				{
					if (dynamicProgrammingMatrix[i, currentIndex] > dynamicProgrammingMatrix[i, currentIndex-1]) currentIndex--;
				}
				else
				{
					min = Math.Min(Math.Min(dynamicProgrammingMatrix[i, currentIndex], dynamicProgrammingMatrix[i, currentIndex-1]), dynamicProgrammingMatrix[i, currentIndex+1]);
					if (min == dynamicProgrammingMatrix[i,currentIndex-1] && min < dynamicProgrammingMatrix[i,currentIndex]) currentIndex--;
					else if (min == dynamicProgrammingMatrix[i,currentIndex+1] && min < dynamicProgrammingMatrix[i,currentIndex]) currentIndex++;
				}

				idealRoute[i] = currentIndex;
			}

			return idealRoute;
		}



		/// <summary>
		/// Uses a dynamic programming algorithm to determine which HORIZONTAL path through the texture is optimal. 
		/// (This was blasted from the vertical function and tweaked.)  
		/// The indices returned are counted from the TOP. So 0 means right along the top edge.
		/// </summary>
		/// <param name="from">The 'from' Bitmap</param>
		/// <param name="to">The 'to' Bitmap</param>
		/// <param name="fromRegion">The overlapping region.  Not the entire image being pasted in, just the overlapping region.</param>
		/// <param name="toUpperLeftCorner">The place it's being pasted to.</param>
		/// <returns></returns>
		public static int[] GetMinErrorBoundaryCutHorizontal(Bitmap from, Bitmap to, Rectangle fromRegion, Point toRegion)
		{
			long[,] dynamicProgrammingMatrix = new long[fromRegion.Width, fromRegion.Height]; // NOTE: This is in (COL, ROW) format now, UNLIKE THE PREVIOUS FUNCTION!!
			// Start at the top and work your way down to the bottom

			long eij = 0;
			for (int j = 0; j < fromRegion.Height; j++)
			{
				dynamicProgrammingMatrix[0, j] = Util.GetError(from.GetPixel(fromRegion.Left, fromRegion.Top + j), to.GetPixel(toRegion.X, toRegion.Y + j) );
			}
			for (int i = 1; i < fromRegion.Width; i++)
			{
				// Pixel on the left
				eij = Util.GetError(from.GetPixel(fromRegion.Left + i, fromRegion.Top + 0), to.GetPixel(toRegion.X + i, toRegion.Y + 0) );
				dynamicProgrammingMatrix[i, 0]
					= eij + Math.Min(dynamicProgrammingMatrix[i-1, 0+1], dynamicProgrammingMatrix[i-1, 0]);
				// Pixel on the right
				eij = Util.GetError(from.GetPixel(fromRegion.Left + i, fromRegion.Top + fromRegion.Height-1), to.GetPixel(toRegion.X + i, toRegion.Y + fromRegion.Height-1) );
				dynamicProgrammingMatrix[i, fromRegion.Height-1]
					= eij + Math.Min(dynamicProgrammingMatrix[i-1, fromRegion.Height-2], dynamicProgrammingMatrix[i-1, fromRegion.Height-1]);

				// Pixels in the middle
				for (int j = 1; j < fromRegion.Height-1; j++)
				{
					eij = Util.GetError(from.GetPixel(fromRegion.Left + i, fromRegion.Top + j), to.GetPixel(toRegion.X + i, toRegion.Y + j) );
					dynamicProgrammingMatrix[i, j]
						= eij + Math.Min(Math.Min(dynamicProgrammingMatrix[i-1, j-1], dynamicProgrammingMatrix[i-1, j]), dynamicProgrammingMatrix[i-1, j+1]);
				}
			}

			// Trace back the ideal route
			int[] idealRoute = new int[fromRegion.Width];
			int currentIndex = 0;
			long min = Int64.MaxValue;
			for (int i = 0; i < fromRegion.Height; i++)
			{
				if (dynamicProgrammingMatrix[fromRegion.Width-1, i] < min) { min = dynamicProgrammingMatrix[fromRegion.Width-1, i]; currentIndex = i; }
			}
			idealRoute[fromRegion.Width-1] = currentIndex;
			for (int i = fromRegion.Width-2; i >= 0; i--)
			{
				if (currentIndex == 0)
				{
					if (dynamicProgrammingMatrix[i, currentIndex] > dynamicProgrammingMatrix[i, currentIndex+1]) currentIndex++;
				}
				else if (currentIndex == fromRegion.Height-1)
				{
					if (dynamicProgrammingMatrix[i, currentIndex] > dynamicProgrammingMatrix[i, currentIndex-1]) currentIndex--;
				}
				else
				{
					min = Math.Min(Math.Min(dynamicProgrammingMatrix[i, currentIndex], dynamicProgrammingMatrix[i, currentIndex-1]), dynamicProgrammingMatrix[i, currentIndex+1]);
					if (min == dynamicProgrammingMatrix[i,currentIndex-1] && min < dynamicProgrammingMatrix[i,currentIndex]) currentIndex--;
					else if (min == dynamicProgrammingMatrix[i,currentIndex+1] && min < dynamicProgrammingMatrix[i,currentIndex]) currentIndex++;
				}

				idealRoute[i] = currentIndex;
			}

			return idealRoute;
		}

		#endregion



		/// <summary>
		/// Returns a sorted array of longs in ascending order
		/// </summary>
		private static long[] Sort(long[]values)
		{
			long[] output = new long[values.Length];
			long temp = 0;
			System.Array.Copy(values, output, values.Length);
			for (int i = 0; i < values.Length; i++) for (int j = 0; j < values.Length-1; j++) if (output[j] > output[j+1]) { temp = output[j+1]; output[j+1] = output[j]; output[j] = temp; }
			return output;
		}


		public static Bitmap QuiltTextureMap(Bitmap startingTextureMap, int quiltingWidth, int quiltingHeight, int blockSize, int blockOverlap, int numCandidateQuiltBlocks, float candidateQuiltErrorTolerance)
		{

			//if (!StartingTextureMapValid) { MessageBox.Show("Please load a valid starting texture map first."); return; }
			if (startingTextureMap.PixelFormat != PixelFormat.Format24bppRgb) { MessageBox.Show("Starting texture map must be 24bppRGB"); return null; }
			
			// We use the alpha channel to indicate whether or not this pixel has been placed down yet.
			Bitmap quiltedTextureMap = new Bitmap(quiltingWidth, quiltingHeight, System.Drawing.Imaging.PixelFormat.Format32bppArgb);


			Point[] CandidatePoints = new Point[numCandidateQuiltBlocks];
			long[]  CandidateErrors = new  long[numCandidateQuiltBlocks];

			Util.Spew("Creating quilt texture map");

			for (int quiltingLocationY = 0; quiltingLocationY < (quiltingHeight-blockSize); quiltingLocationY += (blockSize-blockOverlap))
			{
				for (int quiltingLocationX = 0; quiltingLocationX < (quiltingWidth-blockSize); quiltingLocationX += (blockSize-blockOverlap))
				{

					// Determine the fit of a bunch of random places
					for (int k = 0; k < numCandidateQuiltBlocks; k++)
					{
						CandidatePoints[k] = new Point(r.Next(startingTextureMap.Width-blockSize), r.Next(startingTextureMap.Height-blockSize));
						CandidateErrors[k] = GetPlacementError(startingTextureMap, quiltedTextureMap, new Rectangle(CandidatePoints[k], new Size(blockSize, blockSize)), new Point(quiltingLocationX, quiltingLocationY));
						//Spew("Sample block " + k + " from " + CandidatePoints[k] + " gives result " + CandidateErrors[k]);
					}

					long[] sorted = Sort(CandidateErrors);
					int selectFromBestNCandidateQuiltBlocks = 1;
					long sortedMaxError = (long)(sorted[0]*(1+candidateQuiltErrorTolerance));
					while (selectFromBestNCandidateQuiltBlocks < sorted.Length && sorted[selectFromBestNCandidateQuiltBlocks] < sortedMaxError) selectFromBestNCandidateQuiltBlocks++;
					long oneToSelect = sorted[r.Next(selectFromBestNCandidateQuiltBlocks)];
					int candidateIndex = 0;
					while (CandidateErrors[candidateIndex] != oneToSelect) candidateIndex++;
					//Util.Spew("Chose block " + candidateIndex + " which has error " + CandidateErrors[candidateIndex]);
					Application.DoEvents();
					Point selectedPoint = CandidatePoints[candidateIndex];

					int[] verticalCut = 
						GetMinErrorBoundaryCutVertical(startingTextureMap, quiltedTextureMap, new Rectangle(selectedPoint, new Size(blockOverlap, blockSize)), new Point(quiltingLocationX, quiltingLocationY));
					int[] horizontalCut = 
						GetMinErrorBoundaryCutHorizontal(startingTextureMap, quiltedTextureMap, new Rectangle(selectedPoint, new Size(blockSize, blockOverlap)), new Point(quiltingLocationX, quiltingLocationY));

					PasteSectionRespectingCutsAndBlendSeam(startingTextureMap, quiltedTextureMap, new Rectangle(selectedPoint, new Size(blockSize, blockSize)), new Point(quiltingLocationX, quiltingLocationY), horizontalCut, verticalCut);

					Util.SetBitmapForViewing(quiltedTextureMap, 0);
					Application.DoEvents();
					//Util.RefreshDisplay();
					//System.Threading.Thread.Sleep(1000);

				}
			}

			return quiltedTextureMap;
		}


		
		public static Bitmap TextureTransfer(
			Bitmap sourceTexture, Bitmap sourceCMap,
			Bitmap targetTexture, Bitmap targetCMap,
			Bitmap previousIteration,
			int blockSize, int blockOverlap, 
			int numCandidateQuiltBlocks, float candidateQuiltErrorTolerance,
			float transferAlpha)
		{
			int quiltingWidth = targetTexture.Width;
			int quiltingHeight = targetTexture.Height;
			
			// We use the alpha channel to indicate whether or not this pixel has been placed down yet.
			Bitmap quiltedTextureMap = new Bitmap(quiltingWidth, quiltingHeight, System.Drawing.Imaging.PixelFormat.Format32bppArgb);


			Point[] CandidatePoints = new Point[numCandidateQuiltBlocks];
			long[]  CandidateErrors = new  long[numCandidateQuiltBlocks];

			Util.Spew("Performing texture transfer...");

			for (int quiltingLocationY = 0; quiltingLocationY < (quiltingHeight-blockSize); quiltingLocationY += (blockSize-blockOverlap))
			{
				for (int quiltingLocationX = 0; quiltingLocationX < (quiltingWidth-blockSize); quiltingLocationX += (blockSize-blockOverlap))
				{

					// Determine the fit of a bunch of random places
					for (int k = 0; k < numCandidateQuiltBlocks; k++)
					{
						CandidatePoints[k] = new Point(r.Next(sourceTexture.Width-blockSize), r.Next(sourceTexture.Height-blockSize));
						CandidateErrors[k] = GetPlacementErrorForTextureTransfer(
							sourceTexture, sourceCMap, targetTexture, targetCMap,previousIteration, quiltedTextureMap,
							new Rectangle(CandidatePoints[k], new Size(blockSize, blockSize)), new Point(quiltingLocationX, quiltingLocationY),
							transferAlpha);
					}

					long[] sorted = Sort(CandidateErrors);
					int selectFromBestNCandidateQuiltBlocks = 1;
					long sortedMaxError = (long)(sorted[0]*(1+candidateQuiltErrorTolerance));
					while (selectFromBestNCandidateQuiltBlocks < sorted.Length && sorted[selectFromBestNCandidateQuiltBlocks] < sortedMaxError) selectFromBestNCandidateQuiltBlocks++;
					long oneToSelect = sorted[r.Next(selectFromBestNCandidateQuiltBlocks)];
					int candidateIndex = 0;
					while (CandidateErrors[candidateIndex] != oneToSelect) candidateIndex++;
					//Util.Spew("Chose block " + candidateIndex + " which has error " + CandidateErrors[candidateIndex]);
					Application.DoEvents();
					Point selectedPoint = CandidatePoints[candidateIndex];

					int[] verticalCut = 
						GetMinErrorBoundaryCutVertical(sourceTexture, quiltedTextureMap, new Rectangle(selectedPoint, new Size(blockOverlap, blockSize)), new Point(quiltingLocationX, quiltingLocationY));
					int[] horizontalCut = 
						GetMinErrorBoundaryCutHorizontal(sourceTexture, quiltedTextureMap, new Rectangle(selectedPoint, new Size(blockSize, blockOverlap)), new Point(quiltingLocationX, quiltingLocationY));

					PasteSectionRespectingCutsAndBlendSeam(sourceTexture, quiltedTextureMap, new Rectangle(selectedPoint, new Size(blockSize, blockSize)), new Point(quiltingLocationX, quiltingLocationY), horizontalCut, verticalCut);

					Util.SetBitmapForViewing(quiltedTextureMap, 0);

					//System.Threading.Thread.Sleep(1000);

				}
			}
			Util.Spew("Texture transfer completed.");

			return quiltedTextureMap;
		}

		
		
		#endregion
	}
}
