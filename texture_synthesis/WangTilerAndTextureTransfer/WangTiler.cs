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
	/// Guts of the Wang Tiling application.
	/// Robert Burke, rob@mle.ie
	/// 10 Aug 2003
	/// 
	/// Caveat: this is all "Weekend Project" code.  So it's not necessarily the prettiest thing ever.
	/// Please accept my apologies in advance.
	/// </summary>
	public class WangTiler
	{
		#region Wang Tiling - Cohen et. al. Siggraph 2003 implementation

		/// <summary>
		/// A class which encapsulates a Wang Tile specification.  Look at the constructor to see what it works with:
		/// the set of tiles (Describing the corner and side information), the set of source bitmaps,
		/// and whether or not the tiles to be generated should have a one-pixel overlap in the pixel data.
		/// Contains the WangTileWithBitmapInfo helper class as well.
		/// </summary>
		class WangTileSpecification
		{
			public WangTileSpecification(WangTile[] tiles, Bitmap[] sourceBitmaps, bool onePixelOverlap)
			{
				WangTilesInfo = new WangTileWithBitmapInfo[tiles.Length];
				SourceBitmaps = sourceBitmaps;
				OnePixelOverlap = onePixelOverlap;
				for (int i = 0; i < tiles.Length; i++) WangTilesInfo[i] = new WangTileWithBitmapInfo(tiles[i]);
			}
			public Bitmap[] SourceBitmaps;
			public Rectangle[,] SamplePortions; // sample portions, per bitmap, as in, [bitmap, portion]
			WangTileWithBitmapInfo[] WangTilesInfo;
			public Size WangTileSize;
			bool OnePixelOverlap;

			public WangTile[] WangTiles
			{
				get
				{
					WangTile[] wt = new WangTile[WangTilesInfo.Length];
					for (int i = 0; i < WangTilesInfo.Length; i++) wt[i] = WangTilesInfo[i].WangTile;
					return wt;
				}
			}

			class WangTileWithBitmapInfo
			{
				public WangTileWithBitmapInfo(WangTile wt) { WangTile = wt; }
				public WangTile WangTile;
				public int[] ScanLineNW = null;
				public long MatchingErrorNW = -1;
				public int[] ScanLineNE = null;
				public long MatchingErrorNE = -1;
				public int[] ScanLineSE = null;
				public long MatchingErrorSE = -1;
				public int[] ScanLineSW = null;
				public long MatchingErrorSW = -1;
				public long TotalMatchingError { get { return MatchingErrorNW + MatchingErrorNE + MatchingErrorSE + MatchingErrorSW; } }
			}


			/// <param name="b1">The Bitmap with the region on the left of the scanline</param>
			/// <param name="b2">The Bitmap with the region on the right of the scanline</param>
			private void IncorporateRegionIntoBitmapRespectingScanLine(Bitmap b1, Bitmap b2, Rectangle b1Region, Point b2Region, Bitmap dest, Point destRegion, int[] scanLine)
			{
				for (int i = 0; i < b1Region.Width; i++)
				{
					for (int j = 0; j < b1Region.Height; j++)
					{
						if (i > scanLine[j])
						{
							dest.SetPixel(destRegion.X+i, destRegion.Y+j, b2.GetPixel(b2Region.X+i, b2Region.Y+j));
						}
						else
						{
							dest.SetPixel(destRegion.X+i, destRegion.Y+j, b1.GetPixel(b1Region.X+i, b1Region.Y+j));
						}
					}
				}

			}

			public void SynthesizeBitmaps()
			{
				int buff = this.OnePixelOverlap ? 1 : 0;
				int halfWangTileWidth = WangTileSize.Width/2;
				foreach(WangTileWithBitmapInfo wtbi in WangTilesInfo)
				{
					/// TODO: IF you wanted to blend multiple bitmaps, for example to introduce inhomogeneity, you'd need to first
					/// have computed blended north, west, south, east blend cuts for each possible combination of tiles,
					/// for each possible combination of corners.
					Bitmap newBitmap = new Bitmap(WangTileSize.Width, WangTileSize.Height);
					Rectangle northSample = SamplePortions[0, wtbi.WangTile.ColorN];
					Rectangle eastSample = SamplePortions[0, wtbi.WangTile.ColorE];
					Rectangle southSample = SamplePortions[0, wtbi.WangTile.ColorS];
					Rectangle westSample = SamplePortions[0, wtbi.WangTile.ColorW];
					Point northSampleCenter = new Point(northSample.Left + (northSample.Width/2), northSample.Top + (northSample.Height/2));
					Point eastSampleCenter = new Point(eastSample.Left + (eastSample.Width/2), eastSample.Top + (eastSample.Height/2));
					Point southSampleCenter = new Point(southSample.Left + (southSample.Width/2), southSample.Top + (southSample.Height/2));
					Point westSampleCenter = new Point(westSample.Left + (westSample.Width/2), westSample.Top + (westSample.Height/2));

					IncorporateRegionIntoBitmapRespectingScanLine(SourceBitmaps[0], SourceBitmaps[0],
						new Rectangle(new Point(westSampleCenter.X-buff, westSampleCenter.Y-halfWangTileWidth), new Size(halfWangTileWidth, halfWangTileWidth)), 
						new Point(northSampleCenter.X-halfWangTileWidth, northSampleCenter.Y-buff), 
						newBitmap, new Point(0,0), wtbi.ScanLineNW);
					IncorporateRegionIntoBitmapRespectingScanLine(SourceBitmaps[0], SourceBitmaps[0],
						new Rectangle(new Point(northSampleCenter.X, northSampleCenter.Y-buff), new Size(halfWangTileWidth, halfWangTileWidth)), 
						new Point(eastSampleCenter.X-halfWangTileWidth+buff, eastSampleCenter.Y-halfWangTileWidth), 
						newBitmap, new Point(halfWangTileWidth,0), wtbi.ScanLineNE);
					IncorporateRegionIntoBitmapRespectingScanLine(SourceBitmaps[0], SourceBitmaps[0],
						new Rectangle(new Point(southSampleCenter.X, southSampleCenter.Y-halfWangTileWidth+buff), new Size(halfWangTileWidth, halfWangTileWidth)), // south tile, right side
						new Point(eastSampleCenter.X-halfWangTileWidth+buff, eastSampleCenter.Y), // east tile, top side
						newBitmap, new Point(halfWangTileWidth,halfWangTileWidth), wtbi.ScanLineSE);
					IncorporateRegionIntoBitmapRespectingScanLine(SourceBitmaps[0], SourceBitmaps[0],
						new Rectangle(new Point(westSampleCenter.X-buff, westSampleCenter.Y), new Size(halfWangTileWidth, halfWangTileWidth)), // south tile, left side
						new Point(southSampleCenter.X-halfWangTileWidth, southSampleCenter.Y-halfWangTileWidth+buff), 
						newBitmap, new Point(0,halfWangTileWidth), wtbi.ScanLineSW);
					wtbi.WangTile.WangBitmap = newBitmap;
				}
			}

			public void RecalcScanLines()
			{
				int buff = this.OnePixelOverlap ? 1 : 0;

				int halfWangTileWidth = WangTileSize.Width/2;
				foreach(WangTileWithBitmapInfo wtbi in WangTilesInfo)
				{
					/// TODO: IF you wanted to blend multiple bitmaps, for example to introduce inhomogeneity, you'd need to first
					/// have computed blended north, west, south, east blend cuts for each possible combination of tiles,
					/// for each possible combination of corners.
					Rectangle northSample = SamplePortions[0, wtbi.WangTile.ColorN];
					Rectangle eastSample = SamplePortions[0, wtbi.WangTile.ColorE];
					Rectangle southSample = SamplePortions[0, wtbi.WangTile.ColorS];
					Rectangle westSample = SamplePortions[0, wtbi.WangTile.ColorW];
					Point northSampleCenter = new Point(northSample.Left + (northSample.Width/2), northSample.Top + (northSample.Height/2));
					Point eastSampleCenter = new Point(eastSample.Left + (eastSample.Width/2), eastSample.Top + (eastSample.Height/2));
					Point southSampleCenter = new Point(southSample.Left + (southSample.Width/2), southSample.Top + (southSample.Height/2));
					Point westSampleCenter = new Point(westSample.Left + (westSample.Width/2), westSample.Top + (westSample.Height/2));
					wtbi.ScanLineNW = WangTiler.GetMinErrorBoundaryCutNWToSEDiagonal(SourceBitmaps[0], SourceBitmaps[0],
						new Rectangle(new Point(northSampleCenter.X-halfWangTileWidth, northSampleCenter.Y-buff), new Size(halfWangTileWidth, halfWangTileWidth)), // north tile, left side
						new Point(westSampleCenter.X-buff, westSampleCenter.Y-halfWangTileWidth), // west tile, top side
						out wtbi.MatchingErrorNW);
					wtbi.ScanLineNE = WangTiler.GetMinErrorBoundaryCutNEToSWDiagonal(SourceBitmaps[0], SourceBitmaps[0],
						new Rectangle(new Point(northSampleCenter.X, northSampleCenter.Y-buff), new Size(halfWangTileWidth, halfWangTileWidth)), // north tile, right side
						new Point(eastSampleCenter.X-halfWangTileWidth-buff, eastSampleCenter.Y-halfWangTileWidth), // east tile, top side
						out wtbi.MatchingErrorNE);
					wtbi.ScanLineSE = WangTiler.GetMinErrorBoundaryCutNWToSEDiagonal(SourceBitmaps[0], SourceBitmaps[0],
						new Rectangle(new Point(southSampleCenter.X, southSampleCenter.Y-halfWangTileWidth+buff), new Size(halfWangTileWidth, halfWangTileWidth)), // south tile, right side
						new Point(eastSampleCenter.X-halfWangTileWidth+buff, eastSampleCenter.Y), // east tile, top side
						out wtbi.MatchingErrorSE);
					wtbi.ScanLineSW = WangTiler.GetMinErrorBoundaryCutNEToSWDiagonal(SourceBitmaps[0], SourceBitmaps[0],
						new Rectangle(new Point(southSampleCenter.X-halfWangTileWidth, southSampleCenter.Y-halfWangTileWidth+buff), new Size(halfWangTileWidth, halfWangTileWidth)), // south tile, left side
						new Point(westSampleCenter.X-buff, westSampleCenter.Y), // west tile, top side
						out wtbi.MatchingErrorSW);

				}
			}

			private static Random r = new Random();

			/// <summary>
			/// Returns the total matching error of this set of Wang Tiles.
			/// </summary>
			public long MatchingError 
			{ 
				get 
				{ 
					long totalMatchingError = 0;
					foreach (WangTileWithBitmapInfo wtbi in  WangTilesInfo) totalMatchingError += wtbi.TotalMatchingError;
					return totalMatchingError;
				} 
			}
		}

		/// <summary>
		/// Uses dynamic programming to determine the ideal path through the region diagonally, from the NE corner to the SW.
		/// The int[] array returned is the scan line containing the left-most point at each vertical position.  Thus its
		/// final element is always a 0.
		/// </summary>
		/// <param name="b1Region">Region in the 'from' Bitmap to use</param>
		/// <param name="b2Region">Region in the 'to' Bitmap to use</param>
		private static int[] GetMinErrorBoundaryCutNEToSWDiagonal(Bitmap b1, Bitmap b2, Rectangle b1Region, Point b2Region, out long totalError)
		{
			long[,] dpMat = new long[b1Region.Height, b1Region.Width]; // NOTE: This is in (row, col) format

			int row, col;

			dpMat[b1Region.Height-1, 0] = Util.GetError(
				b1.GetPixel(b1Region.X, b1Region.Y+b1Region.Height-1),
				b2.GetPixel(b2Region.X, b2Region.Y+b1Region.Height-1)
				);
			// do the bottom
			for (int i = 1; i < b1Region.Width; i++)
			{
				dpMat[b1Region.Height-1, i] = dpMat[b1Region.Height-1, i-1] + Util.GetError(
					b1.GetPixel(b1Region.X+i, b1Region.Y+b1Region.Height-1),
					b2.GetPixel(b2Region.X+i, b2Region.Y+b1Region.Height-1)
					);
			}
			// do the left
			for (int i = b1Region.Height-2; i >= 0; i--)
			{
				dpMat[i, 0] = dpMat[i+1, 0] + Util.GetError(
					b1.GetPixel(b1Region.X, b1Region.Y+i),
					b2.GetPixel(b2Region.X, b2Region.Y+i)
					);
			}
			// now do the rest
			for (row = b1Region.Height-2; row >= 0; row--)
			{
				for (col = 1; col < b1Region.Width; col++)
				{
					long prevMinError = Math.Min(Math.Min(dpMat[row, col-1], dpMat[row+1, col-1]), dpMat[row+1, col]);
					dpMat[row, col] = prevMinError + Util.GetError(
						b1.GetPixel(b1Region.X+col, b1Region.Y+row),
						b2.GetPixel(b2Region.X+col, b2Region.Y+row)
						);
				}
			}
			// Record the total error
			totalError = dpMat[0, b1Region.Width-1];


			// Now you need to trace back and produce a scanline.
			int[] scanLine = new int[b1Region.Height];
			col = b1Region.Width-1;
			row = 0;
			while (row < b1Region.Height-1)
			{
				if (col == 0)
				{
					scanLine[row] = 0; row++;
				}
				else
				{
					long minError = Math.Min(Math.Min(dpMat[row, col-1], dpMat[row+1, col-1]), dpMat[row+1, col]);
					if (minError == dpMat[row, col-1])
					{
						col--;
					}
					else
					{
						scanLine[row] = col;
						if (minError == dpMat[row+1, col-1]) col--;
						row++;
					}
				}
			}
			scanLine[b1Region.Height-1] = 0;
			return scanLine;
		}



		/// <summary>
		/// Uses dynamic programming to determine the ideal path through the region diagonally, from the NW corner to the SE.
		/// The int[] array returned is the scan line containing the left-most point at each vertical position.  Thus its
		/// final element is always a 0.
		/// </summary>
		/// <param name="b1Region">Region in the 'from' Bitmap to use</param>
		/// <param name="b2Region">Region in the 'to' Bitmap to use</param>
		private static int[] GetMinErrorBoundaryCutNWToSEDiagonal(Bitmap b1, Bitmap b2, Rectangle b1Region, Point b2Region, out long totalError)
		{
			long[,] dpMat = new long[b1Region.Height, b1Region.Width]; // NOTE: This is in (row, col) format

			int row, col;

			dpMat[b1Region.Height-1, b1Region.Width-1] = Util.GetError(
				b1.GetPixel(b1Region.X+b1Region.Width-1, b1Region.Y+b1Region.Height-1),
				b2.GetPixel(b2Region.X+b1Region.Width-1, b2Region.Y+b1Region.Height-1)
				);
			// do the bottom
			for (int i = b1Region.Width-2; i >= 0; i--)
			{
				dpMat[b1Region.Height-1, i] = dpMat[b1Region.Height-1, i+1] + Util.GetError(
					b1.GetPixel(b1Region.X+i, b1Region.Y+b1Region.Height-1),
					b2.GetPixel(b2Region.X+i, b2Region.Y+b1Region.Height-1)
					);
			}
			// do the right
			for (int i = b1Region.Height-2; i >= 0; i--)
			{
				dpMat[i, b1Region.Width-1] = dpMat[i+1, b1Region.Width-1] + Util.GetError(
					b1.GetPixel(b1Region.X+b1Region.Width-1, b1Region.Y+i),
					b2.GetPixel(b2Region.X+b1Region.Width-1, b2Region.Y+i)
					);
			}
			// now do the rest
			for (row = b1Region.Height-2; row >= 0; row--)
			{
				for (col = b1Region.Width - 2; col >= 0; col--)
				{
					long prevMinError = Math.Min(Math.Min(dpMat[row, col+1], dpMat[row+1, col+1]), dpMat[row+1, col]);
					dpMat[row, col] = prevMinError + Util.GetError(
						b1.GetPixel(b1Region.X+col, b1Region.Y+row),
						b2.GetPixel(b2Region.X+col, b2Region.Y+row)
						);
				}
			}
			// Record the total error
			totalError = dpMat[0, 0];


			// Now you need to trace back and produce a scanline.
			int[] scanLine = new int[b1Region.Height];
			int nextEntry = 0;
			col = 0;
			row = 0;
			while (row < b1Region.Height-1)
			{
				if (col == b1Region.Width-1)
				{
					scanLine[row+1] = b1Region.Width-1; 
					nextEntry = b1Region.Width-1; row++;
				}
				else
				{
					long minError = Math.Min(Math.Min(dpMat[row, col+1], dpMat[row+1, col+1]), dpMat[row+1, col]);
					if (minError == dpMat[row, col+1])
					{
						col++; 
					}
					else
					{
						if (minError == dpMat[row+1, col+1]) col++;
						scanLine[row+1] = col;
						nextEntry = col;
						row++;
					}
				}
			}
			scanLine[b1Region.Height-1] = nextEntry;
			return scanLine;
		}



		
		public static WangTile[] WangTile(WangTile[] wangTiles, Size wangTileSize, 
			Bitmap[] sourceBitmaps, int sourceBitmapBorderWidth, 
			int maxAttempts, 
			long maximumMatchingError,
			bool onePixelOverlapBetweenTiles)
		{

			Size sampleTileSize = new Size(wangTileSize.Width*3/2 + (onePixelOverlapBetweenTiles ? 1 : 0), 
				wangTileSize.Height*3/2 + (onePixelOverlapBetweenTiles ? 1 : 0));

			// determine the number of Sample Portions you'll need
			int numSamplePortions = 0;
			foreach (WangTile wt in wangTiles) numSamplePortions = Math.Max(numSamplePortions, wt.MaxColorNumber);
			numSamplePortions++;

			int numAttempts = 0;
			int numSourceBitmaps = sourceBitmaps.Length;

			WangTileSpecification bestSpecification = null;
			// while you haven't found an appropriate wang tiling..
			while (numAttempts < maxAttempts && (bestSpecification == null || bestSpecification.MatchingError > maximumMatchingError) )
			{
				Util.Spew("Generating Wang Tiles, attempt " + (numAttempts+1) + " of a maximum " + maxAttempts+"...");
				Application.DoEvents();

				WangTileSpecification wts = new WangTileSpecification(wangTiles, sourceBitmaps, onePixelOverlapBetweenTiles);
				numAttempts ++;

				wts.WangTileSize = wangTileSize;
				wts.SamplePortions = new Rectangle[numSourceBitmaps, numSamplePortions];
				// Randomly select new sample portions from each bitmap.
				for (int bmp = 0; bmp < numSourceBitmaps; bmp++)
				{
					int maxX = sourceBitmaps[bmp].Width - sampleTileSize.Width - (2*sourceBitmapBorderWidth);
					int maxY = sourceBitmaps[bmp].Height - sampleTileSize.Height - (2*sourceBitmapBorderWidth);
					for (int i = 0; i < numSamplePortions; i++)
					{
						wts.SamplePortions[bmp, i] = new Rectangle(new Point(sourceBitmapBorderWidth+r.Next(maxX), sourceBitmapBorderWidth+r.Next(maxY)), sampleTileSize);
					}
				}

				wts.RecalcScanLines();
				if (bestSpecification == null || wts.MatchingError < bestSpecification.MatchingError)
				{
					bestSpecification = wts;
				}
			}

			Util.Spew("Synthesizing bitmaps for best Wang Tile specification...");
			Application.DoEvents();

			bestSpecification.SynthesizeBitmaps();

			Util.Spew("Finished generating Wang Tiles.");
			Application.DoEvents();


			return bestSpecification.WangTiles;
		
		}


		
		private static Bitmap AssembleMegaTile(Bitmap[] bitmaps)
		{
			int len = bitmaps.Length;
			int numAcross = (int)Math.Ceiling(Math.Sqrt(len));
		
			int width = bitmaps[0].Width;
			int height = bitmaps[0].Height;
	
			Bitmap finalBitmap = new Bitmap(width*numAcross, height*numAcross);
			int currentBitmap = 0;
			for (int j = 0; j < numAcross; j++)
			{
				for (int i = 0; i < numAcross; i++, currentBitmap++)
				{
					if (currentBitmap < bitmaps.Length)
					{
						Bitmap b = bitmaps[currentBitmap];

						int top = height*j;
						int left = width*i;

						for (int k1 = 0; k1 < width; k1++)
						{
							for (int k2 = 0; k2 < height; k2++)
							{
								finalBitmap.SetPixel(left+k1, top+k2, b.GetPixel(k1, k2));
							}
						}
					}
				}
			}
			return finalBitmap;
		}
		public static void SaveMegaTile(Bitmap[] bitmaps, string fileName)
		{
			Bitmap fila = AssembleMegaTile(bitmaps);
			fila.Save(fileName, System.Drawing.Imaging.ImageFormat.Bmp);
		}

		private static Random r = new Random();

		#endregion

	}


	/// <summary>
	/// Representation of a Wang Tile.  
	/// </summary>
	public class WangTile
	{
		public WangTile(int tileNumber, int colorN, int colorE, int colorS, int colorW,
			int cornerNW, int cornerNE, int cornerSW, int cornerSE) 
		{ 
			TileNumber = tileNumber; ColorN = colorN; ColorE = colorE; ColorS = colorS; ColorW = colorW; 
			CornerNW = cornerNW; CornerNE = cornerNE; CornerSW = cornerSW; CornerSE = cornerSE;
		}
		public int TileNumber;
		public int ColorN;
		public int ColorE;
		public int ColorS;
		public int ColorW;
		public int CornerNW;
		public int CornerNE;
		public int CornerSW;
		public int CornerSE;
		public Bitmap WangBitmap = null;

		public int MaxColorNumber { get { return Math.Max(Math.Max(Math.Max(ColorN, ColorE), ColorS), ColorW); } }

		/// <summary>
		/// Returns a ArrayList of Wang Tiles matching the desired requirements from the list of wangTiles passed in.
		/// -1 can be used for any parameter to indicate "don't care."
		/// </summary>
		public static ArrayList WangTilesMatching(WangTile[] wangTiles, int colorN, int colorE, int colorS, int colorW, int cornerNW, int cornerNE, int cornerSW, int cornerSE)
		{
			ArrayList possibles = new ArrayList();
			foreach (WangTile wt in wangTiles)
			{
				bool ok = true;
				ok = ok && (colorN == wt.ColorN || colorN < 0);
				ok = ok && (colorS == wt.ColorS || colorS < 0);
				ok = ok && (colorE == wt.ColorE || colorE < 0);
				ok = ok && (colorW == wt.ColorW || colorW < 0);
				ok = ok && (cornerNE == wt.CornerSW || cornerNE < 0);
				ok = ok && (cornerNW == wt.CornerSE || cornerNW < 0);
				ok = ok && (cornerSE == wt.CornerNW || cornerSE < 0);
				ok = ok && (cornerSW == wt.CornerNE || cornerSW < 0);
				if (ok) possibles.Add(wt);
			}
			return possibles;
		}
		public static WangTile SelectWangTileMatching(WangTile[] wangTiles, int colorN, int colorE, int colorS, int colorW, int cornerNW, int cornerNE, int cornerSE, int cornerSW)
		{
			ArrayList a = WangTilesMatching(wangTiles, colorN, colorE, colorS, colorW, cornerNW, cornerNE, cornerSE, cornerSW);
			if (a.Count == 0) 
			{
				Console.WriteLine("Warning: no wang tile found matching request: N " + colorN + " E " + colorE + " S " + colorS + " W " + colorW+ " cornerNW " + cornerNW + " cornerNE " + cornerNE + " cornerSE " + cornerSE + " cornerSW " + cornerSW + "!!!");
				return null; 
			}
			else return a[r.Next(a.Count)] as WangTile;
		}
		/// <summary>
		/// Loads from a file where the wang tiles are listed sequentially, four characters indicating the N, E, S, W colours without
		/// spaces, separated by newlines; (presumably followed by the NW, NE, SE, SW corner information but we've ignored this here).  For example:
		/// 0121
		/// 3201
		/// etc.
		/// </summary>
		public static WangTile[] LoadDescriptionsFromFile(string fileName)
		{
			StreamReader sr = File.OpenText(fileName);
			return LoadDescriptionsFromFile(sr);
		}

		public static WangTile[] LoadDescriptionsFromFile(StreamReader sr)
		{

			ArrayList a = new ArrayList();
			int count = 0;
			try
			{
				while (true)
				{
					string s = sr.ReadLine();
					WangTile wt = new WangTile(
						count, 
						Int32.Parse(s.Substring(0,1)),
						Int32.Parse(s.Substring(1,1)),
						Int32.Parse(s.Substring(2,1)),
						Int32.Parse(s.Substring(3,1)),
						-1, 
						-1,
						-1,
						-1
						);
					count++;
					a.Add(wt);
				}
			}
			catch (Exception) { sr.Close(); }
			WangTile[] tiles = new WangTile[a.Count];
			for(int i = 0; i < a.Count; i++) tiles[i] = a[i] as WangTile;
			
			return tiles;
		}

		private static Random r = new Random();
	}



}
