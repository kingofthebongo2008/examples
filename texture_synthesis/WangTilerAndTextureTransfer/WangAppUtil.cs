using System;
using System.Drawing;

namespace WangTileCreator
{
	/// <summary>
	/// Some Utility functions used by both applcations, and some debugging helpers.
	/// Robert Burke, rob@mle.media.mit.edu
	/// 10 Aug 2003
	/// 
	/// Caveat: this is all "Weekend Project" code.  So it's not necessarily the prettiest thing ever.
	/// Please accept my apologies in advance.
	/// </summary>
	public class Util
	{

		/// <summary>
		/// Error function determining the difference between two colors; using the L2 norm for now as in Efros and Freeman.
		/// </summary>
		public static long GetError(Color c1, Color c2)
		{
			if (c1.A == 0 || c2.A == 0) return 0;
			long errorR = ((long)c1.R - (long)c2.R) * ((long)c1.R - (long)c2.R);
			long errorG = ((long)c1.G - (long)c2.G) * ((long)c1.G - (long)c2.G);
			long errorB = ((long)c1.B - (long)c2.B) * ((long)c1.B - (long)c2.B);
			return errorR + errorG + errorB;

		}

		/// <summary>
		/// Blends two colors by taking their average R, G, and B values.  
		/// </summary>
		public static Color BlendColors(Color c1, Color c2)
		{
			return Color.FromArgb(
				((int)c1.R+(int)c2.R)/2,
				((int)c1.G+(int)c2.G)/2,
				((int)c1.B+(int)c2.B)/2
				);
		}

		#region Debug Spew and Refresh Display Delegates -- call RDD() to refresh the display or Spew("THIS!") to spew.
		
		public delegate void RefreshDisplayDelegate();
		public static RefreshDisplayDelegate RDD = null;

		public delegate void SpewDelegate (string s);
		public static SpewDelegate SD = null;

		public delegate void SetBitmapForViewingDelegate(Bitmap b, int viewWindow);
		public static SetBitmapForViewingDelegate SBV = null;

		public static void Spew(string s) { if (SD != null) SD(s); }
		public static void RefreshDisplay() { if (RDD != null) RDD(); }
		public static void SetBitmapForViewing(Bitmap b, int viewWindow) { if (SBV != null) SBV(b, viewWindow); }
		#endregion


	}
}
