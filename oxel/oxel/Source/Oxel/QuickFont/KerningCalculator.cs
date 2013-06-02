using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;

namespace QuickFont
{

    class KerningCalculator
    {

        private struct XLimits
        {
            public int Min;
            public int Max;
        }

        private static int Kerning(QFontGlyph g1, QFontGlyph g2, XLimits[] lim1, XLimits[] lim2)
        {
            int yOffset1 = g1.yOffset;
            int yOffset2 = g2.yOffset;

            int startY = Math.Max(yOffset1, yOffset2);
            int endY = Math.Min(g1.rect.Height + yOffset1, g2.rect.Height + yOffset2);

            int w1 = g1.rect.Width;

            int worstCase = w1;

            //TODO - offset startY, endY by yOffset1 so that lim1[j-yOffset1] can be written as lim1[j], will need another var for yOffset2

            for (int j = startY; j < endY; j++)
                worstCase = Math.Min(worstCase, w1 - lim1[j-yOffset1].Max + lim2[j-yOffset2].Min);


            worstCase = Math.Min(worstCase, g1.rect.Width);
            worstCase = Math.Min(worstCase, g2.rect.Width);

            return worstCase;
        }

        public static Dictionary<String, int> CalculateKerning(char[] charSet, QFontGlyph[] glyphs, List<QBitmap> bitmapPages)
        {
            var kerningPairs = new Dictionary<String, int>();



            //we start by computing the index of the first and last non-empty pixel in each row of each glyph
            XLimits[][] limits = new XLimits[charSet.Length][];
            int maxHeight = 0;
            for (int n = 0; n < charSet.Length; n++)
            {
                var rect = glyphs[n].rect;
                var page = bitmapPages[glyphs[n].page];

                limits[n] = new XLimits[rect.Height];

                maxHeight = Math.Max(rect.Height, maxHeight);

                int yStart = rect.Y;
                int yEnd = rect.Y + rect.Height;
                int xStart = rect.X;
                int xEnd = rect.X + rect.Width;

                for (int j = yStart; j < yEnd; j++)
                {
                    int last = xStart;

                    bool yetToFindFirst = true;
                    for (int i = xStart; i < xEnd; i++)
                    {
                        if (!QBitmap.EmptyAlphaPixel(page.bitmapData, i, j))
                        {

                            if (yetToFindFirst)
                            {
                                limits[n][j - yStart].Min = i - xStart;
                                yetToFindFirst = false;
                            }
                            last = i;
                        }
                    }

                    limits[n][j - yStart].Max = last - xStart;

                    if (yetToFindFirst)
                        limits[n][j - yStart].Min = xEnd - 1;
                }
            }


            //we now bring up each row to the max (or min) of it's two adjacent rows, this is to stop glyphs sliding together too closely
            var tmp = new XLimits[maxHeight];

            for (int n = 0; n < charSet.Length; n++)
            {
                //clear tmp 
                for (int j = 0; j < limits[n].Length; j++)
                    tmp[j] = limits[n][j];

                for (int j = 0; j < limits[n].Length; j++)
                {
                    if(j != 0){
                        tmp[j].Min = Math.Min(limits[n][j - 1].Min, tmp[j].Min);
                        tmp[j].Max = Math.Max(limits[n][j - 1].Max, tmp[j].Max);
                    }

                    if (j != limits[n].Length - 1)
                    {
                        tmp[j].Min = Math.Min(limits[n][j + 1].Min, tmp[j].Min);
                        tmp[j].Max = Math.Max(limits[n][j + 1].Max, tmp[j].Max);
                    }
                    
                }

                for (int j = 0; j < limits[n].Length; j++)
                    limits[n][j] = tmp[j];

            }

            for (int i = 0; i < charSet.Length; i++)
                for (int j = 0; j < charSet.Length; j++)
                    kerningPairs.Add("" + charSet[i] + charSet[j], 1-Kerning(glyphs[i], glyphs[j], limits[i], limits[j]));

            return kerningPairs;
        }

    }
}
