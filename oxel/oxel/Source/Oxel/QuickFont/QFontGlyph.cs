using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;

namespace QuickFont
{
    public class QFontGlyph
    {
       
        /// <summary>
        /// Which texture page the glyph is on
        /// </summary>
        public int page; 
        
        /// <summary>
        /// The rectangle defining the glyphs position on the page
        /// </summary>
        public Rectangle rect;
        
        /// <summary>
        /// How far the glyph would need to be vertically offset to be vertically in line with the tallest glyph in the set of all glyphs
        /// </summary>
        public int yOffset;

        public QFontGlyph(int page, Rectangle rect, int yOffset)
        {
            this.page = page;
            this.rect = rect;
            this.yOffset = yOffset;
        }
    }
}
