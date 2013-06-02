using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace QuickFont
{

    /// <summary>
    /// What settings to use when building the font
    /// </summary>
    public class QFontBuilderConfiguration
    {
        /// <summary>
        /// Whether to use super sampling when building font texture pages
        /// 
        /// 
        /// </summary>
        public int SuperSampleLevels = 1;

        /// <summary>
        /// The standard width of texture pages (the page will
        /// automatically be cropped if there is extra space)
        /// </summary>
        public int PageWidth = 512;

        /// <summary>
        /// The standard height of texture pages (the page will
        /// automatically be cropped if there is extra space)
        /// </summary>
        public int PageHeight = 512;

        /// <summary>
        /// Whether to force texture pages to use a power of two.
        /// </summary>
        public bool ForcePowerOfTwo = true;

        /// <summary>
        /// The margin (on all sides) around glyphs when rendered to
        /// their texture page
        /// </summary>
        public int GlyphMargin = 2;
       
        /// <summary>
        /// Set of characters to support
        /// </summary>
        public string charSet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.:,;'\"(!?)+-*/=_{}[]@~#\\<>|^%$£&";
        
    }
}
