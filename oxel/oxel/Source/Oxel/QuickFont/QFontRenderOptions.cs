using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenTK;
using System.Drawing;
using OpenTK.Graphics;

namespace QuickFont
{


    public enum QFontAlignment { Left=0, Right, Centre, Justify }
    public enum QFontMonospacing { Natural = 0, Yes, No }

    public class QFontRenderOptions
    {
        /// <summary>
        /// The font colour
        /// </summary>
        public Color4 Colour = Color.FromArgb(255,255,255,255);

        /// <summary>
        /// Spacing between characters in units of average glyph width
        /// </summary>
        public float CharacterSpacing = 0.05f;

        /// <summary>
        /// Spacing between words in units of average glyph width
        /// </summary>
        public float WordSpacing = 0.9f;

        /// <summary>
        /// Line spacing in units of max glyph width
        /// </summary>
        public float LineSpacing = 1.0f;


        #region Justify Options

        /// <summary>
        /// When a line of text is justified, space may be inserted between
        /// characters, and between words. 
        /// 
        /// This parameter determines how this choice is weighted:
        /// 
        /// 0.0f => word spacing only
        /// 1.0f => "fairly" distributed between both
        /// > 1.0 => in favour of character spacing
        /// 
        /// This applies to expansions only.
        /// 
        /// </summary>
        public float JustifyCharacterWeightForExpand
        {
            get { return justifyCharWeightForExpand; }
            set { 

                justifyCharWeightForExpand = value;

                if (justifyCharWeightForExpand < 0f)
                    justifyCharWeightForExpand = 0f;
                else if (justifyCharWeightForExpand > 1.0f)
                    justifyCharWeightForExpand = 1.0f;
            }
        }

        private float justifyCharWeightForExpand = 0.5f;


        /// <summary>
        /// When a line of text is justified, space may be removed between
        /// characters, and between words. 
        /// 
        /// This parameter determines how this choice is weighted:
        /// 
        /// 0.0f => word spacing only
        /// 1.0f => "fairly" distributed between both
        /// > 1.0 => in favour of character spacing
        /// 
        /// This applies to contractions only.
        /// 
        /// </summary>
        public float JustifyCharacterWeightForContract
        {
            get { return justifyCharWeightForContract; }
            set
            {

                justifyCharWeightForContract = value;

                if (justifyCharWeightForContract < 0f)
                    justifyCharWeightForContract = 0f;
                else if (justifyCharWeightForContract > 1.0f)
                    justifyCharWeightForContract = 1.0f;
            }
        }

        private float justifyCharWeightForContract = 0.2f;



        /// <summary>
        /// Total justification cap as a fraction of the boundary width.
        /// </summary>
        public float JustifyCapExpand = 0.5f;


        /// <summary>
        /// Total justification cap as a fraction of the boundary width.
        /// </summary>
        public float JustifyCapContract = 0.1f;

        /// <summary>
        /// By what factor justification is penalized for being negative.
        /// 
        /// (e.g. if this is set to 3, then a contraction will only happen
        /// over an expansion if it is 3 of fewer times smaller than the
        /// expansion).
        /// 
        /// 
        /// </summary>
        public float JustifyContractionPenalty = 2;


        /// <summary>
        /// Whether to draw a drop-shadow. Note: this requires
        /// the QFont to have been loaded with a drop shadow to
        /// take effect.
        /// </summary>
        public bool DropShadowActive = false;

        /// <summary>
        /// Offset of the shadow from the font glyphs in units of average glyph width
        /// </summary>
        public Vector2 DropShadowOffset = new Vector2(0.16f, 0.16f);

        /// <summary>
        /// Opacity of drop shadows
        /// </summary>
        public float DropShadowOpacity = 0.5f;


        /// <summary>
        /// Whether to render the font in monospaced mode. If set to "Natural", then 
        /// monospacing will be used if the font loaded font was detected to be monospaced.
        /// </summary>
        public QFontMonospacing Monospacing = QFontMonospacing.Natural;



        #endregion

    }
}
