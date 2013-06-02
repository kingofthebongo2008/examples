using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Drawing.Text;
using System.Drawing.Imaging;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using OpenTK.Graphics;
using System.Runtime.InteropServices;

namespace QuickFont
{



    public class QFont
    {

        private QFontRenderOptions options = new QFontRenderOptions();
        internal QFontData fontData;
        
        public QFontRenderOptions Options
        {
            get { return options; }
            set { options = value; }
        }


        #region Constructors and font builders

        private QFont() { }
        internal QFont(QFontData fontData) { this.fontData = fontData; }
        public QFont(Font font) : this(font, null, null) { }
        public QFont(Font font, QFontBuilderConfiguration config) : this(font, config, null) { }
        public QFont(Font font, QFontShadowConfiguration shadowConfig) : this(font, null, shadowConfig) { }
        public QFont(Font font, QFontBuilderConfiguration config, QFontShadowConfiguration shadowConfig)
        {
            if (config == null)
                config = new QFontBuilderConfiguration();

            fontData = BuildFont(font, config, shadowConfig, null);

            if (shadowConfig != null)
                options.DropShadowActive = true;
        }

        public QFont(string fileName, float size) : this(fileName, size, FontStyle.Regular, null, null) { }
        public QFont(string fileName, float size, FontStyle style) : this(fileName, size, style, null, null) { }
        public QFont(string fileName, float size, QFontBuilderConfiguration config) : this(fileName, size, FontStyle.Regular, config, null) { }
        public QFont(string fileName, float size, QFontShadowConfiguration shadowConfig) : this(fileName, size, FontStyle.Regular, null, shadowConfig) { }
        public QFont(string fileName, float size, FontStyle style, QFontShadowConfiguration shadowConfig) : this(fileName, size, style, null, shadowConfig) { }
        public QFont(string fileName, float size, QFontBuilderConfiguration config, QFontShadowConfiguration shadowConfig) : this(fileName, size, FontStyle.Regular, config, shadowConfig) { }
        public QFont(string fileName, float size, FontStyle style, QFontBuilderConfiguration config, QFontShadowConfiguration shadowConfig)
        {
            PrivateFontCollection pfc = new PrivateFontCollection();
            pfc.AddFontFile(fileName);
            var fontFamily = pfc.Families[0];

            if (!fontFamily.IsStyleAvailable(style))
                throw new ArgumentException("Font file: " + fileName + " does not support style: " +  style );

            if (config == null)
                config = new QFontBuilderConfiguration();

            using (var font = new Font(fontFamily, size * config.SuperSampleLevels, style))
            {
                fontData = BuildFont(font, config, shadowConfig, null);
            }

            if (shadowConfig != null)
                options.DropShadowActive = true;
        }

        public QFont(string fontname, byte[] fontresource, float size) : this(fontname, fontresource, size, FontStyle.Regular, null, null) { }
        public QFont(string fontname, byte[] fontresource, float size, FontStyle style) : this(fontname, fontresource, size, style, (QFontBuilderConfiguration)null, (QFontShadowConfiguration)null) { }
        public QFont(string fontname, byte[] fontresource, float size, QFontBuilderConfiguration config) : this(fontname, fontresource, size, FontStyle.Regular, config, null) { }
        public QFont(string fontname, byte[] fontresource, float size, QFontShadowConfiguration shadowConfig) : this(fontname, fontresource, size, FontStyle.Regular, null, shadowConfig) { }
        public QFont(string fontname, byte[] fontresource, float size, FontStyle style, QFontShadowConfiguration shadowConfig) : this(fontname, fontresource, size, style, null, shadowConfig) { }
        public QFont(string fontname, byte[] fontresource, float size, QFontBuilderConfiguration config, QFontShadowConfiguration shadowConfig) : this(fontname, fontresource, size, FontStyle.Regular, config, shadowConfig) { }
        public QFont(string fontname, byte[] fontresource, float size, FontStyle style, QFontBuilderConfiguration config, QFontShadowConfiguration shadowConfig)
        {
            // This should be probably a field of some class
            PrivateFontCollection pfc = new PrivateFontCollection();

            // allocate memory and copy byte[] to the location
            IntPtr data = Marshal.AllocCoTaskMem(fontresource.Length);
            Marshal.Copy(fontresource, 0, data, fontresource.Length);

            // pass the font to the font collection
            pfc.AddMemoryFont(data, fontresource.Length);

            var fontFamily = pfc.Families[0];

            if (!fontFamily.IsStyleAvailable(style))
                throw new ArgumentException("Font Resource: " + fontname + " does not support style: " + style);

            if (config == null)
                config = new QFontBuilderConfiguration();

            using (var font = new Font(fontFamily, size * config.SuperSampleLevels, style))
            {
                fontData = BuildFont(font, config, shadowConfig, null);
            }

            if (shadowConfig != null)
                options.DropShadowActive = true;

            // Free the unsafe memory
            Marshal.FreeCoTaskMem(data);
        }


        public static void CreateTextureFontFiles(Font font, string newFontName) { CreateTextureFontFiles(font, null); }
        public static void CreateTextureFontFiles(Font font, string newFontName, QFontBuilderConfiguration config)
        {
            var fontData = BuildFont(font, config, null, newFontName);
            Builder.SaveQFontDataToFile(fontData, newFontName);
        }

        



        public static void CreateTextureFontFiles(string fileName, float size, string newFontName) { CreateTextureFontFiles(fileName, size, FontStyle.Regular, null, newFontName); }
        public static void CreateTextureFontFiles(string fileName, float size, FontStyle style, string newFontName) { CreateTextureFontFiles(fileName, size, style, null, newFontName); }
        public static void CreateTextureFontFiles(string fileName, float size, QFontBuilderConfiguration config, string newFontName) { CreateTextureFontFiles(fileName, size, FontStyle.Regular, config, newFontName); }
        public static void CreateTextureFontFiles(string fileName, float size, FontStyle style, QFontBuilderConfiguration config, string newFontName)
        {
            PrivateFontCollection pfc = new PrivateFontCollection();
            pfc.AddFontFile(fileName);
            var fontFamily = pfc.Families[0];

            if (!fontFamily.IsStyleAvailable(style))
                throw new ArgumentException("Font file: " + fileName + " does not support style: " + style);

            QFontData fontData = null;
            if (config == null)
                config = new QFontBuilderConfiguration();


            using(var font = new Font(fontFamily, size * config.SuperSampleLevels, style)){
                fontData  = BuildFont(font, config, null, newFontName);
            }

            Builder.SaveQFontDataToFile(fontData, newFontName);
            
        }

        public static QFont FromQFontFile(string filePath) { return FromQFontFile(filePath, 1.0f, null); }
        public static QFont FromQFontFile(string filePath, QFontShadowConfiguration shadowConfig) { return FromQFontFile(filePath, 1.0f, shadowConfig); }
        public static QFont FromQFontFile(string filePath, float downSampleFactor) { return FromQFontFile(filePath, downSampleFactor,null); } 
        public static QFont FromQFontFile(string filePath,float downSampleFactor, QFontShadowConfiguration shadowConfig)
        {            

            QFont qfont = new QFont();
            qfont.fontData = Builder.LoadQFontDataFromFile(filePath,downSampleFactor,shadowConfig);
            if (shadowConfig != null)
                qfont.options.DropShadowActive = true;
            
            return qfont;
        }
  
        private static QFontData BuildFont(Font font, QFontBuilderConfiguration config, QFontShadowConfiguration shadowConfig, string saveName){

            Builder builder = new Builder(font, config, shadowConfig);
            return builder.BuildFontData(saveName);
        }









        #endregion








        public float LineSpacing
        {
            get { return (float)Math.Ceiling(fontData.maxGlyphHeight * options.LineSpacing); }
        }

        public bool IsMonospacingActive
        {
            get { return fontData.IsMonospacingActive(Options); }
        }


        public float MonoSpaceWidth
        {
            get { return fontData.GetMonoSpaceWidth(Options); }
        }


        public void RenderGlyph(float x, float y, char c, bool isDropShadow)
        {


            var glyph = fontData.CharSetMapping[c];

            if (isDropShadow)
            {
                x -= (int)(glyph.rect.Width * 0.5f);
                y -= (int)(glyph.rect.Height * 0.5f + glyph.yOffset);
            }


            //note can cast drop shadow offset to int, but then you can't move the shadow smoothly...
            if(fontData.dropShadow != null && options.DropShadowActive)
                fontData.dropShadow.RenderGlyph(
                    x + (fontData.meanGlyphWidth * options.DropShadowOffset.X + glyph.rect.Width * 0.5f),
                    y + (fontData.meanGlyphWidth * options.DropShadowOffset.Y + glyph.rect.Height * 0.5f + glyph.yOffset), c, true);



            if (isDropShadow)
            {
                GL.Color4(1.0f, 1.0f, 1.0f, options.DropShadowOpacity);
            }
            else
            {
                GL.Color4(options.Colour);
            }


            TexturePage sheet = fontData.Pages[glyph.page];
            GL.BindTexture(TextureTarget.Texture2D, sheet.GLTexID);


            float tx1 = (float)(glyph.rect.X) / sheet.Width;
            float ty1 = (float)(glyph.rect.Y) / sheet.Height;
            float tx2 = (float)(glyph.rect.X + glyph.rect.Width) / sheet.Width;
            float ty2 = (float)(glyph.rect.Y + glyph.rect.Height) / sheet.Height;

            GL.Begin(BeginMode.Quads);
                GL.TexCoord2(tx1, ty1); GL.Vertex2(x, y + glyph.yOffset);
                GL.TexCoord2(tx1, ty2); GL.Vertex2(x, y + glyph.yOffset + glyph.rect.Height);
                GL.TexCoord2(tx2, ty2); GL.Vertex2(x + glyph.rect.Width, y + glyph.yOffset + glyph.rect.Height);
                GL.TexCoord2(tx2, ty1); GL.Vertex2(x + glyph.rect.Width, y + glyph.yOffset);
            GL.End();

            
        }


        


        private float MeasureNextlineLength(string text)
        {

            float xOffset = 0;
            
            for(int i=0; i < text.Length;i++)
            {
                char c = text[i];

                if (c == '\r' || c == '\n')
                {
                    break;
                }


                if (IsMonospacingActive)
                {
                    xOffset += MonoSpaceWidth;
                }
                else
                {
                    //space
                    if (c == ' ')
                    {
                        xOffset += (float)Math.Ceiling(fontData.meanGlyphWidth * options.WordSpacing);
                    }
                    //normal character
                    else if (fontData.CharSetMapping.ContainsKey(c))
                    {
                        QFontGlyph glyph = fontData.CharSetMapping[c];
                        xOffset += (float)Math.Ceiling(glyph.rect.Width + fontData.meanGlyphWidth * options.CharacterSpacing + fontData.GetKerningPairCorrection(i, text, null));
                    }
                }
            }
            return xOffset;
        }


        public void Print(string text)
        {
            Print(text, QFontAlignment.Left);
        }



        public void Print(string text, QFontAlignment alignment)
        {
            PrintOrMeasure(text, alignment, false);
        }


        public SizeF Measure(string text)
        {
            return Measure(text, QFontAlignment.Left);
        }

        public SizeF Measure(string text, QFontAlignment alignment)
        {
            return PrintOrMeasure(text, alignment, true);
        }


        private SizeF PrintOrMeasure(string text, QFontAlignment alignment, bool measureOnly)
        {

            float maxWidth = 0f; 

            GL.Color4(1.0f, 1.0f, 1.0f, 1.0f);
            GL.Enable(EnableCap.Texture2D);
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);

            float xOffset = 0f;
            float yOffset = 0f;

            text = text.Replace("\r\n", "\r");

            if (alignment == QFontAlignment.Right)
                xOffset -= MeasureNextlineLength(text);
            else if (alignment == QFontAlignment.Centre)
                xOffset -= (int)(0.5f * MeasureNextlineLength(text));

            for(int i = 0; i < text.Length; i++)
            {
                char c = text[i];


                //newline
                if (c == '\r' || c == '\n')
                {
                    yOffset += LineSpacing;
                    xOffset = 0f;

                    if (alignment == QFontAlignment.Right)
                        xOffset -= MeasureNextlineLength(text.Substring(i + 1));
                    else if (alignment == QFontAlignment.Centre)
                        xOffset -= (int)(0.5f * MeasureNextlineLength(text.Substring(i + 1)));

                }
                else
                {


                    //normal character
                    if (c != ' ' && fontData.CharSetMapping.ContainsKey(c))
                    {
                        QFontGlyph glyph = fontData.CharSetMapping[c];
                        if(!measureOnly)
                            RenderGlyph(xOffset, yOffset, c, false);
                    }


                    if (IsMonospacingActive)
                        xOffset += MonoSpaceWidth;
                    else
                    {
                        if (c == ' ')
                            xOffset += (float)Math.Ceiling(fontData.meanGlyphWidth * options.WordSpacing);
                        //normal character
                        else if (fontData.CharSetMapping.ContainsKey(c))
                        {
                            QFontGlyph glyph = fontData.CharSetMapping[c];
                            xOffset += (float)Math.Ceiling(glyph.rect.Width + fontData.meanGlyphWidth * options.CharacterSpacing + fontData.GetKerningPairCorrection(i, text, null));
                        }
                    }

                    maxWidth = Math.Max(xOffset, maxWidth);
                }

            }


            return new SizeF(maxWidth, yOffset + LineSpacing);


        }








        private void RenderWord(float x, float y, TextNode node)
        {

            if (node.Type != TextNodeType.Word)
                return;

            int charGaps = node.Text.Length - 1;
            bool isCrumbleWord = CrumbledWord(node);
            if (isCrumbleWord)
                charGaps++;

            int pixelsPerGap = 0;
            int leftOverPixels = 0;

            if (charGaps != 0)
            {
                pixelsPerGap = (int)node.LengthTweak / charGaps;
                leftOverPixels = (int)node.LengthTweak - pixelsPerGap * charGaps;
            }

            for(int i = 0; i < node.Text.Length; i++){
                char c = node.Text[i];
                if(fontData.CharSetMapping.ContainsKey(c)){
                    var glyph = fontData.CharSetMapping[c];

                    RenderGlyph(x,y,c, false);


                    if (IsMonospacingActive)
                        x += MonoSpaceWidth;
                    else
                        x += (int)Math.Ceiling(glyph.rect.Width + fontData.meanGlyphWidth * options.CharacterSpacing + fontData.GetKerningPairCorrection(i, node.Text, node));

                    x += pixelsPerGap;
                    if (leftOverPixels > 0)
                    {
                        x += 1.0f;
                        leftOverPixels--;
                    }
                    else if (leftOverPixels < 0)
                    {
                        x -= 1.0f;
                        leftOverPixels++;
                    }


                }
            }
        }






        /// <summary>
        /// Computes the length of the next line, and whether the line is valid for
        /// justification.
        /// </summary>
        /// <param name="node"></param>
        /// <param name="maxLength"></param>
        /// <param name="justifable"></param>
        /// <returns></returns>
        private float TextNodeLineLength(TextNode node, float maxLength)
        {

            if (node == null)
                return 0;

            bool atLeastOneNodeCosumedOnLine = false;
            float length = 0;
            for (; node != null; node = node.Next)
            {

                if (node.Type == TextNodeType.LineBreak)
                    break;

                if (SkipTrailingSpace(node, length, maxLength) && atLeastOneNodeCosumedOnLine)
                    break;

                if (length + node.Length <= maxLength || !atLeastOneNodeCosumedOnLine)
                {
                    atLeastOneNodeCosumedOnLine = true;
                    length += node.Length;
                }
                else
                {
                    break;
                }


            }
            return length;
        }


        private bool CrumbledWord(TextNode node)
        {
            return (node.Type == TextNodeType.Word && node.Next != null && node.Next.Type == TextNodeType.Word);  
        }


        /// <summary>
        /// Computes the length of the next line, and whether the line is valid for
        /// justification.
        /// </summary>
        private void JustifyLine(TextNode node, float targetLength)
        {
  
            bool justifiable = false;

            if (node == null)
                return;

            var headNode = node; //keep track of the head node


            //start by finding the length of the block of text that we know will actually fit:

            int charGaps = 0;
            int spaceGaps = 0;

            bool atLeastOneNodeCosumedOnLine = false;
            float length = 0;
            var expandEndNode = node; //the node at the end of the smaller list (before adding additional word)
            for (; node != null; node = node.Next)
            {

                

                if (node.Type == TextNodeType.LineBreak)
                    break;

                if (SkipTrailingSpace(node, length, targetLength) && atLeastOneNodeCosumedOnLine)
                {
                    justifiable = true;
                    break;
                }

                if (length + node.Length < targetLength || !atLeastOneNodeCosumedOnLine)
                {

                    expandEndNode = node;

                    if (node.Type == TextNodeType.Space)
                        spaceGaps++;

                    if (node.Type == TextNodeType.Word)
                    {
                        charGaps += (node.Text.Length - 1);

                        //word was part of a crumbled word, so there's an extra char cap between the two words
                        if (CrumbledWord(node))
                            charGaps++;

                    }

                    atLeastOneNodeCosumedOnLine = true;
                    length += node.Length;
                }
                else
                {
                    justifiable = true;
                    break;
                }

                

            }


            //now we check how much additional length is added by adding an additional word to the line
            float extraLength = 0f;
            int extraSpaceGaps = 0;
            int extraCharGaps = 0;
            bool contractPossible = false;
            TextNode contractEndNode = null;
            for (node = expandEndNode.Next; node != null; node = node.Next)
            {
                

                if (node.Type == TextNodeType.LineBreak)
                    break;

                if (node.Type == TextNodeType.Space)
                {
                    extraLength += node.Length;
                    extraSpaceGaps++;
                } 
                else if (node.Type == TextNodeType.Word)
                {
                    contractEndNode = node;
                    contractPossible = true;
                    extraLength += node.Length;
                    extraCharGaps += (node.Text.Length - 1);
                    break;
                }
            }



            if (justifiable)
            {

                //last part of this condition is to ensure that the full contraction is possible (it is all or nothing with contractions, since it looks really bad if we don't manage the full)
                bool contract = contractPossible && (extraLength + length - targetLength) * options.JustifyContractionPenalty < (targetLength - length) &&
                    ((targetLength - (length + extraLength + 1)) / targetLength > -options.JustifyCapContract); 

                if((!contract && length < targetLength) || (contract && length + extraLength > targetLength))  //calculate padding pixels per word and char
                {

                    if (contract)
                    {
                        length += extraLength + 1; 
                        charGaps += extraCharGaps;
                        spaceGaps += extraSpaceGaps;
                    }

                    

                    int totalPixels = (int)(targetLength - length); //the total number of pixels that need to be added to line to justify it
                    int spacePixels = 0; //number of pixels to spread out amongst spaces
                    int charPixels = 0; //number of pixels to spread out amongst char gaps





                    if (contract)
                    {

                        if (totalPixels / targetLength < -options.JustifyCapContract)
                            totalPixels = (int)(-options.JustifyCapContract * targetLength);
                    }
                    else
                    {
                        if (totalPixels / targetLength > options.JustifyCapExpand)
                            totalPixels = (int)(options.JustifyCapExpand * targetLength);
                    }


                    //work out how to spread pixles between character gaps and word spaces
                    if (charGaps == 0)
                    {
                        spacePixels = totalPixels;
                    }
                    else if (spaceGaps == 0)
                    {
                        charPixels = totalPixels;
                    }
                    else
                    {

                        if(contract)
                            charPixels = (int)(totalPixels * options.JustifyCharacterWeightForContract * charGaps / spaceGaps);
                        else 
                            charPixels = (int)(totalPixels * options.JustifyCharacterWeightForExpand * charGaps / spaceGaps);

         
                        if ((!contract && charPixels > totalPixels) ||
                            (contract && charPixels < totalPixels) )
                            charPixels = totalPixels;

                        spacePixels = totalPixels - charPixels;
                    }


                    int pixelsPerChar = 0;  //minimum number of pixels to add per char
                    int leftOverCharPixels = 0; //number of pixels remaining to only add for some chars

                    if (charGaps != 0)
                    {
                        pixelsPerChar = charPixels / charGaps;
                        leftOverCharPixels = charPixels - pixelsPerChar * charGaps;
                    }


                    int pixelsPerSpace = 0; //minimum number of pixels to add per space
                    int leftOverSpacePixels = 0; //number of pixels remaining to only add for some spaces

                    if (spaceGaps != 0)
                    {
                        pixelsPerSpace = spacePixels / spaceGaps;
                        leftOverSpacePixels = spacePixels - pixelsPerSpace * spaceGaps;
                    }

                    //now actually iterate over all nodes and set tweaked length
                    for (node = headNode; node != null; node = node.Next)
                    {

                        if (node.Type == TextNodeType.Space)
                        {
                            node.LengthTweak = pixelsPerSpace;
                            if (leftOverSpacePixels > 0)
                            {
                                node.LengthTweak += 1;
                                leftOverSpacePixels--;
                            }
                            else if (leftOverSpacePixels < 0)
                            {
                                node.LengthTweak -= 1;
                                leftOverSpacePixels++;
                            }


                        }
                        else if (node.Type == TextNodeType.Word)
                        {
                            int cGaps = (node.Text.Length - 1);
                            if (CrumbledWord(node))
                                cGaps++;

                            node.LengthTweak = cGaps * pixelsPerChar;


                            if (leftOverCharPixels >= cGaps)
                            {
                                node.LengthTweak += cGaps;
                                leftOverCharPixels -= cGaps;
                            }
                            else if (leftOverCharPixels <= -cGaps)
                            {
                                node.LengthTweak -= cGaps;
                                leftOverCharPixels += cGaps;
                            } 
                            else  
                            {
                                node.LengthTweak += leftOverCharPixels;
                                leftOverCharPixels = 0;
                            }
                        }

                        if ((!contract && node == expandEndNode) || (contract && node == contractEndNode))
                            break;

                    }

                }

            }


        }


        /// <summary>
        /// Checks whether to skip trailing space on line because the next word does not
        /// fit.
        /// 
        /// We only check one space - the assumption is that if there is more than one,
        /// it is a deliberate attempt to insert spaces.
        /// </summary>
        /// <param name="node"></param>
        /// <param name="lengthSoFar"></param>
        /// <param name="boundWidth"></param>
        /// <returns></returns>
        private bool SkipTrailingSpace(TextNode node, float lengthSoFar, float boundWidth)
        {

            if (node.Type == TextNodeType.Space && node.Next != null && node.Next.Type == TextNodeType.Word && node.ModifiedLength + node.Next.ModifiedLength + lengthSoFar > boundWidth)
            {
                return true;
            }

            return false;

        }





        /// <summary>
        /// Prints text inside the given bounds.
        /// </summary>
        /// <param name="text"></param>
        /// <param name="bounds"></param>
        /// <param name="alignment"></param>
        public void Print(string text, RectangleF bounds, QFontAlignment alignment)
        {
            var processedText = ProcessText(text, bounds, alignment);
            Print(processedText);
        }


        /// <summary>
        /// Measures the actual width and height of the block of text.
        /// </summary>
        /// <param name="text"></param>
        /// <param name="bounds"></param>
        /// <param name="alignment"></param>
        /// <returns></returns>
        public SizeF Measure(string text, RectangleF bounds, QFontAlignment alignment)
        {
            var processedText = ProcessText(text, bounds, alignment);
            return Measure(processedText);
        }

        /// <summary>
        /// Measures the actual width and height of the block of text
        /// </summary>
        /// <param name="processedText"></param>
        /// <returns></returns>
        public SizeF Measure(ProcessedText processedText)
        {
            return PrintOrMeasure(processedText, true);
        }


        /// <summary>
        /// Creates node list object associated with the text.
        /// </summary>
        /// <param name="text"></param>
        /// <param name="bounds"></param>
        /// <returns></returns>
        public ProcessedText ProcessText(string text, RectangleF bounds, QFontAlignment alignment)
        {
            //TODO: bring justify and alignment calculations in here

            var nodeList = new TextNodeList(text);
            nodeList.MeasureNodes(fontData, options);

            //we "crumble" words that are two long so that that can be split up
            var nodesToCrumble = new List<TextNode>();
            foreach (TextNode node in nodeList)
                if (node.Length >= bounds.Width && node.Type == TextNodeType.Word)
                    nodesToCrumble.Add(node);

            foreach (var node in nodesToCrumble)
                nodeList.Crumble(node, 1);

            //need to measure crumbled words
            nodeList.MeasureNodes(fontData, options);


            var processedText = new ProcessedText();
            processedText.textNodeList = nodeList;
            processedText.bounds = bounds;
            processedText.alignment = alignment;


            return processedText;
        }




        /// <summary>
        /// Prints text as previously processed with a boundary and alignment.
        /// </summary>
        /// <param name="processedText"></param>
        public void Print(ProcessedText processedText)
        {
            PrintOrMeasure(processedText, false);
        }



        private SizeF PrintOrMeasure(ProcessedText processedText, bool measureOnly)
        {
            float maxWidth = 0f;

            if (!measureOnly)
            {
                GL.Color4(1.0f, 1.0f, 1.0f, 1.0f);
                GL.Enable(EnableCap.Texture2D);
                GL.Enable(EnableCap.Blend);
                GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);
            }


            var bounds = processedText.bounds;
            var alignment = processedText.alignment;

            float xOffset = bounds.X;
            float yOffset = bounds.Y;

            var nodeList = processedText.textNodeList;
            for (TextNode node = nodeList.Head; node != null; node = node.Next)
                node.LengthTweak = 0f;  //reset tweaks


            if (alignment == QFontAlignment.Right)
                xOffset -= (float)Math.Ceiling(TextNodeLineLength(nodeList.Head, bounds.Width) - bounds.Width);
            else if (alignment == QFontAlignment.Centre)
                xOffset -= (float)Math.Ceiling(0.5f * TextNodeLineLength(nodeList.Head, bounds.Width) - bounds.Width * 0.5f);
            else if (alignment == QFontAlignment.Justify)
                JustifyLine(nodeList.Head, bounds.Width);




            bool atLeastOneNodeCosumedOnLine = false;
            float length = 0f;
            for (TextNode node = nodeList.Head; node != null; node = node.Next)
            {
                bool newLine = false;

                if (node.Type == TextNodeType.LineBreak)
                {
                    newLine = true;
                }
                else
                {

                    if (SkipTrailingSpace(node, length, bounds.Width) && atLeastOneNodeCosumedOnLine)
                    {
                        newLine = true;
                    }
                    else if (length + node.ModifiedLength <= bounds.Width || !atLeastOneNodeCosumedOnLine)
                    {
                        atLeastOneNodeCosumedOnLine = true;
                        if(!measureOnly)
                            RenderWord(xOffset + length, yOffset, node);
                        length += node.ModifiedLength;

                        maxWidth = Math.Max(length, maxWidth);

                    }
                    else
                    {
                        newLine = true;
                        if (node.Previous != null)
                            node = node.Previous;
                    }

                }

                if (newLine)
                {

                    yOffset += LineSpacing;
                    xOffset = bounds.X;
                    length = 0f;
                    atLeastOneNodeCosumedOnLine = false;

                    if (node.Next != null)
                    {
                        if (alignment == QFontAlignment.Right)
                            xOffset -= (float)Math.Ceiling(TextNodeLineLength(node.Next, bounds.Width) - bounds.Width);
                        else if (alignment == QFontAlignment.Centre)
                            xOffset -= (float)Math.Ceiling(0.5f * TextNodeLineLength(node.Next, bounds.Width) - bounds.Width * 0.5f);
                        else if (alignment == QFontAlignment.Justify)
                            JustifyLine(node.Next, bounds.Width);
                    }
                }

            }


            return new SizeF(maxWidth, yOffset + LineSpacing - bounds.Y);

        }


        /*
        public void Begin()
        {
            ProjectionStack.Begin();
        }

        public void End()
        {
            ProjectionStack.End();
        }*/


        public static void Begin()
        {
            ProjectionStack.Begin();
        }

        public static void End()
        {
            ProjectionStack.End();
        }




    }
}
