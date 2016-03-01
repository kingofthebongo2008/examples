//--------------------------------------------------------------------------------------
// TiledTexturePackTool.cpp
//
// The entry point for the tiled texture pack tool.  This file contains command line
// option parsing code and code that generates a plasma fractal.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "stdafx.h"

//--------------------------------------------------------------------------------------
// Name: DumpTiledTextureFile
// Desc: Gathers information about the given tiled texture and outputs the info to
//       the console.
//--------------------------------------------------------------------------------------
VOID DumpTiledTextureFile( TILEDFILE_HANDLE Handle )
{
    UINT ArrayCount = GetArraySize( Handle );
    UINT LevelCount = GetLevelCount( Handle );
    UINT SubresourceCount = ArrayCount * LevelCount;

    UINT TotalPageCount = 0;

    printf_s( "File 0x%08x, %d array slices, %d mip maps.\n", Handle, ArrayCount, LevelCount );

    for( UINT Array = 0; Array < ArrayCount; ++Array )
    {
        for( UINT Level = 0; Level < LevelCount; ++Level )
        {
            TILEDFILE_LEVEL_DESC LevelDesc;
            GetLevelDesc( Handle, Level, &LevelDesc );

            UINT Subresource = ComputeSubresourceIndex( Handle, Array, Level );

            UINT PagesPopulated = 0;
            for( UINT PageY = 0; PageY < LevelDesc.HeightPages; ++PageY )
            {
                for( UINT PageX = 0; PageX < LevelDesc.WidthPages; ++PageX )
                {
                    VOID* pBuffer = NULL;
                    GetPageData( Handle, Subresource, PageX, PageY, FALSE, &pBuffer );
                    if( pBuffer != NULL )
                    {
                        ++PagesPopulated;
                    }
                }
            }

            printf_s( "Array %d Level %d: %d x %d texels\n\t%d x %d pages, %d x %d page granularity\n\tUsed %d of %d pages\n", 
                Array, 
                Level, 
                LevelDesc.WidthTexels, 
                LevelDesc.HeightTexels, 
                LevelDesc.WidthPages, 
                LevelDesc.HeightPages, 
                LevelDesc.WidthHeightPageBlock,
                LevelDesc.WidthHeightPageBlock,
                PagesPopulated, 
                LevelDesc.WidthPages * LevelDesc.HeightPages );

            TotalPageCount += PagesPopulated;
        }
    }

    printf_s( "%d pages total, %d KB.\n", TotalPageCount, TotalPageCount * 64 );
}


//--------------------------------------------------------------------------------------
// Name: FetchTexel
// Desc: Fetches a single texel from a tiled texture.
//--------------------------------------------------------------------------------------
inline UINT FetchTexel( TILEDFILE_HANDLE hSF, INT X, INT Y, INT TotalWidth, INT TotalHeight )
{
    X = ( X + TotalWidth ) % TotalWidth;
    Y = ( Y + TotalHeight ) % TotalHeight;

    VOID* pTexel = NULL;
    GetTexel( hSF, 0, (UINT)X, (UINT)Y, TRUE, &pTexel );

    USHORT Value = *(USHORT*)pTexel;

    return (UINT)Value;
}


//--------------------------------------------------------------------------------------
// Name: WriteTexel
// Desc: Writes a single texel to a tiled texture.
//--------------------------------------------------------------------------------------
inline VOID WriteTexel( TILEDFILE_HANDLE hSF, INT X, INT Y, UINT Value, INT TotalWidth, INT TotalHeight )
{
    X = ( X + TotalWidth ) % TotalWidth;
    Y = ( Y + TotalHeight ) % TotalHeight;

    VOID* pTexel = NULL;
    GetTexel( hSF, 0, (UINT)X, (UINT)Y, TRUE, &pTexel );

    *(USHORT*)pTexel = (USHORT)Value;
}

//--------------------------------------------------------------------------------------
// Name: BlendAndOffsetTexel
// Desc: Given four corner texel values, generates a center value that is the average of
//       the values, with a random offset proportional to the RectDistance parameter.
//--------------------------------------------------------------------------------------
inline UINT BlendAndOffsetTexel( UINT A, UINT B, UINT C, UINT D, UINT RectDistance, UINT MaxDistance )
{
    //UINT Sum = ( A & 0xFF ) + ( B & 0xFF ) + ( C & 0xFF ) + ( D & 0xFF );
    UINT Sum = A + B + C + D;
    Sum >>= 2;

    INT OffsetMax = ( (INT)RectDistance * 131072 ) / (INT)MaxDistance;

    INT Offset = 0;
    if( OffsetMax > 0 )
    {
        Offset = rand() % OffsetMax;
        Offset -= OffsetMax / 2;
    }

    Sum = min( 65535, max( 0, (INT)Sum + Offset ) );

    //UINT Value = 0xFF000000 | ( Sum << 16 ) | ( Sum << 8 ) | (Sum);
    UINT Value = Sum;
    return Value;
}

//--------------------------------------------------------------------------------------
// Name: ExecuteDiamond2
// Desc: Executes one step of the diamond plasma fractal algorithm.  If the input rect
//       is too small to recurse further, the method returns FALSE.
//--------------------------------------------------------------------------------------
BOOL ExecuteDiamond2( TILEDFILE_HANDLE hSF, INT X1, INT Y1, INT X2, INT Y2, INT TotalWidth, INT TotalHeight )
{
    if( X2 <= X1 + 1 )
    {
        return FALSE;
    }

    if( Y2 <= Y1 + 1 )
    {
        return FALSE;
    }

    INT CenterX = ( X1 + X2 ) / 2;
    INT CenterY = ( Y1 + Y2 ) / 2;

    const UINT MaxEdgeLength = max( TotalWidth, TotalHeight );
    const UINT EdgeLength = max( ( X2 - X1 ), ( Y2 - Y1 ) );

    UINT CornerA = FetchTexel( hSF, X1, Y1, TotalWidth, TotalHeight );
    UINT CornerB = FetchTexel( hSF, X2, Y1, TotalWidth, TotalHeight );
    UINT CornerC = FetchTexel( hSF, X1, Y2, TotalWidth, TotalHeight );
    UINT CornerD = FetchTexel( hSF, X2, Y2, TotalWidth, TotalHeight );

    UINT Center = BlendAndOffsetTexel( CornerA, CornerB, CornerC, CornerD, EdgeLength, MaxEdgeLength );
    if( CenterX == TotalWidth / 2 && CenterY == TotalHeight / 2 )
    {
        Center = 65535;
    }
    WriteTexel( hSF, CenterX, CenterY, Center, TotalWidth, TotalHeight );

    UINT TopEdge = BlendAndOffsetTexel( CornerA, CornerB, CornerA, CornerB, EdgeLength, MaxEdgeLength );
    if( Y1 == 0 )
    {
        TopEdge = 0;
    }
    WriteTexel( hSF, CenterX, Y1, TopEdge, TotalWidth, TotalHeight );

    UINT LeftEdge = BlendAndOffsetTexel( CornerA, CornerC, CornerA, CornerC, EdgeLength, MaxEdgeLength );
    if( X1 == 0 )
    {
        LeftEdge = 0;
    }
    WriteTexel( hSF, X1, CenterY, LeftEdge, TotalWidth, TotalHeight );

    return TRUE;
}

//--------------------------------------------------------------------------------------
// Name: RunPlasmaDiamond
// Desc: Recursively executes the diamond plasma fractal algorithm.
//--------------------------------------------------------------------------------------
VOID RunPlasmaDiamond( TILEDFILE_HANDLE hSF, INT X1, INT Y1, INT X2, INT Y2, INT TotalWidth, INT TotalHeight )
{
    BOOL Result = ExecuteDiamond2( hSF, X1, Y1, X2, Y2, TotalWidth, TotalHeight );

    if( Result )
    {
        INT CenterX = ( X1 + X2 ) / 2;
        INT CenterY = ( Y1 + Y2 ) / 2;
        RunPlasmaDiamond( hSF, CenterX, CenterY, X2, Y2, TotalWidth, TotalHeight );
        RunPlasmaDiamond( hSF, CenterX, Y1, X2, CenterY, TotalWidth, TotalHeight );
        RunPlasmaDiamond( hSF, X1, CenterY, CenterX, Y2, TotalWidth, TotalHeight );
        RunPlasmaDiamond( hSF, X1, Y1, CenterX, CenterY, TotalWidth, TotalHeight );
    }
}

//--------------------------------------------------------------------------------------
// Name: GeneratePlasmaFractal
// Desc: Fills a tiled texture with a plasma fractal.
//--------------------------------------------------------------------------------------
VOID GeneratePlasmaFractal( TILEDFILE_HANDLE hSF, INT Width, INT Height )
{
    RunPlasmaDiamond( hSF, 0, 0, Width, Height, Width, Height );
}


//--------------------------------------------------------------------------------------
// Name: LinearLerp
// Desc: Computes an integer lerp between the output values, given an input value and
//       input value bounds.
//--------------------------------------------------------------------------------------
inline INT LinearLerp( INT Value, INT SmallValue, INT LargeValue, INT SmallOutput, INT LargeOutput )
{
    INT DeltaOutput = LargeOutput - SmallOutput;
    INT DeltaRange = LargeValue - SmallValue;

    INT Output = ( ( Value - SmallValue ) * DeltaOutput ) / DeltaRange;
    if( DeltaOutput < 0 )
    {
        Output = max( DeltaOutput, min( Output, 0 ) );
    }
    else
    {
        Output = max( 0, min( Output, DeltaOutput ) );
    }
    Output += SmallOutput;

    return Output;
}


//--------------------------------------------------------------------------------------
// Name: DarkenColor
// Desc: Computes a 32-bit color value that is up to 12.5% darker than the input color.
//--------------------------------------------------------------------------------------
inline UINT DarkenColor( UINT Color )
{
    BYTE* pColors = (BYTE*)&Color;
    UINT Multiply = 255 - ( rand() % 32 );
    pColors[0] = ( pColors[0] * Multiply ) / 255;
    pColors[1] = ( pColors[1] * Multiply ) / 255;
    pColors[2] = ( pColors[2] * Multiply ) / 255;
    return Color;
}

//--------------------------------------------------------------------------------------
// Name: GenerateMountainColorMap
// Desc: Given an input heightmap as a tiled texture, generates an output diffuse color
//       map as a tiled texture.
//--------------------------------------------------------------------------------------
VOID GenerateMountainColorMap( TILEDFILE_HANDLE hDest, TILEDFILE_HANDLE hSrc )
{
    TILEDFILE_LEVEL_DESC DestLevelDesc;
    TILEDFILE_LEVEL_DESC SrcLevelDesc;
    GetLevelDesc( hDest, 0, &DestLevelDesc );
    GetLevelDesc( hSrc, 0, &SrcLevelDesc );

    assert( SrcLevelDesc.WidthTexels == DestLevelDesc.WidthTexels && SrcLevelDesc.HeightTexels == DestLevelDesc.HeightTexels );

    const UINT RockColor = 0xFF505050;

    for( UINT y = 0; y < DestLevelDesc.HeightTexels; ++y )
    {
        for( UINT x = 0; x < DestLevelDesc.WidthTexels; ++x )
        {
            USHORT* pSrcTexel = NULL;
            GetTexel( hSrc, 0, x, y, FALSE, (VOID**)&pSrcTexel );
            if( pSrcTexel == NULL )
            {
                continue;
            }

            USHORT* pRightTexel = NULL;
            USHORT* pBottomTexel = NULL;
            GetTexel( hSrc, 0, x + 1, y, FALSE, (VOID**)&pRightTexel );
            GetTexel( hSrc, 0, x, y + 1, FALSE, (VOID**)&pBottomTexel );

            INT XDelta = 0;
            if( pRightTexel != NULL )
            {
                XDelta = (INT)*pSrcTexel - (INT)*pRightTexel;
            }
            INT YDelta = 0;
            if( pBottomTexel != NULL )
            {
                YDelta = (INT)*pSrcTexel - (INT)*pBottomTexel;
            }

            INT DeltaHeight = max( abs( XDelta ), abs( YDelta ) );
            INT TexSize = (INT)max( DestLevelDesc.WidthTexels, DestLevelDesc.HeightTexels );

            INT Slope = ( DeltaHeight * TexSize ) / 1024;

            UINT BaseColor = 0xFF000000;
            BOOL Darken = TRUE;

            const USHORT Height = *pSrcTexel;
            if( Height >= 15000 )
            {
                // snow and rock level

                BaseColor = 0xFFFFFFFF;
                INT SnowThreshold = LinearLerp( (INT)Height, 15000, 35000, 256, 1024 );
                if( Slope >= SnowThreshold )
                {
                    BaseColor = RockColor;
                }
                else
                {
                    Darken = FALSE;
                }
            }
            else if( Height >= 2000 )
            {
                // rocks and trees
                BaseColor = 0xFF004000;
                INT RockThreshold = LinearLerp( (INT)Height, 4000, 15000, 512, 0 );
                if( Slope >= RockThreshold )
                {
                    BaseColor = RockColor;
                }
            }
            else if( Height >= 500 )
            {
                // grass and trees
                BaseColor = 0xFF00C000;
                INT RockThreshold = LinearLerp( (INT)Height, 500, 2000, 512, 0 );
                if( Slope >= RockThreshold )
                {
                    BaseColor = 0xFF004000;
                }
            }
            else
            {
                // dirt
                BaseColor = 0xFF402000;
            }

            if( Darken )
            {
                BaseColor = DarkenColor( BaseColor );
            }

            UINT* pDestTexel = NULL;
            GetTexel( hDest, 0, x, y, TRUE, (VOID**)&pDestTexel );
            assert( pDestTexel != NULL );
            *pDestTexel = BaseColor;
        }
    }
}


//--------------------------------------------------------------------------------------
// Name: struct AppOptions
// Desc: Contains settings that can be toggled by command line options to the executable.
//--------------------------------------------------------------------------------------
struct AppOptions
{
    UINT TextureWidth;
    UINT TextureHeight;
    BOOL EndianSwap;
};


//--------------------------------------------------------------------------------------
// Name: PrintHelp
// Desc: Prints a help message to stdout.
//--------------------------------------------------------------------------------------
VOID PrintHelp()
{
    printf_s(
        "TiledTexturePackTool [options]\n"
        "where options are:\n"
        "\t-width [128-16384]       texture width in texels\n"
        "\t-height [128-16384]      texture height in texels\n"
        "\t-endianswap [ppc|intel]  endian swap output for ppc (big endian) or intel (little endian)\n"
        "\n" );
}


//--------------------------------------------------------------------------------------
// Name: ProcessCmdLine
// Desc: Gathers options from the given command line arguments, and initializes settings
//       in the Options struct.
//--------------------------------------------------------------------------------------
UINT ProcessCmdLine( INT argc, _TCHAR* argv[], AppOptions& Options )
{
    if( argc < 2 )
    {
        return 0;
    }

    UINT CommandError = 0;
    INT ArgIndex = 1;
    while( ArgIndex < argc && CommandError == 0 )
    {
        UINT ProcessedArguments = 0;
        const TCHAR* pArg = argv[ArgIndex];
        if( pArg[0] == L'-' || pArg[0] == L'/' )
        {
            const TCHAR* pCommand = pArg + 1;

            const TCHAR* pArgument = NULL;
            if( ArgIndex < ( argc - 1 ) )
            {
                pArgument = argv[ArgIndex + 1];
            }

            if( _tcsicmp( pCommand, TEXT("?") ) == 0 ||
                _tcsicmp( pCommand, TEXT("help") ) == 0 )
            {
                PrintHelp();
                return -1;
            }
            else if( _tcsicmp( pCommand, TEXT("width") ) == 0 )
            {
                if( pArgument == NULL )
                {
                    CommandError = ArgIndex;
                    break;
                }
                Options.TextureWidth = _ttoi( pArgument );
                if( Options.TextureWidth < 128 || Options.TextureWidth > 16384 )
                {
                    CommandError = ArgIndex;
                }
                ProcessedArguments = 1;
            }
            else if( _tcsicmp( pCommand, TEXT("height") ) == 0 )
            {
                if( pArgument == NULL )
                {
                    CommandError = ArgIndex;
                    break;
                }
                Options.TextureHeight = _ttoi( pArgument );
                if( Options.TextureHeight < 128 || Options.TextureHeight > 16384 )
                {
                    CommandError = ArgIndex;
                }
                ProcessedArguments = 1;
            }
            else if( _tcsicmp( pCommand, TEXT("endianswap") ) == 0 )
            {
                if( pArgument == NULL )
                {
                    CommandError = ArgIndex;
                    break;
                }
                if( _tcsicmp( pArgument, TEXT("ppc") ) == 0 )
                {
                    Options.EndianSwap = TRUE;
                }
                else
                {
                    Options.EndianSwap = FALSE;
                }
                ProcessedArguments = 1;
            }
        }
        else
        {
            // must have a command preceding each argument
            CommandError = ArgIndex;
        }

        ArgIndex += ( 1 + ProcessedArguments );
    }

    return CommandError;
}


//--------------------------------------------------------------------------------------
// Name: _tmain
// Desc: Entry point to the executable.
//--------------------------------------------------------------------------------------
int _tmain(int argc, _TCHAR* argv[])
{
    AppOptions Options;
    Options.TextureWidth = 16384;
    Options.TextureHeight = 16384;
    Options.EndianSwap = TRUE;

    UINT Result = ProcessCmdLine( argc, argv, Options );
    if( Result != 0 )
    {
        if( Result == (UINT)-1 )
        {
            return 0;
        }

        if( Result < (UINT)argc )
        {
            printf_s( "Error with command line option: %S\n", argv[Result] );
        }

#ifdef _DEBUG
        _getch();
#endif

        return -1;
    }

    printf_s( "Creating terrain files; width %d height %d, %s endian output.\n", Options.TextureWidth, Options.TextureHeight, Options.EndianSwap ? "big" : "little" );

    GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR gdiplusToken;
    GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

    TILEDFILE_HANDLE hSF = TILEDFILE_INVALID_HANDLE_VALUE;

    hSF = CreateTiledTextureFile( TILED_FORMAT_16BPP_R16, Options.TextureWidth, Options.TextureHeight, 1, 0 );
    printf_s( "Tiled Texture file created with handle %08x\n", hSF );

    {
        BYTE* pDefaultPage = new BYTE[PAGE_SIZE_BYTES];
        memset( pDefaultPage, 0x00, PAGE_SIZE_BYTES );
        SetDefaultPageData( hSF, pDefaultPage );
        delete[] pDefaultPage;
    }

    TILEDFILE_LEVEL_DESC BaseLevelDesc;
    GetLevelDesc( hSF, 0, &BaseLevelDesc );

    srand( 12345678 );

    printf_s( "Generating plasma fractal height map...\n" );
    GeneratePlasmaFractal( hSF, BaseLevelDesc.WidthTexels, BaseLevelDesc.HeightTexels );

    TILEDFILE_HANDLE hNormalMap = CreateTiledTextureFile( TILED_FORMAT_16BPP_R8G8, Options.TextureWidth, Options.TextureHeight, 1, 0 );

    printf_s( "Generating normal map...\n" );
    GenerateNormalMap( hNormalMap, 0, hSF, 0 );

    TILEDFILE_HANDLE hColorMap = CreateTiledTextureFile( TILED_FORMAT_32BPP_R8G8B8A8, Options.TextureWidth, Options.TextureHeight, 1, 0 );
    printf_s( "Generating color map...\n" );
    GenerateMountainColorMap( hColorMap, hSF );

    UINT MipLevelCount = GetLevelCount( hSF );
    for( UINT i = 1; i < MipLevelCount; ++i )
    {
        printf_s( "Generating height map mip level %d...\n", i );
        GenerateMipData( hSF, i );
    }

    DumpTiledTextureFile( hSF );

    printf_s( "Saving file...\n" );
    SaveTiledTextureFile( hSF, Options.EndianSwap, "s_heightmap.sp" );
    printf_s( "File saved.\n" );

    DeleteTiledTextureFile( hSF );

    MipLevelCount = GetLevelCount( hNormalMap );
    
    for( UINT i = 1; i < MipLevelCount; ++i )
    {
        printf_s( "Generating normal map mip level %d...\n", i );
        GenerateMipData( hNormalMap, i );
    }

    DumpTiledTextureFile( hNormalMap );

    printf_s( "Saving file...\n" );
    SaveTiledTextureFile( hNormalMap, Options.EndianSwap, "s_normalmap.sp" );
    printf_s( "File saved.\n" );

    DeleteTiledTextureFile( hNormalMap );

    MipLevelCount = GetLevelCount( hColorMap );

    for( UINT i = 1; i < MipLevelCount; ++i )
    {
        printf_s( "Generating color map mip level %d...\n", i );
        GenerateMipData( hColorMap, i );
    }

    DumpTiledTextureFile( hColorMap );

    printf_s( "Saving file...\n" );
    SaveTiledTextureFile( hColorMap, Options.EndianSwap, "s_diffuse.sp" );
    printf_s( "File saved.\n" );

    DeleteTiledTextureFile( hColorMap );

#ifdef _DEBUG
    _getch();
#endif

    GdiplusShutdown(gdiplusToken);

	return 0;
}

