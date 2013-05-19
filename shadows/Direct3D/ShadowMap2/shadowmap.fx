//-----------------------------------------------------------------------------
// File: ShadowMap.fx
//
// Desc: Effect file for high dynamic range cube mapping sample.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//-----------------------------------------------------------------------------


#define SMAP_SIZE 512


#define SHADOW_EPSILON 0.00005f

float4x4 g_mProj;
float4x4 g_mViewToLight;  // Transform from view space to light projection space
float4   g_vMaterial;
texture  g_txScene;
texture  g_txShadow;
float3   g_vLightPos;  // Light position in view space
float3   g_vLightDir;  // Light direction in view space
float4   g_vLightDiffuse = float4( 1.0f, 1.0f, 1.0f, 1.0f );  // Light diffuse color
float4   g_vLightAmbient = float4( 0.3f, 0.3f, 0.3f, 1.0f );  // Use an ambient light of 0.3
float    g_fCosTheta;  // Cosine of theta of the spot light
float	 g_Light_Space_Far_Z = 100.0f;
float	 g_Light_Space_Near_Z = 1.0f;
float4x4 g_mShadowProj;

float4x4 g_mWorldView;
float4x4 g_mWorldViewLeft;
float4x4 g_mWorldViewRight;

sampler2D g_samScene =
sampler_state
{
    Texture = <g_txScene>;
    MinFilter = Point;
    MagFilter = Linear;
    MipFilter = Linear;
};

sampler2D g_samShadow =
sampler_state
{
    Texture = <g_txShadow>;
    MinFilter = Point;
    MagFilter = Point;
    MipFilter = Point;
    AddressU = Clamp;
    AddressV = Clamp;
};

texture  g_frame_buffer_left;
texture  g_frame_buffer_right;

texture  g_depth_buffer_left;
texture  g_depth_buffer_right;

sampler2D g_sam_frame_buffer_left =
sampler_state
{
    Texture = <g_frame_buffer_left>;
    MinFilter = Point;
    MagFilter = Point;
    MipFilter = Point;
    AddressU = Clamp;
    AddressV = Clamp;
};

sampler2D g_sam_frame_buffer_right =
sampler_state
{
    Texture = <g_frame_buffer_right>;
    MinFilter = Point;
    MagFilter = Point;
    MipFilter = Point;
    AddressU = Clamp;
    AddressV = Clamp;
};

sampler2D g_sam_depth_buffer_left =
sampler_state
{
    Texture = <g_depth_buffer_left>;
    MinFilter = Point;
    MagFilter = Point;
    MipFilter = Point;
    AddressU = Clamp;
    AddressV = Clamp;
};

sampler2D g_sam_depth_buffer_right =
sampler_state
{
    Texture = <g_depth_buffer_right>;
    MinFilter = Point;
    MagFilter = Point;
    MipFilter = Point;
    AddressU = Clamp;
    AddressV = Clamp;
};

float linear_z(float z_ps, float z_vs, float w, float near, float far )
{
    return  ( ( z_vs - near ) / ( far - near ) ) * w ;
}

//-----------------------------------------------------------------------------
// Vertex Shader: VertScene
// Desc: Process vertex for scene
//-----------------------------------------------------------------------------
void VertScene( float4 iPos : POSITION,
                float3 iNormal : NORMAL,
                float2 iTex : TEXCOORD0,
                out float4 oPos : POSITION,
                out float2 Tex : TEXCOORD0,
                out float4 vPos : TEXCOORD1,
                out float3 vNormal : TEXCOORD2,
                out float4 vPosLight : TEXCOORD3 )
{
    //
    // Transform position to view space
    //
    vPos = mul( iPos, g_mWorldView );

    //
    // Transform to screen coord
    //
    oPos = mul( vPos, g_mProj );

    //
    // Compute view space normal
    //
    vNormal = mul( iNormal, (float3x3)g_mWorldView );

    //
    // Propagate texture coord
    //
    Tex = iTex;

    //
    // Transform the position to light projection space, or the
    // projection space as if the camera is looking out from
    // the spotlight.
    //
    vPosLight = mul( vPos, g_mViewToLight );

    float z_vs = vPosLight.z;

    vPosLight = mul( vPosLight, g_mShadowProj );

    vPosLight.z = linear_z (vPosLight.z, z_vs, vPosLight.w, g_Light_Space_Near_Z, g_Light_Space_Far_Z);
}




//-----------------------------------------------------------------------------
// Pixel Shader: PixScene
// Desc: Process pixel (do per-pixel lighting) for enabled scene
//-----------------------------------------------------------------------------
float4 PixScene( float2 Tex : TEXCOORD0,
                 float4 vPos : TEXCOORD1,
                 float3 vNormal : TEXCOORD2,
                 float4 vPosLight : TEXCOORD3 ) : COLOR
{
    float4 Diffuse;

    // vLight is the unit vector from the light to this pixel
    float3 vLight = normalize( float3( vPos - g_vLightPos ) );

    // Compute diffuse from the light
    if( dot( vLight, g_vLightDir ) > g_fCosTheta ) // Light must face the pixel (within Theta)
    {
        // Pixel is in lit area. Find out if it's
        // in shadow using 2x2 percentage closest filtering

        //transform from RT space to texture space.
        float2 ShadowTexC = 0.5 * vPosLight.xy / vPosLight.w + float2( 0.5, 0.5 );
        ShadowTexC.y = 1.0f - ShadowTexC.y;

        // transform to texel space
        float2 texelpos = SMAP_SIZE * ShadowTexC;
        
        // Determine the lerp amounts           
        float2 lerps = frac( texelpos );

        //read in bilerp stamp, doing the shadow checks
        float sourcevals[4];
        sourcevals[0] = (tex2D( g_samShadow, ShadowTexC ) + SHADOW_EPSILON < vPosLight.z / vPosLight.w)? 0.0f: 1.0f;  
        sourcevals[1] = (tex2D( g_samShadow, ShadowTexC + float2(1.0/SMAP_SIZE, 0) ) + SHADOW_EPSILON < vPosLight.z / vPosLight.w)? 0.0f: 1.0f;  
        sourcevals[2] = (tex2D( g_samShadow, ShadowTexC + float2(0, 1.0/SMAP_SIZE) ) + SHADOW_EPSILON < vPosLight.z / vPosLight.w)? 0.0f: 1.0f;  
        sourcevals[3] = (tex2D( g_samShadow, ShadowTexC + float2(1.0/SMAP_SIZE, 1.0/SMAP_SIZE) ) + SHADOW_EPSILON < vPosLight.z / vPosLight.w)? 0.0f: 1.0f;  
        
        // lerp between the shadow values to calculate our light amount
        float LightAmount = lerp( lerp( sourcevals[0], sourcevals[1], lerps.x ),
                                  lerp( sourcevals[2], sourcevals[3], lerps.x ),
                                  lerps.y );
        // Light it
        Diffuse = ( saturate( dot( -vLight, normalize( vNormal ) ) ) * LightAmount * ( 1 - g_vLightAmbient ) + g_vLightAmbient )
                  * g_vMaterial;
    } else
    {
        Diffuse = g_vLightAmbient * g_vMaterial;
    }

    return tex2D( g_samScene, Tex ) * Diffuse;
}




//-----------------------------------------------------------------------------
// Vertex Shader: VertLight
// Desc: Process vertex for the light object
//-----------------------------------------------------------------------------
void VertLight( float4 iPos : POSITION,
                float3 iNormal : NORMAL,
                float2 iTex : TEXCOORD0,
                out float4 oPos : POSITION,
                out float2 Tex : TEXCOORD0 )
{
    //
    // Transform position to view space
    //
    oPos = mul( iPos, g_mWorldView );

    //
    // Transform to screen coord
    //
    oPos = mul( oPos, g_mProj );

    //
    // Propagate texture coord
    //
    Tex = iTex;
}




//-----------------------------------------------------------------------------
// Pixel Shader: PixLight
// Desc: Process pixel for the light object
//-----------------------------------------------------------------------------
float4 PixLight( float2 Tex : TEXCOORD0,
                 float4 vPos : TEXCOORD1 ) : COLOR
{
    return tex2D( g_samScene, Tex );
}




//-----------------------------------------------------------------------------
// Vertex Shader: VertShadow
// Desc: Process vertex for the shadow map
//-----------------------------------------------------------------------------
void VertShadow( float4 Pos : POSITION,
                 float3 Normal : NORMAL,
                 out float4 oPos : POSITION,
                 out float2 Depth : TEXCOORD0 )
{
    //
    // Compute the projected coordinates
    //
    oPos = mul( Pos, g_mWorldView );
    
    float z_vs = oPos.z;

    oPos = mul( oPos, g_mProj );

    oPos.z = linear_z ( oPos.z, z_vs, oPos.w, g_Light_Space_Near_Z, g_Light_Space_Far_Z);

    //
    // Store z and w in our spare texcoord
    //
    Depth.xy = oPos.zw;
}


//-----------------------------------------------------------------------------
// Pixel Shader: PixShadow
// Desc: Process pixel for the shadow map
//-----------------------------------------------------------------------------
void PixShadow( float2 Depth : TEXCOORD0,
                out float4 Color : COLOR )
{
    //
    // Depth is z / w
    //
    Color = Depth.x / Depth.y;
}

//-----------------------------------------------------------------------------
// Vertex Shader: VertDepth
// Desc: Process vertex for scene
//-----------------------------------------------------------------------------
void VertDepthScene
( 
                float4 iPos : POSITION,
                out float4 oPos : POSITION,
                out float4 left_frame : TEXCOORD0,
                out float4 right_frame: TEXCOORD1,
                out float4 middle_frame: TEXCOORD2
)
{
    float4 pos;
    float4 pos_left;
    float4 pos_right;
    //
    // Transform position to view space
    //
    pos = mul( iPos, g_mWorldView );

    pos_left = mul( iPos, g_mWorldViewLeft );
    pos_right = mul( iPos, g_mWorldViewRight );

    //
    // Transform to screen coord
    //
    oPos = mul( pos, g_mProj );

    left_frame = mul( pos_left, g_mProj );
    right_frame = mul( pos_right, g_mProj );
}

//-----------------------------------------------------------------------------
// Pixel Shader: PixShadow
// Desc: Process pixel for the shadow map
//-----------------------------------------------------------------------------
void PixDepthScene( out float4 Color : COLOR, float4 left_frame : TEXCOORD0, float4 right_frame: TEXCOORD1, float4 middle_frame: TEXCOORD2 )
{
    Color = float4( 1, 0, 0, 0);

    left_frame /= left_frame.w;
    right_frame /= right_frame.w;
    middle_frame /= middle_frame.w;

    float left_frame_depth = tex2D( g_sam_depth_buffer_left, left_frame.xy * 0.5 + 0.5).r;
    float right_frame_depth = tex2D( g_sam_depth_buffer_right, right_frame.xy * 0.5 + 0.5).r;

    float4 left_frame_image = tex2D( g_sam_frame_buffer_right, left_frame.xy * 0.5 + 0.5).r;
    float4 right_frame_image = tex2D( g_sam_frame_buffer_right, right_frame.xy * 0.5 + 0.5).r;

    float l = abs ( left_frame_depth - left_frame.z );
    float r = abs ( right_frame_depth - right_frame.z );

    if ( l < 0.0001)
    {
        //both are visibile
        if ( r < 0.0001)
        {
            Color = ( left_frame_image + right_frame_image ) / 2;
        }
        else
        {
            //only left is visibile
            Color = left_frame_image;
        }
    }
    else
    {
        //only right is visible
        if ( r )
        {
            Color = right_frame_image;
        }
        else
        {
            //nothing is visible.
            Color =  l < r ? left_frame_image : right_frame_image;
        }
    }
}



//-----------------------------------------------------------------------------
// Technique: RenderScene
// Desc: Renders scene objects
//-----------------------------------------------------------------------------
technique RenderScene
{
    pass p0
    {
        CullMode = CCW;
        VertexShader = compile vs_3_0 VertScene();
        PixelShader = compile ps_3_0 PixScene();
    }
}




//-----------------------------------------------------------------------------
// Technique: RenderLight
// Desc: Renders the light object
//-----------------------------------------------------------------------------
technique RenderLight
{
    pass p0
    {
        VertexShader = compile vs_3_0 VertLight();
        PixelShader = compile ps_3_0 PixLight();
    }
}




//-----------------------------------------------------------------------------
// Technique: RenderShadow
// Desc: Renders the shadow map
//-----------------------------------------------------------------------------
technique RenderShadow
{
    pass p0
    {
        CullMode = CCW;
        VertexShader = compile vs_3_0 VertShadow();
        PixelShader = compile ps_3_0 PixShadow();
    }
}


//-----------------------------------------------------------------------------
// Technique: RenderScene
// Desc: Renders scene objects
//-----------------------------------------------------------------------------
technique RenderSceneDepth
{
    pass p0
    {
        VertexShader = compile vs_3_0 VertDepthScene();
        PixelShader = compile ps_3_0 PixDepthScene();
    }
}

