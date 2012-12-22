//-----------------------------------------------------------------------------
// File: D3D10\Global Illumination\App.h
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------





#ifndef _APP_H_
#define _APP_H_

#define FRAMEWORK_VERSION 1
#include "../Framework/Version.h"

#include "../Framework/D3D10/D3D10App.h"
#include "../Framework/Model/Model.h"

// Pass indices
#define SHADOW_PASS 0
#define PROBE_PASS  2
#define FINAL_PASS  7

#define LIGHTING_TECH 12

// Size of the probe grid
#define SIZE_X 11
#define SIZE_Y 4
#define SIZE_Z 7

// Scale of the grid in each direction
#define SCALE_X 1.7f
#define SCALE_Y 2.0f
#define SCALE_Z 1.85f

#define SCALE (1.0 / float3(SCALE_X * SIZE_X, SCALE_Y * SIZE_Y, SCALE_Z * SIZE_Z))

#define MAX_PROBE_COUNT (SIZE_X * SIZE_Y * SIZE_Z)
#define PROBE_SLICES (PROBE_COUNT * 6)
#define PROBE_CUBES_PER_PASS 5
#define PROBE_SLICES_PER_PASS (PROBE_CUBES_PER_PASS * 6)

#define PROBE_PASSES_PER_FRAME 8
#define PROBE_CUBES_PER_FRAME (PROBE_PASSES_PER_FRAME * PROBE_CUBES_PER_PASS)

#define PROBE_SIZE 8

#define SHADOWMAP_SIZE 768

// Number of SH bands to use. If this is changed you'll
// have to reconfigure the texture locations in the shader.
#define SH_BANDS 4
#define SH_COEFFS (SH_BANDS * SH_BANDS)
#define SH_COEFF_VECTORS (SH_COEFFS / 4)

#if ((SH_BANDS & 0x1) != 0)
#error "SH_BANDS must be even so that SH_COEFF_VECTORS is an integer"
#endif

// Size of the transfer function SH coefficient cubemap
#define SH_TF_CUBE_SIZE 32

// Number of wall materials
#define WALL_MATERIAL_COUNT 4

// Constant buffers divided by frequency of update. They match the cbuffer declaration in Lighting.fx
struct PerFrame
{
	float4x4 mvp;
	float3 lightPos;
	float intensity;
	float3 camPos;
	float exposure;
	float4 lmModulate[2];
};

struct PerObject
{
	float3 diffuse; float dummy;
	float4 specular;
};

// Vertex layout for the geometry in this sample
struct Vertex
{
	float3 pos;
	float3 normal;
	uint texCoord; // Packed texture coordinate
};

// Vertex layout for the probe SH coefficient computation pass
struct ShVertex
{
	float2 xy;
	int z;
};

// Extend the model class to load resources and render
class MyModel : public Model
{
public:
	MyModel();
	virtual ~MyModel();

	void Setup(ID3D10Device *dev);
	void SetupTexture(ID3D10Device *dev, const int index);
	void SetupMaterial(ID3D10Device *dev, const int index);

	void RenderMaterial(ID3D10Device *dev, const int material, const int count = 1);
	void RenderAll(ID3D10Device *dev);

	virtual bool Load(D3D10Context *context, const TCHAR *name, const TCHAR *texturePath);

private:
	void AddToMaterialRange(uint32 *indices, int &index, const int mat, const uint startMesh, const uint meshCount);

	ID3D10Buffer *m_vertexBuffer;
	ID3D10Buffer *m_indexBuffer;

	ID3D10Texture2D *m_textures[7];
	ID3D10ShaderResourceView *m_textureSRVs[7];

	ID3D10Buffer *m_materials[10];

	int m_materialRange[14];
};

// Main application class
class App : public D3D10App
{
public:
	const TCHAR *GetHomeDirectory(){ return _T("."); }

	void ResetCamera();
	bool OnKeyPress(HWND hwnd, const unsigned int key, const bool pressed);

	bool Create();
	void Destroy();

	bool CreateSHTextures();

	bool Load();
	void Unload();

	void OnRender();

protected:
	float EvaluateTransferFunction(const float3 &normal, const float3 *samples, const int sampleCount, const float *shArray, const int l, const int m) const;

	void SetupScene(const int passIndex);
	void RenderScene(const int passIndex, const float4x4 *mvp, const float4x4 *mvpInv, const int matrixCount);

	// Main effect and some extracted shaders
	ID3D10Effect *m_lighting, *m_shEffect;
	ID3D10PixelShader *m_lightingPS[LIGHTING_TECH];
	ID3D10GeometryShader *m_shadowGS[2];
	ID3D10InputLayout *m_lightingIL;

	ID3D10RasterizerState *m_cullBack, *m_cullNone;

	ID3D10SamplerState *m_baseSS;
	ID3D10SamplerState *m_lightMapSS;

	ID3D10Texture2D *m_shadowMap;
	ID3D10RenderTargetView *m_shadowMapRTV;
	ID3D10ShaderResourceView *m_shadowMapSRV;
	ID3D10SamplerState *m_shadowMapSS;

	ID3D10Texture2D *m_shadowMapDepth;
	ID3D10DepthStencilView *m_shadowMapDepthDSV;

	float3 m_probePos[MAX_PROBE_COUNT];

	int m_probeCount;
	int m_currProbe;
	int m_waitFrames;
	int m_currWallMaterial;

	float m_lmValCeiling1;
	float m_lmValCeiling2;
	float m_lmValWall;
	float m_lightIntensity;
	float m_exposure;


	// Render target for the light probes
	ID3D10Texture3D *m_probes;
	ID3D10RenderTargetView *m_probesRTV;
	ID3D10RenderTargetView *m_probesChunkRTVs[PROBE_PASSES_PER_FRAME];
	ID3D10ShaderResourceView *m_probesSRV;

	// Depth surfaces for the light probes
	ID3D10Texture2D *m_probesDepth;
	ID3D10DepthStencilView *m_probesDepthDSV;
	ID3D10DepthStencilView *m_probesDepthChunkDSVs[PROBE_PASSES_PER_FRAME];



	ID3D10Buffer *m_perFrameCB;
	ID3D10Buffer *m_gsMvpCB;
	ID3D10Buffer *m_gsMvpInvCB;

	ID3D10Texture3D *m_shTable[SH_COEFF_VECTORS];
	ID3D10ShaderResourceView *m_shTableSRV[SH_COEFF_VECTORS];

	ID3D10Texture2D *m_shCube[SH_COEFF_VECTORS];
	ID3D10ShaderResourceView *m_shCubeSRV[SH_COEFF_VECTORS];


	// 3D textures containing the resulting SH coefficients
	ID3D10Texture3D *m_shCoeffs[3][SH_COEFF_VECTORS];
	ID3D10RenderTargetView *m_shCoeffsRTV[3][SH_COEFF_VECTORS];
	ID3D10ShaderResourceView *m_shCoeffsSRV[3][SH_COEFF_VECTORS];
	ID3D10SamplerState *m_shCoeffsSS;

	ID3D10Buffer *m_shCoeffVB;
	ID3D10InputLayout *m_shCoeffIL;

	ID3D10Texture1D *m_texLightGrad;
	ID3D10ShaderResourceView *m_texLightGradSRV;

	Camera m_lightCamera;
	MyModel *m_model;

	bool m_animateLight;
	bool m_showProbes;
	bool m_showHelp;
};

#endif
