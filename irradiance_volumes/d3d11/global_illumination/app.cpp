//-----------------------------------------------------------------------------
// File: D3D10\Global Illumination\App.cpp
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------





#include "App.h"
#include <stdio.h>

D3D11App *app = new App();

const int SAMPLE_COUNT = 256;
float3 samples[SAMPLE_COUNT];

float factorial(const int x)
{
	float f = 1.0f;
	for (int i = 2; i <= x; i++)
	{
		f *= i;
	}

	return f;
}

// Evaluate an Associated Legendre Polynomial P(l, m, x) at x
float P(const int l, const int m, const float x)
{
	float pmm = 1.0f;
	if (m > 0)
	{
		float somx2 = sqrtf((1.0f - x) * (1.0f + x));

		float fact = 1.0;
		for (int i = 1; i <= m; i++)
		{
			pmm *= (-fact) * somx2;
			fact += 2.0;
		}
	}
	if (l == m) return pmm;

	float pmmp1 = x * (2.0f * m + 1.0f) * pmm;
	if (l == m + 1) return pmmp1;

	float pll = 0.0;
	for (int ll = m + 2; ll <= l; ++ll)
	{
		pll = ((2.0f * ll - 1.0f) * x * pmmp1 - (ll + m - 1.0f) * pmm) / (ll - m);
		pmm = pmmp1;
		pmmp1 = pll;
	}

	return pll;
}

// Normalization constant
float K(const int l, const int m)
{
	return sqrtf(((2.0f * l + 1.0f) * factorial(l - m)) / (4.0f * PI * factorial(l + m)));
}

// SH coefficient computation
float SH(const int l, const int m, const float theta, const float phi)
{
	const float sqrt2 = 1.4142135623731f;

	if (m == 0)
		return K(l, 0) * P(l, m, cosf(theta));
	else if (m > 0)
		return sqrt2 * K(l, m) * cosf(m * phi) * P(l, m, cosf(theta));
	else
		return sqrt2 * K(l, -m) * sinf(-m * phi) * P(l, -m, cosf(theta));
}

float SH(const int l, const int m, const float3 &pos)
{
	float len = length(pos);

	float p = atan2f(pos.z, pos.x);
	float t = acosf(pos.y / len);

	return SH(l, m, t, p);
}

float SH_A(const int l, const int m, const float3 &pos)
{
	float d = dot(pos, pos);
	float len = sqrtf(d);

	float p = atan2f(pos.z, pos.x);
	float t = acosf(pos.y / len);

	return SH(l, m, t, p) * powf(d, -1.5f);
}

MyModel::MyModel()
{
	m_vertexBuffer = NULL;
	m_indexBuffer = NULL;
}

MyModel::~MyModel()
{
	for (uint i = 0; i < elementsOf(m_textures); i++)
	{
		SAFE_RELEASE(m_textures[i]);
		SAFE_RELEASE(m_textureSRVs[i]);
	}
	for (uint i = 0; i < elementsOf(m_materials); i++)
	{
		SAFE_RELEASE(m_materials[i]);
	}

	SAFE_RELEASE(m_vertexBuffer);
	SAFE_RELEASE(m_indexBuffer);
}

/*
	Add a mesh from the model into the index range of a material.
	This function is used to arrange meshes in a way to reduce the
	number of draw calls, state changes and get better Hi-Z utilization.
*/
void MyModel::AddToMaterialRange(uint32 *indices, int &index, const int mat, const uint startMesh, const uint meshCount)
{
	for (uint mesh = startMesh; mesh < startMesh + meshCount; mesh++)
	{
		uint base  = m_pMesh[mesh].GetBaseTriangle();
		uint count = m_pMesh[mesh].GetNumTriangles();

		for (uint p = 0; p < count; p++)
		{
			indices[index++] = m_pTriangle[base + p].v[0];
			indices[index++] = m_pTriangle[base + p].v[1];
			indices[index++] = m_pTriangle[base + p].v[2];
		}
	}
	m_materialRange[mat + 1] = index;
}

bool MyModel::Load(D3D11Context *context, const TCHAR *name, const TCHAR *texturePath)
{
	if (!Model::Load(name, texturePath)) return false;

	// Assemble the vertices into the format used in this sample
	Vertex *vertices = new Vertex[m_numVertices];
	for (uint i = 0; i < m_numVertices; i++)
	{
		float3 pos = m_pVPosition[i];
		pos *= 0.2f;
		pos.y -= 6.2f;

		vertices[i].pos = pos;
		vertices[i].normal = m_pVNormal[i];

		// Pack texture coordinates into a uint
		uint tcX = (uint) (m_pVTexcoord[i].x * 1024.0f + 1024.5f);
		uint tcY = (uint) (m_pVTexcoord[i].y * 1024.0f + 1024.5f);
		vertices[i].texCoord = tcX | (tcY << 16);
	}

	/*
		Create the index buffer. We arrange indices to reduce the number
		of draw calls, state changes and get better Hi-Z utilization.
	*/
	uint32 *indices = new uint32[m_numTriangles * 3];

	int index = 0;
	m_materialRange[0] = 0;

	// Untextured materials
	AddToMaterialRange(indices, index, 0,  19, 1); // Hand
	AddToMaterialRange(indices, index, 1,  20, 1); // Ball
	AddToMaterialRange(indices, index, 1,  23, 1); // Horse
	AddToMaterialRange(indices, index, 1,  25, 1); // Sci-Fi weirdo
	AddToMaterialRange(indices, index, 1,  28, 1); // Bench
	AddToMaterialRange(indices, index, 1,  30, 1); // Frame
	AddToMaterialRange(indices, index, 2,  24, 1); // Horse stand
	AddToMaterialRange(indices, index, 2,  26, 1); // Sci-Fi weirdo stand
	AddToMaterialRange(indices, index, 2,  32, 1); // Globe stand
	AddToMaterialRange(indices, index, 3,  3, 15); // Ceiling, Pillars, Stands, Wall lights
	AddToMaterialRange(indices, index, 4,  0,  1); // Walls
	AddToMaterialRange(indices, index, 5,  21, 1); // Teapot
	// Masked materials
	AddToMaterialRange(indices, index, 6,  27, 1); // Globe
	// Textured materials
	AddToMaterialRange(indices, index, 7,  18, 1); // Ball-horse
	AddToMaterialRange(indices, index, 8,  22, 1); // Head
	AddToMaterialRange(indices, index, 9,  29, 1); // Picture
	AddToMaterialRange(indices, index, 10,  1, 1); // Floor
	// Lightmapped materials
	AddToMaterialRange(indices, index, 11, 2,  1); // Ceiling
	AddToMaterialRange(indices, index, 12, 31, 1); // Wall light quads

	if ((m_vertexBuffer = context->CreateVertexBuffer(m_numVertices * sizeof(Vertex), D3D11_USAGE_IMMUTABLE, vertices)) == NULL) return false;
	if ((m_indexBuffer = context->CreateIndexBuffer(m_numTriangles * 3 * sizeof(uint32), D3D11_USAGE_IMMUTABLE, indices)) == NULL) return false;

	delete vertices;
	delete indices;



	// Load textures
	if ((m_textures[0] = (ID3D11Texture2D *) context->LoadTexture(_T("../../Media/Textures/lopal.bmp"     ), &m_textureSRVs[0])) == NULL) return false;
	if ((m_textures[1] = (ID3D11Texture2D *) context->LoadTexture(_T("../../Media/Textures/headpal.bmp"   ), &m_textureSRVs[1])) == NULL) return false;
	if ((m_textures[2] = (ID3D11Texture2D *) context->LoadTexture(_T("../../Media/Textures/picture.dds"   ), &m_textureSRVs[2])) == NULL) return false;
	if ((m_textures[3] = (ID3D11Texture2D *) context->LoadTexture(_T("../../Media/Textures/floor.dds"     ), &m_textureSRVs[3])) == NULL) return false;
	if ((m_textures[4] = (ID3D11Texture2D *) context->LoadTexture(_T("../../Media/Textures/globe.dds"     ), &m_textureSRVs[4])) == NULL) return false;
	if ((m_textures[5] = (ID3D11Texture2D *) context->LoadTexture(_T("../../Media/Textures/wall_lm.bmp"   ), &m_textureSRVs[5])) == NULL) return false;
	if ((m_textures[6] = (ID3D11Texture2D *) context->LoadTexture(_T("../../Media/Textures/ceiling_lm.dds"), &m_textureSRVs[6])) == NULL) return false;

	// Create materials
	PerObject m0 = { float3(0.816f, 0.216f, 0.227f), 0, float4(0.45f, 0.15f, 0.15f, 16.0f) };
	PerObject m1 = { float3(0.435f, 0.443f, 0.682f), 0, float4(0.3f,  0.3f,  0.6f,  16.0f) };
	PerObject m2 = { float3(0.29f,  0.482f, 0.298f), 0, float4(0.15f, 0.3f,  0.15f, 16.0f) };
	PerObject m3 = { float3(0.973f, 0.894f, 0.8f  ), 0, float4(0.5f,  0.5f,  0.5f,  16.0f) };
	PerObject m4 = { float3(1.0f,   0.6f,   0.2f  ), 0, float4(4.0f,  2.4f,  1.6f,  24.0f) };
	PerObject m5 = { float3(1.0f,   1.0f,   1.0f  ), 0, float4(0.3f,  0.4f,  0.6f,   4.0f) };
	PerObject m6 = { float3(0.25f,  0.7f,   0.8f  ), 0, float4(0.7f,  0.7f,  0.8f,   4.0f) };
	PerObject m7 = { float3(0.2f,   0.2f,   0.2f  ), 0, float4(0.7f,  0.7f,  0.7f,  16.0f) };
	PerObject m8 = { float3(0.616f, 0.494f, 0.361f), 0, float4(0.1f,  0.1f,  0.1f,  32.0f) };
	PerObject m9 = { float3(0.5f,   0.5f,   0.5f  ), 0, float4(0.7f,  0.7f,  0.7f,  16.0f) };
	if ((m_materials[0] = context->CreateConstantBuffer(sizeof(PerObject), D3D11_USAGE_IMMUTABLE, &m0)) == NULL) return false;
	if ((m_materials[1] = context->CreateConstantBuffer(sizeof(PerObject), D3D11_USAGE_IMMUTABLE, &m1)) == NULL) return false;
	if ((m_materials[2] = context->CreateConstantBuffer(sizeof(PerObject), D3D11_USAGE_IMMUTABLE, &m2)) == NULL) return false;
	if ((m_materials[3] = context->CreateConstantBuffer(sizeof(PerObject), D3D11_USAGE_IMMUTABLE, &m3)) == NULL) return false;
	if ((m_materials[4] = context->CreateConstantBuffer(sizeof(PerObject), D3D11_USAGE_IMMUTABLE, &m4)) == NULL) return false;
	if ((m_materials[5] = context->CreateConstantBuffer(sizeof(PerObject), D3D11_USAGE_IMMUTABLE, &m5)) == NULL) return false;
	if ((m_materials[6] = context->CreateConstantBuffer(sizeof(PerObject), D3D11_USAGE_IMMUTABLE, &m6)) == NULL) return false;
	if ((m_materials[7] = context->CreateConstantBuffer(sizeof(PerObject), D3D11_USAGE_IMMUTABLE, &m7)) == NULL) return false;
	if ((m_materials[8] = context->CreateConstantBuffer(sizeof(PerObject), D3D11_USAGE_IMMUTABLE, &m8)) == NULL) return false;
	if ((m_materials[9] = context->CreateConstantBuffer(sizeof(PerObject), D3D11_USAGE_IMMUTABLE, &m9)) == NULL) return false;

	return true;
}

void MyModel::Setup(ID3D11DeviceContext *dev)
{
	// Setup buffers and topology for drawing
	UINT stride = sizeof(Vertex);
	UINT offset = 0;
	dev->IASetVertexBuffers(0, 1, &m_vertexBuffer, &stride, &offset);
	dev->IASetIndexBuffer(m_indexBuffer, DXGI_FORMAT_R32_UINT, 0);

	dev->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
}

void MyModel::SetupTexture(ID3D11DeviceContext *dev, const int index)
{
	dev->PSSetShaderResources(0, 1, &m_textureSRVs[index]);
}

void MyModel::SetupMaterial(ID3D11DeviceContext *dev, const int index)
{
	dev->PSSetConstantBuffers(1, 1, &m_materials[index]);
}

void MyModel::RenderMaterial(ID3D11DeviceContext *dev, const int material, const int count)
{
	int start = m_materialRange[material];
	int end = m_materialRange[material + count];
	dev->DrawIndexed(end - start, start, 0);
}

void MyModel::RenderAll(ID3D11DeviceContext *dev)
{
	dev->DrawIndexed(m_numTriangles * 3, 0, 0);
}

void App::ResetCamera()
{
	// Set start position and rotation
	m_camera.SetPosition(float3(-15.9f, -0.57f, 8.05f));
	m_camera.SetRotation(0.0f, 4.4f, 0.0f);
}

bool App::OnKeyPress(HWND hwnd, const unsigned int key, const bool pressed)
{
	// Let framework process key first
	if (D3D11App::OnKeyPress(hwnd, key, pressed)) return true;

	if (pressed)
	{
		switch (key)
		{
		case 'P':
			m_showProbes = !m_showProbes;
			return true;
		case '9':
			m_animateLight = !m_animateLight;
			return true;
		case VK_ADD: // Cycle wall color
			if (!m_waitFrames)
			{
				m_currWallMaterial++;
				if (m_currWallMaterial >= WALL_MATERIAL_COUNT) m_currWallMaterial = 0;
				m_waitFrames = m_probeCount / PROBE_CUBES_PER_FRAME;
			}
			return true;
		case VK_SUBTRACT: // Cycle wall color
			if (!m_waitFrames)
			{
				if (m_currWallMaterial <= 0) m_currWallMaterial = WALL_MATERIAL_COUNT;
				m_currWallMaterial--;
				m_waitFrames = m_probeCount / PROBE_CUBES_PER_FRAME;
			}
			return true;
		case VK_F1: // Toggle help display
			m_showHelp = !m_showHelp;
			return true;
		}
	}

	return false;
}

bool App::Create()
{
	// Compute roughly uniformly distributed sample positions on a unit sphere
	for (int i = 0; i < SAMPLE_COUNT; i++)
	{
		float3 v;
		float len;
		do
		{
			v.x = ((float) rand()) / RAND_MAX * 2 - 1;
			v.y = ((float) rand()) / RAND_MAX * 2 - 1;
			v.z = ((float) rand()) / RAND_MAX * 2 - 1;
			len = dot(v, v);
		}
		while (len < 0.9f * 0.9f || len > 1.1f * 1.1f);

		samples[i] = v * (1.0f / sqrtf(len));
	}


	// Create our rendering context. It'll take care of our D3D10 device and simplify some tasks.
	m_context = new D3D11Context();
	if (!m_context->Create(_T("Global Illumination"), DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_D24_UNORM_S8_UINT, 1280, 720, 4, false)) return false;

	// Let base-class initialize itself as well
	if (!D3D11App::Create()) return false;

	// Setup some camera parameters
	m_camera.SetFrustumExtents(1.5f, 0.1f, 50.0f);

	if (!m_lightCamera.LoadPath(_T("Path.pth"))) return false;
//	m_camera.LoadPath(_T("Path.pth"));


	// Initialize member variables to default values
	m_currProbe = 0;
	m_showProbes = false;
	m_showHelp = false;
	m_animateLight = true;
	m_currWallMaterial = 0;

	m_lmValCeiling1 = 1.3f;
	m_lmValCeiling2 = 1.3f;
	m_lmValWall = 2.0f;
	m_lightIntensity = 1.2f;
	m_exposure = 1.8f;

	return true;
}

void App::Destroy()
{
//	m_camera.SavePath(_T("Path.pth"));

	D3D11App::Destroy();
}

/*
	Compute the light transfer function SH coefficients. We're using a regular diffuse lighting model.
*/
float App::EvaluateTransferFunction(const float3 &normal, const float3 *samples, const int sampleCount, const float *shArray, const int l, const int m) const
{
	int index = l * (l + 1) + m;

	float sh = 0;
	for (int s = 0; s < sampleCount; s++)
	{
		float diffuse = dot(samples[s], normal);
		if (diffuse > 0)
		{
			sh += diffuse * shArray[index * sampleCount + s];
		}
	}
	return sh * (4 * PI / sampleCount);
}

/*
	Create precomputed SH basis function tables. Each channel holds a basis
	function and each slice corresponds to a light probe cubemap face.
*/
bool App::CreateSHTextures()
{
	// Pre-compute the SH basis function values we need first to speed things up a bit
	float *shArray = new float[SH_BANDS * SH_BANDS * SAMPLE_COUNT];

	int index = 0;
	for (int l = 0; l < SH_BANDS; l++)
	{
		for (int m = -l; m <= l; m++)
		{
			for (int s = 0; s < SAMPLE_COUNT; s++)
			{
				shArray[index] = SH(l, m, samples[s]);
				index++;
			}
		}
	}



	// tex0 = SH transfer function cubemap
	// tex1 = SH basis function cubemap
	half4 *tex0 = new half4[SH_TF_CUBE_SIZE * SH_TF_CUBE_SIZE * 6];
	half4 *tex1 = new half4[PROBE_SIZE * PROBE_SIZE * 6];

	int l = 0;
	int m = 0;
	for (int i = 0; i < SH_COEFF_VECTORS; i++)
	{
		// Compute l & m for the next 4 coefficients
		int l4[4], m4[4];
		for (int k = 0; k < 4; k++)
		{
			l4[k] = l;
			m4[k] = m;
			if (m >= l)
			{
				l++;
				m = -l;
			}
			else
			{
				m++;
			}
		}

		uint index0 = 0;
		uint index1 = 0;
		float3 v;
		float4 sh;

		// Positive & negative X faces
		for (v.x = 1; v.x >= -1; v.x -= 2)
		{
			for (int y = 0; y < SH_TF_CUBE_SIZE; y++)
			{
				for (int z = 0; z < SH_TF_CUBE_SIZE; z++)
				{
					v.y =  1 - 2 * float(y + 0.5f) / SH_TF_CUBE_SIZE;
					v.z = (1 - 2 * float(z + 0.5f) / SH_TF_CUBE_SIZE) * v.x;

					float3 normal = normalize(v);
					sh.x = EvaluateTransferFunction(normal, samples, SAMPLE_COUNT, shArray, l4[0], m4[0]);
					sh.y = EvaluateTransferFunction(normal, samples, SAMPLE_COUNT, shArray, l4[1], m4[1]);
					sh.z = EvaluateTransferFunction(normal, samples, SAMPLE_COUNT, shArray, l4[2], m4[2]);
					sh.w = EvaluateTransferFunction(normal, samples, SAMPLE_COUNT, shArray, l4[3], m4[3]);
					tex0[index0++] = sh;
				}
			}
			for (int y = 0; y < PROBE_SIZE; y++)
			{
				for (int z = 0; z < PROBE_SIZE; z++)
				{
					v.y =  1 - 2 * float(y + 0.5f) / PROBE_SIZE;
					v.z = (1 - 2 * float(z + 0.5f) / PROBE_SIZE) * v.x;

					sh.x = SH_A(l4[0], m4[0], v);
					sh.y = SH_A(l4[1], m4[1], v);
					sh.z = SH_A(l4[2], m4[2], v);
					sh.w = SH_A(l4[3], m4[3], v);
					tex1[index1++] = sh;
				}
			}
		}
		// Positive & negative Y faces
		for (v.y = 1; v.y >= -1; v.y -= 2)
		{
			for (int z = 0; z < SH_TF_CUBE_SIZE; z++)
			{
				for (int x = 0; x < SH_TF_CUBE_SIZE; x++)
				{
					v.x =  2 * float(x + 0.5f) / SH_TF_CUBE_SIZE - 1;
					v.z = (2 * float(z + 0.5f) / SH_TF_CUBE_SIZE - 1) * v.y;

					float3 normal = normalize(v);
					sh.x = EvaluateTransferFunction(normal, samples, SAMPLE_COUNT, shArray, l4[0], m4[0]);
					sh.y = EvaluateTransferFunction(normal, samples, SAMPLE_COUNT, shArray, l4[1], m4[1]);
					sh.z = EvaluateTransferFunction(normal, samples, SAMPLE_COUNT, shArray, l4[2], m4[2]);
					sh.w = EvaluateTransferFunction(normal, samples, SAMPLE_COUNT, shArray, l4[3], m4[3]);
					tex0[index0++] = sh;
				}
			}

			for (int z = 0; z < PROBE_SIZE; z++)
			{
				for (int x = 0; x < PROBE_SIZE; x++)
				{
					v.x =  2 * float(x + 0.5f) / PROBE_SIZE - 1;
					v.z = (2 * float(z + 0.5f) / PROBE_SIZE - 1) * v.y;

					sh.x = SH_A(l4[0], m4[0], v);
					sh.y = SH_A(l4[1], m4[1], v);
					sh.z = SH_A(l4[2], m4[2], v);
					sh.w = SH_A(l4[3], m4[3], v);
					tex1[index1++] = sh;
				}
			}
		}
		// Positive & negative Z faces
		for (v.z = 1; v.z >= -1; v.z -= 2)
		{
			for (int y = 0; y < SH_TF_CUBE_SIZE; y++)
			{
				for (int x = 0; x < SH_TF_CUBE_SIZE; x++)
				{
					v.x = (2 * float(x + 0.5f) / SH_TF_CUBE_SIZE - 1) * v.z;
					v.y =  1 - 2 * float(y + 0.5f) / SH_TF_CUBE_SIZE;

					float3 normal = normalize(v);
					sh.x = EvaluateTransferFunction(normal, samples, SAMPLE_COUNT, shArray, l4[0], m4[0]);
					sh.y = EvaluateTransferFunction(normal, samples, SAMPLE_COUNT, shArray, l4[1], m4[1]);
					sh.z = EvaluateTransferFunction(normal, samples, SAMPLE_COUNT, shArray, l4[2], m4[2]);
					sh.w = EvaluateTransferFunction(normal, samples, SAMPLE_COUNT, shArray, l4[3], m4[3]);
					tex0[index0++] = sh;
				}
			}

			for (int y = 0; y < PROBE_SIZE; y++)
			{
				for (int x = 0; x < PROBE_SIZE; x++)
				{
					v.x = (2 * float(x + 0.5f) / PROBE_SIZE - 1) * v.z;
					v.y =  1 - 2 * float(y + 0.5f) / PROBE_SIZE;
					sh.x = SH_A(l4[0], m4[0], v);
					sh.y = SH_A(l4[1], m4[1], v);
					sh.z = SH_A(l4[2], m4[2], v);
					sh.w = SH_A(l4[3], m4[3], v);
					tex1[index1++] = sh;
				}
			}
		}

		if ((m_shCube[i] = m_context->CreateTextureCube(tex0, DXGI_FORMAT_R16G16B16A16_FLOAT, SH_TF_CUBE_SIZE, &m_shCubeSRV[i])) == NULL) return false;
		if ((m_shTable[i] = m_context->CreateTexture3D(tex1, DXGI_FORMAT_R16G16B16A16_FLOAT, PROBE_SIZE, PROBE_SIZE, 6, &m_shTableSRV[i])) == NULL) return false;
	}

	delete tex0;
	delete tex1;

	delete shArray;

	return true;
}


bool App::Load()
{
	// Nvidia has a driver bug with comparison texture lookups
	bool useNvidiaWorkaround = (m_context->GetVendorID() == VENDOR_NVIDIA);

	// Pass on some #define values to the shader
	D3D_SHADER_MACRO effectDefines[] =
	{
		DEF_MACRO(SH_COEFF_VECTORS),
		DEF_MACRO(PROBE_SLICES_PER_PASS),
		DEF_MACRO(SCALE),
		{ "NVIDIA_WORKAROUND", useNvidiaWorkaround? "1" : "0" },
		{ NULL, NULL },
	};
	if ((m_lighting = m_context->LoadEffect(L"Lighting.fx", effectDefines)) == NULL) return false;

	// Use our own constant buffer management for best performance
	if ((m_perFrameCB = m_context->CreateEffectConstantBuffer(m_lighting, "PerFrame")) == NULL) return false;
	if ((m_gsMvpCB    = m_context->CreateEffectConstantBuffer(m_lighting, "GSMvp"   )) == NULL) return false;
	if ((m_gsMvpInvCB = m_context->CreateEffectConstantBuffer(m_lighting, "GSMvpInv")) == NULL) return false;


	/*
		Query some of the shaders from the effect. This is done to reduce the overhead while changing materials so
		we can just set the right texture, constant buffer or shader instead of having everything set all the time.
	*/
	D3DX11_PASS_SHADER_DESC desc;
	for (int i = 0; i < LIGHTING_TECH; i++)
	{
		ID3DX11EffectPass *fxPass = m_lighting->GetTechniqueByIndex(0)->GetPassByIndex(i);
		fxPass->GetPixelShaderDesc(&desc);
		if (FAILED(desc.pShaderVariable->GetPixelShader(0, &m_lightingPS[i]))) return false;
		if (i < 2)
		{
			fxPass->GetGeometryShaderDesc(&desc);
			if (FAILED(desc.pShaderVariable->GetGeometryShader(0, &m_shadowGS[i]))) return false;
		}
	}


	// Manually load the light gradient texture because the D3DX function don't handle 1D textures
	ubyte grad[256 * 4];
	FILE *file = _tfopen(TEX_PATH _T("light_grad.dds"), _T("rb"));
	if (file == NULL) return false;
	fseek(file, 128, SEEK_SET);
	fread(grad, 1, sizeof(grad), file);
	fclose(file);
	if ((m_texLightGrad = (ID3D11Texture1D *) m_context->CreateTexture1D(grad, DXGI_FORMAT_R8G8B8A8_UNORM, 256, 1, &m_texLightGradSRV)) == NULL) return false;


	// Load the model
	m_model = new MyModel();
	if (!m_model->Load(m_context, MODEL_PATH _T("GIroom.am"), TEX_PATH)) return false;


	// Define the input layout
	D3D11_INPUT_ELEMENT_DESC layout[] =
	{
		{ "Vertex", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0, offsetOf(Vertex, pos),    D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "Normal", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, offsetOf(Vertex, normal), D3D11_INPUT_PER_VERTEX_DATA, 0 },
	};
	if ((m_lightingIL = m_context->CreateInputLayout(m_lighting->GetTechniqueByIndex(0)->GetPassByIndex(0), layout, elementsOf(layout))) == NULL) return false;

	// Create two rasterizer states so we can switch between back/none face culling.
	if ((m_cullBack = m_context->CreateRasterizerState(D3D11_CULL_BACK)) == NULL) return false;
	if ((m_cullNone = m_context->CreateRasterizerState(D3D11_CULL_NONE)) == NULL) return false;


	// Load the SH coefficient computation shader
	D3D_SHADER_MACRO shDefines[] =
	{
		DEF_MACRO(PROBE_SIZE),
		{ NULL, NULL },
	};
	if ((m_shEffect = m_context->LoadEffect(_T("SH.fx"), shDefines)) == NULL) return false;


	// Define the input layout for the SH shader
	D3D11_INPUT_ELEMENT_DESC shLayout[] =
	{
		{ "XY", 0, DXGI_FORMAT_R32G32_FLOAT, 0, offsetOf(ShVertex, xy), D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "Z",  0, DXGI_FORMAT_R32_UINT,     0, offsetOf(ShVertex,  z), D3D11_INPUT_PER_VERTEX_DATA, 0 },
	};
	if ((m_shCoeffIL = m_context->CreateInputLayout(m_shEffect->GetTechniqueByName("SH")->GetPassByName("Main"), shLayout, elementsOf(shLayout))) == NULL) return false;


	if (!CreateSHTextures()) return false;


	// Create the shadow map
	if ((m_shadowMap = m_context->CreateRenderTargetCube(DXGI_FORMAT_R16_UNORM, SHADOWMAP_SIZE, 1, 1, &m_shadowMapRTV, NULL, &m_shadowMapSRV)) == NULL) return false;
	if ((m_shadowMapDepth = m_context->CreateDepthTargetCube(DXGI_FORMAT_D16_UNORM, SHADOWMAP_SIZE, 1, 1, &m_shadowMapDepthDSV, NULL)) == NULL) return false;

	// Comparison filters are broken in current Nvidia drivers
	if (useNvidiaWorkaround)
	{
		if ((m_shadowMapSS = m_context->CreateSamplerState(D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT, D3D11_TEXTURE_ADDRESS_CLAMP, 0.0f)) == NULL) return false;
	}
	else
	{
		if ((m_shadowMapSS = m_context->CreateSamplerState(D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT, D3D11_TEXTURE_ADDRESS_CLAMP, 0.0f)) == NULL) return false;
	}

	// Create volume render targets that will hold the final SH values from the light probes
	for (int i = 0; i < SH_COEFF_VECTORS; i++)
	{
		for (int c = 0; c < 3; c++)
		{
			if ((m_shCoeffs[c][i] = m_context->CreateRenderTarget3D(DXGI_FORMAT_R16G16B16A16_FLOAT, SIZE_X, SIZE_Y, SIZE_Z, 1, &m_shCoeffsRTV[c][i], &m_shCoeffsSRV[c][i])) == NULL) return false;
		}
	}
	if ((m_shCoeffsSS = m_context->CreateSamplerState(D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT, D3D11_TEXTURE_ADDRESS_CLAMP, 0.0f)) == NULL) return false;


	// Compute light probe positions and create geometry for updating the SH volume render targets
	bool probes[SIZE_Z][SIZE_Y][SIZE_X];
	memset(probes, 1, sizeof(probes));

	// Clear out some empty space
	for (int y = 1; y < 3; y++)
	{
		for (int x = 3; x < 9; x++)
		{
			probes[3][y][x] = false;
		}
	}
	probes[3][2][9] = false;


	ShVertex coords[MAX_PROBE_COUNT + PROBE_CUBES_PER_FRAME - 1];
	m_probeCount = 0;
	for (int z = 0; z < SIZE_Z; z++)
	{
		for (int y = 0; y < SIZE_Y; y++)
		{
			for (int x = 0; x < SIZE_X; x++)
			{
				if (probes[z][y][x])
				{
					coords[m_probeCount].xy = float2((2.0f * x + 1.0f) / SIZE_X - 1.0f, 1.0f - (2.0f * y + 1.0f) / SIZE_Y);
					coords[m_probeCount].z = z;

					m_probePos[m_probeCount] = float3(SCALE_X * (2.0f * x + 1.0f - SIZE_X), SCALE_Y * (2.0f * y + 1.0f - SIZE_Y), SCALE_Z * (2.0f * z + 1.0f - SIZE_Z));

					m_probeCount++;
				}
			}
		}
	}
	// On first frame we wait until all probes have been updated before displaying
	m_waitFrames = m_probeCount / PROBE_CUBES_PER_FRAME;

	// Shuffle the probes to avoid a "moving line" as probes are updated over several frames
	for (int i = 0; i < 16384; i++)
	{
		int i0 = rand() % m_probeCount;
		int i1 = rand() % m_probeCount;

		ShVertex temp = coords[i0];
		coords[i0] = coords[i1];
		coords[i1] = temp;

		float3 tmp = m_probePos[i0];
		m_probePos[i0] = m_probePos[i1];
		m_probePos[i1] = tmp;
	}


	// Pad the array with the first vertices so we can wrap around without issuing multiple draw calls
	for (int i = 0; i < PROBE_CUBES_PER_FRAME - 1; i++)
	{
		coords[m_probeCount + i] = coords[i];
	}

	// Create the geometry for the SH coefficient pass
	if ((m_shCoeffVB = m_context->CreateVertexBuffer((m_probeCount + PROBE_CUBES_PER_FRAME - 1) * sizeof(ShVertex), D3D11_USAGE_IMMUTABLE, coords)) == NULL) return false;

	// Create the sampler states for base textures and lightmaps
	if ((m_baseSS = m_context->CreateSamplerState(D3D11_FILTER_ANISOTROPIC, D3D11_TEXTURE_ADDRESS_WRAP)) == NULL) return false;
	if ((m_lightMapSS = m_context->CreateSamplerState(D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT, D3D11_TEXTURE_ADDRESS_CLAMP, 0.0f)) == NULL) return false;

	/*
		Create light probe render target.
		The light probes are packed into a large 3D texture where each slice is a light probe cubemap face. By using a 3D texture
		we can render to multiple "cubemaps" in a single pass.
	*/
	if ((m_probes = m_context->CreateRenderTarget3D(DXGI_FORMAT_R11G11B10_FLOAT, PROBE_SIZE, PROBE_SIZE, PROBE_PASSES_PER_FRAME * PROBE_SLICES_PER_PASS, 1, &m_probesRTV, &m_probesSRV)) == NULL) return false;
	// Make an RTV for each chunk of probes that we'll render per pass
	for (int i = 0; i < PROBE_PASSES_PER_FRAME; i++)
	{
		if ((m_probesChunkRTVs[i] = m_context->CreateRenderTargetView3D(m_probes, DXGI_FORMAT_R11G11B10_FLOAT, i * PROBE_SLICES_PER_PASS, PROBE_SLICES_PER_PASS)) == NULL) return false;
	}

	// Make a depth buffer for the light probes
	if ((m_probesDepth = m_context->CreateDepthTarget2D(DXGI_FORMAT_D16_UNORM, PROBE_SIZE, PROBE_SIZE, PROBE_SLICES_PER_PASS * PROBE_PASSES_PER_FRAME, 1, 1, &m_probesDepthDSV)) == NULL) return false;
	for (int i = 0; i < PROBE_PASSES_PER_FRAME; i++)
	{
		if ((m_probesDepthChunkDSVs[i] = m_context->CreateDepthStencilView2D(m_probesDepth, DXGI_FORMAT_D16_UNORM, i * PROBE_SLICES_PER_PASS, PROBE_SLICES_PER_PASS)) == NULL) return false;
	}



	/*
		Initially clear all render targets
	*/
	ID3D11DeviceContext *dev = m_context->GetDeviceContext();

	dev->ClearRenderTargetView(m_probesRTV, float4(0, 0, 0, 0));
	dev->ClearRenderTargetView(m_shadowMapRTV, float4(0, 0, 0, 0));

	m_context->Clear(float4(0, 0, 0, 0), D3D11_CLEAR_DEPTH, 1.0f, 0);

	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < SH_COEFF_VECTORS; i++)
		{
			dev->ClearRenderTargetView(m_shCoeffsRTV[c][i], float4(0, 0, 0, 0));
		}
	}

	return true;
}

void App::Unload()
{
	delete m_model;

	// Clean up all our resources
	SAFE_RELEASE(m_lighting);
	SAFE_RELEASE(m_shEffect);

	for (int i = 0; i < elementsOf(m_lightingPS); i++)
	{
		SAFE_RELEASE(m_lightingPS[i]);
	}
	for (int i = 0; i < elementsOf(m_shadowGS); i++)
	{
		SAFE_RELEASE(m_shadowGS[i]);
	}

	SAFE_RELEASE(m_baseSS);
	SAFE_RELEASE(m_lightMapSS);

	SAFE_RELEASE(m_lightingIL);

	SAFE_RELEASE(m_cullBack);
	SAFE_RELEASE(m_cullNone);

	SAFE_RELEASE(m_shadowMap);
	SAFE_RELEASE(m_shadowMapRTV);
	SAFE_RELEASE(m_shadowMapSRV);
	SAFE_RELEASE(m_shadowMapSS);

	SAFE_RELEASE(m_shadowMapDepth);
	SAFE_RELEASE(m_shadowMapDepthDSV);

	SAFE_RELEASE(m_probesDepth);
	SAFE_RELEASE(m_probesDepthDSV);

	SAFE_RELEASE(m_probes);
	SAFE_RELEASE(m_probesRTV);
	SAFE_RELEASE(m_probesSRV);

	for (int i = 0; i < PROBE_PASSES_PER_FRAME; i++)
	{
		SAFE_RELEASE(m_probesChunkRTVs[i]);
	}
	for (int i = 0; i < PROBE_PASSES_PER_FRAME; i++)
	{
		SAFE_RELEASE(m_probesDepthChunkDSVs[i]);
	}

	for (int i = 0; i < SH_COEFF_VECTORS; i++)
	{
		SAFE_RELEASE(m_shTable[i]);
		SAFE_RELEASE(m_shTableSRV[i]);

		SAFE_RELEASE(m_shCube[i]);
		SAFE_RELEASE(m_shCubeSRV[i]);

		for (int c = 0; c < 3; c++)
		{
			SAFE_RELEASE(m_shCoeffs[c][i]);
			SAFE_RELEASE(m_shCoeffsRTV[c][i]);
			SAFE_RELEASE(m_shCoeffsSRV[c][i]);
		}
	}
	SAFE_RELEASE(m_shCoeffsSS);

	SAFE_RELEASE(m_texLightGrad);
	SAFE_RELEASE(m_texLightGradSRV);

	SAFE_RELEASE(m_perFrameCB);
	SAFE_RELEASE(m_gsMvpCB);
	SAFE_RELEASE(m_gsMvpInvCB);

	SAFE_RELEASE(m_shCoeffVB);
	SAFE_RELEASE(m_shCoeffIL);
}

// Setup for rendering the scene
void App::SetupScene(const int passIndex)
{
	ID3D11DeviceContext *dev = m_context->GetDeviceContext();

	m_lighting->GetTechniqueByIndex(0)->GetPassByIndex(passIndex)->Apply(0, dev);

	dev->RSSetState(m_cullBack);
	dev->IASetInputLayout(m_lightingIL);

	m_model->Setup(dev);

	ID3D11SamplerState *samplers[] = { m_shadowMapSS, m_baseSS, m_lightMapSS, m_shCoeffsSS };
	dev->PSSetSamplers(0, 4, samplers);

	if (passIndex >= PROBE_PASS) // Shadow depth pass doesn't need any of these textures
	{
		ID3D11ShaderResourceView *textures[2 + 4 * SH_COEFF_VECTORS];
		ID3D11ShaderResourceView **dest = textures;

		*dest++ = m_shadowMapSRV;
		*dest++ = m_texLightGradSRV;

		// SH coefficient volumes
		for (int c = 0; c < 3; c++)
		{
			for (int i = 0; i < SH_COEFF_VECTORS; i++)
			{
				*dest++ = m_shCoeffsSRV[c][i];
			}
		}
		// SH coefficient cubemaps
		for (int i = 0; i < SH_COEFF_VECTORS; i++)
		{
			*dest++ = m_shCubeSRV[i];
		}

		dev->PSSetShaderResources(1, elementsOf(textures), textures);
	}
}

void App::RenderScene(const int passIndex, const float4x4 *mvp, const float4x4 *mvpInv, const int matrixCount)
{
	ID3D11DeviceContext *dev = m_context->GetDeviceContext();

	// Update array of mvp matrices
	float4x4 *mvpArray;
	if (mvp)
	{
		D3D11_MAPPED_SUBRESOURCE resource;
		dev->Map( m_gsMvpCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &resource );
		mvpArray = reinterpret_cast<float4x4*> ( resource.pData );
			memcpy(mvpArray, mvp, matrixCount * sizeof(float4x4));
		dev->Unmap( m_gsMvpCB, 0 );
	}
	if (mvpInv)
	{
		D3D11_MAPPED_SUBRESOURCE resource;
		dev->Map(m_gsMvpInvCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &resource);
		mvpArray = reinterpret_cast<float4x4*> ( resource.pData );
			memcpy(mvpArray, mvpInv, matrixCount * sizeof(float4x4));
		dev->Unmap(m_gsMvpInvCB, 0 );
	}

	if (passIndex >= PROBE_PASS)
	{
		/*
			Untextured materials
		*/
		dev->PSSetShader(m_lightingPS[passIndex], 0, 0);

		// Hand
		m_model->SetupMaterial(dev, 9);
		m_model->RenderMaterial(dev, 0);

		// Ball + Horse + Sci-Fi weirdo + Bench + Frame
		m_model->SetupMaterial(dev, 4);
		m_model->RenderMaterial(dev, 1);

		// Horse stand + Sci-Fi weirdo stand + Globe stand
		m_model->SetupMaterial(dev, 7);
		m_model->RenderMaterial(dev, 2);

		// Ceiling + Pillars + Stands + Wall lights
		m_model->SetupMaterial(dev, 5);
		m_model->RenderMaterial(dev, 3);

		// Walls
		m_model->SetupMaterial(dev, m_currWallMaterial);
		m_model->RenderMaterial(dev, 4);

		dev->RSSetState(m_cullNone);

		// Teapot
		m_model->SetupMaterial(dev, 8);
		m_model->RenderMaterial(dev, 5);


		/*
			Masked materials
		*/
		dev->PSSetShader(m_lightingPS[passIndex + 1], 0, 0);

		// Globe
		m_model->SetupMaterial(dev, 6);
		m_model->SetupTexture(dev, 4);
		m_model->RenderMaterial(dev, 6);

		dev->RSSetState(m_cullBack);


		/*
			Textured materials
		*/
		dev->PSSetShader(m_lightingPS[passIndex + 2], 0, 0);

		// Ball-horse
		m_model->SetupTexture(dev, 0);
		m_model->RenderMaterial(dev, 7);

		// Head
		m_model->SetupTexture(dev, 1);
		m_model->RenderMaterial(dev, 8);

		// Picture
		m_model->SetupTexture(dev, 2);
		m_model->RenderMaterial(dev, 9);

		// Floor
		m_model->SetupTexture(dev, 3);
		m_model->RenderMaterial(dev, 10);


		/*
			Lightmapped materials
		*/
		dev->PSSetShader(m_lightingPS[passIndex + 3],0, 0);

		// Ceiling
		m_model->SetupMaterial(dev, 5);
		m_model->SetupTexture(dev, 6);
		m_model->RenderMaterial(dev, 11);

		dev->PSSetShader(m_lightingPS[passIndex + 4], 0, 0);

		// Wall light quads
		m_model->SetupMaterial(dev, m_currWallMaterial);
		m_model->SetupTexture(dev, 5);
		m_model->RenderMaterial(dev, 12);
	}
	else
	{
		/*
			Shadow depth pass
		*/
		dev->GSSetShader(m_shadowGS[1], 0, 0);
		dev->PSSetShader(m_lightingPS[1],0, 0);
		dev->RSSetState(m_cullNone);
		m_model->SetupTexture(dev, 4);

		// Teapot + Globe
		m_model->RenderMaterial(dev, 5, 2);

		dev->GSSetShader(m_shadowGS[0], 0, 0);
		dev->PSSetShader(m_lightingPS[0], 0, 0);
		dev->RSSetState(m_cullBack);

		// The rest of the objects
		m_model->RenderMaterial(dev, 1, 4);
		m_model->RenderMaterial(dev, 7, 6);
		m_model->RenderMaterial(dev, 0);
	}
}

void App::OnRender()
{
	if (m_keys['1']) m_lmValCeiling1 = max(m_lmValCeiling1 - 0.8f * m_frameTime, 0.0f);
	if (m_keys['2']) m_lmValCeiling1 = min(m_lmValCeiling1 + 0.8f * m_frameTime, 2.5f);
	if (m_keys['3']) m_lmValCeiling2 = max(m_lmValCeiling2 - 0.8f * m_frameTime, 0.0f);
	if (m_keys['4']) m_lmValCeiling2 = min(m_lmValCeiling2 + 0.8f * m_frameTime, 2.5f);
	if (m_keys['5']) m_lmValWall = max(m_lmValWall - 1.0f * m_frameTime, 0.0f);
	if (m_keys['6']) m_lmValWall = min(m_lmValWall + 1.0f * m_frameTime, 4.0f);
	if (m_keys['7']) m_lightIntensity = max(m_lightIntensity - 1.5f * m_frameTime, 0.0f);
	if (m_keys['8']) m_lightIntensity = min(m_lightIntensity + 1.5f * m_frameTime, 3.0f);
	if (m_keys[VK_NEXT])  m_exposure = max(m_exposure * powf(8.0f, -m_frameTime), 0.02f);
	if (m_keys[VK_PRIOR]) m_exposure = min(m_exposure * powf(8.0f,  m_frameTime), 8.0f);

	// NULL SRVs for resetting
	ID3D11ShaderResourceView *null[3 + 4 * SH_COEFF_VECTORS];
	memset(null, 0, sizeof(null));


	// Update camera
	UpdateCameraPosition(8.0f * m_frameTime);
	float4x4 mvp = m_camera.GetModelViewProjection();
	float3 camPos = m_camera.GetPosition();

	// Get light position from looping camera path
	static float3 lightPos;
	if (m_animateLight)
	{
		static float t = 0;
		t += 0.8f * m_frameTime;
		m_lightCamera.GetNodeAt(t, &lightPos, NULL, true);
	}


	// Update per frame constants
	PerFrame *pf;

	ID3D11DeviceContext *dev = m_context->GetDeviceContext();
	D3D11_MAPPED_SUBRESOURCE resource;

	dev->Map( m_perFrameCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &resource);
		pf = reinterpret_cast<PerFrame*> ( resource.pData );
		pf->mvp = mvp;
		pf->lightPos = lightPos;
		pf->camPos = camPos;
		pf->intensity = m_lightIntensity;
		pf->exposure = m_exposure;
		pf->lmModulate[0] = float4(m_lmValCeiling1, m_lmValCeiling2, 0, 0);
		pf->lmModulate[1] = float4(m_lmValWall, 0.0f, 0, 0);
	dev->Unmap( m_perFrameCB, 0 );

	float4x4 proj = CubemapProjectionMatrix(0.03f, 50.0f);
	float4x4 cubeMvp[PROBE_SLICES_PER_PASS];
	float4x4 cubeMvpInv[PROBE_SLICES_PER_PASS];

	/*
		Shadow map pass
	*/
	if (m_lightIntensity > 0.0f)
	{
		for (int i = 0; i < 6; i++)
		{
			float4x4 mv = CubemapModelviewMatrix(i);
			mv.translate(-lightPos);
			cubeMvp[i] = proj * mv;
		}

		D3D11_VIEWPORT vpShadowMap = { 0, 0, SHADOWMAP_SIZE, SHADOWMAP_SIZE, 0, 1 };
		dev->RSSetViewports(1, &vpShadowMap);

		dev->ClearDepthStencilView(m_shadowMapDepthDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
		dev->OMSetRenderTargets(1, &m_shadowMapRTV, m_shadowMapDepthDSV);

		SetupScene(SHADOW_PASS);
		RenderScene(SHADOW_PASS, cubeMvp, NULL, 6);

		dev->OMSetRenderTargets(0, NULL, NULL);
	}



	/*
		Light probe passes
	*/
	D3D11_VIEWPORT vpProbes = { 0, 0, PROBE_SIZE, PROBE_SIZE, 0, 1 };
	dev->RSSetViewports(1, &vpProbes);

	SetupScene(PROBE_PASS);

	dev->ClearDepthStencilView(m_probesDepthDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);

	int i = m_currProbe;
	for (int n = 0; n < PROBE_PASSES_PER_FRAME; n++)
	{
		// Compute all matrices for the chunk
		for (int c = 0; c < PROBE_CUBES_PER_PASS; c++)
		{
			for (int f = 0; f < 6; f++)
			{
				float4x4 mv = CubemapModelviewMatrix(f);
				mv.translate(-m_probePos[i]);
				float4x4 mvp = proj * mv;
				cubeMvp[c * 6 + f] = mvp;
				// This matrix transforms from screen space into world space.
				cubeMvpInv[c * 6 + f] = !mvp * translate(-1.0f, 1.0f, 0.0f) * scale4(2.0f / PROBE_SIZE, -2.0f / PROBE_SIZE, 1.0f);
			}
			i++;
			if (i >= m_probeCount) i = 0;
		}

		dev->OMSetRenderTargets(1, &m_probesChunkRTVs[n], 0 ) ;//m_probesDepthChunkDSVs[n]);

		RenderScene(PROBE_PASS, cubeMvp, cubeMvpInv, PROBE_SLICES_PER_PASS);
	}


	/*
		Update the SH coefficient volumes
	*/
	dev->PSSetShaderResources(0, elementsOf(null), null);

	D3D11_VIEWPORT shVP = { 0, 0, SIZE_X, SIZE_Y, 0, 1 };
	dev->RSSetViewports(1, &shVP);

	m_context->SetEffect(m_shEffect);
	m_context->Apply("SH", "Main");

	dev->IASetInputLayout(m_shCoeffIL);
	dev->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

	UINT stride = sizeof(ShVertex);
	UINT offset = 0;
	dev->IASetVertexBuffers(0, 1, &m_shCoeffVB, &stride, &offset);

	for (int i = 0; i < SH_COEFF_VECTORS; i++)
	{
		ID3D11RenderTargetView *rt[] = { m_shCoeffsRTV[0][i], m_shCoeffsRTV[1][i], m_shCoeffsRTV[2][i]};
		dev->OMSetRenderTargets(3, rt, NULL);

		dev->PSSetShaderResources(0, 1, &m_shTableSRV[i]);
		if (i == 0) dev->PSSetShaderResources(1, 1, &m_probesSRV);

		dev->Draw(PROBE_CUBES_PER_FRAME, m_currProbe);
	}



	// Step forward for the next frame
	m_currProbe += PROBE_CUBES_PER_FRAME;
	if (m_currProbe >= m_probeCount) m_currProbe -= m_probeCount;



	if (m_waitFrames)
	{
		m_waitFrames--;

		dev->PSSetShaderResources(0, elementsOf(null), null);

		m_context->Present();
		return;
	}

	m_context->SetRTToBackBuffer();


	/*
		Final pass rendering direct and indirect lighting to the backbuffer
	*/
	m_context->Clear(float4(0, 0, 0, 0), D3D11_CLEAR_DEPTH, 1.0f, 0);

		SetupScene(FINAL_PASS);
		RenderScene(FINAL_PASS, NULL, NULL, 0);


		if (m_lightIntensity > 0.0f) RenderBillboards(&lightPos, 1, 0.3f, float4(1, 1, 1, powf(m_lightIntensity / 3.0f, 0.25f)));
		if (m_showProbes) RenderBillboards((float3 *) &m_probePos, m_probeCount, 0.08f, float4(0.2f, 0.2f, 1.0f, 1));

		RenderCameraPath();
		RenderGUI();

		if (m_showHelp)
		{
			float4 color(1.0f, 0.6f, 0.25f, 1.0f);
			m_mainFont.DrawText(dev,
				"Controls:\n\n"
				"-, + - cycle wall color\n"
				"1, 2 - control ceiling light 1\n"
				"3, 4 - control ceiling light 2\n"
				"5, 6 - control wall lights\n"
				"7, 8 - control dynamic light\n"
				"9 - toggle dynamic light animation\n"
				"PgUp/PgDn - control exposure\n"
				"P - visualize probes",
				-0.65f, 0.0f, 0.1f, 0.1f, HA_LEFT, VA_CENTER, &color);
		}

		dev->PSSetShaderResources(0, elementsOf(null), null);

	m_context->Present();
}
