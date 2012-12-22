//-----------------------------------------------------------------------------
// File: Framework\Util\Model.cpp
// Copyright (c) 2005 ATI Technologies Inc. All rights reserved.
//-----------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include "D3D10Model.h"
#include "../D3D10/D3D10Context.h"

D3D10Model::D3D10Model()
{
	m_boneMatrixConstantType = CONSTANT_BUFFER;

	m_techConstantBuffer = NULL;
	m_boneMatrixConstantBuffer = NULL;
	
	m_techBuffer = NULL;
	m_boneMatrixBuffer = NULL;
	m_boneMatrixBufferRV = NULL;
	m_boneMatrixBufferVar = NULL;

	m_IL = NULL;
	m_IB = NULL;
	m_VB = NULL;

	m_ppMeshEffect = NULL;
	m_pD3D10Material = NULL;
}

D3D10Model::~D3D10Model()
{
	Clean();
}

void D3D10Model::Clean()
{
	SAFE_RELEASE(m_boneMatrixBuffer);
	SAFE_RELEASE(m_boneMatrixBufferRV);

	if (m_ppMeshEffect)
	{
		for (uint i = 0; i<m_numMeshs; i++)
		{
			SAFE_RELEASE(m_ppMeshEffect[i]);
		}
		delete [] m_ppMeshEffect;
	}
	if (m_pD3D10Material)
	{
		for (uint i = 0; i<m_numMaterials; i++)
		{
			SAFE_RELEASE(m_pD3D10Material[i].diffuseMap);
			SAFE_RELEASE(m_pD3D10Material[i].diffuseMapView);
			SAFE_RELEASE(m_pD3D10Material[i].bumpMap);
			SAFE_RELEASE(m_pD3D10Material[i].bumpMapView);
		}
		delete [] m_pD3D10Material;
	}
	SAFE_RELEASE(m_VB);
	SAFE_RELEASE(m_IB);
	SAFE_RELEASE(m_IL);
}

bool D3D10Model::Load(ID3D10Device* pdev, const TCHAR *name, const TCHAR *texture_path)
{
	if (!Model::Load(name, texture_path))
		return false;
	
	LoadTextures(pdev);
	// Create MeshEffect array
	m_ppMeshEffect = new ID3D10Effect*[m_numMeshs];
	for (uint i=0; i<m_numMeshs; i++)
		m_ppMeshEffect[i] = NULL;
	
	return true;
}

bool D3D10Model::Load(ID3D10Device* pdev, const TCHAR *name, const TCHAR *texture_path, ID3D10Effect *eff, BoneMatrixConstantType t)
{
	if (!Model::Load(name, texture_path))
		return false;
	
	LoadTextures(pdev);

	m_boneMatrixConstantType = t;

	//Load effect file
	m_ppMeshEffect = new ID3D10Effect*[m_numMeshs];
	for (unsigned int i=0; i<m_numMeshs; i++)
		SetMeshEffect(i, eff);

	D3D10_BUFFER_DESC vbdesc;
	D3D10_SHADER_RESOURCE_VIEW_DESC SRVDesc;
	switch (m_animationType)
	{
		case SKINNING_ANIMATION:
			switch (m_boneMatrixConstantType)
			{
				case CONSTANT_BUFFER:
					//Create Constant buffer
					m_techConstantBuffer = eff->GetTechniqueByName( "techConstantBuffer" );
					m_boneMatrixConstantBuffer = eff->GetVariableByName( "boneMatrixConstantBuffer" )->AsMatrix();
					break;

				case BUFFER:
					//Create buffer
					m_techBuffer = eff->GetTechniqueByName( "techBuffer" );
					m_boneMatrixBufferVar = eff->GetVariableByName( "boneMatrixBuffer" )->AsShaderResource();

					ZeroMemory( &vbdesc, sizeof(vbdesc) );
					vbdesc.ByteWidth = m_numBones*sizeof(float4x4);
					vbdesc.Usage = D3D10_USAGE_DYNAMIC;
					vbdesc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
					vbdesc.CPUAccessFlags = D3D10_CPU_ACCESS_WRITE;
					vbdesc.MiscFlags = 0;
					pdev->CreateBuffer(&vbdesc, NULL, &m_boneMatrixBuffer);
					
					ZeroMemory( &SRVDesc, sizeof(SRVDesc) );
					SRVDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
					SRVDesc.ViewDimension = D3D10_SRV_DIMENSION_BUFFER;
					SRVDesc.Buffer.ElementOffset = 0;
					SRVDesc.Buffer.ElementWidth = m_numBones*4;
					pdev->CreateShaderResourceView(m_boneMatrixBuffer, &SRVDesc, &m_boneMatrixBufferRV);
					break;
			}
			break;

		case STATIC_GEOMETRY:
			m_techStatic = eff->GetTechniqueByName( "techStatic" );
			break;
	}	

	return true;
}

bool D3D10Model::Load(ID3D10Device* pdev, const TCHAR *name, const TCHAR *texture_path, LPCWSTR shader, BoneMatrixConstantType t)
{
	ID3D10Effect* pEffect;
	ID3D10Blob *ppErrors;
	if (SUCCEEDED(D3DX10CreateEffectFromFile(shader, NULL, NULL, "fx_4_0", D3D10_SHADER_ENABLE_STRICTNESS, D3D10_EFFECT_SINGLE_THREADED, pdev, NULL, NULL, &pEffect, &ppErrors, NULL)))
	{

		/*ID3D10Blob *assembly = NULL;
		D3D10DisassembleEffect(pEffect, FALSE, &assembly);
		if (assembly)
		{
			// Write the assembly to file
			FILE *file = _tfopen(_T("Dump.txt"), _T("wb"));
			if (file)
			{
				fwrite(assembly->GetBufferPointer(), assembly->GetBufferSize(), 1, file);
				fclose(file);
			}

			// Write the assembly to debug output
			//OutputDebugStringA((char *) assembly->GetBufferPointer());

			assembly->Release();
		}*/

	}
	else
	{
		char *lpError = (char*) ppErrors->GetBufferPointer();
		OutputDebugStringA(lpError);
		return false;
	}
	if (!D3D10Model::Load(pdev, name, texture_path, pEffect, t))
		return false;

	SAFE_RELEASE(pEffect);

	return true;
}

bool D3D10Model::LoadMeshEffect(ID3D10Device* pdev, int i, LPCWSTR shader)
{
	if (!m_ppMeshEffect)
		return false;

	//Load effect file
	ID3D10Blob *ppErrors;
	if (FAILED(D3DX10CreateEffectFromFile(shader, NULL, NULL, "fx_4_0", D3D10_SHADER_ENABLE_STRICTNESS, D3D10_EFFECT_SINGLE_THREADED, pdev, NULL, NULL, &m_ppMeshEffect[i], &ppErrors, NULL)))
	{
		char *lpError = (char*)ppErrors->GetBufferPointer();
		return false;
	}
	return true;
}

void D3D10Model::LoadTextures(ID3D10Device* pdev)
{
	TCHAR path[128];

	m_pD3D10Material = new D3D10Material[m_numMaterials];
	for (DWORD i=0; i<m_numMaterials; i++)
	{
		m_pD3D10Material[i].diffuseMap = NULL;
		m_pD3D10Material[i].bumpMap = NULL;
		m_pD3D10Material[i].diffuseMapView = NULL;
		m_pD3D10Material[i].bumpMapView = NULL;

		_stprintf(path, _T("%s%s"), m_texturePath, m_pMaterial[i].diffuseMapFile);
		if (!FAILED(D3DX10CreateTextureFromFile(pdev, path, NULL, NULL, (ID3D10Resource **)&m_pD3D10Material[i].diffuseMap, NULL)))
		{
			D3D10_TEXTURE2D_DESC desc;
			m_pD3D10Material[i].diffuseMap->GetDesc(&desc);

			D3D10_SHADER_RESOURCE_VIEW_DESC srvDesc;
			ZeroMemory(&srvDesc, sizeof(srvDesc));
			srvDesc.Format = desc.Format;
			srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2D;
			srvDesc.Texture2D.MipLevels = desc.MipLevels;

			pdev->CreateShaderResourceView(m_pD3D10Material[i].diffuseMap, &srvDesc, &m_pD3D10Material[i].diffuseMapView);
		}
		_stprintf(path, _T("%s%s"), m_texturePath, m_pMaterial[i].bumpMapFile);
		if (!FAILED(D3DX10CreateTextureFromFile(pdev, path, NULL, NULL, (ID3D10Resource **)&m_pD3D10Material[i].bumpMap, NULL)))
		{
			D3D10_TEXTURE2D_DESC desc;
			m_pD3D10Material[i].diffuseMap->GetDesc(&desc);

			D3D10_SHADER_RESOURCE_VIEW_DESC srvDesc;
			ZeroMemory(&srvDesc, sizeof(srvDesc));
			srvDesc.Format = desc.Format;
			srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2D;
			srvDesc.Texture2D.MipLevels = desc.MipLevels;

			pdev->CreateShaderResourceView(m_pD3D10Material[i].bumpMap, &srvDesc, &m_pD3D10Material[i].bumpMapView);
		}
	}
}

bool D3D10Model::CreateInputLayout(ID3D10Device* pdev)
{
	const D3D10_INPUT_ELEMENT_DESC animationlayout4[] =
	{
		{ "NORMAL",         0, DXGI_FORMAT_R32G32B32_FLOAT,       0,    0,  D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD",       0, DXGI_FORMAT_R32G32_FLOAT,          0,    12, D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "POSITION",       0, DXGI_FORMAT_R32G32_FLOAT,          0,    20, D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "POSITION",       1, DXGI_FORMAT_R32G32B32_FLOAT,       0,    28, D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "POSITION",       2, DXGI_FORMAT_R32G32B32_FLOAT,       0,    40, D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "POSITION",       3, DXGI_FORMAT_R32G32B32_FLOAT,       0,    52, D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "BLENDWEIGHT",    0, DXGI_FORMAT_R32G32B32A32_FLOAT,    0,    64, D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "BLENDINDICES",   0, DXGI_FORMAT_R32G32B32A32_FLOAT,    0,    80, D3D10_INPUT_PER_VERTEX_DATA, 0 },
	};

	const D3D10_INPUT_ELEMENT_DESC animationlayout[] =
	{
		{ "NORMAL",         0, DXGI_FORMAT_R32G32B32_FLOAT,       0,    0,  D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD",       0, DXGI_FORMAT_R32G32_FLOAT,          0,    12, D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "POSITION",       0, DXGI_FORMAT_R32G32B32_FLOAT,       0,    20, D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "BLENDWEIGHT",    0, DXGI_FORMAT_R32G32B32A32_FLOAT,    0,    32, D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "BLENDINDICES",   0, DXGI_FORMAT_R32G32B32A32_FLOAT,    0,    48, D3D10_INPUT_PER_VERTEX_DATA, 0 },
	};

	const D3D10_INPUT_ELEMENT_DESC modellayout[] =
	{
		{ "POSITION",       0, DXGI_FORMAT_R32G32B32_FLOAT,       0,    0,  D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "NORMAL",         0, DXGI_FORMAT_R32G32B32_FLOAT,       0,    12, D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD",       0, DXGI_FORMAT_R32G32_FLOAT,          0,    24, D3D10_INPUT_PER_VERTEX_DATA, 0 },
	};

	D3D10_PASS_DESC PassDesc;
	switch (m_animationType)
	{
		case SKINNING_ANIMATION:
			switch (m_boneMatrixConstantType)
			{
				case CONSTANT_BUFFER:
					m_techConstantBuffer->GetPassByIndex( 0 )->GetDesc( &PassDesc );
					if (FAILED( pdev->CreateInputLayout( animationlayout, (sizeof(animationlayout)/sizeof(animationlayout[0])), PassDesc.pIAInputSignature, PassDesc.IAInputSignatureSize, &m_IL)) )
						return false;
					break;

				case BUFFER:
					m_techBuffer->GetPassByIndex( 0 )->GetDesc( &PassDesc );
					if (FAILED( pdev->CreateInputLayout( animationlayout, (sizeof(animationlayout)/sizeof(animationlayout[0])), PassDesc.pIAInputSignature, PassDesc.IAInputSignatureSize, &m_IL)) )
						return false;
					break;
			}
			break;

		case STATIC_GEOMETRY:
			m_techStatic->GetPassByIndex( 0 )->GetDesc( &PassDesc );
			if (FAILED( pdev->CreateInputLayout( modellayout, 3, PassDesc.pIAInputSignature, PassDesc.IAInputSignatureSize, &m_IL)) )
				return false;
			break;
	}
	return true;
}

bool D3D10Model::CreateVertexBuffer(ID3D10Device* pdev)
{
	D3D10_BUFFER_DESC bd;
	D3D10_SUBRESOURCE_DATA initData;
	float* dest;
	float* odest;

	switch (m_animationType)
	{
		case SKINNING_ANIMATION:
			switch (m_boneMatrixConstantType)
			{
				case CONSTANT_BUFFER:
				case BUFFER:
					odest = dest = new float[m_numVertices*(sizeof(BLENDVERTEX)/sizeof(float))];
					for (unsigned int i=0; i<m_numVertices; i++) 
					{
						(*dest++) = m_pVNormal[i].x;
						(*dest++) = m_pVNormal[i].y;
						(*dest++) = m_pVNormal[i].z;
						(*dest++) = m_pVTexcoord[i].x;
						(*dest++) = m_pVTexcoord[i].y;

						//for (unsigned int j=0; j<4; j++)
						for (unsigned int j=0; j<1; j++)
						{
							if (j < m_pVSkinningData[i].numBones)
							{
								(*dest++) = m_pVSkinningData[i].pBlendData[j].offset.x;
								(*dest++) = m_pVSkinningData[i].pBlendData[j].offset.y;
								(*dest++) = m_pVSkinningData[i].pBlendData[j].offset.z;
							}
							else
							{
								(*dest++) = 0.0f;
								(*dest++) = 0.0f;
								(*dest++) = 0.0f;
							}
						}
						
						for (unsigned int j=0; j<4; j++)
						{
							if (j < m_pVSkinningData[i].numBones)
							{
								(*dest++) = m_pVSkinningData[i].pBlendData[j].weight;
							}
							else
							{
								(*dest++) = 0.0f;
							}
						}
						for (unsigned int j=0; j<4; j++)
						{
							if (j < m_pVSkinningData[i].numBones)
							{
								(*dest++) = (float)m_pVSkinningData[i].pBlendData[j].bone;
							}
							else
							{
								(*dest++) = 0.0f;
							}
						}
					}
					bd.Usage = D3D10_USAGE_IMMUTABLE;
					bd.ByteWidth = m_numVertices*sizeof(BLENDVERTEX);
					bd.BindFlags = D3D10_BIND_VERTEX_BUFFER;
					bd.CPUAccessFlags = 0;
					bd.MiscFlags = 0;
					initData.pSysMem = odest;
					if (FAILED(pdev->CreateBuffer(&bd, &initData, &m_VB))) 
					{
						delete [] odest;
						return false;
					}
					delete [] odest;
					break;
			}
			break;
		
		case STATIC_GEOMETRY:
			odest = dest = new float[m_numVertices*8];
			for (unsigned int i=0; i<m_numVertices; i++)
			{
				(*dest++) = m_pVPosition[i].x*1.0f;
				(*dest++) = m_pVPosition[i].y*1.0f;
				(*dest++) = m_pVPosition[i].z*1.0f;
				(*dest++) = m_pVNormal[i].x;
				(*dest++) = m_pVNormal[i].y;
				(*dest++) = m_pVNormal[i].z;
				(*dest++) = m_pVTexcoord[i].x;
				(*dest++) = m_pVTexcoord[i].y;
			}
			bd.Usage = D3D10_USAGE_IMMUTABLE;
			bd.ByteWidth = m_numVertices*8*(sizeof(float));
			bd.BindFlags = D3D10_BIND_VERTEX_BUFFER;
			bd.CPUAccessFlags = 0;
			bd.MiscFlags = 0;
			initData.pSysMem = odest;
			if (FAILED(pdev->CreateBuffer(&bd, &initData, &m_VB))) 
			{
				delete [] odest;
				return false;
			}
			delete [] odest;
			break;
	}
	return true;
}

bool D3D10Model::CreateIndexBuffer(ID3D10Device* pdev)
{
	D3D10_BUFFER_DESC bd;
	D3D10_SUBRESOURCE_DATA initData;
	DWORD numIndices;
	unsigned int* iDest; 
	unsigned int* oiDest;

	numIndices = 0;
	for (unsigned int i=0; i<m_numMeshs; i++)
	{
		for (unsigned int j=0; j<m_pMesh[i].GetNumPrimitives(); j++)
		{
			m_pMesh[i].GetPrimitive(j).baseIndex = numIndices;
			numIndices += m_pMesh[i].GetPrimitive(j).numIndices;
		}
	}

	oiDest = iDest = new unsigned int[numIndices];
	for (unsigned int i=0; i<m_numMeshs; i++)
	{
		for (unsigned int j=0; j<m_pMesh[i].GetNumPrimitives(); j++)
		{
			for (unsigned int k=0; k<m_pMesh[i].GetPrimitive(j).numIndices; k++)
			{
				(*iDest++) = m_pMesh[i].GetPrimitive(j).pIndices[k];
			}
		}
	}
	// Create the index buffer
	bd.Usage = D3D10_USAGE_IMMUTABLE;
	bd.ByteWidth = numIndices*sizeof(DWORD);
	bd.BindFlags = D3D10_BIND_INDEX_BUFFER;
	bd.CPUAccessFlags = 0;
	bd.MiscFlags = 0;
	initData.pSysMem = oiDest;
	if (FAILED(pdev->CreateBuffer(&bd, &initData, &m_IB))) 
	{
		delete [] oiDest;
		return false;
	}
	delete [] oiDest;
	return true;
}

bool D3D10Model::CreateResources(ID3D10Device* pdev)
{
	if (!CreateInputLayout(pdev)) return false;
	if (!CreateVertexBuffer(pdev)) return false;
	if (!CreateIndexBuffer(pdev)) return false;

	return true;
}

void D3D10Model::Render(ID3D10Device* pdev, const float4x4& mvp, const float3& cmaerapos) 
{
	UINT stride[1];
	UINT offset[1] = {0};
	float4x4* pMatrices;

	pdev->IASetInputLayout(m_IL);
	pdev->IASetIndexBuffer(m_IB, DXGI_FORMAT_R32_UINT, 0);
	switch (m_animationType)
	{
		case SKINNING_ANIMATION:
			stride[0] = sizeof(BLENDVERTEX);
			pdev->IASetVertexBuffers(0, 1, &m_VB, stride, offset);
			break;

		case STATIC_GEOMETRY:
			stride[0] = sizeof(STATICVERTEX);
			pdev->IASetVertexBuffers(0, 1, &m_VB, stride, offset);
			break;
	}

	for (unsigned int i=0; i<m_numMeshs; i++)
	{
		switch (m_animationType)
		{
			case SKINNING_ANIMATION:
				switch (m_boneMatrixConstantType)
				{
					case CONSTANT_BUFFER:
						m_techConstantBuffer->GetPassByName("P0")->Apply(0);
						for (DWORD b=0; b<m_numBones; b++)
						{
							m_boneMatrixConstantBuffer->SetMatrixArray( (float*)&(GetBone(b).GetTransformMatrix()), b, 1);
						}
						break;

					case BUFFER:
						m_techBuffer->GetPassByName("P0")->Apply(0);
						m_boneMatrixBuffer->Map( D3D10_MAP_WRITE_DISCARD, 0, (void**)&pMatrices );
						for (DWORD b=0; b<m_numBones; b++)
						{
							pMatrices[b] = GetBone(b).GetTransformMatrix();
						}
						m_boneMatrixBuffer->Unmap();
						m_boneMatrixBufferVar->SetResource( m_boneMatrixBufferRV );
						break;
				}
				break;

			case STATIC_GEOMETRY:
				m_ppMeshEffect[i]->GetTechniqueByName("techStatic")->GetPassByName("P0")->Apply(0);
				break;
		}
		m_ppMeshEffect[i]->GetVariableByName("mvp")->AsMatrix()->SetMatrix((float *)&mvp);
		m_ppMeshEffect[i]->GetVariableByName("cameraPos")->AsVector()->SetFloatVector((float *)&cmaerapos);
		if (m_pMesh[i].GetMaterialIndex() >= 0)
			m_ppMeshEffect[i]->GetVariableByName("diffuseMap")->AsShaderResource()->SetResource(m_pD3D10Material[m_pMesh[i].GetMaterialIndex()].diffuseMapView);

		for (unsigned int p=0; p<m_pMesh[i].GetNumPrimitives(); p++) 
		{
			pdev->IASetPrimitiveTopology(m_pMesh[i].GetPrimitive(p).type);
			pdev->DrawIndexed(m_pMesh[i].GetPrimitive(p).numIndices, m_pMesh[i].GetPrimitive(p).baseIndex, 0);
		}
	}
}
