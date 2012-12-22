//-----------------------------------------------------------------------------
// File: Framework\Model\Model.cpp
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include "Model.h"

Model::Model()
{
	m_animationType = STATIC_GEOMETRY;
	m_numVertices = 0;
	m_pVPosition = NULL;
	m_pVNormal = NULL;
	m_pVTangent = NULL;
	m_pVBinormal = NULL;
	m_pVTexcoord = NULL;
	m_pVColor = NULL;
	m_pVSkinningData = NULL;
	m_numTriangles = 0;
	m_pTriangle = NULL;

	m_numMaterials = 0;
	m_numMeshs = 0;
	m_numBones = 0;
	m_pMaterial = NULL;
	m_pMesh = NULL;
	m_ppMeshHie = NULL;
	m_pBone = NULL;
	m_ppBoneHie = NULL;
}

Model::~Model()
{
	Clean();
}

void Model::Clean()
{
	delete [] m_pVPosition;
	delete [] m_pVNormal;
	delete [] m_pVTangent;
	delete [] m_pVBinormal;
	delete [] m_pVTexcoord;
	delete [] m_pVColor;
	for (DWORD v = 0; v < m_numVertices; v++)
	{
		delete [] m_pVSkinningData[v].pBlendData;
	}
	delete [] m_pVSkinningData;
	delete [] m_pTriangle;

	delete [] m_pMaterial;
	delete [] m_pMesh;
	delete [] m_pBone;
	delete [] m_ppMeshHie;
	delete [] m_ppBoneHie;
}

bool Model::Load(const TCHAR *name, const TCHAR *texture_path) 
{
	FILE *fp;
	DWORD Version;
	OffsetTable offsetTable;

	fp = _tfopen(name, _T("rb"));
	if (!fp)
		return false;

	char Header[9] = "AtiModel";
	char ReadMark[9];

	fread(ReadMark, 1, 8, fp);
	ReadMark[8] = '\0';
	if (stricmp(ReadMark, Header))
		goto ErrorExit;

	fread(&m_animationType, sizeof(int), 1, fp);
	fread(&Version, sizeof(DWORD), 1, fp);

	fread(&offsetTable, sizeof(OffsetTable), 1, fp);

	if (texture_path)
		_tcscpy(m_texturePath, texture_path);
	else
		_tcscpy(m_texturePath, _T(""));

	fseek(fp, offsetTable.materialChunk, SEEK_SET);
	fread(&m_numMaterials, sizeof(DWORD), 1, fp);
	m_pMaterial = new Material[m_numMaterials];
	for (DWORD i = 0; i < m_numMaterials; i++)
	{
		fseek(fp, 64, SEEK_CUR);
		fseek(fp, sizeof(float4), SEEK_CUR);
		fseek(fp, sizeof(float4), SEEK_CUR);
		fseek(fp, sizeof(float), SEEK_CUR);

#ifdef _UNICODE
		char buf[64];
		fread(buf, 1, 64, fp);
		MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, buf, -1, m_pMaterial[i].diffuseMapFile, sizeof(m_pMaterial[i].diffuseMapFile));
#else
		fread(m_pMaterial[i].diffuseMapFile, 1, 64, fp);
#endif

		fread(buf, 1, 64, fp);
#ifdef _UNICODE
		fread(buf, 1, 64, fp);
		MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, buf, -1, m_pMaterial[i].bumpMapFile, sizeof(m_pMaterial[i].bumpMapFile));
#else
		fread(m_pMaterial[i].bumpMapFile, 1, 64, fp);
#endif
		fread(buf, 1, 64, fp);
		fread(buf, 1, 64, fp);
	}

	fseek(fp, offsetTable.vertexChunk, SEEK_SET);
	fread(&m_numVertices, sizeof(DWORD), 1, fp);
	m_pVPosition = new float3[m_numVertices];
	m_pVNormal = new float3[m_numVertices];
	m_pVTangent = new float3[m_numVertices];
	m_pVBinormal = new float3[m_numVertices];
	m_pVTexcoord = new float2[m_numVertices];
	m_pVColor = new float4[m_numVertices];
	m_pVSkinningData = new SkinningData[m_numVertices];
	for (DWORD i = 0; i < m_numVertices; i++)
	{
		m_pVSkinningData[i].pBlendData = NULL;
		m_pVSkinningData[i].numBones = 0;
	}
	fread(m_pVPosition, sizeof(float3), m_numVertices, fp);
	fread(m_pVNormal, sizeof(float3), m_numVertices, fp);
	fread(m_pVTangent, sizeof(float3), m_numVertices, fp);
	fread(m_pVBinormal, sizeof(float3), m_numVertices, fp);
	fread(m_pVTexcoord, sizeof(float2), m_numVertices, fp);
	fread(m_pVColor, sizeof(float4), m_numVertices, fp);
	for (DWORD i = 0; i<m_numVertices; i++)
	{
		fread(&m_pVSkinningData[i].numBones, sizeof(DWORD), 1, fp);
		if (m_pVSkinningData[i].numBones)
		{
			m_pVSkinningData[i].pBlendData = new BlendData[m_pVSkinningData[i].numBones];
			for (DWORD b = 0; b < m_pVSkinningData[i].numBones; b++)
			{
				fread(&m_pVSkinningData[i].pBlendData[b].bone, sizeof(DWORD), 1, fp);
				fread(&m_pVSkinningData[i].pBlendData[b].offset, sizeof(float3), 1, fp);
				fread(&m_pVSkinningData[i].pBlendData[b].weight, sizeof(float), 1, fp);
			}
		}
	}

	fseek(fp, offsetTable.triangleChunk, SEEK_SET);
	fread(&m_numTriangles, sizeof(DWORD), 1, fp);
	m_pTriangle = new Triangle[m_numTriangles];
	for (DWORD i = 0; i < m_numTriangles; i++)
	{
		fread(&m_pTriangle[i].v[0], sizeof(DWORD), 1, fp);
		fread(&m_pTriangle[i].v[1], sizeof(DWORD), 1, fp);
		fread(&m_pTriangle[i].v[2], sizeof(DWORD), 1, fp);
		fread(&m_pTriangle[i].normal, sizeof(float3), 1, fp);
	}

	fseek(fp, offsetTable.meshChunk, SEEK_SET);
	fread(&m_numMeshs, sizeof(DWORD), 1, fp);
	m_pMesh = new Mesh[m_numMeshs];
	for (DWORD i = 0; i < m_numMeshs; i++)
	{
		m_pMesh[i].m_ID = i;
		fread(&m_pMesh[i].m_Name, 1, 64, fp);
		fread(&m_pMesh[i].m_materialIndex, sizeof(int), 1, fp);
		fread(&m_pMesh[i].m_baseVertex, sizeof(DWORD), 1, fp);
		fread(&m_pMesh[i].m_numVertices, sizeof(DWORD), 1, fp);
		fread(&m_pMesh[i].m_baseTriangle, sizeof(DWORD), 1, fp);
		fread(&m_pMesh[i].m_numTriangles, sizeof(DWORD), 1, fp);

		fread(&m_pMesh[i].m_numBones, sizeof(DWORD), 1, fp);
		m_pMesh[i].m_pBone = new DWORD[m_pMesh[i].m_numBones];
		fread(m_pMesh[i].m_pBone, sizeof(DWORD), m_pMesh[i].m_numBones, fp);

		fread(&m_pMesh[i].m_parent, sizeof(int),1, fp);
		fread(&m_pMesh[i].m_numChildren, sizeof(DWORD), 1, fp);
		m_pMesh[i].m_pChildren = new DWORD[m_pMesh[i].m_numChildren];
		fread(m_pMesh[i].m_pChildren, sizeof(DWORD), m_pMesh[i].m_numChildren, fp);

		fread(&m_pMesh[i].m_numPrimitives, sizeof(DWORD), 1, fp);
		m_pMesh[i].m_pPrimitive = new Primitive[m_pMesh[i].m_numPrimitives];
		for (DWORD p = 0; p < m_pMesh[i].m_numPrimitives; p++)
		{
			int Type;
			fread(&Type, sizeof(int), 1, fp);
			switch (Type)
			{
				case PL_TRIANGLE_STRIP:
					m_pMesh[i].m_pPrimitive[p].type = D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
					break;

				case PL_TRIANGLE_LIST:
					m_pMesh[i].m_pPrimitive[p].type = D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
					break;

				case PL_TRIANGLE_FAN:
					//pMesh[i].m_pPrimitive[p].type = TRIANGLE_FAN;
					break;
			}
			fread(&m_pMesh[i].m_pPrimitive[p].numIndices, sizeof(DWORD), 1, fp);
			m_pMesh[i].m_pPrimitive[p].pIndices = new DWORD[m_pMesh[i].m_pPrimitive[p].numIndices];
			fread(m_pMesh[i].m_pPrimitive[p].pIndices, sizeof(DWORD), m_pMesh[i].m_pPrimitive[p].numIndices, fp);
		}

		fread(&m_pMesh[i].m_animation.m_positionKeys.numKeys, sizeof(DWORD), 1, fp);
		m_pMesh[i].m_animation.m_positionKeys.pKey = new AnimationKey[m_pMesh[i].m_animation.m_positionKeys.numKeys];
		for (DWORD p = 0; p < m_pMesh[i].m_animation.m_positionKeys.numKeys; p++)
		{
			fread(&m_pMesh[i].m_animation.m_positionKeys.pKey[p].startTime, sizeof(int), 1, fp);
			fread(&m_pMesh[i].m_animation.m_positionKeys.pKey[p].endTime, sizeof(int), 1, fp);
			fread(m_pMesh[i].m_animation.m_positionKeys.pKey[p].value, sizeof(float), 3, fp);
		}
		fread(&m_pMesh[i].m_animation.m_rotationKeys.numKeys, sizeof(DWORD), 1, fp);
		m_pMesh[i].m_animation.m_rotationKeys.pKey = new AnimationKey[m_pMesh[i].m_animation.m_rotationKeys.numKeys];
		for (DWORD p = 0; p < m_pMesh[i].m_animation.m_rotationKeys.numKeys; p++)
		{
			fread(&m_pMesh[i].m_animation.m_rotationKeys.pKey[p].startTime, sizeof(int), 1, fp);
			fread(&m_pMesh[i].m_animation.m_rotationKeys.pKey[p].endTime, sizeof(int), 1, fp);
			fread(m_pMesh[i].m_animation.m_rotationKeys.pKey[p].value, sizeof(float), 4, fp);
		}
		fread(&m_pMesh[i].m_animation.m_scaleKeys.numKeys, sizeof(DWORD), 1, fp);
		m_pMesh[i].m_animation.m_scaleKeys.pKey = new AnimationKey[m_pMesh[i].m_animation.m_scaleKeys.numKeys];
		for (DWORD p = 0; p < m_pMesh[i].m_animation.m_scaleKeys.numKeys; p++)
		{
			fread(&m_pMesh[i].m_animation.m_scaleKeys.pKey[p].startTime, sizeof(int), 1, fp);
			fread(&m_pMesh[i].m_animation.m_scaleKeys.pKey[p].endTime, sizeof(int), 1, fp);
			fread(m_pMesh[i].m_animation.m_scaleKeys.pKey[p].value, sizeof(float), 3, fp);
		}
	}

	fseek(fp, offsetTable.boneChunk, SEEK_SET);
	fread(&m_numBones, sizeof(DWORD), 1, fp);
	m_pBone = new Bone[m_numBones];
	for (DWORD i = 0; i < m_numBones; i++)
	{
		m_pBone[i].m_ID = i;
		fseek(fp, 64, SEEK_CUR);
		fread(&m_pBone[i].m_parent, sizeof(int),1, fp);
		fread(&m_pBone[i].m_numChildren, sizeof(DWORD), 1, fp);
		if (m_pBone[i].m_numChildren)
		{
			m_pBone[i].m_pChildren = new DWORD[m_pBone[i].m_numChildren];
			fread(m_pBone[i].m_pChildren, sizeof(DWORD), m_pBone[i].m_numChildren, fp);
		}
		fread(&m_pBone[i].m_animation.m_positionKeys.numKeys, sizeof(DWORD), 1, fp);
		m_pBone[i].m_animation.m_positionKeys.pKey = new AnimationKey[m_pBone[i].m_animation.m_positionKeys.numKeys];
		for (DWORD p = 0; p < m_pBone[i].m_animation.m_positionKeys.numKeys; p++)
		{
			fread(&m_pBone[i].m_animation.m_positionKeys.pKey[p].startTime, sizeof(int), 1, fp);
			fread(&m_pBone[i].m_animation.m_positionKeys.pKey[p].endTime, sizeof(int), 1, fp);
			fread(m_pBone[i].m_animation.m_positionKeys.pKey[p].value, sizeof(float), 3, fp);
		}
		fread(&m_pBone[i].m_animation.m_rotationKeys.numKeys, sizeof(DWORD), 1, fp);
		m_pBone[i].m_animation.m_rotationKeys.pKey = new AnimationKey[m_pBone[i].m_animation.m_rotationKeys.numKeys];
		for (DWORD p = 0; p < m_pBone[i].m_animation.m_rotationKeys.numKeys; p++)
		{
			fread(&m_pBone[i].m_animation.m_rotationKeys.pKey[p].startTime, sizeof(int), 1, fp);
			fread(&m_pBone[i].m_animation.m_rotationKeys.pKey[p].endTime, sizeof(int), 1, fp);
			fread(m_pBone[i].m_animation.m_rotationKeys.pKey[p].value, sizeof(float), 4, fp);
		}
		fread(&m_pBone[i].m_animation.m_scaleKeys.numKeys, sizeof(DWORD), 1, fp);
		m_pBone[i].m_animation.m_scaleKeys.pKey = new AnimationKey[m_pBone[i].m_animation.m_scaleKeys.numKeys];
		for (DWORD p = 0; p < m_pBone[i].m_animation.m_scaleKeys.numKeys; p++)
		{
			fread(&m_pBone[i].m_animation.m_scaleKeys.pKey[p].startTime, sizeof(int), 1, fp);
			fread(&m_pBone[i].m_animation.m_scaleKeys.pKey[p].endTime, sizeof(int), 1, fp);
			fread(m_pBone[i].m_animation.m_scaleKeys.pKey[p].value, sizeof(float), 3, fp);
		}
	}
	fclose(fp);

	if (m_numMeshs)
	{
		m_ppMeshHie = new Transform*[m_numMeshs];
		BuildMeshHie();
	}

	if (m_numBones)
	{
		m_ppBoneHie = new Transform*[m_numBones];
		BuildBoneHie();
		//ReorderBone();
	}

	return true;

ErrorExit:
	fclose(fp);
	return false;
}

void Model::RecursionMeshTraverse(Transform **lphie, Transform *lpct)
{
	lphie[m_traverseIndex++] = lpct;
	for (DWORD i = 0; i < lpct->m_numChildren; i++)
	{
		RecursionMeshTraverse(lphie, &m_pMesh[lpct->m_pChildren[i]]);
	}
}

void Model::BuildMeshHie()
{
	DWORD i;
	for (i = 0; i < m_numMeshs; i++)
	{
		if (m_pMesh[i].m_parent == -1)
			break;
	}
	m_traverseIndex = 0;
	RecursionMeshTraverse(m_ppMeshHie, &m_pMesh[i]);
}

void Model::RecursionBoneTraverse(Transform **lphie, Transform *lpct)
{
	lphie[m_traverseIndex++] = lpct;
	for (DWORD i = 0; i < lpct->m_numChildren; i++)
	{
		RecursionBoneTraverse(lphie, &m_pBone[lpct->m_pChildren[i]]);
	}
}

void Model::BuildBoneHie()
{
	DWORD i;
	if (m_numBones)
	{
		for (i=0; i<m_numBones; i++)
		{
			if (m_pBone[i].m_parent == -1)
				break;
		}
		m_traverseIndex = 0;
		RecursionBoneTraverse(m_ppBoneHie, &m_pBone[i]);
	}
}


//void Model::ReorderBone() 
//{
//	Bone* pNewBone = new Bone[m_numBones];
//	int* Map;
//
//	Map = new int[m_numBones];
//	for (uint i=0; i<m_numBones; ++i)
//	{
//		pNewBone[i] = m_pBone[m_ppBoneHie[i]->m_ID];
//		Map[m_ppBoneHie[i]->m_ID] = i;
//	}	
//	delete [] m_pBone;
//	m_pBone = pNewBone;	
//
//	for (uint i=0; i<m_numBones; ++i)
//	{
//		if (m_pBone[i].m_parent >= 0)
//			m_pBone[i].m_parent = Map[m_pBone[i].m_parent];
//	}
//	for (uint i=0; i<m_numVertices; i++)
//	{
//		for (uint b=0; b<m_pVSkinningData[i].numBones; b++)
//		{
//			m_pVSkinningData[i].pBlendData[b].bone = Map[m_pVSkinningData[i].pBlendData[b].bone];
//		}
//	}
//	delete [] Map;
//	for (uint i=0; i<m_numBones; ++i)
//	{
//		m_ppBoneHie[i] = &m_pBone[i];
//	}
//}

bool Model::Play(float step, bool bloop)
{
	Transform::Play(step, bloop);
	switch (m_animationType)
	{
		case STATIC_GEOMETRY:
			break;

		case SKINNING_ANIMATION:
			for (DWORD i = 0; i < m_numBones; i++)
			{
				m_ppBoneHie[i]->Play(step, bloop);
				if (m_ppBoneHie[i]->m_parent == -1)
					m_ppBoneHie[i]->SetTransformMatrix(m_ppBoneHie[i]->m_animation.GetTransformMatrix() * m_transformMatrix);
				else
					m_ppBoneHie[i]->SetTransformMatrix(m_ppBoneHie[i]->m_animation.GetTransformMatrix() * m_pBone[m_ppBoneHie[i]->m_parent].GetTransformMatrix());
			}
			break;

		case HIERARCHY_ANIMATION:
			for (DWORD i = 0; i < m_numMeshs; i++)
			{
				m_ppMeshHie[i]->Play(step, bloop);
				if (m_ppMeshHie[i]->m_parent == -1)
					m_ppMeshHie[i]->SetTransformMatrix(m_ppMeshHie[i]->m_animation.GetTransformMatrix() * m_transformMatrix);
				else
					m_ppMeshHie[i]->SetTransformMatrix(m_ppMeshHie[i]->m_animation.GetTransformMatrix() * m_pMesh[m_ppMeshHie[i]->m_parent].GetTransformMatrix());
			}
			break;

		case SHAPE_ANIMATION:
			for (DWORD i = 0; i < m_numMeshs; i++)
			{
				m_ppMeshHie[i]->Play(step, bloop);
				if (m_ppMeshHie[i]->m_parent == -1)
					m_ppMeshHie[i]->SetTransformMatrix(m_ppMeshHie[i]->m_animation.GetTransformMatrix() * m_transformMatrix);
				else
					m_ppMeshHie[i]->SetTransformMatrix(m_ppMeshHie[i]->m_animation.GetTransformMatrix() * m_pMesh[m_ppMeshHie[i]->m_parent].GetTransformMatrix());
			}
			break;
	}
	return true;
}

bool Model::PlayAndSaveAnimationKey(float step, bool bloop)
{
	Transform::PlayAndSaveAnimationKey(step, bloop);
	for (DWORD i = 0; i < m_numBones; i++)
		m_pBone[i].PlayAndSaveAnimationKey(step, bloop);

	return true;
}

void Model::SetRange(float s, float e)
{
	Transform::SetRange(s, e);
	switch (m_animationType)
	{
		case STATIC_GEOMETRY:
			break;

		case SKINNING_ANIMATION:
			for (DWORD i = 0; i < m_numBones; i++)
			{
				m_pBone[i].SetRange(s, e);
			}
			break;

		case HIERARCHY_ANIMATION:
			for (DWORD i = 0; i < m_numMeshs; i++)
			{
				m_pMesh[i].SetRange(s, e);
			}
			break;

		case SHAPE_ANIMATION:
			for (DWORD i = 0; i < m_numMeshs; i++)
			{
				m_pMesh[i].SetRange(s, e);
			}
			break;
	}
}

void Model::SetTime(float t)
{
	Transform::SetTime(t);
	switch (m_animationType)
	{
		case STATIC_GEOMETRY:
			break;

		case SKINNING_ANIMATION:
			for (DWORD i = 0; i < m_numBones; i++)
			{
				m_pBone[i].SetTime(t);
			}
			break;

		case HIERARCHY_ANIMATION:
			for (DWORD i = 0; i < m_numMeshs; i++)
			{
				m_pMesh[i].SetTime(t);
			}
			break;

		case SHAPE_ANIMATION:
			for (DWORD i = 0; i < m_numMeshs; i++)
			{
				m_pMesh[i].SetTime(t);
			}
			break;
	}
}

void Model::UpdateVertices(float4 *pv)
{
	float4 P;
	switch (m_animationType)
	{
		case STATIC_GEOMETRY:
			for (DWORD i = 0; i < m_numMeshs; i++)
			{
				for (DWORD v = m_pMesh[i].m_baseVertex; v < m_pMesh[i].m_baseVertex + m_pMesh[i].m_numVertices; v++)
				{
					pv[v].x = m_pVPosition[v].x;
					pv[v].y = m_pVPosition[v].y;
					pv[v].z = m_pVPosition[v].z;
					pv[i].w = 1.0f;
				}
			}
			break;

		case SKINNING_ANIMATION:
			for (DWORD i = 0; i < m_numVertices; i++)
			{
				if (m_pVSkinningData[i].numBones)
				{
					if (m_pVSkinningData[i].numBones > 1)
					{
						pv[i].x = pv[i].y = pv[i].z = 0.0f;
						for (DWORD j = 0; j < m_pVSkinningData[i].numBones; j++)
						{
							float4 P(m_pVSkinningData[i].pBlendData[j].offset, 1.0f);
							P = transpose(m_pBone[m_pVSkinningData[i].pBlendData[j].bone].GetTransformMatrix()) * P;
							P *= m_pVSkinningData[i].pBlendData[j].weight;
							pv[i] += P;
						}
						pv[i].w = 1.0f;
					}
					else
					{
						float4 P(m_pVSkinningData[i].pBlendData[0].offset, 1.0f);
						pv[i] = transpose(m_pBone[m_pVSkinningData[i].pBlendData[0].bone].GetTransformMatrix()) * P;
						pv[i].w = 1.0f;
					}
				}
			}
			break;

		case HIERARCHY_ANIMATION:
			for (DWORD i = 0; i < m_numMeshs; i++)
			{
				for (DWORD v = m_pMesh[i].m_baseVertex; v < m_pMesh[i].m_baseVertex + m_pMesh[i].m_numVertices; v++)
				{
					float4 P(m_pVPosition[v], 1.0f);
					pv[v] = transpose(m_pMesh[i].GetTransformMatrix()) * P;
					pv[i].w = 1.0f;
				}
			}
			break;

		case SHAPE_ANIMATION:
			for (DWORD i = 0; i < m_numMeshs; i++)
			{
				for (DWORD v = m_pMesh[i].m_baseVertex; v < m_pMesh[i].m_baseVertex + m_pMesh[i].m_numVertices; v++)
				{
					float4 P(m_pVPosition[v], 1.0f);
					pv[v] = transpose(m_pMesh[i].GetTransformMatrix()) * P;
					pv[i].w = 1.0f;
				}
			}
			break;
	}
}

Mesh &Model::GetMesh(const char *name)
{
	for (DWORD i = 0; i < m_numMeshs; i++)
	{
		if (!strcmp(name, m_pMesh[i].GetName()))
			return m_pMesh[i];
	}
	return m_pMesh[0];
}
