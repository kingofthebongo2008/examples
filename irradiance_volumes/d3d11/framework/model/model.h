//-----------------------------------------------------------------------------
// File: Framework\Model\Model.h
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------

#ifndef _MODEL_H_
#define _MODEL_H_

#include "../math/Vector.h"
#include "Mesh.h"
#include "Bone.h"

enum AnimationType
{
	STATIC_GEOMETRY = 0,
	SKINNING_ANIMATION,
	SHAPE_ANIMATION,
	HIERARCHY_ANIMATION,
};

struct Material
{
	TCHAR diffuseMapFile[64];
	TCHAR bumpMapFile[64];
};

struct BlendData
{
	DWORD bone;
	float3 offset;
	float weight;
};

struct SkinningData
{
	DWORD numBones;
	BlendData *pBlendData;
};

struct Triangle
{
	DWORD v[3];
	float3 normal;
};

struct OffsetTable
{
	DWORD materialChunk;
	DWORD vertexChunk;
	DWORD triangleChunk;
	DWORD meshChunk;
	DWORD boneChunk;
};

class Model : public Transform 
{
public:
	float3 *m_pVPosition;
	float3 *m_pVNormal;
	float3 *m_pVTangent;
	float3 *m_pVBinormal;
	float2 *m_pVTexcoord;
	float4 *m_pVColor;
protected:
	TCHAR m_texturePath[64];
	int m_animationType;
	DWORD m_numVertices;
	
	SkinningData *m_pVSkinningData;
	DWORD m_numTriangles;
	Triangle *m_pTriangle;

	DWORD m_numMaterials;
	Material *m_pMaterial;
	DWORD m_numMeshs;
	Mesh *m_pMesh;
	Transform **m_ppMeshHie;
	DWORD m_numBones;
	Bone *m_pBone;
	Transform **m_ppBoneHie;
	DWORD m_traverseIndex;
protected:
	void RecursionMeshTraverse(Transform **lphie, Transform *lpct);
	void RecursionBoneTraverse(Transform **lphie, Transform *lpct);
	void BuildMeshHie();
	void BuildBoneHie();
	//void ReorderBone();

public:
	Model();
	virtual ~Model();
	virtual void Clean();

	inline const int GetAnimationType(){ return m_animationType; }
	virtual bool Load(const TCHAR *name, const TCHAR *texture_path);

	bool Play(float step = 1.0f, bool bloop = true);
	bool PlayAndSaveAnimationKey(float step = 1.0f, bool bloop = true);
	void UpdateVertices(float4 *pv);
	void SetRange(float s, float e);
	void SetTime(float t);

	inline const Triangle &GetTriangle(const uint index) const { return m_pTriangle[index]; }

	inline Mesh &GetMesh(const int index){ return m_pMesh[index]; }
	Mesh &GetMesh(const char *name);
	inline Bone &GetBone(const int index){ return m_pBone[index]; }
	inline const DWORD GetVertexCount() const { return m_numVertices; }
	inline const DWORD GetTriangleCount() const { return m_numTriangles; }
	inline const DWORD GetBoneCount() const { return m_numBones; }
	inline const DWORD GetMeshCount() const { return m_numMeshs; }

	inline Material &GetMaterial(const uint index){ return m_pMaterial[index]; }
};

#endif // _MODEL_H_
