//-----------------------------------------------------------------------------
// File: Framework\Model\Mesh.h
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------

#ifndef _MESH_H_
#define _MESH_H_

#include "../math/Vector.h"
#include "Transform.h"

enum
{
	PL_TRIANGLE_LIST = 0,
	PL_TRIANGLE_STRIP,
	PL_TRIANGLE_FAN,
};

struct Primitive
{
	D3D10_PRIMITIVE_TOPOLOGY type;
	DWORD baseIndex;
	DWORD numIndices;
	DWORD *pIndices;
};

class Mesh : public Transform
{
protected:
	char m_Name[64];
	int m_materialIndex;

	DWORD m_baseVertex, m_numVertices;
	DWORD m_baseTriangle, m_numTriangles;

	DWORD m_numBones;
	DWORD *m_pBone;

	DWORD m_numPrimitives;
	Primitive *m_pPrimitive;

public:
	Mesh();
	virtual ~Mesh();

	inline const char *GetName() const { return m_Name; }
	inline const int GetMaterialIndex() const { return m_materialIndex; }
	inline const DWORD GetNumPrimitives() const { return m_numPrimitives; }
	inline Primitive &GetPrimitive(const int index) { return m_pPrimitive[index]; }
	inline const DWORD GetNumVertices() const { return m_numVertices; }
	inline const DWORD GetBaseVertex() const { return m_baseVertex; }
	inline const DWORD GetNumTriangles() const { return m_numTriangles; }
	inline const DWORD GetBaseTriangle() const { return m_baseTriangle; }

friend class Model;
};

#endif // _MESH_H_
