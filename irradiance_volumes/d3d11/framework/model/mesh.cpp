//-----------------------------------------------------------------------------
// File: Framework\Model\Mesh.cpp
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------

#include "Mesh.h"

Mesh::Mesh()
{
	m_materialIndex = -1;
	m_baseVertex =  m_numVertices = 0;
	m_baseTriangle = m_numTriangles = 0;
	m_pPrimitive = NULL;
	m_pBone = NULL;
}

Mesh::~Mesh()
{
	for (DWORD i = 0; i < m_numPrimitives; i++)
		delete [] m_pPrimitive[i].pIndices;

	delete [] m_pPrimitive;
	delete [] m_pBone;
}

//void Mesh::Render(ID3D10Device* pdev) 
//{	
//	for (unsigned int i=0; i<m_numPrimitives; i++) 
//	{		
//		pdev->IASetPrimitiveTopology(m_pPrimitive[i].type);
//		pdev->DrawIndexed(m_pPrimitive[i].numIndices, m_pPrimitive[i].baseIndex, 0);
//	}
//}