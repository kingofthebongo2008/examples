//-----------------------------------------------------------------------------
// File: Framework\Util\Model.h
// Copyright (c) 2005 ATI Technologies Inc. All rights reserved.
//-----------------------------------------------------------------------------
#ifndef _D3D10MODEL_H_
#define _D3D10MODEL_H_

#include <d3d10.h>
#include <d3dx10.h>
#include "model.h"

enum BoneMatrixConstantType
{
	CONSTANT_BUFFER = 0,
	BUFFER,
	NUM_BONE_MATRIX_CONSTNAT_TYPE,
};

typedef struct{
	float3 Normal;
	float2 TexCoord;
	float3 Position[4];
	float4 BlendWeight;
	float4 BlendIndices;	
}BLENDVERTEX4;

typedef struct{
	float3 Normal;
	float2 TexCoord;
	float3 Position;
	float4 BlendWeight;
	float4 BlendIndices;	
}BLENDVERTEX;

typedef struct{
	float3 Pos;
	float3 Normal;
	float2 TexCoord;	
}STATICVERTEX;

struct D3D10Material
{
	ID3D10Texture2D* diffuseMap;
	ID3D10ShaderResourceView* diffuseMapView;
	ID3D10Texture2D* bumpMap;
	ID3D10ShaderResourceView* bumpMapView;
};

class D3D10Model : public Model 
{
public:
	
protected:
	BoneMatrixConstantType m_boneMatrixConstantType;

	// material
	D3D10Material* m_pD3D10Material;
	// effects
	ID3D10Effect** m_ppMeshEffect;
	// static 
	ID3D10EffectTechnique* m_techStatic;
	//Constant buffer
	ID3D10EffectTechnique* m_techConstantBuffer;	
	ID3D10EffectMatrixVariable* m_boneMatrixConstantBuffer;
	//Buffer
	ID3D10EffectTechnique* m_techBuffer;
	ID3D10Buffer* m_boneMatrixBuffer; 
	ID3D10ShaderResourceView* m_boneMatrixBufferRV;
	ID3D10EffectShaderResourceVariable* m_boneMatrixBufferVar;
	
	ID3D10InputLayout* m_IL;
	ID3D10Buffer* m_VB;
	ID3D10Buffer* m_IB;	
protected:	
public:
	D3D10Model();
	virtual ~D3D10Model();
	virtual void Clean();

	inline BoneMatrixConstantType GetBoneConstantType() const {return m_boneMatrixConstantType; };
	virtual bool Load(ID3D10Device* pdev, const TCHAR *name, const TCHAR *texture_path);
	virtual bool Load(ID3D10Device* pdev, const TCHAR *name, const TCHAR *texture_path, LPCWSTR shader, BoneMatrixConstantType t);
	virtual bool Load(ID3D10Device* pdev, const TCHAR *name, const TCHAR *texture_path, ID3D10Effect *eff, BoneMatrixConstantType t);
	void LoadTextures(ID3D10Device* pdev); 
	virtual bool CreateInputLayout(ID3D10Device* pdev); 
	virtual bool CreateVertexBuffer(ID3D10Device* pdev);
	virtual bool CreateResources(ID3D10Device* pdev);	
	bool CreateIndexBuffer(ID3D10Device* pdev);	
	virtual void Render(ID3D10Device* pdev, const float4x4& mvp, const float3& cmaerapos);	

	inline void  SetInputLayout(ID3D10InputLayout* l) {m_IL = l; l->AddRef();};
	inline const ID3D10InputLayout* GetInputLayout() {return m_IL; };
	inline void SetVertexBuffer(ID3D10Buffer* vb) {m_VB = vb; vb->AddRef();};
	inline const ID3D10Buffer* GetVertexBuffer() {return m_VB; };
	inline const ID3D10Buffer* GetIndexBuffer() {return m_IB; };
	inline const D3D10Material& GetMaterial(int i) {return m_pD3D10Material[i]; };
	inline ID3D10Effect* GetMeshEffect(int i) {return m_ppMeshEffect[i]; };
	inline void SetMeshEffect(int i, ID3D10Effect* pe) {m_ppMeshEffect[i] = pe; pe->AddRef();};
	bool LoadMeshEffect(ID3D10Device* pdev, int i, LPCWSTR shader);
};

#endif // _D3DMODEL_H_
