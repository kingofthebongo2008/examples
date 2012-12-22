//-----------------------------------------------------------------------------
// File: Framework\Model\Transform.h
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------

#ifndef _TRANSFORM_H_
#define _TRANSFORM_H_

#include <d3d10.h>
#include <d3dx10.h>
#include "../math/Vector.h"
#include "Animation.h"

class Transform
{
public:
	int m_ID;
protected:
	float4x4 m_transformMatrix;
	Animation m_animation;
	int m_parent;
	DWORD m_numChildren;
	DWORD *m_pChildren;

public:
	Transform();
	virtual ~Transform();

	inline bool Play(float step = 1.0f, bool bloop = true){ return m_animation.Play(step, bloop); }
	inline bool PlayAndSaveAnimationKey(float step = 1.0f, bool bloop = true){ return m_animation.PlayAndSaveAnimationKey(step, bloop); }
	inline void SetPosition(const float3 &p){ m_transformMatrix.elem[3][0] = p.x; m_transformMatrix.elem[3][1] = p.y; m_transformMatrix.elem[3][2] = p.z; }
	inline void GetPosition(float3 &p) const { p.x = m_transformMatrix.elem[3][0]; p.y = m_transformMatrix.elem[3][1]; p.z = m_transformMatrix.elem[3][2]; }
	inline void SetTransformMatrix(const float4x4 &m){ m_transformMatrix = m; }
	inline const float4x4 &GetTransformMatrix() const { return m_transformMatrix; }
	inline void SetRange(const float s, const float e){ m_animation.SetRange(s, e); }
	inline void SetTime(const float t){ m_animation.SetTime(t); }
	inline const float GetTime() const { return m_animation.GetTime(); }
	inline const DWORD GetNumChildren() const { return m_numChildren; }
	inline const DWORD GetChild(const int c) const { return m_pChildren[c]; }
	inline Animation& GetAnimation(){ return m_animation; }
	inline int GetParent() const { return m_parent; }

friend class Model;
};

#endif // _TRANSFORM_H_
