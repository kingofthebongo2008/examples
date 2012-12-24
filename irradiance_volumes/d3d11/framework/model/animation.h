//-----------------------------------------------------------------------------
// File: Framework\Model\Animation.h
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------

#ifndef _ANIMATION_H_
#define _ANIMATION_H_

#include <d3d11.h>
#include "../math/Vector.h"

struct AnimationKey
{
	int startTime, endTime;
	float value[4];
};

struct AnimationKeys
{
	DWORD numKeys;
	AnimationKey *pKey;
};

class Animation 
{
public:
	float3 m_CurrentKey;
	float3 m_CurrentKeyLerp;
protected:
	float m_currentTime, m_rangeStart, m_rangeEnd;
	float4x4 m_transformMatrix;
	AnimationKeys m_positionKeys, m_rotationKeys, m_scaleKeys;
	
public:
	Animation();
	virtual ~Animation();

	void GetAnimationKeyValue(AnimationKeys &ak, float time, float *lpf, int num);
	inline void SetRange(float s, float e){ m_rangeStart = s; m_rangeEnd = e; }
	inline void SetTime(float t){ m_currentTime = t; }
	inline const float GetTime() const { return m_currentTime;};
	bool Play(float step, bool bloop = true);
	inline const float4x4 &GetTransformMatrix() const { return m_transformMatrix; }
	inline AnimationKeys &GetRotationKeys(){ return m_rotationKeys; }
	inline AnimationKeys &GetPositionKeys(){ return m_positionKeys; }
	inline AnimationKeys &GetScaleKeys(){ return m_scaleKeys; }

	void GetCurrentAnimationKeyAndLerp(float *key, float *lerp, AnimationKeys &ak, float time);
	void GetCurrentAnimationKeys();
	bool PlayAndSaveAnimationKey(float step, bool bloop);

protected:
	void CalculateTransformMatrix();

	friend class Model;
};

void Quaternion2Mat4(float4x4 &m, float *q);
void Mat42Quaternion(float4 &q, float4x4 &m);
//void Mat42Quaternion(half4& q, float4x4& m);

#endif // _ANIMATION_H_
