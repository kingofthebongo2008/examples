//-----------------------------------------------------------------------------
// File: Framework\Model\Animation.cpp
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------

#include "Animation.h"

void Quaternion2Mat4(float4x4& m, float* q)
{
	float s, xs, ys, zs, wx, wy, wz, xx, xy, xz, yy, yz, zz;

	s = 2.0f;
	//s = 2.0f/(q[0]*q[0] + q[0]*q[1] + q[2]*q[2] + q[3]*q[3]);

	xs = q[0]*s;        ys = q[1]*s;        zs = q[2]*s;
	wx = q[3]*xs;       wy = q[3]*ys;       wz = q[3]*zs;
	xx = q[0]*xs;       xy = q[0]*ys;       xz = q[0]*zs;
	yy = q[1]*ys;       yz = q[1]*zs;       zz = q[2]*zs;

	
	m.elem[0][0] = 1.0f - (yy + zz);
	m.elem[0][1] = xy + wz;
	m.elem[0][2] = xz - wy;

	m.elem[1][0] = xy - wz;
	m.elem[1][1] = 1.0f - (xx + zz);
	m.elem[1][2] = yz + wx;
	
	m.elem[2][0] = xz + wy;
	m.elem[2][1] = yz - wx;
	m.elem[2][2] = 1.0f - (xx + yy);

	m.elem[3][0] = m.elem[3][1] = m.elem[3][2] = 0.0f;
	m.elem[3][3] = 1.0f;
}

void Mat42Quaternion(float4& q, float4x4& m)
{
	float tr, s;
	int   i, j, k;
	float lpM[4][4], lpQ[3];
	int   nxt[3] = {1, 2, 0};

	tr = m.elem[0][0] + m.elem[1][1] + m.elem[2][2];
	if (tr > 0.0f)
	{
		s = (float)sqrtf(tr + 1.0f);
		q.w = s * 0.5f;

		s = 0.5f/s;
		q.x = (m.elem[1][2] - m.elem[2][1])*s;
		q.y = (m.elem[2][0] - m.elem[0][2])*s;
		q.z = (m.elem[0][1] - m.elem[1][0])*s;
	}
	else
	{
		memcpy(lpM, &m, sizeof(float4x4));
		i = 0;
		if (lpM[1][1] > lpM[0][0])
			i = 1;
		if (lpM[2][2] > lpM[i][i])
			i = 2;
		j = nxt[i];
		k = nxt[j];

		s = (float)sqrtf(lpM[i][i] - ( lpM[j][j] + lpM[k][k] ) + 1.0f);

		lpQ[i] = s*0.5f;
		s = 0.5f/s;
		q.w = (lpM[j][k] - lpM[k][j])*s;

		lpQ[j] = (lpM[i][j] + lpM[j][i])*s;
		lpQ[k] = (lpM[i][k] + lpM[k][i])*s;
		q.x = lpQ[0];
		q.y = lpQ[1];
		q.z = lpQ[2];
	}
}

Animation::Animation()
{
	m_rangeStart = m_rangeEnd = 0.0f;
	m_currentTime = 0.0f;
	m_positionKeys.numKeys = 0;
	m_positionKeys.pKey = NULL;
	m_rotationKeys.numKeys = 0;
	m_rotationKeys.pKey = NULL;
	m_scaleKeys.numKeys = 0;
	m_scaleKeys.pKey = NULL;
	m_transformMatrix = Identity4();
	m_CurrentKey = float3(0, 0, 0);
}

Animation::~Animation()
{
	delete [] m_positionKeys.pKey;
	delete [] m_rotationKeys.pKey;
	delete [] m_scaleKeys.pKey;
}

void Animation::GetAnimationKeyValue(AnimationKeys &ak, float time, float *lpf, int num) 
{
	float         Offset;
	DWORD         i, Key;
	int           Top, Bottom, Mid;
	AnimationKey *lpCK;
	float        *lpV1, *lpV2, *lpR = lpf;
	
	if (ak.numKeys >= 2)
	{
		Top = 0;
		Key = ak.numKeys;
		Bottom = ak.numKeys;
		while (Top <= Bottom)
		{
			Mid = ((Top + Bottom) >> 1);
			if (time >= ak.pKey[Mid].endTime)
			{
				Top = Mid + 1;
			}
			else if (time < ak.pKey[Mid].startTime)
			{
				Bottom = Mid - 1;
			}
			else
			{
				Key = Mid;
				break;
			}
		}
		lpCK = &ak.pKey[Key];
		if (Key >= ak.numKeys-1)
		{
			memcpy(lpf, ak.pKey[Key].value, sizeof(float) * num);
			return;
		}
		lpV1 = lpCK->value;
		lpV2 = (lpCK + 1)->value;

		Offset = (time-lpCK->startTime)/(lpCK->endTime - lpCK->startTime);
		for(i = num; i > 0; i--)
		{		
			*lpR = (*lpV1) + (((*lpV2) - (*lpV1))*Offset);
			lpV1++;
			lpV2++;
			lpR++;
		}
	}
	else
	{
		memcpy(lpf, ak.pKey[0].value, sizeof(float) * num);
	}
}

void Animation::CalculateTransformMatrix() 
{
	float value[4];
	m_transformMatrix = Identity4();
	if (m_rotationKeys.numKeys)
	{
		GetAnimationKeyValue(m_rotationKeys, m_currentTime, value, 4);
		Quaternion2Mat4(m_transformMatrix, value);
	}
	if (m_scaleKeys.numKeys)
	{
		GetAnimationKeyValue(m_scaleKeys, m_currentTime, value, 3);
		m_transformMatrix.elem[0][0] *= value[0];
		m_transformMatrix.elem[1][0] *= value[1];
		m_transformMatrix.elem[2][0] *= value[2];

		m_transformMatrix.elem[0][1] *= value[0];
		m_transformMatrix.elem[1][1] *= value[1];
		m_transformMatrix.elem[2][1] *= value[2];

		m_transformMatrix.elem[0][2] *= value[0];
		m_transformMatrix.elem[1][2] *= value[1];
		m_transformMatrix.elem[2][2] *= value[2];
	}
	if (m_positionKeys.numKeys)
	{
		GetAnimationKeyValue(m_positionKeys, m_currentTime, value, 3);
		m_transformMatrix.elem[3][0] += value[0];
		m_transformMatrix.elem[3][1] += value[1];
		m_transformMatrix.elem[3][2] += value[2];
	}
}

bool Animation::Play(float step, bool bloop) 
{
	bool bReachEnd = false;

	CalculateTransformMatrix();
	m_currentTime += step;
	if (bloop)
	{
		if (m_currentTime >= m_rangeEnd)
		{
			m_currentTime = m_rangeStart;
			bReachEnd = true;
		}
	}
	else
	{
		if (m_currentTime >= m_rangeEnd)
		{
			m_currentTime = m_rangeEnd;
			bReachEnd = true;
		}
	}
	return bReachEnd;
}

void Animation::GetCurrentAnimationKeyAndLerp(float *key, float* lerp, AnimationKeys &ak, float time) 
{
	DWORD           Key;
	int             Top, Bottom, Mid;
	AnimationKey   *lpCK;	
	
	if (ak.numKeys >= 2)
	{
		Top = 0;
		Key = ak.numKeys;
		Bottom = ak.numKeys;
		while (Top <= Bottom)
		{
			Mid = ((Top + Bottom) >> 1);
			if (time >= ak.pKey[Mid].endTime)
			{
				Top = Mid + 1;
			}
			else if (time < ak.pKey[Mid].startTime)
			{
				Bottom = Mid - 1;
			}
			else
			{
				Key = Mid;
				break;
			}
		}

		if (Key >= ak.numKeys-1)
		{
			*key = (float)(ak.numKeys-1);
			*lerp = 1.0f;
			return;
		}
		lpCK = &ak.pKey[Key];
		*key = (float)Key;
		*lerp = (time-lpCK->startTime)/(lpCK->endTime - lpCK->startTime);
	}
	else
	{
		*key = 0;
		*lerp = 0.0f;
	}
}

void Animation::GetCurrentAnimationKeys()
{
	if (m_rotationKeys.numKeys)
	{
		GetCurrentAnimationKeyAndLerp(&m_CurrentKey.y, &m_CurrentKeyLerp.y, m_rotationKeys, m_currentTime);
	}	
	if (m_scaleKeys.numKeys)
	{
		GetCurrentAnimationKeyAndLerp(&m_CurrentKey.z, &m_CurrentKeyLerp.z, m_scaleKeys, m_currentTime);
	}
	if (m_positionKeys.numKeys)
	{
		GetCurrentAnimationKeyAndLerp(&m_CurrentKey.x, &m_CurrentKeyLerp.x, m_positionKeys, m_currentTime);
	}
}

bool Animation::PlayAndSaveAnimationKey(float step, bool bloop)
{
	bool bReachEnd = false;

	GetCurrentAnimationKeys();
	m_currentTime += step;
	if (bloop)
	{
		if (m_currentTime >= m_rangeEnd)
		{
			m_currentTime = m_rangeStart;
			bReachEnd = true;
		}
	}
	else
	{
		if (m_currentTime >= m_rangeEnd)
		{
			m_currentTime = m_rangeEnd;
			bReachEnd = true;
		}
	}
	return bReachEnd;
}
