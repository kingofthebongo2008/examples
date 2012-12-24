//-----------------------------------------------------------------------------
// File: Framework\Math\Camera.h
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------





#ifndef _CAMERA_H_
#define _CAMERA_H_

#include "Vector.h"
#include "../Util/Array.h"

struct PathNode
{
	float3 pos;
	float rotX, rotY, rotZ;
};

class Camera
{
public:
	Camera();

	void SetViewport(const int width, const int height);
	void SetFrustumExtents(const float fov, const float zNear, const float zFar);

	void SetPosition(const float3 &position);
	void SetRotation(const float rotX, const float rotY, const float rotZ);

	void UpdateRotation(const float deltaX, const float deltaY);
	void LookAt(const float3 &look, const float3 &up);

	const float3 &GetPosition() const { return m_position; }
	const float GetRotationX() const { return m_rotX; }
	const float GetRotationY() const { return m_rotY; }
	const float GetRotationZ() const { return m_rotZ; }
	void GetViewBaseVectorsXY(float3 &dx, float3 &dy, float3 &dz) const;

	const float GetFOV() const {return m_fov; };
	const float GetzNear() const {return m_zNear; };
	const float GetzFar() const {return m_zFar; };
	const float GetAspect() const {return ((float)m_width/(float)m_height); };
	const float4x4 &GetModelView();
	const float4x4 &GetProjection();
	const float4x4 &GetModelViewProjection();
	const float4x4 GetSkyboxMVP();

	void GetBaseVectors(float3 *dx, float3 *dy, float3 *dz);

	void AddPathNode();
	bool RemovePathNode();

	bool LoadPath(const TCHAR *fileName);
	bool SavePath(const TCHAR *fileName) const;

	bool GetNodeAt(const float time, float3 *position, float3 *angles, const bool looping) const;
	bool SetCameraToPathTime(const float time, const bool looping);

	const float3 &GetPathNodePosition(const uint index);
	uint GetPathNodeCount() const { return m_pathNodes.GetCount(); }

private:
	float3 m_position;
	float m_rotX, m_rotY, m_rotZ;

	float m_fov, m_zNear, m_zFar;
	int m_width, m_height;

	float4x4 m_modelview;
	float4x4 m_projection;
	float4x4 m_modelviewProjection;

	bool m_modelviewDirty;
	bool m_projectionDirty;
	bool m_modelviewProjectionDirty;

	Array <PathNode> m_pathNodes;

};

#endif // _CAMERA_H_
