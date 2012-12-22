//-----------------------------------------------------------------------------
// File: Framework\Math\Camera.cpp
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------





#include "Camera.h"
#include <stdio.h>


Camera::Camera()
{
	// Initialize variables
	m_position = float3(0, 0, 0);
	m_rotX = 0;
	m_rotY = 0;
	m_rotZ = 0;
	m_fov   = 1.5f;
	m_zNear = 0.1f;
	m_zFar  = 10.0f;
	m_width  = 1;
	m_height = 1;

	// Initially dirty so we don't have to initialize the matrices
	m_modelviewDirty = true;
	m_projectionDirty = true;
	m_modelviewProjectionDirty = true;
}

void Camera::SetViewport(const int width, const int height)
{
	m_width = width;
	m_height = height;

	// Viewport change alters the projection matrix
	m_projectionDirty = true;
	m_modelviewProjectionDirty = true;
}

void Camera::SetFrustumExtents(const float fov, const float zNear, const float zFar)
{
	m_fov = fov;
	m_zNear = zNear;
	m_zFar = zFar;

	// Frstum change alters the projection matrix
	m_projectionDirty = true;
	m_modelviewProjectionDirty = true;
}

void Camera::SetPosition(const float3 &position)
{
	// Position change alters the modelview matrix
	m_position = position;
	m_modelviewDirty = true;
	m_modelviewProjectionDirty = true;
}

void Camera::SetRotation(const float rotX, const float rotY, const float rotZ)
{
	// Rotation change alters the modelview matrix
	m_rotX = rotX;
	m_rotY = rotY;
	m_rotZ = rotZ;
	m_modelviewDirty = true;
	m_modelviewProjectionDirty = true;
}

void Camera::UpdateRotation(const float deltaX, const float deltaY)
{
	SetRotation(m_rotX + deltaX, m_rotY + deltaY, m_rotZ);	
}

void Camera::LookAt(const float3 &look, const float3 &up)
{
	float3 Dir = look - m_position;
	Dir = normalize(Dir);
	float3 dv = Dir + up;
	dv = normalize(dv);
	float f = dot(dv, Dir);
	if (f == 0.0f)
		f = 0.1f;
	float3 dn = f * Dir;
	float3 Up = dv - dn;
	Up = normalize(Up);
	float3 Right = cross(Up, Dir);

	m_modelview = float4x4(
		float4(Right, -dot(Right, m_position)),
		float4(Up, -dot(Up, m_position)),
		float4(Dir, -dot(Dir, m_position)),
		float4(0.0f, 0.0f, 0.0f, 1.0f)
	);
	m_modelviewDirty = false;
}

void Camera::GetViewBaseVectorsXY(float3 &dx, float3 &dy, float3 &dz) const
{
	float cosX = cosf(m_rotX), sinX = sinf(m_rotX);
	float cosY = cosf(m_rotY), sinY = sinf(m_rotY);

	dx = float3(cosY, 0, sinY);
	dy = float3(-sinX * sinY,  cosX, sinX * cosY);
	dz = float3(-cosX * sinY, -sinX, cosX * cosY);
}

const float4x4 &Camera::GetModelView()
{
	// Recompute matrix if neccesary
	if (m_modelviewDirty)
	{
		m_modelview = rotateZXY4(-m_rotX, -m_rotY, -m_rotZ);
		m_modelview.translate(-m_position);
		m_modelviewDirty = false;
	}

	return m_modelview;
}

const float4x4 &Camera::GetProjection()
{
	// Recompute matrix if neccesary
	if (m_projectionDirty)
	{
		m_projection = PerspectiveMatrixX(m_fov, m_width, m_height, m_zNear, m_zFar);
		m_projectionDirty = false;
	}

	return m_projection;
}

const float4x4 &Camera::GetModelViewProjection()
{
	if (m_modelviewProjectionDirty)
	{
		// Recompute matrix if neccesary
		if (m_modelviewDirty)
		{
			m_modelview = rotateZXY4(-m_rotX, -m_rotY, -m_rotZ);
			m_modelview.translate(-m_position);
			m_modelviewDirty = false;
		}
		// Recompute matrix if neccesary
		if (m_projectionDirty)
		{
			m_projection = PerspectiveMatrixX(m_fov, m_width, m_height, m_zNear, m_zFar);
			m_projectionDirty = false;
		}

		m_modelviewProjection = m_projection * m_modelview;
		m_modelviewProjectionDirty = false;
	}

	return m_modelviewProjection;
}

const float4x4 Camera::GetSkyboxMVP()
{
	if (m_modelviewProjectionDirty)
	{
		// Recompute matrix if neccesary
		if (m_modelviewDirty)
		{
			m_modelview = rotateZXY4(-m_rotX, -m_rotY, -m_rotZ);
			m_modelview.translate(-m_position);
			m_modelviewDirty = false;
		}
		// Recompute matrix if neccesary
		if (m_projectionDirty)
		{
			m_projection = PerspectiveMatrixX(m_fov, m_width, m_height, m_zNear, m_zFar);
			m_projectionDirty = false;
		}

		m_modelviewProjection = m_projection * m_modelview;
		m_modelviewProjectionDirty = false;
	}

	// Take the translation out of the MVP matrix
	float4x4 temp = m_modelviewProjection;
	temp.translate(m_position);

	return temp;
}

void Camera::GetBaseVectors(float3 *dx, float3 *dy, float3 *dz)
{
	// Recompute matrix if neccesary
	if (m_modelviewDirty)
	{
		m_modelview = rotateZXY4(-m_rotX, -m_rotY, -m_rotZ);
		m_modelview.translate(-m_position);
		m_modelviewDirty = false;
	}

	if (dx) *dx = m_modelview.getRightVec();
	if (dy) *dy = m_modelview.getUpVec();
	if (dz) *dz = m_modelview.getForwardVec();
}

void Camera::AddPathNode()
{
	PathNode node;
	node.pos = m_position;
	node.rotX = m_rotX;
	node.rotY = m_rotY;
	node.rotZ = m_rotZ;

	m_pathNodes.Add(node);
}

bool Camera::RemovePathNode()
{
	uint count = m_pathNodes.GetCount();
	if (count == 0) return false;

	m_pathNodes.FastRemove(count - 1);
	return true;
}

bool Camera::LoadPath(const TCHAR *fileName)
{
	FILE *file = _tfopen(fileName, _T("rb"));
	if (file == NULL) return false;

	uint count = m_pathNodes.GetCount();
	fread(&count, sizeof(count), 1, file);
	m_pathNodes.SetCount(count);
	fread(m_pathNodes.GetArray(), count * sizeof(PathNode), 1, file);
	fclose(file);

	return true;
}

bool Camera::SavePath(const TCHAR *fileName) const
{
	FILE *file = _tfopen(fileName, _T("wb"));
	if (file == NULL) return false;

	uint count = m_pathNodes.GetCount();
	fwrite(&count, sizeof(count), 1, file);
	fwrite(m_pathNodes.GetArray(), count * sizeof(PathNode), 1, file);
	fclose(file);

	return true;
}

int clamp(int x, const int len){
	if (x < 0) return 0;
	if (x >= len)
		return len - 1;
	else
		return x;
}

int Wrap(int x, const int len){
	x %= len;
	if (x < 0) x += len;
	return x;
}

bool Camera::GetNodeAt(const float time, float3 *position, float3 *angles, const bool looping) const
{
	int count = m_pathNodes.GetCount();
	if (count == 0) return false;

	int i = (int) time;
	float f = time - i;

	int i0, i1, i2, i3;
	if (looping)
	{
		i0 = Wrap(i - 1, count);
		i1 = Wrap(i,     count);
		i2 = Wrap(i + 1, count);
		i3 = Wrap(i + 2, count);
	}
	else
	{
		i0 = clamp(i - 1, count);
		i1 = clamp(i,     count);
		i2 = clamp(i + 1, count);
		i3 = clamp(i + 2, count);
	}

	if (position)
	{
		*position = herp(m_pathNodes[i0].pos, m_pathNodes[i1].pos, m_pathNodes[i2].pos, m_pathNodes[i3].pos, f, 0, 0);
	}

	if (angles)
	{
		float3 a0 = float3(m_pathNodes[i0].rotX, m_pathNodes[i0].rotY, m_pathNodes[i0].rotZ);
		float3 a1 = float3(m_pathNodes[i1].rotX, m_pathNodes[i1].rotY, m_pathNodes[i1].rotZ);
		float3 a2 = float3(m_pathNodes[i2].rotX, m_pathNodes[i2].rotY, m_pathNodes[i2].rotZ);
		float3 a3 = float3(m_pathNodes[i3].rotX, m_pathNodes[i3].rotY, m_pathNodes[i3].rotZ);

		*angles = herp(a0, a1, a2, a3, f, 0, 0);
	}

	return true;
}

bool Camera::SetCameraToPathTime(const float time, const bool looping)
{
	float3 pos, angles;
	if (GetNodeAt(time, &pos, &angles, looping))
	{
		SetPosition(pos);
		SetRotation(angles.x, angles.y, angles.z);
		return true;
	}

	return false;
}

const float3 &Camera::GetPathNodePosition(const uint index)
{
	return m_pathNodes[index].pos;
}
