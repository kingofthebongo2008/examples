//-----------------------------------------------------------------------------
// File: Framework\Model\Transform.cpp
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------

#include "Transform.h"

Transform::Transform()
{
	m_transformMatrix = Identity4();
	m_parent = -1;
	m_numChildren = 0;
	m_pChildren = NULL;
}

Transform::~Transform()
{
	delete [] m_pChildren;
}
