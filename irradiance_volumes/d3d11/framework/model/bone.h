//-----------------------------------------------------------------------------
// File: Framework\Model\Bone.h
// Copyright (c) 2007 Advanced Micro Devices Inc. All rights reserved.
//-----------------------------------------------------------------------------





#ifndef _BONE_H_
#define _BONE_H_

#include "../math/Vector.h"
#include "Transform.h"

class Bone : public Transform
{
public:
	Bone();
	virtual ~Bone();
};

#endif // _BONE_H_
