//--------------------------------------------------------------------------------------
// Copyright 2013 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.
//
//--------------------------------------------------------------------------------------

#include "HelperSSE.h"
#include "CPUTCamera.h"

HelperSSE::HelperSSE()
{
}

HelperSSE::~HelperSSE()
{
}

__m128 HelperSSE::TransformCoords(const __m128 *v, __m128 *m)
{
	__m128 vResult = ssp_shuffle_ps(*v, *v, _MM_SHUFFLE(0,0,0,0));
    vResult = ssp_mul_ps(vResult, m[0]);

    __m128 vTemp = ssp_shuffle_ps(*v, *v, _MM_SHUFFLE(1,1,1,1));
    vTemp = ssp_mul_ps(vTemp, m[1]);
    
	vResult = ssp_add_ps(vResult, vTemp);
    vTemp = ssp_shuffle_ps(*v, *v, _MM_SHUFFLE(2,2,2,2));
    
	vTemp = ssp_mul_ps(vTemp, m[2]);
    vResult = ssp_add_ps(vResult, vTemp);
    
	vResult = ssp_add_ps(vResult, m[3]);
	return vResult;
}


void HelperSSE::MatrixMultiply(const __m128 *m1, const __m128 *m2, __m128 *result)
{
	__m128 X, Y, Z, W;
	float *mat = (float*)m1;
	 

	X = ssp_set1_ps(*(mat + 0));
	Y = ssp_set1_ps(*(mat + 1));
	Z = ssp_set1_ps(*(mat + 2));
	W = ssp_set1_ps(*(mat + 3));

	result[0] = ssp_mul_ps(X, m2[0]);
	result[0] = ssp_add_ps(result[0], ssp_mul_ps(Y, m2[1]));
	result[0] = ssp_add_ps(result[0], ssp_mul_ps(Z, m2[2]));
	result[0] = ssp_add_ps(result[0], ssp_mul_ps(W, m2[3]));

	X = ssp_set1_ps(*(mat + 4));
	Y = ssp_set1_ps(*(mat + 5));
	Z = ssp_set1_ps(*(mat + 6));
	W = ssp_set1_ps(*(mat + 7));

	result[1] = ssp_mul_ps(X, m2[0]);
	result[1] = ssp_add_ps(result[1], ssp_mul_ps(Y, m2[1]));
	result[1] = ssp_add_ps(result[1], ssp_mul_ps(Z, m2[2]));
	result[1] = ssp_add_ps(result[1], ssp_mul_ps(W, m2[3]));

	X = ssp_set1_ps(*(mat + 8));
	Y = ssp_set1_ps(*(mat + 9));
	Z = ssp_set1_ps(*(mat + 10));
	W = ssp_set1_ps(*(mat + 11));

	result[2] = ssp_mul_ps(X, m2[0]);
	result[2] = ssp_add_ps(result[2], ssp_mul_ps(Y, m2[1]));
	result[2] = ssp_add_ps(result[2], ssp_mul_ps(Z, m2[2]));
	result[2] = ssp_add_ps(result[2], ssp_mul_ps(W, m2[3]));
	
	X = ssp_set1_ps(*(mat + 12));
	Y = ssp_set1_ps(*(mat + 13));
	Z = ssp_set1_ps(*(mat + 14));
	W = ssp_set1_ps(*(mat + 15));

	result[3] = ssp_mul_ps(X, m2[0]);
	result[3] = ssp_add_ps(result[3], ssp_mul_ps(Y, m2[1]));
	result[3] = ssp_add_ps(result[3], ssp_mul_ps(Z, m2[2]));
	result[3] = ssp_add_ps(result[3], ssp_mul_ps(W, m2[3]));
	
}

void BoxTestSetupSSE::Init(const __m128 viewMatrix[4], const __m128 projMatrix[4], const float4x4 &viewportMatrix, CPUTCamera *pCamera, float occludeeSizeThreshold)
{
	__m128 viewPortMatrix[4];
	viewPortMatrix[0] = ssp_loadu_ps((float*)&viewportMatrix.r0);
	viewPortMatrix[1] = ssp_loadu_ps((float*)&viewportMatrix.r1);
	viewPortMatrix[2] = ssp_loadu_ps((float*)&viewportMatrix.r2);
	viewPortMatrix[3] = ssp_loadu_ps((float*)&viewportMatrix.r3);

	MatrixMultiply(viewMatrix, projMatrix, mViewProjViewport);
	MatrixMultiply(mViewProjViewport, viewPortMatrix, mViewProjViewport);

	mpCamera = pCamera;

	float fov = pCamera->GetFov();
	float tanOfHalfFov = tanf(fov * 0.5f);
	radiusThreshold = occludeeSizeThreshold * occludeeSizeThreshold * tanOfHalfFov;
} 