using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenTK;

namespace Oxel
{
    /*!
    **
    ** Copyright (c) 2009 by John W. Ratcliff mailto:jratcliffscarab@gmail.com
    **
    ** The MIT license:
    **
    ** Permission is hereby granted, FREE of charge, to any person obtaining a copy
    ** of this software and associated documentation files (the "Software"), to deal
    ** in the Software without restriction, including without limitation the rights
    ** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    ** copies of the Software, and to permit persons to whom the Software is furnished
    ** to do so, subject to the following conditions:
    **
    ** The above copyright notice and this permission notice shall be included in all
    ** copies or substantial portions of the Software.

    ** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    ** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    ** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    ** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    ** WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    ** CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    */
    public class Eigen
    {
        public float[][] mElement;
        public float[] m_afDiag;
        public float[] m_afSubd;
        public bool m_bIsRotation;

        public Eigen()
        {
            mElement = new float[3][]
                {
                    new float[3],
                    new float[3],
                    new float[3]
                };
            m_afDiag = new float[3];
            m_afSubd = new float[3];
        }

        public void DecrSortEigenStuff()
        {
            Tridiagonal(); //diagonalize the matrix.
            QLAlgorithm(); //
            DecreasingSort();
            GuaranteeRotation();
        }

        public void Tridiagonal()
        {
            float fM00 = mElement[0][0];
            float fM01 = mElement[0][1];
            float fM02 = mElement[0][2];
            float fM11 = mElement[1][1];
            float fM12 = mElement[1][2];
            float fM22 = mElement[2][2];

            m_afDiag[0] = fM00;
            m_afSubd[2] = 0;
            if (fM02 != (float)0.0)
            {
                float fLength = (float)Math.Sqrt(fM01 * fM01 + fM02 * fM02);
                float fInvLength = ((float)1.0) / fLength;
                fM01 *= fInvLength;
                fM02 *= fInvLength;
                float fQ = ((float)2.0) * fM01 * fM12 + fM02 * (fM22 - fM11);
                m_afDiag[1] = fM11 + fM02 * fQ;
                m_afDiag[2] = fM22 - fM02 * fQ;
                m_afSubd[0] = fLength;
                m_afSubd[1] = fM12 - fM01 * fQ;
                mElement[0][0] = (float)1.0;
                mElement[0][1] = (float)0.0;
                mElement[0][2] = (float)0.0;
                mElement[1][0] = (float)0.0;
                mElement[1][1] = fM01;
                mElement[1][2] = fM02;
                mElement[2][0] = (float)0.0;
                mElement[2][1] = fM02;
                mElement[2][2] = -fM01;
                m_bIsRotation = false;
            }
            else
            {
                m_afDiag[1] = fM11;
                m_afDiag[2] = fM22;
                m_afSubd[0] = fM01;
                m_afSubd[1] = fM12;
                mElement[0][0] = (float)1.0;
                mElement[0][1] = (float)0.0;
                mElement[0][2] = (float)0.0;
                mElement[1][0] = (float)0.0;
                mElement[1][1] = (float)1.0;
                mElement[1][2] = (float)0.0;
                mElement[2][0] = (float)0.0;
                mElement[2][1] = (float)0.0;
                mElement[2][2] = (float)1.0;
                m_bIsRotation = true;
            }
        }

        public bool QLAlgorithm()
        {
            const int iMaxIter = 32;

            for (int i0 = 0; i0 < 3; i0++)
            {
                int i1;
                for (i1 = 0; i1 < iMaxIter; i1++)
                {
                    int i2;
                    for (i2 = i0; i2 <= (3 - 2); i2++)
                    {
                        float fTmp = Math.Abs(m_afDiag[i2]) + Math.Abs(m_afDiag[i2 + 1]);
                        if (Math.Abs(m_afSubd[i2]) + fTmp == fTmp)
                            break;
                    }
                    if (i2 == i0)
                    {
                        break;
                    }

                    float fG = (m_afDiag[i0 + 1] - m_afDiag[i0]) / (((float)2.0) * m_afSubd[i0]);
                    float fR = (float)Math.Sqrt(fG * fG + (float)1.0);
                    if (fG < (float)0.0)
                    {
                        fG = m_afDiag[i2] - m_afDiag[i0] + m_afSubd[i0] / (fG - fR);
                    }
                    else
                    {
                        fG = m_afDiag[i2] - m_afDiag[i0] + m_afSubd[i0] / (fG + fR);
                    }
                    float fSin = (float)1.0, fCos = (float)1.0, fP = (float)0.0;
                    for (int i3 = i2 - 1; i3 >= i0; i3--)
                    {
                        float fF = fSin * m_afSubd[i3];
                        float fB = fCos * m_afSubd[i3];
                        if (Math.Abs(fF) >= Math.Abs(fG))
                        {
                            fCos = fG / fF;
                            fR = (float)Math.Sqrt(fCos * fCos + (float)1.0);
                            m_afSubd[i3 + 1] = fF * fR;
                            fSin = ((float)1.0) / fR;
                            fCos *= fSin;
                        }
                        else
                        {
                            fSin = fF / fG;
                            fR = (float)Math.Sqrt(fSin * fSin + (float)1.0);
                            m_afSubd[i3 + 1] = fG * fR;
                            fCos = ((float)1.0) / fR;
                            fSin *= fCos;
                        }
                        fG = m_afDiag[i3 + 1] - fP;
                        fR = (m_afDiag[i3] - fG) * fSin + ((float)2.0) * fB * fCos;
                        fP = fSin * fR;
                        m_afDiag[i3 + 1] = fG + fP;
                        fG = fCos * fR - fB;
                        for (int i4 = 0; i4 < 3; i4++)
                        {
                            fF = mElement[i4][i3 + 1];
                            mElement[i4][i3 + 1] = fSin * mElement[i4][i3] + fCos * fF;
                            mElement[i4][i3] = fCos * mElement[i4][i3] - fSin * fF;
                        }
                    }
                    m_afDiag[i0] -= fP;
                    m_afSubd[i0] = fG;
                    m_afSubd[i2] = (float)0.0;
                }
                if (i1 == iMaxIter)
                {
                    return false;
                }
            }
            return true;
        }

        public void DecreasingSort()
        {
            //sort eigenvalues in decreasing order, e[0] >= ... >= e[iSize-1]
            for (int i0 = 0, i1; i0 <= 3 - 2; i0++)
            {
                // locate maximum eigenvalue
                i1 = i0;
                float fMax = m_afDiag[i1];
                int i2;
                for (i2 = i0 + 1; i2 < 3; i2++)
                {
                    if (m_afDiag[i2] > fMax)
                    {
                        i1 = i2;
                        fMax = m_afDiag[i1];
                    }
                }

                if (i1 != i0)
                {
                    // swap eigenvalues
                    m_afDiag[i1] = m_afDiag[i0];
                    m_afDiag[i0] = fMax;
                    // swap eigenvectors
                    for (i2 = 0; i2 < 3; i2++)
                    {
                        float fTmp = mElement[i2][i0];
                        mElement[i2][i0] = mElement[i2][i1];
                        mElement[i2][i1] = fTmp;
                        m_bIsRotation = !m_bIsRotation;
                    }
                }
            }
        }

        public void GuaranteeRotation()
        {
            if (!m_bIsRotation)
            {
                // change sign on the first column
                for (int iRow = 0; iRow < 3; iRow++)
                {
                    mElement[iRow][0] = -mElement[iRow][0];
                }
            }
        }
    }

    public static class BestFit
    {
        public enum FitStrategy
        {
            FS_FAST_FIT,   // just computes the diagonals only, can be off substantially at times.
            FS_MEDIUM_FIT, // rotates on one axis to converge to a solution.
            FS_SLOW_FIT,   // rotates on all three axes to find the best fit.
            FS_SLOWEST_FIT,
        };

        public static bool ComputeBestFitPlane(Vector3[] points, out Plane plane)
        {
            Vector3 origin = new Vector3(0, 0, 0);

            for (int i = 0; i < points.Length; i++)
            {
                origin = origin + points[i];
            }

            float recip = 1.0f / points.Length; // reciprocol of total weighting

            origin.X *= recip;
            origin.Y *= recip;
            origin.Z *= recip;

            float fSumXX = 0;
            float fSumXY = 0;
            float fSumXZ = 0;

            float fSumYY = 0;
            float fSumYZ = 0;
            float fSumZZ = 0;

            Vector3 kDiff;
            for (int i = 0; i < points.Length; i++)
            {
                Vector3 p = points[i];

                kDiff.X = (p.X - origin.X); // apply vertex weighting!
                kDiff.Y = (p.Y - origin.Y);
                kDiff.Z = (p.Z - origin.Z);

                fSumXX += kDiff.X * kDiff.X; // sume of the squares of the differences.
                fSumXY += kDiff.X * kDiff.Y; // sume of the squares of the differences.
                fSumXZ += kDiff.X * kDiff.Z; // sume of the squares of the differences.

                fSumYY += kDiff.Y * kDiff.Y;
                fSumYZ += kDiff.Y * kDiff.Z;
                fSumZZ += kDiff.Z * kDiff.Z;
            }

            fSumXX *= recip;
            fSumXY *= recip;
            fSumXZ *= recip;
            fSumYY *= recip;
            fSumYZ *= recip;
            fSumZZ *= recip;

            // setup the eigensolver
            Eigen kES = new Eigen();
            kES.mElement[0][0] = fSumXX;
            kES.mElement[0][1] = fSumXY;
            kES.mElement[0][2] = fSumXZ;

            kES.mElement[1][0] = fSumXY;
            kES.mElement[1][1] = fSumYY;
            kES.mElement[1][2] = fSumYZ;

            kES.mElement[2][0] = fSumXZ;
            kES.mElement[2][1] = fSumYZ;
            kES.mElement[2][2] = fSumZZ;

            // compute eigenstuff, smallest eigenvalue is in last position
            kES.DecrSortEigenStuff();

            Vector3 kNormal = new Vector3();
            kNormal.X = kES.mElement[0][2];
            kNormal.Y = kES.mElement[1][2];
            kNormal.Z = kES.mElement[2][2];

            // the minimum energy
            plane = new Plane(kNormal, 0 - Vector3.Dot(kNormal, origin));
            return true;
        }

        // computes the OBB for this set of points relative to this transform matrix.
        public static void ComputeOBB(Vector3[] points, ref Matrix4 matrix, out float[] sides)
        {
            AABBf aabb = new AABBf();

            Matrix4 matrixInverse = matrix;
            matrixInverse.Invert();
            for (int i = 0; i < points.Length; i++)
            {
                Vector3 t = Vector3.Transform(points[i], matrixInverse); // inverse rotate translate
                aabb.Add(t);
            }

            sides = new float[3];
            sides[0] = aabb.MaxX - aabb.MinX;
            sides[1] = aabb.MaxY - aabb.MinY;
            sides[2] = aabb.MaxZ - aabb.MinZ;

            Vector3 center = new Vector3();
            center.X = sides[0] * 0.5f + aabb.MinX;
            center.Y = sides[1] * 0.5f + aabb.MinY;
            center.Z = sides[2] * 0.5f + aabb.MinZ;

            Vector3 ocenter = Vector3.Transform(center, matrix);

            matrix = matrix * Matrix4.CreateTranslation(ocenter);
        }

        public static void ComputeBestFitOBB(Vector3[] points, out float[] sides, out Matrix4 matrix, FitStrategy strategy)
        {
            matrix = Matrix4.Identity;
            AABBf aabb = new AABBf();
            for (int i = 0; i < points.Length; i++)
                aabb.Add(points[i]);

            float avolume = (aabb.MaxX - aabb.MinX) * (aabb.MaxY - aabb.MinY) * (aabb.MaxZ - aabb.MinZ);

            Plane plane;
            ComputeBestFitPlane(points, out plane);

            plane.ToMatrix(ref matrix);
            ComputeOBB(points, ref matrix, out sides);

            Matrix4 refmatrix = new Matrix4();
            refmatrix = matrix;

            float volume = sides[0] * sides[1] * sides[2];

            float stepSize = 5;
            switch (strategy)
            {
                case BestFit.FitStrategy.FS_FAST_FIT:
                    stepSize = 13; // 15 degree increments
                    break;
                case BestFit.FitStrategy.FS_MEDIUM_FIT:
                    stepSize = 7; // 10 degree increments
                    break;
                case BestFit.FitStrategy.FS_SLOW_FIT:
                    stepSize = 3; // 5 degree increments
                    break;
                case BestFit.FitStrategy.FS_SLOWEST_FIT:
                    stepSize = 1; // 1 degree increments
                    break;
            }

            Quaternion quat = new Quaternion();
            for (float a = 0; a < 180; a += stepSize)
            {
                Matrix4 temp;
                Matrix4.CreateRotationY(MathHelper.DegreesToRadians(a), out temp);
                QuaternionEx.CreateFromMatrix(ref temp, ref quat);

                Matrix4 pmatrix;
                Matrix4.Mult(ref temp, ref refmatrix, out pmatrix);

                float[] psides;
                ComputeOBB(points, ref pmatrix, out psides);
                float v = psides[0] * psides[1] * psides[2];
                if (v < volume)
                {
                    volume = v;
                    matrix = pmatrix;
                    sides[0] = psides[0];
                    sides[1] = psides[1];
                    sides[2] = psides[2];
                }
            }

            if (avolume < volume)
            {
                matrix = Matrix4.CreateTranslation(
                    (aabb.MinX + aabb.MaxX) * 0.5f,
                    (aabb.MinY + aabb.MaxY) * 0.5f,
                    (aabb.MinZ + aabb.MaxZ) * 0.5f);
                sides[0] = aabb.MaxX - aabb.MinX;
                sides[1] = aabb.MaxY - aabb.MinY;
                sides[2] = aabb.MaxZ - aabb.MinZ;
            }
        }
    }
}