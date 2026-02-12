#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>

#include "nbody_io.hh"
#include "nbody_generate.hh"
#include "time_experiment.hh"
#include <arm_neon.h>

#ifndef __GNUC__
#define __restrict__
#endif

// basic data type for position, velocity, acceleration
const int M = 3;
typedef double double3[M]; // pad up for later use with SIMD
const int B = 128;				 // block size for tiling

/*const double gamma = 6.674E-11;*/
const double G = 1.0;
const double epsilon2 = 1E-10;

/** \brief compute acceleration vector from position and masses (vanilla version)
 *
 * This version works on structure of arrays data layout
 * Executes \sum_{i=0}^{n-1} (n-i-1)*26 = n(n-1)*13
 * flops including 1 division and one square root
 */
void acceleration(int n, double *__restrict__ x, double *__restrict__ m, double *__restrict__ a)
{
	for (int i = 0; i < n; i++)
		for (int j = i + 1; j < n; j++)
		{
			double d0 = x[j] - x[i];
			double d1 = x[n + j] - x[n + i];
			double d2 = x[2 * n + j] - x[2 * n + i];
			double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
			double r = sqrt(r2);
			double invfact = G / (r * r2);
			double factori = m[i] * invfact;
			double factorj = m[j] * invfact;
			a[i] += factorj * d0;
			a[n + i] += factorj * d1;
			a[2 * n + i] += factorj * d2;
			a[j] -= factori * d0;
			a[n + j] -= factori * d1;
			a[2 * n + j] -= factori * d2;
		}
}

/** \brief compute acceleration vector from position and masses (vectorized using 1x2 masses)
 *
 * This version works on structure of arrays data layout
 * and it is conceptually easy to extend to arbitrary simd width
 */
void acceleration_blocked_vectorized_new (int n, double *__restrict__ x, double *__restrict__ m, double *__restrict__ aglobal)
{
	// simd definitions
	const int simd_width = 2;
	using simd_type = float64x2_t;

	// make private acceleration vectors to accumulate to
	// for omp parallel version 
	double *aI;
	aI = new (std::align_val_t(64)) double[3 * B];
	double *aJ;
	aJ = new (std::align_val_t(64)) double[3 * B];

	// simd registers
	simd_type XI0, XI1, XI2; // broadcasted coordinates of body i
	simd_type DJ0, DJ1, DJ2; // position of bodies j, ..., j+simd_width-1 
	simd_type AJ0, AJ1, AJ2; // accelerations of bodies j, ..., j+simd_width-1 
	simd_type R, R3;         // distances, squared distances and cubed distances
	simd_type MI,MJ;         // broadcasted mass of body i, and individual masses j
	simd_type	S,F;           // scalar factors
	simd_type	VECG;          // gravitational constant as vector
	VECG = vmovq_n_f64(G);   // broadcast for gravitational constant

	// for conversion from simd_type to scalar
	double factor[simd_width];
	double distance0[simd_width];
	double distance1[simd_width];
	double distance2[simd_width];

	for (int I = 0; I < n; I += B)
	{
		// clear accelerations for whole block row
		for (int i = 0; i < 3 * B; ++i)
			aI[i] = 0.0;

		// diagonal block (I,I) is handled in the standard way exploiting symmetry
		for (int i = I; i < I + B; i++)
			for (int j = i + 1; j < I + B; j++)
			{
				double d0 = x[j] - x[i];
				double d1 = x[n + j] - x[n + i];
				double d2 = x[2 * n + j] - x[2 * n + i];
				double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
				double r = sqrt(r2);
				double invfact = G / (r * r2);
				double factori = m[i] * invfact;
				double factorj = m[j] * invfact;
				aI[i - I] += factorj * d0; // updates private vector
				aI[i - I + B] += factorj * d1;
				aI[i - I + 2 * B] += factorj * d2;
				aI[j - I] -= factori * d0;
				aI[j - I + B] -= factori * d1;
				aI[j - I + 2 * B] -= factori * d2;
			}

		// blocks J>I can also exploit symmetry
		for (int J = I + B; J < n; J += B)
		{
			// clear accelerations for whole block column
			for (int j = 0; j < 3 * B; ++j)
				aJ[j] = 0.0;

			for (int i = I; i < I + B; i += 1)
			{
				// load position of body i. This can be reused for the whole block row.
				XI0 = vmovq_n_f64(x[i]);				 // load *broadcast* for x-coordinate
				XI1 = vmovq_n_f64(x[i + n]);		 // load *broadcast* for y-coordinate
				XI2 = vmovq_n_f64(x[i + 2 * n]); // load *broadcast* for z-coordinate
				MI  = vmovq_n_f64(m[i]);         // load *broadcast* for mass i

				for (int j = J; j < J + B; j += simd_width)
				{
					// compute interaction of body i and j,j+1 (here simd_width = 2)

					// load positions of simd_width particles starting at j
					DJ0 = vld1q_f64(&(x[j]));					// x coordinates
					DJ1 = vld1q_f64(&(x[j + n]));			// y coordinates
					DJ2 = vld1q_f64(&(x[j + 2 * n])); // z coordinates
					DJ0 = vsubq_f64(DJ0, XI0);				// difference of x coordinates for simd_width bodies
					DJ1 = vsubq_f64(DJ1, XI1);				// difference of y coordinates for simd_width bodies
					DJ2 = vsubq_f64(DJ2, XI2);				// difference of z coordinates for simd_width bodies

					// distances of body i to all bodies j,...,j+w-1 with power 1,2,3
					R3 = vmovq_n_f64(epsilon2);			 // regularization
					R3 = vfmaq_f64(R3, DJ0, DJ0);		 // fma
					R3 = vfmaq_f64(R3, DJ1, DJ1);		 // fma
					R3 = vfmaq_f64(R3, DJ2, DJ2);		 // fma
					R = vsqrtq_f64(R3); // compute square roots; the advantage is, we do not need the result immediately
					R3 = vmulq_f64(R3, R);           // R to power 3

					// now compute all scalar factors between i and all bodies j,...,j+w-1 
					S = vdivq_f64(VECG,R3);          // these are the scalar factors up to the masses

					// now we update the accelerations of the masses j,...,j+w-1
					AJ0 = vld1q_f64(&(aJ[j - J]));				// load x coordinates
					AJ1 = vld1q_f64(&(aJ[j - J + B]));		// load y coordinates
					AJ2 = vld1q_f64(&(aJ[j - J + 2 * B]));// load z coordinates
					F = vmulq_f64(MI, S);                 // scalar factors with masses
          AJ0 = vfmsq_f64(AJ0, F, DJ0);         // fms
          AJ1 = vfmsq_f64(AJ1, F, DJ1);         // fms
          AJ2 = vfmsq_f64(AJ2, F, DJ2);         // fms
					vst1q_f64(&(aJ[j - J]), AJ0);         // store result
					vst1q_f64(&(aJ[j - J + B]), AJ1);			// store result
					vst1q_f64(&(aJ[j - J + 2 * B]), AJ2); // store result

					// scalar updates for mass i
					MJ = vld1q_f64(&(m[j]));		     			// simd_width masses from j; prefetch
					F = vmulq_f64(MJ, S);                 // scalar factors with masses
					// F, DJ0, DJ1, DJ2 each contain contributions from two bodies j,j+1
					vst1q_f64(factor,F);
					vst1q_f64(distance0,DJ0);
					vst1q_f64(distance1,DJ1);
					vst1q_f64(distance2,DJ2);
					aI[i-I] += factor[0]*distance0[0];
					aI[i-I] += factor[1]*distance0[1];
					aI[i-I+B] += factor[0]*distance1[0];
					aI[i-I+B] += factor[1]*distance1[1];
					aI[i-I+2*B] += factor[0]*distance2[0];
					aI[i-I+2*B] += factor[1]*distance2[1];
				}
			}
			// update accelerations for block J
			for (int j = 0; j < B; ++j)
				aglobal[J + j] += aJ[j];
			for (int j = 0; j < B; ++j)
				aglobal[J + j + n] += aJ[j + B];
			for (int j = 0; j < B; ++j)
				aglobal[J + j + 2 * n] += aJ[j + 2 * B];
		} // end J loop
		// update accelerations of block I
		for (int i = 0; i < B; ++i)
			aglobal[I + i] += aI[i];
		for (int i = 0; i < B; ++i)
			aglobal[I + i + n] += aI[i + B];
		for (int i = 0; i < B; ++i)
			aglobal[I + i + 2 * n] += aI[i + 2 * B];
	} // end I loop
	// delete thread local data
	delete[] aI;
	delete[] aJ;
}

/** \brief compute acceleration vector from position and masses (vectorized using 2x2 masses)
 *
 * This version works on structure of arrays data layout
 */
void acceleration_blocked_vectorized(int n, double *__restrict__ x, double *__restrict__ m, double *__restrict__ aglobal)
{
	// make private acceleration vectors to accumulate to
	double *aI;
	aI = new (std::align_val_t(64)) double[3 * B];
	double *aJ;
	aJ = new (std::align_val_t(64)) double[3 * B];

	float64x2_t XJ0, XJ1, XJ2;
	float64x2_t XI0, XI1, XI2;
	float64x2_t D0, D1, D2;
	float64x2_t R0, R1;
	float64x2_t A0, A1, A2;
	float64x2_t S, M;

	for (int I = 0; I < n; I += B)
	{
		// clear accelerations for whole block row
		for (int i = 0; i < 3 * B; ++i)
			aI[i] = 0.0;

		// diagonal block (I,I) is handled in the standard way exploiting symmetry
		for (int i = I; i < I + B; i++)
			for (int j = i + 1; j < I + B; j++)
			{
				double d0 = x[j] - x[i];
				double d1 = x[n + j] - x[n + i];
				double d2 = x[2 * n + j] - x[2 * n + i];
				double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
				double r = sqrt(r2);
				double invfact = G / (r * r2);
				double factori = m[i] * invfact;
				double factorj = m[j] * invfact;
				aI[i - I] += factorj * d0; // updates private vector
				aI[i - I + B] += factorj * d1;
				aI[i - I + 2 * B] += factorj * d2;
				aI[j - I] -= factori * d0;
				aI[j - I + B] -= factori * d1;
				aI[j - I + 2 * B] -= factori * d2;
			}

		// blocks J>I can also exploit symmetry
		for (int J = I + B; J < n; J += B)
		{
			// clear accelerations for whole block column
			for (int j = 0; j < 3 * B; ++j)
				aJ[j] = 0.0;

			for (int i = I; i < I + B; i += 2)
				for (int j = J; j < J + B; j += 2)
				{
					// compute interaction of 2x2 masses i,i+1 and j,j+1

					// load positions of two particles starting at j
					XJ0 = vld1q_f64(&(x[j]));					// x coordinates
					XJ1 = vld1q_f64(&(x[j + n]));			// y coordinates
					XJ2 = vld1q_f64(&(x[j + 2 * n])); // z coordinates

					// difference to mass i
					XI0 = vmovq_n_f64(x[i]);				 // load broadcast for x-coordinate
					XI1 = vmovq_n_f64(x[i + n]);		 // load broadcast for y-coordinate
					XI2 = vmovq_n_f64(x[i + 2 * n]); // load broadcast for z-coordinate
					D0 = vsubq_f64(XJ0, XI0);				 // distance j,j+1-i, x-coordinate
					D1 = vsubq_f64(XJ1, XI1);				 // distance j,j+1-i, y-coordinate
					D2 = vsubq_f64(XJ2, XI2);				 // distance j,j+1-i, z-coordinate
					A0 = vmovq_n_f64(epsilon2);			 // now compute squared distances
					A0 = vfmaq_f64(A0, D0, D0);			 // fma
					A0 = vfmaq_f64(A0, D1, D1);			 // fma
					A0 = vfmaq_f64(A0, D2, D2);			 // fma
					// now A0 = ( ||x_j-x_i||^2 ; ||x_{j+1}-x_i||^2)
					R0 = vsqrtq_f64(A0); // compute square roots; the advantage is, we do not need the result immediately

					// difference to mass i+1
					XI0 = vmovq_n_f64(x[i + 1]);				 // load broadcast for x-coordinate
					XI1 = vmovq_n_f64(x[i + 1 + n]);		 // load broadcast for y-coordinate
					XI2 = vmovq_n_f64(x[i + 1 + 2 * n]); // load broadcast for z-coordinate
					D0 = vsubq_f64(XJ0, XI0);						 // distance j,j+1-i+1, x-coordinate
					D1 = vsubq_f64(XJ1, XI1);						 // distance j,j+1-i+1, y-coordinate
					D2 = vsubq_f64(XJ2, XI2);						 // distance j,j+1-i+1, z-coordinate
					A1 = vmovq_n_f64(epsilon2);					 // now compute squared distances
					A1 = vfmaq_f64(A1, D0, D0);					 // fma
					A1 = vfmaq_f64(A1, D1, D1);					 // fma
					A1 = vfmaq_f64(A1, D2, D2);					 // fma
					// now A1 = ( ||x_j-x_{i+1}||^2 ; ||x_{j+1}-x_{i+1}||^2)
					R1 = vsqrtq_f64(A1); // compute square roots; the advantage is, we do not need the result immediately

					// now we proceed to compute the scalar factors s_{i,j} = G/r_{i,j}^3
					A0 = vmulq_f64(A0, R0); // now A0 is distance^3
					A1 = vmulq_f64(A1, R1); // now A1 is distance^3
					D0 = vmovq_n_f64(G);		// load broadcast for G
					R0 = vdivq_f64(D0, A0);
					R1 = vdivq_f64(D0, A1);
					// Now R0, R1 contain the scalar factors in the following form
					// R0 = (s_{j,i} ; s_{j+1,i})
					// R1 = (s_{j,i+1} ; s_{j+1,i+1})

					// now we update the accelerations of the four masses

					// FIRST contribution to acceleration of masses j,j+1 *from* masses i and i+1
					// load acceleration of masses j,j+1
					A0 = vld1q_f64(&(aJ[j - J]));					// x coordinates
					A1 = vld1q_f64(&(aJ[j - J + B]));			// y coordinates
					A2 = vld1q_f64(&(aJ[j - J + 2 * B])); // z coordinates

					// contribution from mass i
					// XJ* still contains the positions of j,j+1. But we need to recompute the differences
					XI0 = vmovq_n_f64(x[i]);				 // load broadcast for x-coordinate
					XI1 = vmovq_n_f64(x[i + n]);		 // load broadcast for y-coordinate
					XI2 = vmovq_n_f64(x[i + 2 * n]); // load broadcast for z-coordinate
					// build difference vectors
					D0 = vsubq_f64(XI0, XJ0); // distance i-j,j+1, x-coordinate
					D1 = vsubq_f64(XI1, XJ1); // distance i-j,j+1, y-coordinate
					D2 = vsubq_f64(XI2, XJ2); // distance i-j,j+1, z-coordinate
					// build scalar factors
					S = vmovq_n_f64(m[i]);
					S = vmulq_f64(S, R0);
					// update
					A0 = vfmaq_f64(A0, S, D0); // fma
					A1 = vfmaq_f64(A1, S, D1); // fma
					A2 = vfmaq_f64(A2, S, D2); // fma

					// contribution from mass i+1
					// XJ* still contains the positions of j,j+1. But we need to recompute the differences
					XI0 = vmovq_n_f64(x[i + 1]);				 // load broadcast for x-coordinate
					XI1 = vmovq_n_f64(x[i + 1 + n]);		 // load broadcast for y-coordinate
					XI2 = vmovq_n_f64(x[i + 1 + 2 * n]); // load broadcast for z-coordinate
					// build difference vectors
					D0 = vsubq_f64(XI0, XJ0); // distance i+1-j,j+1, x-coordinate
					D1 = vsubq_f64(XI1, XJ1); // distance i+1-j,j+1, y-coordinate
					D2 = vsubq_f64(XI2, XJ2); // distance i+1-j,j+1, z-coordinate
					// build scalar factors
					S = vmovq_n_f64(m[i + 1]);
					S = vmulq_f64(S, R1);
					// update
					A0 = vfmaq_f64(A0, S, D0); // fma
					A1 = vfmaq_f64(A1, S, D1); // fma
					A2 = vfmaq_f64(A2, S, D2); // fma

					// store acceleration of masses j,j+1
					vst1q_f64(&(aJ[j - J]), A0);
					vst1q_f64(&(aJ[j - J + B]), A1);
					vst1q_f64(&(aJ[j - J + 2 * B]), A2);

					// SECOND contribution to acceleration of masses i,i+1 *from* masses j and j+1
					// load coordinates of i,i+1 in SIMD register
					XI0 = vld1q_f64(&(x[i]));					// x coordinates
					XI1 = vld1q_f64(&(x[i + n]));			// y coordinates
					XI2 = vld1q_f64(&(x[i + 2 * n])); // z coordinates

					// contribution from mass j
					XJ0 = vmovq_n_f64(x[j]);				 // load broadcast for x-coordinate
					XJ1 = vmovq_n_f64(x[j + n]);		 // load broadcast for y-coordinate
					XJ2 = vmovq_n_f64(x[j + 2 * n]); // load broadcast for z-coordinate
					// build difference vectors
					D0 = vsubq_f64(XJ0, XI0); // distance i+1-j,j+1, x-coordinate
					D1 = vsubq_f64(XJ1, XI1); // distance i+1-j,j+1, y-coordinate
					D2 = vsubq_f64(XJ2, XI2); // distance i+1-j,j+1, z-coordinate
					// build scalar factors
					M = vmovq_n_f64(m[j]);
					S = R0;
					S = vcopyq_laneq_f64(S, 1, R1, 0);
					S = vmulq_f64(S, M);
					// load acceleration of masses i,i+1
					A0 = vld1q_f64(&(aI[i - I]));					// x coordinates
					A1 = vld1q_f64(&(aI[i - I + B]));			// y coordinates
					A2 = vld1q_f64(&(aI[i - I + 2 * B])); // z coordinates
					// update
					A0 = vfmaq_f64(A0, S, D0); // fma
					A1 = vfmaq_f64(A1, S, D1); // fma
					A2 = vfmaq_f64(A2, S, D2); // fma

					// contribution from mass j+1
					XJ0 = vmovq_n_f64(x[j + 1]);				 // load broadcast for x-coordinate
					XJ1 = vmovq_n_f64(x[j + 1 + n]);		 // load broadcast for y-coordinate
					XJ2 = vmovq_n_f64(x[j + 1 + 2 * n]); // load broadcast for z-coordinate
					// build difference vectors
					D0 = vsubq_f64(XJ0, XI0); // distance i+1-j,j+1, x-coordinate
					D1 = vsubq_f64(XJ1, XI1); // distance i+1-j,j+1, y-coordinate
					D2 = vsubq_f64(XJ2, XI2); // distance i+1-j,j+1, z-coordinate
					// build scalar factors
					M = vmovq_n_f64(m[j + 1]);
					S = R1;
					S = vcopyq_laneq_f64(S, 0, R0, 1);
					S = vmulq_f64(S, M);
					// update
					A0 = vfmaq_f64(A0, S, D0); // fma
					A1 = vfmaq_f64(A1, S, D1); // fma
					A2 = vfmaq_f64(A2, S, D2); // fma

					// store acceleration of masses i,i+1
					vst1q_f64(&(aI[i - I]), A0);
					vst1q_f64(&(aI[i - I + B]), A1);
					vst1q_f64(&(aI[i - I + 2 * B]), A2);
				}
			// update accelerations for block J
			for (int j = 0; j < B; ++j)
				aglobal[J + j] += aJ[j];
			for (int j = 0; j < B; ++j)
				aglobal[J + j + n] += aJ[j + B];
			for (int j = 0; j < B; ++j)
				aglobal[J + j + 2 * n] += aJ[j + 2 * B];
		} // end J loop

		// update accelerations of block I
		for (int i = 0; i < B; ++i)
			aglobal[I + i] += aI[i];
		for (int i = 0; i < B; ++i)
			aglobal[I + i + n] += aI[i + B];
		for (int i = 0; i < B; ++i)
			aglobal[I + i + 2 * n] += aI[i + 2 * B];

	} // end I loop
	// delete thread local data
	delete[] aI;
	delete[] aJ;
}

/** \brief do one time step with leapfrog
 *
 * does n*(n-1)*13 + 12n flops
 */
void leapfrog(int n, double dt, double *__restrict__ x, double *__restrict__ v, double *__restrict__ m, double *__restrict__ a)
{
	// update position: 6n flops
	for (int i = 0; i < 3 * n; i++)
		x[i] += dt * v[i];

	// save and clear acceleration
	for (int i = 0; i < 3 * n; i++)
		a[i] = 0.0;

	// compute new acceleration: n*(n-1)*13 flops
	// acceleration(n,x,m,a);
	acceleration_blocked_vectorized_new(n, x, m, a);

	// update velocity: 6n flops
	for (int i = 0; i < 3 * n; i++)
		v[i] += dt * a[i];
}

template <typename T>
size_t alignment(const T *p)
{
	for (size_t m = 64; m > 1; m /= 2)
		if (((size_t)p) % m == 0)
			return m;
	return 1;
}

// functions for AoS <-> SoA transformation
void copy(double *to, double3 *from, size_t n)
{
	for (size_t i = 0; i < n; ++i)
		for (size_t j = 0; j < 3; ++j)
			to[j * n + i] = from[i][j];
}
void copy(double3 *to, double *from, size_t n)
{
	for (size_t i = 0; i < n; ++i)
		for (size_t j = 0; j < 3; ++j)
			to[i][j] = from[j * n + i];
}

int main(int argc, char **argv)
{
	int n;							// number of bodies in the system
	double *m;					// array for maasses
	double3 *x;					// array for positions
	double3 *v;					// array for velocites
	double3 *a;					// array for accelerations
	int timesteps;			// final time step number
	int k;							// time step number
	int mod;						// files are written when k is a multiple of mod
	char basename[256]; // common part of file name
	char name[256];			// filename with number
	FILE *file;					// C style file hande
	double t;						// current time
	double dt;					// time step

	// command line for restarting
	if (argc == 5)
	{
		sscanf(argv[1], "%s", &basename);
		sscanf(argv[2], "%d", &k);
		sscanf(argv[3], "%d", &timesteps);
		sscanf(argv[4], "%d", &mod);
	}
	else if (argc == 6) // command line for starting with initial condition
	{
		sscanf(argv[1], "%s", &basename);
		sscanf(argv[2], "%d", &n);
		sscanf(argv[3], "%d", &timesteps);
		sscanf(argv[4], "%lg", &dt);
		sscanf(argv[5], "%d", &mod);
	}
	else // invalid command line, print usage
	{
		std::cout << "usage: " << std::endl;
		std::cout << "nbody_vanilla <basename> <load step> <final step> <every>" << std::endl;
		std::cout << "nbody_vanilla <basename> <nbodies> <timesteps> <timestep> <every>" << std::endl;
		return 1;
	}

	// set up computation from file
	if (argc == 5)
	{
		sprintf(name, "%s_%06d.vtk", basename, k);
		file = fopen(name, "r");
		if (file == NULL)
		{
			std::cout << "could not open file " << std::string(basename) << " aborting" << std::endl;
			return 1;
		}
		n = get_vtk_numbodies(file);
		rewind(file);
		x = new (std::align_val_t(64)) double3[n];
		v = new (std::align_val_t(64)) double3[n];
		m = new (std::align_val_t(64)) double[n];
		read_vtk_file_double(file, n, x, v, m, &t, &dt);
		fclose(file);
		k *= mod; // adjust step number
		std::cout << "loaded " << n << "bodies from file " << std::string(basename) << std::endl;
	}
	// set up computation from initial condition
	if (argc == 6)
	{
		x = new (std::align_val_t(64)) double3[n];
		v = new (std::align_val_t(64)) double3[n];
		m = new (std::align_val_t(64)) double[n];
		// plummer(n,17,x,v,m);
		two_plummer(n, 17, x, v, m);
		// cube(n,17,1.0,100.0,0.1,x,v,m);
		std::cout << "initialized " << n << " bodies" << std::endl;
		k = 0;
		t = 0.0;
		printf("writing %s_%06d.vtk \n", basename, k);
		sprintf(name, "%s_%06d.vtk", basename, k);
		file = fopen(name, "w");
		write_vtk_file_double(file, n, x, v, m, t, dt);
		fclose(file);
	}
	if (n % B != 0)
	{
		std::cout << n << " is not a multiple of the block size " << B << std::endl;
		exit(1);
	}
	if (B % 8 != 0)
	{
		std::cout << B << "=B is not a multiple of 4 " << std::endl;
		exit(1);
	}

	// switch to SoA data layout in 1d array
	double *X;
	X = new (std::align_val_t(64)) double[3 * n];
	double *V;
	V = new (std::align_val_t(64)) double[3 * n];
	double *A;
	A = new (std::align_val_t(64)) double[3 * n];

	// explicitly fill/clear padded values
	for (int i = 0; i < n; i++)
		for (int j = 3; j < M; j++)
			x[i][j] = v[i][j] = 0.0;

	// copy initial values
	copy(X, x, n);
	copy(V, v, n);

	// initialize timestep and write first file
	std::cout << "step=" << k << " finalstep=" << timesteps << " time=" << t << " dt=" << dt << std::endl;
	auto start = get_time_stamp();

	// do time steps
	k += 1;
	for (; k <= timesteps; k++)
	{
		leapfrog(n, dt, X, V, m, A);
		t += dt;
		if (k % mod == 0)
		{
			auto stop = get_time_stamp();
			double elapsed = get_duration_seconds(start, stop);
			double flop = mod * (13.0 * n * (n - 1.0) + 12.0 * n);
			printf("%g seconds for %g ops = %g GFLOPS \n", elapsed, flop, flop / elapsed / 1E9);

			printf("writing %s_%06d.vtk \n", basename, k / mod);
			sprintf(name, "%s_%06d.vtk", basename, k / mod);
			file = fopen(name, "w");
			copy(x, X, n);
			copy(v, V, n);
			write_vtk_file_double(file, n, x, v, m, t, dt);
			fclose(file);

			start = get_time_stamp();
		}
	}

	return 0;
}
