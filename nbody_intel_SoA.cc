#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <array>

#include "nbody_io.hh"
#include "nbody_generate.hh"
#include "time_experiment.hh"
#include "vcl/vectorclass.h"

#ifndef __GNUC__
#define __restrict__
#endif

// basic data type for position, velocity, acceleration
const int M = 3;
typedef double double3[M]; // pad up for later use with SIMD
const int B = 32;					 // block size for tiling

/*const double gamma = 6.674E-11;*/
const double G = 1.0;
const double epsilon2 = 1E-10;

/** \brief compute acceleration vector from position and masses
 *
 * Vanilla version
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

/** \brief compute acceleration vector from position and masses
 *
 * Vanilla version blocked
 * Executes \sum_{i=0}^{n-1} (n-i-1)*26 = n(n-1)*13
 * flops including 1 division and one square root
 */
void acceleration_blocked(int n, double *__restrict__ x, double *__restrict__ m, double *__restrict__ a)
{
	for (int I = 0; I < n; I += B)
	{
		// diagonal block
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
				a[i] += factorj * d0;
				a[n + i] += factorj * d1;
				a[2 * n + i] += factorj * d2;
				a[j] -= factori * d0;
				a[n + j] -= factori * d1;
				a[2 * n + j] -= factori * d2;
			}
		// upper diagonal full blocks
		for (int J = I + B; J < n; J += B)
			for (int j = J; j < J + B; j++)
				for (int i = I; i < I + B; i++)
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
}

/** \brief compute acceleration vector from position and masses
 *
 * Vanilla version blocked nonsymmetric
 * Executes n*n*19 flops
 * flops including 1 division and one square root
 */
void acceleration_blocked_full(int n, double *__restrict__ x, double *__restrict__ m, double *__restrict__ a)
{
	for (int I = 0; I < n; I += B)
		for (int J = 0; J < n; J += B)
			for (int j = J; j < J + B; j++)
				for (int i = I; i < I + B; i++)
				{
					double d0 = x[j] - x[i];
					double d1 = x[n + j] - x[n + i];
					double d2 = x[2 * n + j] - x[2 * n + i];
					double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
					double r = sqrt(r2);
					double invfact = G / (r * r2);
					double factorj = m[j] * invfact;
					a[i] += factorj * d0;
					a[n + i] += factorj * d1;
					a[2 * n + i] += factorj * d2;
				}
}

template <size_t simd_width>
struct SIMDSelector
{
	const size_t width = simd_width;
};
template <>
struct SIMDSelector<2>
{
	const std::string name = "SSE2";
	const size_t simd_width = 2;
	const size_t simd_registers = 16;
	typedef Vec2d SIMDType;
};
template <>
struct SIMDSelector<4>
{
	const std::string name = "AVX2";
	const size_t simd_width = 4;
	const size_t simd_registers = 16;
	typedef Vec4d SIMDType;
};
template <>
struct SIMDSelector<8>
{
	const std::string name = "AVX512";
	const size_t simd_width = 8;
	const size_t simd_registers = 32;
	typedef Vec8d SIMDType;
};

/** \brief compute acceleration vector from position and masses
 *
 * Vectorized version blocked: works on Wx1 masses; no transpose needed; but symmetry is not exploited
 * Executes n*n*19 flops
 * flops including 1 division and one square root
 */
template <size_t simd_width>
void acceleration_blocked_vectorized(int n, double *__restrict__ x, double *__restrict__ m, double *__restrict__ aglobal)
{
	const size_t W = simd_width;															 // SIMD width
	using VecWd = typename SIMDSelector<simd_width>::SIMDType; // SIMD type

#pragma omp parallel shared(aglobal), firstprivate(n, x, m)
	{
		// 16 registers!
		VecWd Xi0, Xi1, Xi2;
		VecWd Ai0, Ai1, Ai2;
		VecWd Di0, Di1, Di2;
		VecWd M, R3;

		double xj0;
		double xj1;
		double xj2;
		double mj;

		// make private acceleration vector to accumulate to
		std::vector<double> a(3 * n, 0.0);

#pragma omp for
		for (int I = 0; I < n; I += B)
			for (int J = 0; J < n; J += B)
				for (int i = I; i < I + B; i += W)
				{
					// load data of four masses from i
					Xi0.load(&x[i]);
					Xi1.load(&x[n + i]);
					Xi2.load(&x[2 * n + i]);
					Ai0.load(&a[i]);
					Ai1.load(&a[n + i]);
					Ai2.load(&a[2 * n + i]);

					// prefetching of scalar(!) quantities
					xj0 = x[J];
					xj1 = x[n + J];
					xj2 = x[2 * n + J];
					mj = G * m[J];

					// loop over masses j
					for (int j = J; j < J + B - 1; j += 1)
					{
						// now we compute the interaction of W masses with the mass j
						// distance vectors
						Di0 = VecWd(xj0); // hope that these are loaded now
						Di1 = VecWd(xj1);
						Di2 = VecWd(xj2);
						xj0 = x[j + 1]; // prefetch
						xj1 = x[n + j + 1];
						xj2 = x[2 * n + j + 1];

						Di0 -= Xi0;
						Di1 -= Xi1;
						Di2 -= Xi2;

						// compute W distances^3
						R3 = VecWd(epsilon2);
						R3 = mul_add(Di0, Di0, R3);
						R3 = mul_add(Di1, Di1, R3);
						R3 = mul_add(Di2, Di2, R3);
						M = sqrt(R3);
						R3 *= M;

						// update acceleration
						M = VecWd(mj);
						mj = G * m[j + 1]; // prefetch
						M /= R3;
						Ai0 = mul_add(Di0, M, Ai0);
						Ai1 = mul_add(Di1, M, Ai1);
						Ai2 = mul_add(Di2, M, Ai2);
					}

					// last interaction
					{
						// now we compute the interaction of W masses with the mass j
						// distance vectors
						Di0 = VecWd(xj0); // hope that these are loaded now
						Di1 = VecWd(xj1);
						Di2 = VecWd(xj2);

						Di0 -= Xi0;
						Di1 -= Xi1;
						Di2 -= Xi2;

						// compute W distances^3
						R3 = VecWd(epsilon2);
						R3 = mul_add(Di0, Di0, R3);
						R3 = mul_add(Di1, Di1, R3);
						R3 = mul_add(Di2, Di2, R3);
						M = sqrt(R3);
						R3 *= M;

						// update acceleration
						M = VecWd(mj);
						M /= R3;
						Ai0 = mul_add(Di0, M, Ai0);
						Ai1 = mul_add(Di1, M, Ai1);
						Ai2 = mul_add(Di2, M, Ai2);
					}

					// write back accelerations
					Ai0.store(&a[i]);
					Ai1.store(&a[n + i]);
					Ai2.store(&a[2 * n + i]);
				}

				// now we need to reduce the private accelerations
#pragma omp critical
		{
			for (int i = 0; i < 3 * n; i++)
				aglobal[i] += a[i];
		}
	} // end parallel regions
}

/** \brief compute acceleration vector from position and masses
 *
 * Vectorized version blocked: works on Wx2 masses; no transpose needed; but symmetry is not exploited
 * Executes n*n*19 flops
 * flops including 1 division and one square root
 */
template <size_t simd_width>
void acceleration_blocked_vectorized_interleaved(int n, double *__restrict__ x, double *__restrict__ m, double *__restrict__ aglobal)
{
	const size_t W = simd_width;															 // SIMD width
	using VecWd = typename SIMDSelector<simd_width>::SIMDType; // SIMD type

#pragma omp parallel shared(aglobal), firstprivate(n, x, m)
	{
		// 16 registers!
		VecWd Xi0, Xi1, Xi2;
		VecWd Ai0, Ai1, Ai2;
		VecWd Di0[2], Di1[2], Di2[2];
		VecWd M[2], R3[2];

		double xj0[2];
		double xj1[2];
		double xj2[2];
		double mj[2];

		// make private acceleration vector to accumulate to
		std::vector<double> a(3 * n, 0.0);

#pragma omp for
		for (int I = 0; I < n; I += B)
			for (int J = 0; J < n; J += B)
				for (int i = I; i < I + B; i += W)
				{
					// load data of mass i
					Xi0.load(&x[i]);
					Xi1.load(&x[n + i]);
					Xi2.load(&x[2 * n + i]);
					Ai0.load(&a[i]);
					Ai1.load(&a[n + i]);
					Ai2.load(&a[2 * n + i]);

					// prefetching of scalar(!) quantities
					xj0[0] = x[J];
					xj0[1] = x[J + 1];
					xj1[0] = x[n + J];
					xj1[1] = x[n + J + 1];
					xj2[0] = x[2 * n + J];
					xj2[1] = x[2 * n + J + 1];
					mj[0] = G * m[J];
					mj[1] = G * m[J + 1];

					// loop over masses j
					for (int j = J; j < J + B - 2; j += 2)
					{
						// now we compute the interaction of W masses with the mass j
						// distance vectors
						Di0[0] = VecWd(xj0[0]); // hope that these are loaded now
						Di0[1] = VecWd(xj0[1]); // hope that these are loaded now
						xj0[0] = x[j + 2];			// prefetch
						xj0[1] = x[j + 3];			// prefetch

						Di1[0] = VecWd(xj1[0]);
						Di1[1] = VecWd(xj1[1]);
						xj1[0] = x[n + j + 2];
						xj1[1] = x[n + j + 3];

						Di2[0] = VecWd(xj2[0]);
						Di2[1] = VecWd(xj2[1]);
						xj2[0] = x[2 * n + j + 2];
						xj2[1] = x[2 * n + j + 3];

						Di0[0] -= Xi0;
						Di0[1] -= Xi0;
						Di1[0] -= Xi1;
						Di1[1] -= Xi1;
						Di2[0] -= Xi2;
						Di2[1] -= Xi2;

						// compute W distances^3
						R3[0] = VecWd(epsilon2);
						R3[1] = VecWd(epsilon2);
						R3[0] = mul_add(Di0[0], Di0[0], R3[0]);
						R3[1] = mul_add(Di0[1], Di0[1], R3[1]);
						R3[0] = mul_add(Di1[0], Di1[0], R3[0]);
						R3[1] = mul_add(Di1[1], Di1[1], R3[1]);
						R3[0] = mul_add(Di2[0], Di2[0], R3[0]);
						R3[1] = mul_add(Di2[1], Di2[1], R3[1]);
						M[0] = sqrt(R3[0]);
						M[1] = sqrt(R3[1]);
						R3[0] *= M[0];
						R3[1] *= M[1];

						// update acceleration
						M[0] = VecWd(mj[0]);
						mj[0] = G * m[j + 2]; // prefetch
						M[1] = VecWd(mj[1]);
						mj[1] = G * m[j + 3]; // prefetch

						M[0] /= R3[0];
						M[1] /= R3[1];
						Ai0 = mul_add(Di0[0], M[0], Ai0);
						Ai1 = mul_add(Di1[0], M[0], Ai1);
						Ai2 = mul_add(Di2[0], M[0], Ai2);
						Ai0 = mul_add(Di0[1], M[1], Ai0);
						Ai1 = mul_add(Di1[1], M[1], Ai1);
						Ai2 = mul_add(Di2[1], M[1], Ai2);
					}

					// last interaction
					{
						// now we compute the interaction of W masses with the mass j
						// distance vectors
						Di0[0] = VecWd(xj0[0]); // hope that these are loaded now
						Di0[1] = VecWd(xj0[1]); // hope that these are loaded now
						Di1[0] = VecWd(xj1[0]);
						Di1[1] = VecWd(xj1[1]);
						Di2[0] = VecWd(xj2[0]);
						Di2[1] = VecWd(xj2[1]);

						Di0[0] -= Xi0;
						Di0[1] -= Xi0;
						Di1[0] -= Xi1;
						Di1[1] -= Xi1;
						Di2[0] -= Xi2;
						Di2[1] -= Xi2;

						// compute W distances^3
						R3[0] = VecWd(epsilon2);
						R3[1] = VecWd(epsilon2);
						R3[0] = mul_add(Di0[0], Di0[0], R3[0]);
						R3[1] = mul_add(Di0[1], Di0[1], R3[1]);
						R3[0] = mul_add(Di1[0], Di1[0], R3[0]);
						R3[1] = mul_add(Di1[1], Di1[1], R3[1]);
						R3[0] = mul_add(Di2[0], Di2[0], R3[0]);
						R3[1] = mul_add(Di2[1], Di2[1], R3[1]);
						M[0] = sqrt(R3[0]);
						M[1] = sqrt(R3[1]);
						R3[0] *= M[0];
						R3[1] *= M[1];

						// update acceleration
						M[0] = VecWd(mj[0]);
						M[1] = VecWd(mj[1]);
						M[0] /= R3[0];
						M[1] /= R3[1];
						Ai0 = mul_add(Di0[0], M[0], Ai0);
						Ai1 = mul_add(Di1[0], M[0], Ai1);
						Ai2 = mul_add(Di2[0], M[0], Ai2);
						Ai0 = mul_add(Di0[1], M[1], Ai0);
						Ai1 = mul_add(Di1[1], M[1], Ai1);
						Ai2 = mul_add(Di2[1], M[1], Ai2);
					}

					// write back accelerations
					Ai0.store(&a[i]);
					Ai1.store(&a[n + i]);
					Ai2.store(&a[2 * n + i]);
				}

				// now we need to reduce the private accelerations
#pragma omp critical
		{
			for (int i = 0; i < 3 * n; i++)
				aglobal[i] += a[i];
		}
	} // end parallel regions
}

/** \brief compute acceleration vector from position and masses
 *
 * Vectorized version blocked: works on 4x4 masses; symmetry is exploited; transpose for scalar factors needed
 * Executes \sum_{i=0}^{n-1} (n-i-1)*26 = n(n-1)*13
 * flops including 1 division and one square root
 */
void acceleration_4x4(int n, double *__restrict__ x, double *__restrict__ m, double *__restrict__ a)
{
	Vec4d X0, X1, X2;
	Vec4d D0, D1, D2;
	Vec4d R0, R1, R2, R3;
	Vec4d A0, A1, A2, A3;
	Vec4d S, Y;

	for (int I = 0; I < n; I += B)
	{
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
				a[i] += factorj * d0;
				a[n + i] += factorj * d1;
				a[2 * n + i] += factorj * d2;
				a[j] -= factori * d0;
				a[n + j] -= factori * d1;
				a[2 * n + j] -= factori * d2;
			}

		// blocks J>I can also exploit symmetry but are full now
		for (int J = I + B; J < n; J += B)
			for (int i = I; i < I + B; i += 4)
				for (int j = J; j < J + B; j += 4)
				{
					// compute interaction of 4x4 masses i,i+1,i+2,i+3 and j,j+1,j+1,j+3

					// load positions of four particles starting at j
					X0.load(&(x[j]));				 // x coordinates
					X1.load(&(x[j + n]));		 // y coordinates
					X2.load(&(x[j + 2 * n])); // z coordinates

					// difference to mass i
					D0 = Vec4d(x[i]);				 // load broadcast for x-coordinate
					D1 = Vec4d(x[i + n]);		 // load broadcast for y-coordinate
					D2 = Vec4d(x[i + 2 * n]); // load broadcast for z-coordinate
					D0 = X0-D0;				 // distance j,...-i, x-coordinate
					D1 = X1-D1;				 // distance j,...-i, y-coordinate
					D2 = X2-D2;				 // distance j,...-i, z-coordinate
					A0 = Vec4d(epsilon2);			 // now compute squared distances
					A0 = mul_add(D0,D0,A0);
					A0 = mul_add(D1,D1,A0);
					A0 = mul_add(D2,D2,A0); // now A0=(||xj-xi||^2, ||xj+1-xi||^2, ||xj+2-xi||^2, ||xj+3-xi||^2)
					R0 = sqrt(A0);          // now R0=(||xj-xi||,   ||xj+1-xi||,   ||xj+2-xi||,   ||xj+3-xi||  )

					// difference to mass i+1
					D0 = Vec4d(x[i+1]);				 // load broadcast for x-coordinate
					D1 = Vec4d(x[i+1 + n]);		 // load broadcast for y-coordinate
					D2 = Vec4d(x[i+1 + 2 * n]); // load broadcast for z-coordinate
					D0 = X0-D0;				 // distance j,...-i, x-coordinate
					D1 = X1-D1;				 // distance j,...-i, y-coordinate
					D2 = X2-D2;				 // distance j,...-i, z-coordinate
					A1 = Vec4d(epsilon2);			 // now compute squared distances
					A1 = mul_add(D0,D0,A1);
					A1 = mul_add(D1,D1,A1);
					A1 = mul_add(D2,D2,A1); // now A1=(||xj-xi+1||^2, ||xj+1-xi+1||^2, ||xj+2-xi+1||^2, ||xj+3-xi+1||^2)
					R1 = sqrt(A1);          // now R1=(||xj-xi+1||,   ||xj+1-xi+1||,   ||xj+2-xi+1||,   ||xj+3-xi+1||  )

					// difference to mass i+2
					D0 = Vec4d(x[i+2]);				 // load broadcast for x-coordinate
					D1 = Vec4d(x[i+2 + n]);		 // load broadcast for y-coordinate
					D2 = Vec4d(x[i+2 + 2 * n]); // load broadcast for z-coordinate
					D0 = X0-D0;				 // distance j,...-i, x-coordinate
					D1 = X1-D1;				 // distance j,...-i, y-coordinate
					D2 = X2-D2;				 // distance j,...-i, z-coordinate
					A2 = Vec4d(epsilon2);			 // now compute squared distances
					A2 = mul_add(D0,D0,A2);
					A2 = mul_add(D1,D1,A2);
					A2 = mul_add(D2,D2,A2); // now A2=(||xj-xi+2||^2, ||xj+1-xi+2||^2, ||xj+2-xi+2||^2, ||xj+3-xi+2||^2)
					R2 = sqrt(A2);          // now R2=(||xj-xi+2||,   ||xj+1-xi+2||,   ||xj+2-xi+2||,   ||xj+3-xi+2||  )

					// difference to mass i+3
					D0 = Vec4d(x[i+3]);				 // load broadcast for x-coordinate
					D1 = Vec4d(x[i+3 + n]);		 // load broadcast for y-coordinate
					D2 = Vec4d(x[i+3 + 2 * n]); // load broadcast for z-coordinate
					D0 = X0-D0;				 // distance j,...-i, x-coordinate
					D1 = X1-D1;				 // distance j,...-i, y-coordinate
					D2 = X2-D2;				 // distance j,...-i, z-coordinate
					A3 = Vec4d(epsilon2);			 // now compute squared distances
					A3 = mul_add(D0,D0,A3);
					A3 = mul_add(D1,D1,A3);
					A3 = mul_add(D2,D2,A3); // now A3=(||xj-xi+3||^2, ||xj+1-xi+3||^2, ||xj+2-xi+3||^2, ||xj+3-xi+3||^2)
					R3 = sqrt(A3);          // now R3=(||xj-xi+3||,   ||xj+1-xi+3||,   ||xj+2-xi+3||,   ||xj+3-xi+3||  )

					// now we proceed to compute the scalar factors s_{i,j} = G/r_{i,j}^3
					D0 = Vec4d(G);		// load broadcast for G, D0 is unused
					A0 = A0*R0; // Ai contains third powers
					A1 = A1*R1;
					A2 = A2*R2;
					A3 = A3*R3;
					R0 = D0/A0; // R0 = ( s_{j,i}   ; s_{j+1,i}  ; s_{j+2,i}  ; s_{j+3,i} )
					R1 = D0/A1; // R1 = ( s_{j,i+1} ; s_{j+1,i+1}; s_{j+2,i+1}; s_{j+3,i+1} )
					R2 = D0/A2; // R2 = ( s_{j,i+2} ; s_{j+1,i+2}; s_{j+2,i+2}; s_{j+3,i+2} )
					R3 = D0/A3; // R3 = ( s_{j,i+3} ; s_{j+1,i+3}; s_{j+2,i+3}; s_{j+3,i+3} )


					// now we update the accelerations of the 2*4 masses

					// FIRST contribution to acceleration of masses j,j+1,j+2,j+3 *from* masses i,i+1,i+2,i+3
					// load acceleration of masses j,j+1
					A0.load(&(a[j]));				 // x coordinates
					A1.load(&(a[j + n]));		 // y coordinates
					A2.load(&(a[j + 2 * n])); // z coordinates

					// contribution from mass i
					// X* still contains the positions of j,j+1,j+2,j+3. But we need to recompute the differences
					D0 = Vec4d(x[i]);				 // load broadcast for x-coordinate
					D1 = Vec4d(x[i + n]);		 // load broadcast for y-coordinate
					D2 = Vec4d(x[i + 2 * n]); // load broadcast for z-coordinate
					D0 -= X0;	// distance i-j,... x-coordinate
					D1 -= X1;	// distance i-j,... y-coordinate
					D2 -= X2;	// distance i-j,... z-coordinate
					// build scalar factors
					S = Vec4d(m[i]);
					S *= R0;
					// update
					A0 = mul_add(S,D0,A0);
					A1 = mul_add(S,D1,A1);
					A2 = mul_add(S,D2,A2);

					// contribution from mass i+1
					// X* still contains the positions of j,j+1,j+2,j+3. But we need to recompute the differences
					D0 = Vec4d(x[i+1]);				 // load broadcast for x-coordinate
					D1 = Vec4d(x[i+1 + n]);		 // load broadcast for y-coordinate
					D2 = Vec4d(x[i+1 + 2 * n]); // load broadcast for z-coordinate
					D0 -= X0;	// distance i-j,... x-coordinate
					D1 -= X1;	// distance i-j,... y-coordinate
					D2 -= X2;	// distance i-j,... z-coordinate
					// build scalar factors
					S = Vec4d(m[i+1]);
					S *= R1;
					// update
					A0 = mul_add(S,D0,A0);
					A1 = mul_add(S,D1,A1);
					A2 = mul_add(S,D2,A2);

					// contribution from mass i+2
					// X* still contains the positions of j,j+1,j+2,j+3. But we need to recompute the differences
					D0 = Vec4d(x[i+2]);				 // load broadcast for x-coordinate
					D1 = Vec4d(x[i+2 + n]);		 // load broadcast for y-coordinate
					D2 = Vec4d(x[i+2 + 2 * n]); // load broadcast for z-coordinate
					D0 -= X0;	// distance i-j,... x-coordinate
					D1 -= X1;	// distance i-j,... y-coordinate
					D2 -= X2;	// distance i-j,... z-coordinate
					// build scalar factors
					S = Vec4d(m[i+2]);
					S *= R2;
					// update
					A0 = mul_add(S,D0,A0);
					A1 = mul_add(S,D1,A1);
					A2 = mul_add(S,D2,A2);

					// contribution from mass i+3
					// X* still contains the positions of j,j+1,j+2,j+3. But we need to recompute the differences
					D0 = Vec4d(x[i+3]);				 // load broadcast for x-coordinate
					D1 = Vec4d(x[i+3 + n]);		 // load broadcast for y-coordinate
					D2 = Vec4d(x[i+3 + 2 * n]); // load broadcast for z-coordinate
					D0 -= X0;	// distance i-j,... x-coordinate
					D1 -= X1;	// distance i-j,... y-coordinate
					D2 -= X2;	// distance i-j,... z-coordinate
					// build scalar factors
					S = Vec4d(m[i+3]);
					S *= R3;
					// update
					A0 = mul_add(S,D0,A0);
					A1 = mul_add(S,D1,A1);
					A2 = mul_add(S,D2,A2);

					// store acceleration of masses j,j+1,j+2,j+3
					A0.store(&(a[j]));				 // x coordinates
					A1.store(&(a[j + n]));		 // y coordinates
					A2.store(&(a[j + 2 * n])); // z coordinates


					// SECOND contribution to acceleration of masses i,i+1,i+2,i+3 *from* masses j, j+1, j+2, j+3
					// load coordinates of i,i+1,i+2,i+3 in SIMD register; we did not have this before
					X0.load(&(x[i]));					// x coordinates
					X1.load(&(x[i + n]));			// y coordinates
					X2.load(&(x[i + 2 * n])); // z coordinates
					// load acceleration
					A0.load(&(a[i]));				 // x coordinates
					A1.load(&(a[i + n]));		 // y coordinates
					A2.load(&(a[i + 2 * n])); // z coordinates

					// contribution from mass j
					D0 = Vec4d(x[j]);				  // load broadcast for x-coordinate
					D1 = Vec4d(x[j + n]);		  // load broadcast for y-coordinate
					D2 = Vec4d(x[j + 2 * n]); // load broadcast for z-coordinate
					// build difference vectors
					D0 -= X0; // distance j-i,i+1,i+2,i+3 x-coordinate
					D1 -= X1; // distance j-i,i+1,i+2,i+3 y-coordinate
					D2 -= X2; // distance j-i,i+1,i+2,i+3 z-coordinate
					// build scalar factors: we need a transpose here
					// now the transpose; unused registers: A3,Y
		  		A3 = blend4<0,4,V_DC,V_DC>(R0,R1); // scalar factor column j
		  		Y = blend4<V_DC,V_DC,0,4>(R2,R3); // scalar factor column j
					S = blend4<0,1,6,7>(A3,Y);
					Y = Vec4d(m[j]);
					S *= Y;
					// update
					A0 = mul_add(S,D0,A0);
					A1 = mul_add(S,D1,A1);
					A2 = mul_add(S,D2,A2);

					// contribution from mass j+1
					D0 = Vec4d(x[j+1]);				  // load broadcast for x-coordinate
					D1 = Vec4d(x[j+1 + n]);		  // load broadcast for y-coordinate
					D2 = Vec4d(x[j+1 + 2 * n]); // load broadcast for z-coordinate
					// build difference vectors
					D0 -= X0; // distance j+1-i,i+1,i+2,i+3 x-coordinate
					D1 -= X1; // distance j+1-i,i+1,i+2,i+3 y-coordinate
					D2 -= X2; // distance j+1-i,i+1,i+2,i+3 z-coordinate
					// build scalar factors: we need a transpose here
					// now the transpose; unused registers: A3,Y
		  		A3 = blend4<1,5,V_DC,V_DC>(R0,R1); // scalar factor column j
		  		Y = blend4<V_DC,V_DC,1,5>(R2,R3); // scalar factor column j
					S = blend4<0,1,6,7>(A3,Y);
					Y = Vec4d(m[j+1]);
					S *= Y;
					// update
					A0 = mul_add(S,D0,A0);
					A1 = mul_add(S,D1,A1);
					A2 = mul_add(S,D2,A2);

					// contribution from mass j+2
					D0 = Vec4d(x[j+2]);				  // load broadcast for x-coordinate
					D1 = Vec4d(x[j+2 + n]);		  // load broadcast for y-coordinate
					D2 = Vec4d(x[j+2 + 2 * n]); // load broadcast for z-coordinate
					// build difference vectors
					D0 -= X0; // distance j+2-i,i+1,i+2,i+3 x-coordinate
					D1 -= X1; // distance j+2-i,i+1,i+2,i+3 y-coordinate
					D2 -= X2; // distance j+2-i,i+1,i+2,i+3 z-coordinate
					// build scalar factors: we need a transpose here
					// now the transpose; unused registers: A3,Y
		  		A3 = blend4<2,6,V_DC,V_DC>(R0,R1); // scalar factor column j
		  		Y = blend4<V_DC,V_DC,2,6>(R2,R3); // scalar factor column j
					S = blend4<0,1,6,7>(A3,Y);
					Y = Vec4d(m[j+2]);
					S *= Y;
					// update
					A0 = mul_add(S,D0,A0);
					A1 = mul_add(S,D1,A1);
					A2 = mul_add(S,D2,A2);

					// contribution from mass j+3
					D0 = Vec4d(x[j+3]);				  // load broadcast for x-coordinate
					D1 = Vec4d(x[j+3 + n]);		  // load broadcast for y-coordinate
					D2 = Vec4d(x[j+3 + 2 * n]); // load broadcast for z-coordinate
					// build difference vectors
					D0 -= X0; // distance j+3-i,i+1,i+2,i+3 x-coordinate
					D1 -= X1; // distance j+3-i,i+1,i+2,i+3 y-coordinate
					D2 -= X2; // distance j+3-i,i+1,i+2,i+3 z-coordinate
					// build scalar factors: we need a transpose here
					// now the transpose; unused registers: A3,Y
		  		A3 = blend4<3,7,V_DC,V_DC>(R0,R1); // scalar factor column j
		  		Y = blend4<V_DC,V_DC,3,7>(R2,R3); // scalar factor column j
					S = blend4<0,1,6,7>(A3,Y);
					Y = Vec4d(m[j+3]);
					S *= Y;
					// update
					A0 = mul_add(S,D0,A0);
					A1 = mul_add(S,D1,A1);
					A2 = mul_add(S,D2,A2);

					// store acceleration of masses i,i+1,i+2,i+3
					A0.store(&(a[i]));
					A1.store(&(a[i + n]));
					A2.store(&(a[i + 2 * n]));
				}
	}
}

void
transpose(std::array<Vec2d, 2>& A)
{
  Vec2d T = blend2<1, 2>(A[0], A[1]); // A01, A10

  A[0] = blend2<0, 3>(A[0], T);       // A00, A10
  A[1] = blend2<2, 1>(A[1], T);       // A01, A11
}

void
transpose(std::array<Vec4d, 4>& A)
{
  std::array<Vec4d, 4> T;
  T[0] = blend4<0, 4, 2, 6>(A[0], A[1]); // A00, A10, A02, A12
  T[1] = blend4<1, 5, 3, 7>(A[0], A[1]); // A01, A11, A03, A13
  T[2] = blend4<0, 4, 2, 6>(A[2], A[3]); // A20, A30, A22, A32
  T[3] = blend4<1, 5, 3, 7>(A[2], A[3]); // A21, A31, A23, A33

  A[0] = blend4<0, 1, 4, 5>(T[0], T[2]); // A00, A10, A20, A30
  A[1] = blend4<0, 1, 4, 5>(T[1], T[3]); // A01, A11, A21, A31
  A[2] = blend4<2, 3, 6, 7>(T[0], T[2]); // A02, A12, A22, A32
  A[3] = blend4<2, 3, 6, 7>(T[1], T[3]); // A03, A13, A23, A33
}

void
transpose(std::array<Vec8d, 8>& A)
{
  std::array<Vec8d, 8> T;
  T[0] = blend8<0, 8, 2, 10, 4, 12, 6, 14>(A[0], A[1]); // 00, 10, 02, 12, 04, 14, 06, 16
  T[1] = blend8<1, 9, 3, 11, 5, 13, 7, 15>(A[0], A[1]); // 01, 11, 03, 13, 05, 15, 07, 17
  T[2] = blend8<0, 8, 2, 10, 4, 12, 6, 14>(A[2], A[3]); // 20, 30, 22, 32, 24, 34, 26, 36
  T[3] = blend8<1, 9, 3, 11, 5, 13, 7, 15>(A[2], A[3]); // 21, 31, 23, 33, 25, 35, 27, 37
  T[4] = blend8<0, 8, 2, 10, 4, 12, 6, 14>(A[4], A[5]); // 40, 50, 42, 52, 44, 54, 46, 56
  T[5] = blend8<1, 9, 3, 11, 5, 13, 7, 15>(A[4], A[5]); // 41, 51, 43, 53, 45, 55, 47, 57
  T[6] = blend8<0, 8, 2, 10, 4, 12, 6, 14>(A[6], A[7]); // 60, 70, 62, 72, 64, 74, 66, 76
  T[7] = blend8<1, 9, 3, 11, 5, 13, 7, 15>(A[6], A[7]); // 61, 71, 63, 73, 65, 75, 67, 77

  A[0] = blend8<0,  1,  8,  9,  4,  5, 12, 13>(T[0], T[2]); // 00, 10, 20, 30, 04, 14, 24, 34
  A[1] = blend8<0,  1,  8,  9,  4,  5, 12, 13>(T[1], T[3]); // 01, 11, 21, 31, 05, 15, 25, 35
  A[2] = blend8<2,  3, 10, 11,  6,  7, 14, 15>(T[0], T[2]); // 02, 12, 22, 32, 06, 16, 26, 36
  A[3] = blend8<2,  3, 10, 11,  6,  7, 14, 15>(T[1], T[3]); // 03, 13, 23, 33, 07, 17, 27, 37
  A[4] = blend8<0,  1,  8,  9,  4,  5, 12, 13>(T[4], T[6]); // 40, 50, 60, 70, 44, 54, 64, 74
  A[5] = blend8<0,  1,  8,  9,  4,  5, 12, 13>(T[5], T[7]); // 41, 51, 61, 71, 45, 55, 65, 75
  A[6] = blend8<2,  3, 10, 11,  6,  7, 14, 15>(T[4], T[6]); // 42, 52, 62, 72, 46, 56, 66, 76
  A[7] = blend8<2,  3, 10, 11,  6,  7, 14, 15>(T[5], T[7]); // 43, 53, 63, 73, 47, 57, 67, 77

  T[0] = blend8< 0,  1,  2,  3,  8,  9, 10, 11>(A[0], A[4]); // 00, 10, 20, 30, 40, 50, 60, 70
  T[1] = blend8< 0,  1,  2,  3,  8,  9, 10, 11>(A[1], A[5]); // 01, 11, 21, 31, 41, 51, 61, 71
  T[2] = blend8< 0,  1,  2,  3,  8,  9, 10, 11>(A[2], A[6]); // 02, 12, 22, 32, 42, 52, 62, 72
  T[3] = blend8< 0,  1,  2,  3,  8,  9, 10, 11>(A[3], A[7]); // 03, 13, 23, 33, 43, 53, 63, 73
  T[4] = blend8<12, 13, 14, 15,  4,  5,  6,  7>(A[4], A[0]); // 04, 14, 24, 34, 44, 54, 64, 74
  T[5] = blend8<12, 13, 14, 15,  4,  5,  6,  7>(A[5], A[1]); // 05, 15, 25, 35, 45, 55, 65, 75
  T[6] = blend8<12, 13, 14, 15,  4,  5,  6,  7>(A[6], A[2]); // 06, 16, 26, 36, 46, 56, 66, 76
  T[7] = blend8<12, 13, 14, 15,  4,  5,  6,  7>(A[7], A[3]); // 07, 17, 27, 37, 47, 57, 67, 77

  A = T;
}



/** \brief compute acceleration vector from position and masses
 *
 * Vectorized version blocked: works on WxW masses; symmetry is exploited; transpose for scalar factors needed
 * Executes \sum_{i=0}^{n-1} (n-i-1)*26 = n(n-1)*13
 * flops including 1 division and one square root
 */
template<size_t simd_width>
void
acceleration_WxW(int n,
                 double* __restrict__ x,
                 double* __restrict__ m,
                 double* __restrict__ a)
{
  const size_t W = simd_width;                               // SIMD width
  using VecWd = typename SIMDSelector<simd_width>::SIMDType; // SIMD type

  {
    for (int I = 0; I < n; I += B) {
      // diagonal block
      for (int i = I; i < I + B; i++) {
        for (int j = i + 1; j < I + B; j++) {
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

      for (int J = I; J < n; J += B) {
        for (int i = I; i < I + B; i += W) {
          std::array<VecWd, 3> Xi;
          Xi[0].load(&x[i]);
          Xi[1].load(&x[n + i]);
          Xi[2].load(&x[2 * n + i]);
          for (int j = J; j < J + B; j += W) {
            // load acceleration for i-index
            std::array<VecWd, 3> A;
            A[0].load(&a[i]);
            A[1].load(&a[n + i]);
            A[2].load(&a[2 * n + i]);
            std::array<VecWd, W> N;
            for (int w = 0; w != W; w += 1) {
              std::array<VecWd, 3> Dj{ VecWd(x[j + w]),
                                       VecWd(x[n + j + w]),
                                       VecWd(x[2 * n + j + w]) };
              Dj[0] -= Xi[0];
              Dj[1] -= Xi[1];
              Dj[2] -= Xi[2];
              N[w] = VecWd(epsilon2);
              N[w] = mul_add(Dj[0], Dj[0], N[w]);
              N[w] = mul_add(Dj[1], Dj[1], N[w]);
              N[w] = mul_add(Dj[2], Dj[2], N[w]);
              // calculate row of N := G/d_ji^3 (note that this is transposed -> ji vs ij!)
              N[w] = VecWd(G) / (N[w] * sqrt(N[w]));
              // update acceleration for mass i (note that we take adventage of the transposed N)
              VecWd M(m[i + w]);
              A[0] = mul_add(M * Dj[0], N[w], A[0]);
              A[1] = mul_add(M * Dj[1], N[w], A[1]);
              A[2] = mul_add(M * Dj[2], N[w], A[2]);
            }
            // now we need to transpose N
            transpose(N); // N_ji -> N_ij
            // store acceleration for i-index
            A[0].store(&a[i]);
            A[1].store(&a[n + i]);
            A[2].store(&a[2 * n + i]);
            // load acceleration for j-index
            A[0].load(&a[j]);
            A[1].load(&a[n + j]);
            A[2].load(&a[2 * n + j]);

            std::array<VecWd, 3> Xj;
            Xj[0].load(&x[j]);
            Xj[1].load(&x[n + j]);
            Xj[2].load(&x[2 * n + j]);
            // update acceleration for mass j (note that we need the non-transposed version of N)
            for (int w = 0; w != W; w += 1) {
              std::array<VecWd, 3> Di{ VecWd(x[i + w]),
                                       VecWd(x[n + i + w]),
                                       VecWd(x[2 * n + i + w]) };
              Di[0] -= Xj[0];
              Di[1] -= Xj[1];
              Di[2] -= Xj[2];
              VecWd M(m[j + w]);
              A[0] = mul_add(M * Di[0], N[w], A[0]);
              A[1] = mul_add(M * Di[1], N[w], A[1]);
              A[2] = mul_add(M * Di[2], N[w], A[2]);
            }
            // store acceleration for j-index
            A[0].store(&a[j]);
            A[1].store(&a[n + j]);
            A[2].store(&a[2 * n + j]);
          }
        }
      }
    }
  }
}

#ifdef __AVX512F__
void acceleration_blocked_vectorized_512(int n, double *__restrict__ x, double *__restrict__ m, double *__restrict__ aglobal)
{
	const size_t W = 8;																// SIMD width
	using VecWd = typename SIMDSelector<W>::SIMDType; // SIMD type

#pragma omp parallel shared(aglobal), firstprivate(n, x, m)
	{

		// 26 registers!
		VecWd Xi0, Xi1, Xi2;
		VecWd Ai0, Ai1, Ai2;
		VecWd Di0[4], Di1[4], Di2[4];
		VecWd M[4], R3[4];

		Vec4d xj0;
		Vec4d xj1;
		Vec4d xj2;
		Vec4d mj;
		Vec4d g = Vec4d(G);

		// make private acceleration vector to accumulate to
		std::vector<double> a(3 * n, 0.0);

#pragma omp for
		for (int I = 0; I < n; I += B)
			for (int J = 0; J < n; J += B)
				for (int i = I; i < I + B; i += W)
				{
					// load data of mass i
					Xi0.load(&x[i]);
					Xi1.load(&x[n + i]);
					Xi2.load(&x[2 * n + i]);
					Ai0.load(&a[i]);
					Ai1.load(&a[n + i]);
					Ai2.load(&a[2 * n + i]);

					// prefetching
					xj0.load(&x[J]);
					xj1.load(&x[n + J]);
					xj2.load(&x[2 * n + J]);
					mj.load(&m[J]);

					// loop over masses j
					for (int j = J; j < J + B - 4; j += 4)
					{
						// now we compute the interaction of W masses with the mass j
						// distance vectors
						Di0[0] = VecWd(xj0[0]); // hope that these are loaded now
						Di0[1] = VecWd(xj0[1]); // hope that these are loaded now
						Di0[2] = VecWd(xj0[2]); // hope that these are loaded now
						Di0[3] = VecWd(xj0[3]); // hope that these are loaded now
						xj0.load(&x[j + 4]);
						Di1[0] = VecWd(xj1[0]);
						Di1[1] = VecWd(xj1[1]);
						Di1[2] = VecWd(xj1[2]);
						Di1[3] = VecWd(xj1[3]);
						xj1.load(&x[n + j + 4]);
						Di2[0] = VecWd(xj2[0]);
						Di2[1] = VecWd(xj2[1]);
						Di2[2] = VecWd(xj2[2]);
						Di2[3] = VecWd(xj2[3]);
						xj2.load(&x[2 * n + j + 4]);

						Di0[0] -= Xi0;
						Di0[1] -= Xi0;
						Di0[2] -= Xi0;
						Di0[3] -= Xi0;
						Di1[0] -= Xi1;
						Di1[1] -= Xi1;
						Di1[2] -= Xi1;
						Di1[3] -= Xi1;
						Di2[0] -= Xi2;
						Di2[1] -= Xi2;
						Di2[2] -= Xi2;
						Di2[3] -= Xi2;

						// compute W distances^3
						R3[0] = VecWd(epsilon2);
						R3[1] = VecWd(epsilon2);
						R3[2] = VecWd(epsilon2);
						R3[3] = VecWd(epsilon2);
						R3[0] = mul_add(Di0[0], Di0[0], R3[0]);
						R3[1] = mul_add(Di0[1], Di0[1], R3[1]);
						R3[2] = mul_add(Di0[2], Di0[2], R3[2]);
						R3[3] = mul_add(Di0[3], Di0[3], R3[3]);
						R3[0] = mul_add(Di1[0], Di1[0], R3[0]);
						R3[1] = mul_add(Di1[1], Di1[1], R3[1]);
						R3[2] = mul_add(Di1[2], Di1[2], R3[2]);
						R3[3] = mul_add(Di1[3], Di1[3], R3[3]);
						R3[0] = mul_add(Di2[0], Di2[0], R3[0]);
						R3[1] = mul_add(Di2[1], Di2[1], R3[1]);
						R3[2] = mul_add(Di2[2], Di2[2], R3[2]);
						R3[3] = mul_add(Di2[3], Di2[3], R3[3]);
						M[0] = sqrt(R3[0]);
						M[1] = sqrt(R3[1]);
						M[2] = sqrt(R3[2]);
						M[3] = sqrt(R3[3]);
						R3[0] *= M[0];
						R3[1] *= M[1];
						R3[2] *= M[2];
						R3[3] *= M[3];

						// update acceleration
						mj *= g;
						M[0] = VecWd(mj[0]);
						M[1] = VecWd(mj[1]);
						M[2] = VecWd(mj[2]);
						M[3] = VecWd(mj[3]);
						mj.load(&m[j + 4]);
						M[0] /= R3[0];
						M[1] /= R3[1];
						M[2] /= R3[2];
						M[3] /= R3[3];
						Ai0 = mul_add(Di0[0], M[0], Ai0);
						Ai1 = mul_add(Di1[0], M[0], Ai1);
						Ai2 = mul_add(Di2[0], M[0], Ai2);
						Ai0 = mul_add(Di0[1], M[1], Ai0);
						Ai1 = mul_add(Di1[1], M[1], Ai1);
						Ai2 = mul_add(Di2[1], M[1], Ai2);
						Ai0 = mul_add(Di0[2], M[2], Ai0);
						Ai1 = mul_add(Di1[2], M[2], Ai1);
						Ai2 = mul_add(Di2[2], M[2], Ai2);
						Ai0 = mul_add(Di0[3], M[3], Ai0);
						Ai1 = mul_add(Di1[3], M[3], Ai1);
						Ai2 = mul_add(Di2[3], M[3], Ai2);
					}

					// last interaction
					{
						// now we compute the interaction of W masses with the mass j
						// distance vectors
						Di0[0] = VecWd(xj0[0]); // hope that these are loaded now
						Di0[1] = VecWd(xj0[1]); // hope that these are loaded now
						Di0[2] = VecWd(xj0[2]); // hope that these are loaded now
						Di0[3] = VecWd(xj0[3]); // hope that these are loaded now
						Di1[0] = VecWd(xj1[0]);
						Di1[1] = VecWd(xj1[1]);
						Di1[2] = VecWd(xj1[2]);
						Di1[3] = VecWd(xj1[3]);
						Di2[0] = VecWd(xj2[0]);
						Di2[1] = VecWd(xj2[1]);
						Di2[2] = VecWd(xj2[2]);
						Di2[3] = VecWd(xj2[3]);

						Di0[0] -= Xi0;
						Di0[1] -= Xi0;
						Di0[2] -= Xi0;
						Di0[3] -= Xi0;
						Di1[0] -= Xi1;
						Di1[1] -= Xi1;
						Di1[2] -= Xi1;
						Di1[3] -= Xi1;
						Di2[0] -= Xi2;
						Di2[1] -= Xi2;
						Di2[2] -= Xi2;
						Di2[3] -= Xi2;

						// compute W distances^3
						R3[0] = VecWd(epsilon2);
						R3[1] = VecWd(epsilon2);
						R3[2] = VecWd(epsilon2);
						R3[3] = VecWd(epsilon2);
						R3[0] = mul_add(Di0[0], Di0[0], R3[0]);
						R3[1] = mul_add(Di0[1], Di0[1], R3[1]);
						R3[2] = mul_add(Di0[2], Di0[2], R3[2]);
						R3[3] = mul_add(Di0[3], Di0[3], R3[3]);
						R3[0] = mul_add(Di1[0], Di1[0], R3[0]);
						R3[1] = mul_add(Di1[1], Di1[1], R3[1]);
						R3[2] = mul_add(Di1[2], Di1[2], R3[2]);
						R3[3] = mul_add(Di1[3], Di1[3], R3[3]);
						R3[0] = mul_add(Di2[0], Di2[0], R3[0]);
						R3[1] = mul_add(Di2[1], Di2[1], R3[1]);
						R3[2] = mul_add(Di2[2], Di2[2], R3[2]);
						R3[3] = mul_add(Di2[3], Di2[3], R3[3]);
						M[0] = sqrt(R3[0]);
						M[1] = sqrt(R3[1]);
						M[2] = sqrt(R3[2]);
						M[3] = sqrt(R3[3]);
						R3[0] *= M[0];
						R3[1] *= M[1];
						R3[2] *= M[2];
						R3[3] *= M[3];

						// update acceleration
						M[0] = VecWd(mj[0]);
						M[1] = VecWd(mj[1]);
						M[2] = VecWd(mj[2]);
						M[3] = VecWd(mj[3]);
						M[0] /= R3[0];
						M[1] /= R3[1];
						M[2] /= R3[2];
						M[3] /= R3[3];
						Ai0 = mul_add(Di0[0], M[0], Ai0);
						Ai1 = mul_add(Di1[0], M[0], Ai1);
						Ai2 = mul_add(Di2[0], M[0], Ai2);
						Ai0 = mul_add(Di0[1], M[1], Ai0);
						Ai1 = mul_add(Di1[1], M[1], Ai1);
						Ai2 = mul_add(Di2[1], M[1], Ai2);
						Ai0 = mul_add(Di0[2], M[2], Ai0);
						Ai1 = mul_add(Di1[2], M[2], Ai1);
						Ai2 = mul_add(Di2[2], M[2], Ai2);
						Ai0 = mul_add(Di0[3], M[3], Ai0);
						Ai1 = mul_add(Di1[3], M[3], Ai1);
						Ai2 = mul_add(Di2[3], M[3], Ai2);
					}

					// write back accelerations
					Ai0.store(&a[i]);
					Ai1.store(&a[n + i]);
					Ai2.store(&a[2 * n + i]);
				}

				// now we need to reduce the private accelerations
#pragma omp critical
		{
			for (int i = 0; i < 3 * n; i++)
				aglobal[i] += a[i];
		}
	} // end parallel regions
}
#endif

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
	// acceleration_4x4(n,x,m,a);
	acceleration_WxW<4>(n,x,m,a);
	// acceleration_blocked(n,x,m,a);
	// acceleration_blocked_full(n,x,m,a);
	// acceleration_blocked_vectorized<2>(n,x,m,a);
	// acceleration_blocked_vectorized_interleaved<4>(n, x, m, a);
	// acceleration_blocked_vectorized_512(n,x,m,a);

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

	std::cout << "memory in one tile acceleration kernel: " << (B * 10) * 8 << " bytes" << std::endl;
	std::cout << "memory in acceleration kernel: " << n * 7 * 8 << " bytes" << std::endl;
	std::cout << "memory in leapfrog kernel: " << n * 10 * 8 << " bytes" << std::endl;
	std::cout << "memory total: " << n * 16 * 8 << " bytes" << std::endl;
	std::cout << "flops in one iteration: " << 19.0 * n * n + 12.0 * n << std::endl;
	std::cout << "total intensity: " << (13.0 * n * (n - 1.0) + 12.0 * n) / (n * 10 * 8) << " flops/byte" << std::endl;
	std::cout << "tile intensity: " << 19.0 * B * B / (B * 10 * 8) << " flops/byte" << std::endl;

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
			double flop2 = mod * (19.0 * n * n + 12.0 * n);
			printf("%g seconds for %g ops = %g (%g) GFLOPS \n", elapsed, flop, flop / elapsed / 1E9, flop2 / elapsed / 1E9);

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
