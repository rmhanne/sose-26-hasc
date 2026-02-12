#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <mutex>

#include "nbody_io.hh"
#include "nbody_generate.hh"
#include "time_experiment.hh"

// This assumes that vector class library is available in the directory vcl
#include "vcl/vectorclass.h"

#ifdef _OPENMP
#include <omp.h> // headers for runtime if available
#endif

#ifndef __GNUC__
#define __restrict__
#endif

/*
 * This version uses the "structure of arrays" (SoA) data layout.
 * Since the generate and file I/O functions use the "array of structures" (AoS) layout,
 * the data needs to be converted after (reading) or before (writing) using these functions.
 */

// basic data type for position, velocity, acceleration
const int M = 3;
typedef double double3[M]; // pad up for later use with SIMD
const int B = 128;				 // block size for tiling

const double G = 1.0;
const double epsilon2 = 1E-10;

/** \brief compute acceleration vector from position and masses
 *
 * Sequential blocked version
 * Executes \sum_{i=0}^{n-1} (n-i-1)*26 = n(n-1)*13
 * flops including 1 division and one square root
 */
void acceleration_blocked_SoA(int n, double *__restrict__ x, double *__restrict__ m, double *__restrict__ a)
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

/** \brief compute acceleration vector from position and masses (blocked sequential version)
 *
 * This version works on structure of arrays data layout
 * Executes \sum_{i=0}^{n-1} (n-i-1)*26 = n(n-1)*13
 * flops including 1 division and one square root
 */
void acceleration_blocked_buffered_SoA(int n, double *__restrict__ x, double *__restrict__ m, double *__restrict__ aglobal)
{
	// make private acceleration vectors to accumulate to
	double *aI;
	aI = new (std::align_val_t(64)) double[3 * B];
	double *aJ;
	aJ = new (std::align_val_t(64)) double[3 * B];

	// parallel loop over block rows with *cyclic* partitioning and dynamic scheduling
	for (int I = 0; I < n; I += B)
	{
		// clear accelerations for whole block row
		for (int i = 0; i < 3 * B; ++i)
			aI[i] = 0.0;

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
				aI[i - I] += factorj * d0; // updates private vector
				aI[i - I + B] += factorj * d1;
				aI[i - I + 2 * B] += factorj * d2;
				aI[j - I] -= factori * d0;
				aI[j - I + B] -= factori * d1;
				aI[j - I + 2 * B] -= factori * d2;
			}

		// blocks in upper triangle
		for (int J = I + B; J < n; J += B)
		{
			// clear accelerations for whole block column
			for (int j = 0; j < 3 * B; ++j)
				aJ[j] = 0.0;

			for (int i = I; i < I + B; i++)
				for (int j = J; j < J + B; j++)
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
					aJ[j - J] -= factori * d0;
					aJ[j - J + B] -= factori * d1;
					aJ[j - J + 2 * B] -= factori * d2;
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

/** \brief compute acceleration vector from position and masses (blocked OMP parallel version)
 *
 * This version works on structure of arrays data layout
 * Executes \sum_{i=0}^{n-1} (n-i-1)*26 = n(n-1)*13
 * flops including 1 division and one square root
 */
void acceleration_blocked_omp_SoA(int n, double *__restrict__ x, double *__restrict__ m, double *__restrict__ aglobal)
{
	std::vector<std::mutex> mutexes(n / B); // one mutex per block

#pragma omp parallel firstprivate(n, x, m, aglobal)
	{
		// make private acceleration vectors to accumulate to
		// these are then added to aglobal in critical sections
		double *aI;
		aI = new (std::align_val_t(64)) double[3 * B];
		double *aJ;
		aJ = new (std::align_val_t(64)) double[3 * B];

		// parallel loop over block rows with *cyclic* partitioning and dynamic scheduling
#pragma omp for schedule(dynamic, 1)
		for (int I = 0; I < n; I += B)
		{
			// clear accelerations for whole block row
			for (int i = 0; i < 3 * B; ++i)
				aI[i] = 0.0;

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
					aI[i - I] += factorj * d0; // updates private vector
					aI[i - I + B] += factorj * d1;
					aI[i - I + 2 * B] += factorj * d2;
					aI[j - I] -= factori * d0;
					aI[j - I + B] -= factori * d1;
					aI[j - I + 2 * B] -= factori * d2;
				}

			// blocks in upper triangle
			for (int J = I + B; J < n; J += B)
			{
				// clear accelerations for whole block column
				for (int j = 0; j < 3 * B; ++j)
					aJ[j] = 0.0;

				for (int i = I; i < I + B; i++)
					for (int j = J; j < J + B; j++)
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
						aJ[j - J] -= factori * d0;
						aJ[j - J + B] -= factori * d1;
						aJ[j - J + 2 * B] -= factori * d2;
					}

				// update accelerations for block J
				// #pragma omp critical
				std::lock_guard<std::mutex> ul{mutexes[J / B]};
				{
					for (int j = 0; j < B; ++j)
						aglobal[J + j] += aJ[j];
					for (int j = 0; j < B; ++j)
						aglobal[J + j + n] += aJ[j + B];
					for (int j = 0; j < B; ++j)
						aglobal[J + j + 2 * n] += aJ[j + 2 * B];
				}
			} // end J loop

			// update accelerations of block I
			// #pragma omp critical
			std::lock_guard<std::mutex> ul{mutexes[I / B]};
			{
				for (int i = 0; i < B; ++i)
					aglobal[I + i] += aI[i];
				for (int i = 0; i < B; ++i)
					aglobal[I + i + n] += aI[i + B];
				for (int i = 0; i < B; ++i)
					aglobal[I + i + 2 * n] += aI[i + 2 * B];
			}
		} // end I loop
		// delete thread local data
		delete[] aI;
		delete[] aJ;
	}
}

template<size_t simd_width>
struct SIMDSelector
{
};
template<>
struct SIMDSelector<2>
{
  static const size_t simd_width = 2;
  static const size_t simd_registers = 16;
  static const size_t num_rows = 1;
  typedef Vec2d SIMDType;
};
template<>
struct SIMDSelector<4>
{
  static const size_t simd_width = 4;
  static const size_t simd_registers = 16;
  static const size_t num_rows = 1;
  typedef Vec4d SIMDType;
};
template<>
struct SIMDSelector<8>
{
  static const size_t simd_width = 8;
  static const size_t simd_registers = 32;
  static const size_t num_rows = 2;
  typedef Vec8d SIMDType;
};

/** \brief compute acceleration vector from position and masses (vectorized using num_rows x simd_width masses)
 *
 * This version works on structure of arrays data layout
 */
template<int simd_width>
void acceleration_blocked_omp_vectorized_SoA_new(int n, double *__restrict__ x, double *__restrict__ m, double *__restrict__ aglobal)
{
	std::vector<std::mutex> mutexes(n / B); // one mutex per block

#pragma omp parallel firstprivate(n, x, m, aglobal)
	{
		// simd definitions
  		using simd_type = typename SIMDSelector<simd_width>::SIMDType;
  		constexpr int num_rows = SIMDSelector<simd_width>::num_rows;

		// make private acceleration vectors to accumulate to
		// for omp parallel version
		double *aI;
		aI = new (std::align_val_t(64)) double[3 * B];
		double *aJ;
		aJ = new (std::align_val_t(64)) double[3 * B];

		// simd registers: 8*num_rows + 8
		simd_type XI0[num_rows]; // broadcasted coordinates of body i,...,i+num_rows-1
		simd_type XI1[num_rows]; // broadcasted coordinates of body i,...,i+num_rows-1
		simd_type XI2[num_rows]; // broadcasted coordinates of body i,...,i+num_rows-1
		simd_type XJ0, XJ1, XJ2; // position of bodies j, ..., j+simd_width-1
		simd_type DJ0[num_rows]; // distances of bodies j, ..., j+simd_width-1
		simd_type DJ1[num_rows]; // distances of bodies j, ..., j+simd_width-1
		simd_type DJ2[num_rows]; // distances of bodies j, ..., j+simd_width-1
		simd_type AJ0, AJ1, AJ2; // accelerations to bodies j, ..., j+simd_width-1
		simd_type MI[num_rows];	 // broadcasted mass of body i,...,i+num_rows-1
		simd_type MJ;            // individual masses j
		simd_type S[num_rows];   // scalar factors
		simd_type F;	 					 // scalar factors

		// for conversion from simd_type to scalar
		double factor[num_rows][simd_width];
		double distance0[num_rows][simd_width];
		double distance1[num_rows][simd_width];
		double distance2[num_rows][simd_width];

#pragma omp for schedule(dynamic, 1)
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

				for (int i = I; i < I + B; i += num_rows)
				{
					// load position of body i. This can be reused for the whole block row.
					for (int k=0; k<num_rows; ++k)
					{
					  XI0[k] = simd_type(x[i+k]);
					  XI1[k] = simd_type(x[i+k+n]);
					  XI2[k] = simd_type(x[i+k+2*n]);
					  MI[k] = simd_type(m[i+k]);					// load *broadcast* for mass masses
					}
					for (int j = J; j < J + B; j += simd_width)
					{
						// compute interaction of num_rows x simd_width bodies

						// load positions of simd_width particles starting at j
						XJ0.load(&(x[j]));		   // x coordinates
						XJ1.load(&(x[j + n]));	   // y coordinates
						XJ2.load(&(x[j + 2 * n])); // z coordinates

						// distances of body i to all bodies j,...,j+w-1 with power 1,2,3
						for (int k=0; k<num_rows; ++k)
						{
							DJ0[k] = XJ0 - XI0[k];// difference of x coordinates for simd_width bodies to k
							DJ1[k] = XJ1 - XI1[k];// difference of x coordinates for simd_width bodies to k
							DJ2[k] = XJ2 - XI2[k];// difference of x coordinates for simd_width bodies to k
							S[k] = simd_type(epsilon2);		// regularization
							S[k] = mul_add(DJ0[k], DJ0[k], S[k]); // fma
							S[k] = mul_add(DJ1[k], DJ1[k], S[k]); // fma
							S[k] = mul_add(DJ2[k], DJ2[k], S[k]); // fma
							F = sqrt(S[k]);		  // compute square roots;
							S[k] *= F;		      // R to power 3
							F = simd_type(G);	  // broadcast for gravitational constant
							S[k] = F / S[k];      // these are the scalar factors up to the masses
						}

						// now we update the accelerations of the masses j,...,j+w-1
						AJ0.load(&(aJ[j - J]));			// load x coordinates
						AJ1.load(&(aJ[j - J + B]));		// load y coordinates
						AJ2.load(&(aJ[j - J + 2 * B])); // load z coordinates
						for (int k=0; k<num_rows; ++k)
						{
							F = MI[k] * S[k];					 // scalar factors with masses
							AJ0 = nmul_add(F, DJ0[k], AJ0);		 // fms
							AJ1 = nmul_add(F, DJ1[k], AJ1);		 // fms
							AJ2 = nmul_add(F, DJ2[k], AJ2);		 // fms
						}
						AJ0.store(&aJ[j - J]);		   // store result
						AJ1.store(&aJ[j - J + B]);	   // store result
						AJ2.store(&aJ[j - J + 2 * B]); // store result

						// scalar updates for mass i, ..., i+num_rows-1
						MJ.load(&m[j]); // simd_width masses from j, ..., j+simd_width-1
						for (int k=0; k<num_rows; ++k)
						{
							F = MJ * S[k];		 // scalar factors with masses
							F.store(factor[k]);
							DJ0[k].store(distance0[k]);
							DJ1[k].store(distance1[k]);
							DJ2[k].store(distance2[k]);
							for (int l=0; l<simd_width; ++l)
							{
								aI[i+k - I] += factor[k][l] * distance0[k][l];
								aI[i+k - I + B] += factor[k][l] * distance1[k][l];
								aI[i+k - I + 2 * B] += factor[k][l] * distance2[k][l];
							}
						}
					}
				}
				std::lock_guard<std::mutex> ul{mutexes[J / B]};
				{
					// update accelerations for block J
					for (int j = 0; j < B; ++j)
						aglobal[J + j] += aJ[j];
					for (int j = 0; j < B; ++j)
						aglobal[J + j + n] += aJ[j + B];
					for (int j = 0; j < B; ++j)
						aglobal[J + j + 2 * n] += aJ[j + 2 * B];
				}
			} // end J loop
			std::lock_guard<std::mutex> ul{mutexes[I / B]};
			{
				// update accelerations of block I
				for (int i = 0; i < B; ++i)
					aglobal[I + i] += aI[i];
				for (int i = 0; i < B; ++i)
					aglobal[I + i + n] += aI[i + B];
				for (int i = 0; i < B; ++i)
					aglobal[I + i + 2 * n] += aI[i + 2 * B];
			}
		} // end I loop
		// delete thread local data
		delete[] aI;
		delete[] aJ;
	}
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
	// acceleration_blocked_SoA(n, x, m, a);
	// acceleration_blocked_buffered_SoA(n, x, m, a);
	// acceleration_blocked_omp_SoA(n, x, m, a);
	 acceleration_blocked_omp_vectorized_SoA_new<8>(n, x, m, a);

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
#pragma omp for schedule(static, 1)
	for (int I = 0; I < n; I += B)
		for (int i = I; i < I + B; i++)
			for (size_t j = 0; j < 3; ++j)
				to[j * n + i] = from[i][j];
}
void copy(double3 *to, double *from, size_t n)
{
#pragma omp for schedule(static, 1)
	for (int I = 0; I < n; I += B)
		for (int i = I; i < I + B; i++)
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
	char basename[128]; // common part of file name
	char name[256];			// filename with number
	FILE *file;					// C style file hande
	double t;						// current time
	double dt;					// time step

	// command line for restarting
	if (argc == 5)
	{
		sscanf(argv[1], "%s", (char *)&basename);
		sscanf(argv[2], "%d", &k);
		sscanf(argv[3], "%d", &timesteps);
		sscanf(argv[4], "%d", &mod);
	}
	else if (argc == 6) // command line for starting with initial condition
	{
		sscanf(argv[1], "%s", (char *)&basename);
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
		// plummer(n, 17, x, v, m);
		two_plummer(n, 17, x, v, m);
		//  cube(n,17,1.0,100.0,0.1,x,v,m);
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
	auto ekin0 = ekin(n, m, v);
	auto epot0 = epot(n, m, x, G);
	std::cout << "ekin=" << ekin0 << " epot=" << epot0 << " etot=" << ekin0 + epot0 << std::endl;

	// std::cout << "size of mutex is " << sizeof(std::mutex) << std::endl;

	// initialize timestep and write first file
	std::cout << "step=" << k << " finalstep=" << timesteps << " time=" << t << " dt=" << dt << std::endl;
	int P;
#pragma omp parallel
	{
		P = omp_get_num_threads();
	}
	std::cout << "using " << P << " threads" << std::endl;
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

			auto ekin1 = ekin(n, m, v);
			auto epot1 = epot(n, m, x, G);
			std::cout << "ekin=" << ekin1 << " epot=" << epot1 << " etot=" << ekin1 + epot1
								<< " ratio=" << (ekin1 + epot1) / (ekin0 + epot0) << std::endl;

			start = get_time_stamp();
		}
	}

	delete[] a;
	delete[] x;
	delete[] v;
	delete[] m;
	delete[] A;
	delete[] X;
	delete[] V;

	return 0;
}
