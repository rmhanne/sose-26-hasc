#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>

#include "nbody_io.hh"
#include "nbody_generate.hh"
#include "time_experiment.hh"
#include <arm_neon.h>

#ifndef __GNUC__
#define __restrict__
#endif

// basic data type for position, velocity, acceleration
const int M = 4;
typedef double double3[M]; // pad up for later use with SIMD
const int B = 32;					 // block size for tiling

/*const double gamma = 6.674E-11;*/
const double G = 1.0;
const double epsilon2 = 1E-10;

void acceleration(int n, double3 *__restrict__ x, double *__restrict__ m, double3 *__restrict__ a)
{
	float64x2_t A00, A01, A10, A11, A20, A21, A30, A31;
	float64x2_t B00, B01, B10, B11, B20, B21, B30, B31;

	for (int I = 0; I < n; I += B)
	{
		// block (I,I)
		for (int i = I; i < I + B; i++)
			for (int j = i + 1; j < I + B; j++)
			{
				double d0 = x[j][0] - x[i][0];
				double d1 = x[j][1] - x[i][1];
				double d2 = x[j][2] - x[i][2];
				double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
				double r = sqrt(r2);
				double invfact = G / (r * r2);
				double factori = m[i] * invfact;
				double factorj = m[j] * invfact;
				a[i][0] += factorj * d0;
				a[i][1] += factorj * d1;
				a[i][2] += factorj * d2;
				a[j][0] -= factori * d0;
				a[j][1] -= factori * d1;
				a[j][2] -= factori * d2;
			}

		// blocks J>I
		for (int J = I + B; J < n; J += B)
			for (int i = I; i < I + B; i += 1)
				for (int j = J; j < J + B; j += 4)
				{
					// load x_i (there are not enough registers to keep it)
					B30 = vld1q_f64(&(x[i][0]));
					B31 = vld1q_f64(&(x[i][2]));

					// load positions of four consecutive points from j
					A00 = vld1q_f64(&(x[j][0]));
					A01 = vld1q_f64(&(x[j][2]));
					A10 = vld1q_f64(&(x[j + 1][0]));
					A11 = vld1q_f64(&(x[j + 1][2]));
					A20 = vld1q_f64(&(x[j + 2][0]));
					A21 = vld1q_f64(&(x[j + 2][2]));
					A30 = vld1q_f64(&(x[j + 3][0]));
					A31 = vld1q_f64(&(x[j + 3][2]));

					// subtract to get distances
					A00 = vsubq_f64(A00, B00);
					A01 = vsubq_f64(A01, B01);
					A10 = vsubq_f64(A10, B00);
					A01 = vsubq_f64(A11, B01);
					A20 = vsubq_f64(A20, B00);
					A01 = vsubq_f64(A21, B01);
					A30 = vsubq_f64(A30, B00);
					A01 = vsubq_f64(A31, B01);
					// now A00-A31 contains direction vectors from x_j+k to x_i

					// transpose
					B00 = A00;
					B00 = vcopyq_laneq_f64(B00, 1, A10, 0);
					B10 = A10;
					B10 = vcopyq_laneq_f64(B10, 0, A00, 1);

					B20 = A01;
					B20 = vcopyq_laneq_f64(B20, 1, A11, 0);
					B30 = A11;
					B30 = vcopyq_laneq_f64(B30, 0, A01, 1);

					B01 = A20;
					B01 = vcopyq_laneq_f64(B01, 1, A30, 0);
					B11 = A30;
					B11 = vcopyq_laneq_f64(B11, 0, A20, 1);

					B21 = A21;
					B21 = vcopyq_laneq_f64(B21, 1, A31, 0);
					B31 = A31;
					B31 = vcopyq_laneq_f64(B31, 0, A21, 1);

					// now we can compute the four scalar distances and store them in B30/B31 which is now zero
					B30 = vmovq_n_f64(epsilon2);
					B31 = vmovq_n_f64(epsilon2);
					B30 = vfmaq_f64(B30, B00, B00); // fused multiply add
					B31 = vfmaq_f64(B31, B01, B01);
					B30 = vfmaq_f64(B30, B10, B10);
					B31 = vfmaq_f64(B31, B11, B11);
					B30 = vfmaq_f64(B30, B20, B20);
					B31 = vfmaq_f64(B31, B21, B21);

					// square roots can be stored in B00/B01
					B00 = vsqrtq_f64(B30);
					B01 = vsqrtq_f64(B31);

					// compute r^3 and store it in B30/B31
					B30 = vmulq_f64(B30, B00);
					B31 = vmulq_f64(B31, B01);

					// now the division
					B20 = vmovq_n_f64(G); // load broadcast for G
					B30 = vdivq_f64(B20, B30);
					B31 = vdivq_f64(B20, B31);
					// now B30/B31 stores the factors G / (r * r2) for interaction of x_i with x_j, x_j+1, x_j+2, x_j+3

					// now we can update the accelerations

					// first we compute contributions for mass i by the four masses j,j+1,j+2,j+3
					// load acceleration for mass i in B00/B01
					B00 = vld1q_f64(&(a[i][0]));
					B01 = vld1q_f64(&(a[i][2]));
					// load four masses for bodies j,j+1,j+2,j+3
					B10 = vld1q_f64(&(m[j]));
					B11 = vld1q_f64(&(m[j + 2]));
					// compute scalar factors and store them in B10/B11
					B10 = vmulq_f64(B10, B30);
					B11 = vmulq_f64(B11, B31);
					// now compute four contributions
					B20 = vmovq_n_f64(vgetq_lane_f64(B10, 0)); // factor for mass j
					B00 = vfmaq_f64(B00, B20, A00);
					B01 = vfmaq_f64(B01, B20, A01);
					B20 = vmovq_n_f64(vgetq_lane_f64(B10, 1)); // factor for mass j
					B00 = vfmaq_f64(B00, B20, A10);
					B01 = vfmaq_f64(B01, B20, A11);
					B20 = vmovq_n_f64(vgetq_lane_f64(B11, 0)); // factor for mass j
					B00 = vfmaq_f64(B00, B20, A20);
					B01 = vfmaq_f64(B01, B20, A21);
					B20 = vmovq_n_f64(vgetq_lane_f64(B11, 1)); // factor for mass j
					B00 = vfmaq_f64(B00, B20, A30);
					B01 = vfmaq_f64(B01, B20, A21);
					// and store the result
					vst1q_f64(&(a[i][0]), B00);
					vst1q_f64(&(a[i][2]), B01);

					// load broadcast of mass i, we need it four times
					B10 = vmovq_n_f64(m[i]);

					// now the contribution to mass j from mass i
					B00 = vld1q_f64(&(a[j][0]));
					B01 = vld1q_f64(&(a[j][2]));
					B20 = vmovq_n_f64(vgetq_lane_f64(B30, 0)); // inverse factor
					B20 = vmulq_f64(B10, B20);								 // mass * inverse factor
					B00 = vfmsq_f64(B00, B20, A00);
					B01 = vfmsq_f64(B01, B20, A01);
					vst1q_f64(&(a[j][0]), B00);
					vst1q_f64(&(a[j][2]), B01);

					// now the contribution to mass j+1 from mass i
					B00 = vld1q_f64(&(a[j + 1][0]));
					B01 = vld1q_f64(&(a[j + 1][2]));
					B20 = vmovq_n_f64(vgetq_lane_f64(B30, 1)); // inverse factor
					B20 = vmulq_f64(B10, B20);								 // mass * inverse factor
					B00 = vfmsq_f64(B00, B20, A10);
					B01 = vfmsq_f64(B01, B20, A11);
					vst1q_f64(&(a[j + 1][0]), B00);
					vst1q_f64(&(a[j + 1][2]), B01);

					// now the contribution to mass j+2 from mass i
					B00 = vld1q_f64(&(a[j + 2][0]));
					B01 = vld1q_f64(&(a[j + 2][2]));
					B20 = vmovq_n_f64(vgetq_lane_f64(B31, 0)); // inverse factor
					B20 = vmulq_f64(B10, B20);								 // mass * inverse factor
					B00 = vfmsq_f64(B00, B20, A20);
					B01 = vfmsq_f64(B01, B20, A21);
					vst1q_f64(&(a[j + 2][0]), B00);
					vst1q_f64(&(a[j + 2][2]), B01);

					// now the contribution to mass j+3 from mass i
					B00 = vld1q_f64(&(a[j + 3][0]));
					B01 = vld1q_f64(&(a[j + 3][2]));
					B20 = vmovq_n_f64(vgetq_lane_f64(B31, 1)); // inverse factor
					B20 = vmulq_f64(B10, B20);								 // mass * inverse factor
					B00 = vfmsq_f64(B00, B20, A30);
					B01 = vfmsq_f64(B01, B20, A31);
					vst1q_f64(&(a[j + 3][0]), B00);
					vst1q_f64(&(a[j + 3][2]), B01);
				}
	}
}

/** \brief do one time step with leapfrog
 *
 * does n*(n-1)*13 + 12n flops
 */
void leapfrog(int n, double dt, double3 *__restrict__ x, double3 *__restrict__ v, double *__restrict__ m, double3 *__restrict__ a)
{
	// update position: 6n flops
	for (int i = 0; i < n; i++)
	{
		x[i][0] += dt * v[i][0];
		x[i][1] += dt * v[i][1];
		x[i][2] += dt * v[i][2];
	}

	// save and clear acceleration
	for (int i = 0; i < n; i++)
		for (int j = 0; j < M; j++)
			a[i][j] = 0.0;

	// compute new acceleration: n*(n-1)*13 flops
	acceleration(n, x, m, a);

	// update velocity: 6n flops
	for (int i = 0; i < n; i++)
	{
		v[i][0] += dt * a[i][0];
		v[i][1] += dt * a[i][1];
		v[i][2] += dt * a[i][2];
	}
}

template <typename T>
size_t alignment(const T *p)
{
	for (size_t m = 64; m > 1; m /= 2)
		if (((size_t)p) % m == 0)
			return m;
	return 1;
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
	if (B % 4 != 0)
	{
		std::cout << B << "=B is not a multiple of 4 " << std::endl;
		exit(1);
	}
	if (M != 4)
	{
		std::cout << M << "=M is not 4 " << std::endl;
		exit(1);
	}

	// allocate acceleration vector
	a = new (std::align_val_t(64)) double3[n];

	// explicitly fill/clear padded values
	// this is important to ensure that B30/B31 is zero after transpose
	for (int i = 0; i < n; i++)
		for (int j = 3; j < M; j++)
			x[i][j] = v[i][j] = a[i][j] = 0.0;

	// report alignment
	std::cout << "x aligned at " << alignment(x) << std::endl;
	std::cout << "v aligned at " << alignment(v) << std::endl;
	std::cout << "a aligned at " << alignment(a) << std::endl;
	std::cout << "m aligned at " << alignment(m) << std::endl;

	// initialize timestep and write first file
	std::cout << "step=" << k << " finalstep=" << timesteps << " time=" << t << " dt=" << dt << std::endl;
	auto start = get_time_stamp();

	// do time steps
	k += 1;
	for (; k <= timesteps; k++)
	{
		leapfrog(n, dt, x, v, m, a);
		t += dt;
		if (k % mod == 0)
		{
			auto stop = get_time_stamp();
			double elapsed = get_duration_seconds(start, stop);
			double flop = mod * (13.0 * n * (n - 1.0) + 12.0 * n);
			printf("%g seconds for %g ops = %g GFLOPS\n", elapsed, flop, flop / elapsed / 1E9);

			printf("writing %s_%06d.vtk \n", basename, k / mod);
			sprintf(name, "%s_%06d.vtk", basename, k / mod);
			file = fopen(name, "w");
			write_vtk_file_double(file, n, x, v, m, t, dt);
			fclose(file);

			start = get_time_stamp();
		}
	}

	return 0;
}
