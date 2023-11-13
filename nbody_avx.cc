#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>

#include "nbody_io.hh"
#include "nbody_generate.hh"
#include "time_experiment.hh"
#include "vcl/vectorclass.h"

#ifndef __GNUC__
#define __restrict__
#endif

// basic data type for position, velocity, acceleration
const int M=4;
typedef double double3[M]; // pad up for later use with SIMD
const int B=64; // block size for tiling

/*const double gamma = 6.674E-11;*/
const double G = 1.0;
const double epsilon2 = 1E-10;

/** \brief compute acceleration vector from position and masses
 * 
 * Executes \sum_{i=0}^{n-1} (n-i-1)*26 = n(n-1)*13
 * flops including 1 division and one square root
 */
void acceleration2 (int n, double3* __restrict__ x, double* __restrict__ m, double3* __restrict__ a)
{
  for (int i=0; i<n; i++)
    for (int j=i+1; j<n; j++)
      {
        double d0 = x[j][0]-x[i][0];
        double d1 = x[j][1]-x[i][1];
        double d2 = x[j][2]-x[i][2]; 
        double r2 = d0*d0 + d1*d1 + d2*d2 + epsilon2;
        double r = sqrt(r2);
        double invfact = G/(r*r2);
        double factori = m[i]*invfact;
        double factorj = m[j]*invfact;
        a[i][0] += factorj*d0;
        a[i][1] += factorj*d1;
        a[i][2] += factorj*d2;
        a[j][0] -= factori*d0;
        a[j][1] -= factori*d1;
        a[j][2] -= factori*d2;
      }
}

/** \brief compute acceleration vector from position and masses
 *
 * Executes \sum_{i=0}^{n-1} (n-i-1)*26 = n(n-1)*13
 * flops including 1 division and one square root
 * Blocked version working on tiles of size BxB
 */
void acceleration_blocked(int n, double3 *__restrict__ x, double *__restrict__ m, double3 *__restrict__ a)
{
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
        for (int j = J; j < J + B; j += 1)
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
  }
}

/** \brief compute acceleration vector from position and masses
 * 
 * Executes \sum_{i=0}^{n-1} (n-i-1)*26 = n(n-1)*13
 * flops including 1 division and one square root
 * AVX2 version working on 2x4 masses 
 *  - exploits symmetry
 *  - needs transpose
 */
void acceleration (int n, double3* __restrict__ x, double* __restrict__ m, double3* __restrict__ a)
{
  Vec4d A0,A1;
  Vec4d D0,D1,D2,D3; // distances
  Vec4d E0,E1,E2,E3; // distances
  Vec4d S0,S,U,T0,T1,T2;
  
  for (int I=0; I<n; I+=B)
    {
      // block (I,I)
      for (int i=I; i<I+B; i++)
	{
	  for (int j=i+1; j<I+B; j++)
	    {
	      double d0 = x[j][0]-x[i][0];
	      double d1 = x[j][1]-x[i][1];
	      double d2 = x[j][2]-x[i][2]; 
	      double r2 = d0*d0 + d1*d1 + d2*d2 + epsilon2;
	      double r = sqrt(r2);
	      double invfact = G/(r*r2);
	      double factori = m[i]*invfact;
	      double factorj = m[j]*invfact;
	      a[i][0] += factorj*d0;
	      a[i][1] += factorj*d1;
	      a[i][2] += factorj*d2;
	      a[j][0] -= factori*d0;
	      a[j][1] -= factori*d1;
	      a[j][2] -= factori*d2;
	    }
	}
      // blocks J>I
      for (int J=I+B; J<n; J+=B)
	{
	  for (int i=I; i<I+B; i+=2)
	    {
	      for (int j=J; j<J+B; j+=4)
		{
		  // distances 2x4 masses
		  T0.load(&x[i][0]);
		  T1.load(&x[i+1][0]);
		  D0.load(&x[j][0]);
		  D1.load(&x[j+1][0]);
		  D2.load(&x[j+2][0]);
		  D3.load(&x[j+3][0]);
		  E0 = D0;
		  E1 = D1;
		  E2 = D2;
		  E3 = D3;
		  D0 -= T0;
		  D1 -= T0;
		  D2 -= T0;
		  D3 -= T0;
		  E0 -= T1;
		  E1 -= T1;
		  E2 -= T1;
		  E3 -= T1;
		  
		  // transpose first batch
		  S = blend4<0,4,1,5>(D0,D1);
		  U = blend4<0,4,1,5>(D2,D3);
		  T0 = blend4<0,1,4,5>(S,U); // all 0 components
		  T1 = blend4<2,3,6,7>(S,U); // all 1 components
		  S = blend4<2,6,V_DC,V_DC>(D0,D1);
		  U = blend4<2,6,V_DC,V_DC>(D2,D3);
		  T2 = blend4<0,1,4,5>(S,U); // all 2 components

		  // the norms first batch
		  S0 = Vec4d(epsilon2);
		  S0 = mul_add(T0,T0,S0);
		  S0 = mul_add(T1,T1,S0);
		  S0 = mul_add(T2,T2,S0);

		  // transpose second batch
		  S = blend4<0,4,1,5>(E0,E1);
		  U = blend4<0,4,1,5>(E2,E3);
		  T0 = blend4<0,1,4,5>(S,U); // all 0 components
		  T1 = blend4<2,3,6,7>(S,U); // all 1 components
		  S = blend4<2,6,V_DC,V_DC>(E0,E1);
		  U = blend4<2,6,V_DC,V_DC>(E2,E3);
		  T2 = blend4<0,1,4,5>(S,U); // all 2 components

		  // the norms second batch
		  S = Vec4d(epsilon2);
		  S = mul_add(T0,T0,S);
		  S = mul_add(T1,T1,S);
		  S = mul_add(T2,T2,S);

		  // sqrt first batch
		  U = sqrt(S0);
		  U *= S0; // now U contains r^3
		  T0 = Vec4d(G);
		  S0 =T0/U; // now S is the inverse factor for four pairs first batch

		  // sqrt second batch
		  U = sqrt(S);
		  U *= S; // now U contains r^3
		  S =T0/U; // now S is the inverse factor for four pairs second batch

		  // update both rows from all columns
		  A0.load(&a[i][0]);
		  A1.load(&a[i+1][0]);
		  T2 = Vec4d(m[j]); // mass col j		  
		  U = permute4<0,0,0,0>(S0); // scalar factor column j
		  T0 = T2*U;
		  A0 = mul_add(T0,D0,A0);
		  U = permute4<0,0,0,0>(S); // scalar factor column j
		  T1 = T2*U;
		  A1 = mul_add(T1,E0,A1);

		  T2 = Vec4d(m[j+1]); // mass col j+1		  
		  U = permute4<1,1,1,1>(S0); // scalar factor column j
		  T0 = T2*U;
		  A0 = mul_add(T0,D1,A0);
		  U = permute4<1,1,1,1>(S); // scalar factor column j
		  T1 = T2*U;
		  A1 = mul_add(T1,E1,A1);

		  T2 = Vec4d(m[j+2]); // mass col j+2		  
		  U = permute4<2,2,2,2>(S0); // scalar factor column j
		  T0 = T2*U;
		  A0 = mul_add(T0,D2,A0);
		  U = permute4<2,2,2,2>(S); // scalar factor column j
		  T1 = T2*U;
		  A1 = mul_add(T1,E2,A1);

		  T2 = Vec4d(m[j+3]); // mass col j+3		  
		  U = permute4<3,3,3,3>(S0); // scalar factor column j
		  T0 = T2*U;
		  A0 = mul_add(T0,D3,A0);
		  U = permute4<3,3,3,3>(S); // scalar factor column j
		  T1 = T2*U;
		  A1 = mul_add(T1,E3,A1);
		  A0.store(&a[i][0]);
		  A1.store(&a[i+1][0]);

		  // now update all columns from both rows
		  T0 = Vec4d(m[i]); // row 0 mass
		  T1 = Vec4d(m[i+1]); // row 1 mass
		  A0.load(&a[j][0]);
		  U = permute4<0,0,0,0>(S0); // scalar factor column j row 0
		  T2 = T0*U;
		  A0 = nmul_add(T2,D0,A0);
		  U = permute4<0,0,0,0>(S); // scalar factor column j row 1
		  T2 = T1*U;
		  A0 = nmul_add(T2,E0,A0);
		  A0.store(&a[j][0]);

		  A0.load(&a[j+1][0]);
		  U = permute4<1,1,1,1>(S0); // scalar factor column j+1 row 0
		  T2 = T0*U;
		  A0 = nmul_add(T2,D1,A0);
		  U = permute4<1,1,1,1>(S); // scalar factor column j+1 row 1
		  T2 = T1*U;
		  A0 = nmul_add(T2,E1,A0);
		  A0.store(&a[j+1][0]);

		  A0.load(&a[j+2][0]);
		  U = permute4<2,2,2,2>(S0); // scalar factor column j+2 row 0
		  T2 = T0*U;
		  A0 = nmul_add(T2,D2,A0);
		  U = permute4<2,2,2,2>(S); // scalar factor column j+2 row 1
		  T2 = T1*U;
		  A0 = nmul_add(T2,E2,A0);
		  A0.store(&a[j+2][0]);

		  A0.load(&a[j+3][0]);
		  U = permute4<3,3,3,3>(S0); // scalar factor column j+3 row 0
		  T2 = T0*U;
		  A0 = nmul_add(T2,D3,A0);
		  U = permute4<3,3,3,3>(S); // scalar factor column j+3 row 1
		  T2 = T1*U;
		  A0 = nmul_add(T2,E3,A0);
		  A0.store(&a[j+3][0]);
		}
	    }
	}
    }
}

/** \brief compute acceleration vector from position and masses
 * 
 * Executes \sum_{i=0}^{n-1} (n-i-1)*26 = n(n-1)*13
 * flops including 1 division and one square root
 *   - Exploits symmetry
 *   - works on 1x4 masses at at time
 *   - delays write back of acceleration
 */
void acceleration3 (int n, double3* __restrict__ x, double* __restrict__ m, double3* __restrict__ a)
{
  Vec4d Ai; // acceleration row
  Vec4d D0,D1,D2,D3; // distances
  Vec4d A0,A1,A2,A3; // accelerations columns
  Vec4d S,U,T0,T1,T2; // temporaries

  // loop over rows of blocks
  for (int I=0; I<n; I+=B)
    {
      // block (I,I) diagonal block, treat classically
      for (int i=I; i<I+B; i++)
	for (int j=i+1; j<I+B; j++)
	  {
	    double d0 = x[j][0]-x[i][0];
	    double d1 = x[j][1]-x[i][1];
	    double d2 = x[j][2]-x[i][2]; 
	    double r2 = d0*d0 + d1*d1 + d2*d2 + epsilon2;
	    double r = sqrt(r2);
	    double invfact = G/(r*r2);
	    double factori = m[i]*invfact;
	    double factorj = m[j]*invfact;
	    a[i][0] += factorj*d0;
	    a[i][1] += factorj*d1;
	    a[i][2] += factorj*d2;
	    a[j][0] -= factori*d0;
	    a[j][1] -= factori*d1;
	    a[j][2] -= factori*d2;
	  }
      // blocks J>I; offdiagonal block. Particles are not the same
      for (int J=I+B; J<n; J+=B) // loop over columns of blocks
	for (int j=J; j<J+B; j+=4) // loop over 4 particles
	  {
	    // update accelerations of four particles
	    A0.load(&a[j][0]);
	    A1.load(&a[j+1][0]);
	    A2.load(&a[j+2][0]);
	    A3.load(&a[j+3][0]);

	    // loop over particles in row
	    for (int i=I; i<I+B; i+=1)
	      {
		// distances 1x4 masses
		T0.load(&x[i][0]); // position particle i
		D0.load(&x[j][0]); // positions of all particles j...j+3
		D1.load(&x[j+1][0]);
		D2.load(&x[j+2][0]);
		D3.load(&x[j+3][0]);
		D0 -= T0; // distance vector; is needed later
		D1 -= T0;
		D2 -= T0;
		D3 -= T0;
		  
		// transpose distances
		S = blend4<0,4,1,5>(D0,D1);
		U = blend4<0,4,1,5>(D2,D3);
		T0 = blend4<0,1,4,5>(S,U); // all 0 components
		T1 = blend4<2,3,6,7>(S,U); // all 1 components
		S = blend4<2,6,V_DC,V_DC>(D0,D1);
		U = blend4<2,6,V_DC,V_DC>(D2,D3);
		T2 = blend4<0,1,4,5>(S,U); // all 2 components

		// the norms first batch
		S = Vec4d(epsilon2);
		S = mul_add(T0,T0,S);
		S = mul_add(T1,T1,S);
		S = mul_add(T2,T2,S); // now S contains the norms 

		// determine inverse factors
		U = sqrt(S);
		U *= S; // now U contains r^3
		T0 = Vec4d(G);
		S =T0/U; // now S is the inverse factor for four particle pairs

		// now update accelerations
		Ai.load(&a[i][0]); // particle i, and we have particles j,...,j+3 in A0,..., A3
		
		// update Ai with all j
		T2 = Vec4d(m[j]); // mass col j		  
		U = permute4<0,0,0,0>(S); // scalar factor column j
		T0 = T2*U;
		Ai = mul_add(T0,D0,Ai);

		T2 = Vec4d(m[j+1]); // mass col j+1		  
		U = permute4<1,1,1,1>(S); // scalar factor column j
		T0 = T2*U;
		Ai = mul_add(T0,D1,Ai);

		T2 = Vec4d(m[j+2]); // mass col j+2		  
		U = permute4<2,2,2,2>(S); // scalar factor column j
		T0 = T2*U;
		Ai = mul_add(T0,D2,Ai);

		T2 = Vec4d(m[j+3]); // mass col j+3		  
		U = permute4<3,3,3,3>(S); // scalar factor column j
		T0 = T2*U;
		Ai = mul_add(T0,D3,Ai);

		Ai.store(&a[i][0]);

		// now update all columns from row i
		T0 = Vec4d(m[i]); // row 0 mass
		U = permute4<0,0,0,0>(S); // scalar factor column j row 0
		T2 = T0*U;
		A0 = nmul_add(T2,D0,A0);

		U = permute4<1,1,1,1>(S); // scalar factor column j+1 row 0
		T2 = T0*U;
		A1 = nmul_add(T2,D1,A1);

		U = permute4<2,2,2,2>(S); // scalar factor column j+2 row 0
		T2 = T0*U;
		A2 = nmul_add(T2,D2,A2);

		U = permute4<3,3,3,3>(S); // scalar factor column j+3 row 0
		T2 = T0*U;
		A3 = nmul_add(T2,D3,A3);
	      }

	    // update accelerations of four particles
	    A0.store(&a[j][0]);
	    A1.store(&a[j+1][0]);
	    A2.store(&a[j+2][0]);
	    A3.store(&a[j+3][0]);
	  }
    }
}



/** \brief do one time step with leapfrog
 *
 * does n*(n-1)*13 + 12n flops
 */
void leapfrog (int n, double dt, double3* __restrict__ x, double3* __restrict__ v, double* __restrict__ m, double3* __restrict__ a)
{
  // update position: 6n flops
  for (int i=0; i<n; i++)
    {
      x[i][0] += dt*v[i][0];
      x[i][1] += dt*v[i][1];
      x[i][2] += dt*v[i][2];
    }

  // save and clear acceleration
  for (int i=0; i<n; i++)
    for (int j=0; j<M; j++)
      a[i][j] = 0.0;
  
  // compute new acceleration: n*(n-1)*13 flops
  acceleration3(n,x,m,a);

  // update velocity: 6n flops
  for (int i=0; i<n; i++)
    {
      v[i][0] += dt*a[i][0];
      v[i][1] += dt*a[i][1];
      v[i][2] += dt*a[i][2];
    }
}

template<typename T>
size_t alignment (const T *p)
{
  for (size_t m=64; m>1; m/=2)
    if (((size_t)p)%m==0) return m;
  return 1;
}

int main (int argc, char** argv)
{
  int n;              // number of bodies in the system
  double *m;          // array for maasses
  double3 *x;         // array for positions
  double3 *v;         // array for velocites
  double3 *a;         // array for accelerations
  int timesteps;      // final time step number
  int k;              // time step number
  int mod;            // files are written when k is a multiple of mod 
  char basename[256]; // common part of file name
  char name[256];     // filename with number 
  FILE *file;         // C style file hande
  double t;           // current time
  double dt;          // time step

  // command line for restarting
  if (argc==5)
    {
      sscanf(argv[1],"%s",&basename);
      sscanf(argv[2],"%d",&k);
      sscanf(argv[3],"%d",&timesteps);
      sscanf(argv[4],"%d",&mod);
    }
  else if (argc==6) // command line for starting with initial condition
    {
      sscanf(argv[1],"%s",&basename);
      sscanf(argv[2],"%d",&n);
      sscanf(argv[3],"%d",&timesteps);
      sscanf(argv[4],"%lg",&dt);
      sscanf(argv[5],"%d",&mod);
    }
  else // invalid command line, print usage
    {
      std::cout << "usage: " << std::endl;
      std::cout << "nbody_vanilla <basename> <load step> <final step> <every>" << std::endl;
      std::cout << "nbody_vanilla <basename> <nbodies> <timesteps> <timestep> <every>" << std::endl;
      return 1;
    }
  
  // set up computation from file
  if (argc==5)
    {
      sprintf(name,"%s_%06d.vtk",basename,k);
      file = fopen(name,"r");
      if (file==NULL)
        {
          std::cout << "could not open file " << std::string(basename) << " aborting" << std::endl;
          return 1;
        }
      n = get_vtk_numbodies(file);
      rewind(file);
      x = new (std::align_val_t(64)) double3[n];
      v = new (std::align_val_t(64)) double3[n];
      m = new (std::align_val_t(64)) double[n];
      read_vtk_file_double(file,n,x,v,m,&t,&dt);
      fclose(file);
      k *= mod; // adjust step number
      std::cout << "loaded " << n << "bodies from file " << std::string(basename) << std::endl;
    }
  // set up computation from initial condition
  if (argc==6)
    {
      x = new (std::align_val_t(64)) double3[n];
      v = new (std::align_val_t(64)) double3[n];
      m = new (std::align_val_t(64)) double[n];
      //plummer(n,17,x,v,m);
      two_plummer(n,17,x,v,m);
      //cube(n,17,1.0,100.0,0.1,x,v,m);
      std::cout << "initialized " << n << " bodies" << std::endl;
      k = 0;
      t = 0.0;
      printf("writing %s_%06d.vtk \n",basename,k);
      sprintf(name,"%s_%06d.vtk",basename,k);
      file = fopen(name,"w");
      write_vtk_file_double(file,n,x,v,m,t,dt);
      fclose(file);
    }
  if (n%B!=0)
    {
      std::cout << n << " is not a multiple of the block size " << B << std::endl;
      exit(1);
    }
  if (B%4!=0)
    {
      std::cout << B << "=B is not a multiple of 4 " << std::endl;
      exit(1);
    }
  if (M!=4)
    {
      std::cout << M << "=M is not 4 " << std::endl;
      exit(1);
    }
  
  // allocate acceleration vector
  a = new (std::align_val_t(64)) double3[n];
  // explicitly fill/clear padded values
  for (int i=0; i<n; i++)
    for (int j=3; j<M; j++)
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
  for (; k<=timesteps; k++)
    {
      leapfrog(n,dt,x,v,m,a);
      t += dt;
      if (k%mod==0)
        {
          auto stop = get_time_stamp();
          double elapsed = get_duration_seconds(start,stop);
          double flop = mod*(13.0*n*(n-1.0)+12.0*n);
          printf("%g seconds for %g ops = %g GFLOPS\n",elapsed,flop,flop/elapsed/1E9);

          printf("writing %s_%06d.vtk \n",basename,k/mod);                 
          sprintf(name,"%s_%06d.vtk",basename,k/mod);
          file = fopen(name,"w");
          write_vtk_file_double(file,n,x,v,m,t,dt);
          fclose(file);
                  
          start = get_time_stamp();
        }
    }

  return 0;
}
