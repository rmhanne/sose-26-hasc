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
const int M=3;
typedef double double3[M]; // pad up for later use with SIMD
const int B=32; // block size for tiling
const int W=4; // SIMD width

/*const double gamma = 6.674E-11;*/
const double G = 1.0;
const double epsilon2 = 1E-10;

/** \brief compute acceleration vector from position and masses
 * 
 * Executes \sum_{i=0}^{n-1} (n-i-1)*26 = n(n-1)*13
 * flops including 1 division and one square root
 */
void acceleration (int n, double* __restrict__ x, double* __restrict__ m, double* __restrict__ a)
{
  for (int i=0; i<n; i++)
    for (int j=i+1; j<n; j++)
      {
        double d0 = x[j]-x[i];
        double d1 = x[n+j]-x[n+i];
        double d2 = x[2*n+j]-x[2*n+i]; 
        double r2 = d0*d0 + d1*d1 + d2*d2 + epsilon2;
        double r = sqrt(r2);
        double invfact = G/(r*r2);
        double factori = m[i]*invfact;
        double factorj = m[j]*invfact;
        a[i] += factorj*d0;
        a[n+i] += factorj*d1;
        a[2*n+i] += factorj*d2;
        a[j] -= factori*d0;
        a[n+j] -= factori*d1;
        a[2*n+j] -= factori*d2;
      }
}

void acceleration_blocked (int n, double* __restrict__ x, double* __restrict__ m, double* __restrict__ a)
{
  for (int I=0; I<n; I+=B)
    {
      // diagonal block
      for (int i=I; i<I+B; i++)
	for (int j=i+1; j<I+B; j++)
	  {
	    double d0 = x[j]-x[i];
	    double d1 = x[n+j]-x[n+i];
	    double d2 = x[2*n+j]-x[2*n+i]; 
	    double r2 = d0*d0 + d1*d1 + d2*d2 + epsilon2;
	    double r = sqrt(r2);
	    double invfact = G/(r*r2);
	    double factori = m[i]*invfact;
	    double factorj = m[j]*invfact;
	    a[i] += factorj*d0;
	    a[n+i] += factorj*d1;
	    a[2*n+i] += factorj*d2;
	    a[j] -= factori*d0;
	    a[n+j] -= factori*d1;
	    a[2*n+j] -= factori*d2;
	  }
      // upper diagonal full blocks
      for (int J=I+B; J<n; J+=B)
	for (int j=J; j<J+B; j++)
	  for (int i=I; i<I+B; i++)
	    {
	      double d0 = x[j]-x[i];
	      double d1 = x[n+j]-x[n+i];
	      double d2 = x[2*n+j]-x[2*n+i]; 
	      double r2 = d0*d0 + d1*d1 + d2*d2 + epsilon2;
	      double r = sqrt(r2);
	      double invfact = G/(r*r2);
	      double factori = m[i]*invfact;
	      double factorj = m[j]*invfact;
	      a[i] += factorj*d0;
	      a[n+i] += factorj*d1;
	      a[2*n+i] += factorj*d2;
	      a[j] -= factori*d0;
	      a[n+j] -= factori*d1;
	      a[2*n+j] -= factori*d2;
	    }
    }
}

void acceleration_blocked_full (int n, double* __restrict__ x, double* __restrict__ m, double* __restrict__ a)
{
  for (int I=0; I<n; I+=B)
    for (int J=0; J<n; J+=B)
      for (int j=J; j<J+B; j++)
	for (int i=I; i<I+B; i++)
	  {
	    double d0 = x[j]-x[i];
	    double d1 = x[n+j]-x[n+i];
	    double d2 = x[2*n+j]-x[2*n+i]; 
	    double r2 = d0*d0 + d1*d1 + d2*d2 + epsilon2;
	    double r = sqrt(r2);
	    double invfact = G/(r*r2);
	    double factorj = m[j]*invfact;
	    a[i] += factorj*d0;
	    a[n+i] += factorj*d1;
	    a[2*n+i] += factorj*d2;
	  }
}

void acceleration_blocked_vectorized (int n, double* __restrict__ x, double* __restrict__ m, double* __restrict__ a)
{
  using VecWd = Vec4d;
  VecWd Xi0,Xi1,Xi2;
  VecWd Ai0,Ai1,Ai2;
  VecWd Di0,Di1,Di2;
  VecWd R3;
  VecWd S,T,U; // auxiiliary registers
    
  for (int I=0; I<n; I+=B)
    for (int J=0; J<n; J+=B)
      for (int i=I; i<I+B; i+=W)
	{
	  // load data of mass i
	  Xi0.load(&x[i]);
	  Xi1.load(&x[n+i]);
	  Xi2.load(&x[2*n+i]);
	  Ai0.load(&a[i]);
	  Ai1.load(&a[n+i]);
	  Ai2.load(&a[2*n+i]);

	  // loop over masses j
	  for (int j=J; j<J+B; ++j)
	    {
	      // now we compute the interaction of W masses with the mass j
	      // distance vectors
	      Di0.load(&x[j]);
	      Di1.load(&x[n+j]);
	      Di2.load(&x[2*n+j]);
	      Di0 -= Xi0;
	      Di1 -= Xi1;
	      Di2 -= Xi2;

	      // compute W distances^3
	      R3 = VecWd(epsilon2);
	      R3 = mul_add(Di0,Di0,R3);
	      R3 = mul_add(Di1,Di1,R3);
	      R3 = mul_add(Di2,Di2,R3);
	      U = sqrt(R3);
	      R3 *= U;
	      
	      // update acceleration
	      S = VecWd(m[j]);
	      T = VecWd(G);
	      U = T/R3; U *= S;
	      Ai0 = mul_add(Di0,U,Ai0);
	      Ai1 = mul_add(Di1,U,Ai1);
	      Ai2 = mul_add(Di2,U,Ai2);
	    }

	  // write back accelerations
	  Ai0.store(&a[i]);
	  Ai1.store(&a[n+i]);
	  Ai2.store(&a[2*n+i]);
	}
}

/** \brief do one time step with leapfrog
 *
 * does n*(n-1)*13 + 12n flops
 */
void leapfrog (int n, double dt, double* __restrict__ x, double* __restrict__ v, double* __restrict__ m, double* __restrict__ a)
{
  // update position: 6n flops
  for (int i=0; i<3*n; i++)
    x[i] += dt*v[i];

  // save and clear acceleration
  for (int i=0; i<3*n; i++)
    a[i] = 0.0;
  
  // compute new acceleration: n*(n-1)*13 flops
  //acceleration(n,x,m,a);
  // acceleration_blocked(n,x,m,a);
  // acceleration_blocked_full(n,x,m,a);
   acceleration_blocked_vectorized(n,x,m,a);

  // update velocity: 6n flops
  for (int i=0; i<3*n; i++)
    v[i] += dt*a[i];
}

template<typename T>
size_t alignment (const T *p)
{
  for (size_t m=64; m>1; m/=2)
    if (((size_t)p)%m==0) return m;
  return 1;
}


// functions for AoS <-> SoA transformation
void copy (double* to, double3* from, size_t n)
{
  for (size_t i=0; i<n; ++i)
    for (size_t j=0; j<3; ++j)
      to[j*n+i] = from[i][j];
}
void copy (double3* to, double* from, size_t n)
{
  for (size_t i=0; i<n; ++i)
    for (size_t j=0; j<3; ++j)
      to[i][j] = from[j*n+i];
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
  if (B%W!=0)
    {
      std::cout << B << "=B is not a multiple of 4 " << std::endl;
      exit(1);
    }

  // switch to SoA data layout in 1d array
  double *X;
  X = new (std::align_val_t(64)) double[3*n];
  double *V;
  V = new (std::align_val_t(64)) double[3*n];
  double *A;
  A = new (std::align_val_t(64)) double[3*n];
  
  // explicitly fill/clear padded values
  for (int i=0; i<n; i++)
    for (int j=3; j<M; j++)
      x[i][j] = v[i][j] = 0.0;

  // copy initial values
  copy(X,x,n);
  copy(V,v,n);

  // initialize timestep and write first file
  std::cout << "step=" << k << " finalstep=" << timesteps << " time=" << t << " dt=" << dt << std::endl;
  auto start = get_time_stamp();

  // do time steps
  k += 1;
  for (; k<=timesteps; k++)
    {
      leapfrog(n,dt,X,V,m,A);
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
	  copy(x,X,n);
	  copy(v,V,n);
          write_vtk_file_double(file,n,x,v,m,t,dt);
          fclose(file);
                  
          start = get_time_stamp();
        }
    }

  return 0;
}
