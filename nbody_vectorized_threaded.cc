#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include <thread>
#include <xmmintrin.h>

#include "nbody_io.hh"
#include "nbody_generate.hh"
#include "time_experiment.hh"
#include "vcl/vectorclass.h"
#include "Barrier.hh"

#ifndef __GNUC__
#define __restrict__
#endif

// basic data type for position, velocity, acceleration
const int M=4;
typedef double double3[M]; // pad up for later use with SIMD
const int B=32; // block size for tiling

/*const double gamma = 6.674E-11;*/
const double G = 1.0;
const double epsilon2 = 1E-10;

// version 3: 1x4 interaction, column major
// with nthreads threads
void acceleration3 (int nthreads, int rank, Barrier& barrier, int n, double3* __restrict__ x, double* __restrict__ m, double3* __restrict__ a)
{
  Vec4d Ai; // acceleration row
  Vec4d D0,D1,D2,D3; // distances
  Vec4d A0,A1,A2,A3; // accelerations columns
  Vec4d S,U,T0,T1,T2; // temporaries

  // loop over colors
  for (int color=0; color<nthreads; color+=1)
    {
      // synchronize with others before starting with color
      barrier.wait(rank);
      
      // cyclic loop over rows of blocks for each rank
      for (int II=0; II<n; II+=nthreads*B)
	{
	  int I=II+rank*B;
	  
	  // treat BxB block in diagonal nthreadsBxnthreadsB block
	  if (color==0)
	    {
	      // really a diagonal block
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
	    }
	  else if (rank+color<nthreads)
	    {
	      // blocks J>I; offdiagonal block. Particles are not the same
	      int J=I+color*B;
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
			// distances 2x4 masses
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
	  
	  // blocks J>I; offdiagonal block. Particles are not the same
	  for (int JJ=II+nthreads*B; JJ<n; JJ+=nthreads*B) // loop over columns of blocks
	    {
	      int J=JJ+B*(rank+color)%nthreads;
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
		      // distances 2x4 masses
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
    }
}

/** \brief do one time step with leapfrog
 *
 * does n*(n-1)*13 + 12n flops
 */
void leapfrog (int nthreads, int rank, Barrier& barrier, int n, double dt, double3* __restrict__ x, double3* __restrict__ v, double* __restrict__ m, double3* __restrict__ a)
{
  // Note: We follow the "synchronize before" paradigm
  // we assume we enter synchronized here
  
  barrier.wait(rank);
  if (rank==0)
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
	a[i][0] = a[i][1] = a[i][2] = 0.0;
    }

  // parallel phase
  // compute new acceleration: n*(n-1)*13 flops
  acceleration3(nthreads,rank,barrier,n,x,m,a);

  // update velocity: 6n flops
  barrier.wait(rank);
  if (rank==0)
    for (int i=0; i<n; i++)
      {
	v[i][0] += dt*a[i][0];
	v[i][1] += dt*a[i][1];
	v[i][2] += dt*a[i][2];
      }
}

struct GlobalContext
{
  int nthreads; // number of threads involved

  // input data
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

  // shared state
  Barrier barrier;

  GlobalContext (int P)
    : nthreads(P), barrier(P)
  {}
};

void simulation (std::shared_ptr<GlobalContext> context, int rank)
{
  context->barrier.wait(rank);
  auto start = get_time_stamp();

  // do time steps
  int k = context->k + 1;
  for (; k<=context->timesteps; k++)
    {
      leapfrog(context->nthreads,rank,context->barrier,context->n,context->dt,context->x,context->v,context->m,context->a);
      if (rank==0) context->t += context->dt;
	  
      if (k%context->mod==0)
	{
	  context->barrier.wait(rank);
	  auto stop = get_time_stamp();
	  double elapsed = get_duration_seconds(start,stop);
	  double flop = context->mod*(13.0*context->n*(context->n-1.0)+12.0*context->n);
	  if (rank==0)
	    {
	      printf("%g seconds for %g ops = %g GFLOPS\n",elapsed,flop,flop/elapsed/1E9);	      
	      printf("writing %s_%06d.vtk \n",context->basename,k/context->mod);                 
	      sprintf(context->name,"%s_%06d.vtk",context->basename,k/context->mod);
	      context->file = fopen(context->name,"w");
	      write_vtk_file_double(context->file,context->n,context->x,context->v,context->m,context->t,context->dt);
	      fclose(context->file);
	    }
	  start = get_time_stamp();
	}
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
  _mm_setcsr(_mm_getcsr() | 0x8040);  // Both
 
  int nthreads;
  if (argc==6)
    {
      sscanf(argv[5],"%d",&nthreads);
    }
  else if (argc==7) // command line for starting with initial condition
    {
      sscanf(argv[6],"%d",&nthreads);
    }
  else // invalid command line, print usage
    {
      std::cout << "usage: " << std::endl;
      std::cout << "nbody_vanilla <basename> <load step> <final step> <every> <nthreads>" << std::endl;
      std::cout << "nbody_vanilla <basename> <nbodies> <timesteps> <timestep> <every> <nthreads>" << std::endl;
      return 1;
    }
    
  // get global context shared by all threads
  auto context = std::make_shared<GlobalContext>(nthreads);

  // command line for restarting
  if (argc==6)
    {
      sscanf(argv[1],"%s",&context->basename);
      sscanf(argv[2],"%d",&context->k);
      sscanf(argv[3],"%d",&context->timesteps);
      sscanf(argv[4],"%d",&context->mod);
    }
  else if (argc==7) // command line for starting with initial condition
    {
      sscanf(argv[1],"%s",&context->basename);
      sscanf(argv[2],"%d",&context->n);
      sscanf(argv[3],"%d",&context->timesteps);
      sscanf(argv[4],"%lg",&context->dt);
      sscanf(argv[5],"%d",&context->mod);
    }

  // check sizes
  if (context->n%(nthreads*B)!=0)
    {
      std::cout << context->n << " is not a multiple of the block size times nthreads" << nthreads*B << std::endl;
      exit(1);
    }
  if (context->n%B!=0)
    {
      std::cout << context->n << " is not a multiple of the block size " << B << std::endl;
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

  // set up computation from file
  if (argc==6)
    {
      sprintf(context->name,"%s_%06d.vtk",context->basename,context->k);
      context->file = fopen(context->name,"r");
      if (context->file==NULL)
        {
          std::cout << "could not open file " << std::string(context->basename) << " aborting" << std::endl;
          return 1;
        }
      context->n = get_vtk_numbodies(context->file);
      rewind(context->file);
      context->x = new (std::align_val_t(64)) double3[context->n];
      context->v = new (std::align_val_t(64)) double3[context->n];
      context->m = new (std::align_val_t(64)) double[context->n];
      // context->x = static_cast<double3*>(calloc(context->n,sizeof(double3)));
      // context->v = static_cast<double3*>(calloc(context->n,sizeof(double3)));
      // context->m = static_cast<double*>(calloc(context->n,sizeof(double)));
      read_vtk_file_double(context->file,context->n,context->x,context->v,context->m,&context->t,&context->dt);
      fclose(context->file);
      context->k *= context->mod; // adjust step number
      std::cout << "loaded " << context->n << "bodies from file " << std::string(context->basename) << std::endl;
    }
  // set up computation from initial condition
  if (argc==7)
    {
      context->x = new (std::align_val_t(64)) double3[context->n];
      context->v = new (std::align_val_t(64)) double3[context->n];
      context->m = new (std::align_val_t(64)) double[context->n];
      // context->x = static_cast<double3*>(calloc(context->n,sizeof(double3)));
      // context->v = static_cast<double3*>(calloc(context->n,sizeof(double3)));
      // context->m = static_cast<double*>(calloc(context->n,sizeof(double)));
      //plummer(n,17,x,v,m);
      two_plummer(context->n,17,context->x,context->v,context->m);
      //cube(n,17,1.0,100.0,0.1,x,v,m);
      std::cout << "initialized " << context->n << " bodies" << std::endl;
      context->k = 0;
      context->t = 0.0;
      printf("writing %s_%06d.vtk \n",context->basename,context->k);
      sprintf(context->name,"%s_%06d.vtk",context->basename,context->k);
      context->file = fopen(context->name,"w");
      write_vtk_file_double(context->file,context->n,context->x,context->v,context->m,context->t,context->dt);
      fclose(context->file);
    }
  
  // allocate acceleration vector
  context->a = new (std::align_val_t(64)) double3[context->n];
  //context->a = static_cast<double3*>(calloc(context->n,sizeof(double3)));
  // explicitly fill/clear padded values
  for (int i=0; i<context->n; i++)
    {
      for (int j=3; j<M; j++)
	context->x[i][j] = context->v[i][j] = context->a[i][j] = 0.0;
    }

  // report alignment
  std::cout << "x aligned at " << alignment(context->x) << std::endl;
  std::cout << "v aligned at " << alignment(context->v) << std::endl;
  std::cout << "a aligned at " << alignment(context->a) << std::endl;
  std::cout << "m aligned at " << alignment(context->m) << std::endl;

  // report simulation parameters
  std::cout << "step=" << context->k << " finalstep=" << context->timesteps << " time=" << context->t << " dt=" << context->dt << std::endl;

  // now start the threads and wait for
  // the computation to finish
  std::vector<std::thread> th;
  for (int i=0; i<nthreads; ++i)
    th.push_back(std::thread{simulation,context,i});
  for (int i=0; i<nthreads; ++i)
    th[i].join();

  delete[] context->x;
  delete[] context->v;
  delete[] context->a;
  delete[] context->m;
  
  return 0;
}
