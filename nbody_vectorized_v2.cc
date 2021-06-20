#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>

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
const int B=40; // block size for tiling

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

template<size_t simd_width>
struct SIMDSelector
{
  const size_t width = simd_width;
};
template<>
struct SIMDSelector<2>
{
  const std::string name = "SSE2";
  const size_t simd_width = 2;
  const size_t simd_registers = 16;
  typedef Vec2d SIMDType;
};
template<>
struct SIMDSelector<4>
{
  const std::string name = "AVX2";
  const size_t simd_width = 4;
  const size_t simd_registers = 16;
  typedef Vec4d SIMDType;
};
template<>
struct SIMDSelector<8>
{
  const std::string name = "AVX512";
  const size_t simd_width = 8;
  const size_t simd_registers = 32;
  typedef Vec8d SIMDType;
};

template<size_t simd_width>
void acceleration_blocked_vectorized (int n, double* __restrict__ x, double* __restrict__ m, double* __restrict__ aglobal)
{
  const size_t W=simd_width; // SIMD width
  using VecWd = typename SIMDSelector<simd_width>::SIMDType; // SIMD type

#pragma omp parallel shared(aglobal), firstprivate(n,x,m)
  {
    // 16 registers!
    VecWd Xi0,Xi1,Xi2;
    VecWd Ai0,Ai1,Ai2;
    VecWd Di0,Di1,Di2;
    VecWd M,R3;
    
    double xj0;
    double xj1;
    double xj2;
    double mj;

    // make private acceleration vector to accumulate to
    std::vector<double> a(3*n,0.0);
    
#pragma omp for
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
	  
	    // prefetching of scalar(!) quantities
	    xj0 = x[J];
	    xj1 = x[n+J];
	    xj2 = x[2*n+J];
	    mj = G*m[J];

	    // loop over masses j
	    for (int j=J; j<J+B-1; j+=1)
	      {
		// now we compute the interaction of W masses with the mass j
		// distance vectors
		Di0 = VecWd(xj0); // hope that these are loaded now
		Di1 = VecWd(xj1);
		Di2 = VecWd(xj2);
		xj0 = x[j+1]; // prefetch
		xj1 = x[n+j+1];
		xj2 = x[2*n+j+1];

		Di0 -= Xi0;
		Di1 -= Xi1;
		Di2 -= Xi2;

		// compute W distances^3
		R3 = VecWd(epsilon2);
		R3 = mul_add(Di0,Di0,R3);
		R3 = mul_add(Di1,Di1,R3);
		R3 = mul_add(Di2,Di2,R3);
		M = sqrt(R3);
		R3 *= M;
	      
		// update acceleration
		M = VecWd(mj);
		mj = G*m[j+1]; // prefetch
		M /= R3;
		Ai0 = mul_add(Di0,M,Ai0);
		Ai1 = mul_add(Di1,M,Ai1);
		Ai2 = mul_add(Di2,M,Ai2);
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
	      R3 = mul_add(Di0,Di0,R3);
	      R3 = mul_add(Di1,Di1,R3);
	      R3 = mul_add(Di2,Di2,R3);
	      M = sqrt(R3);
	      R3 *= M;
	      
	      // update acceleration
	      M = VecWd(mj);
	      M /= R3;
	      Ai0 = mul_add(Di0,M,Ai0);
	      Ai1 = mul_add(Di1,M,Ai1);
	      Ai2 = mul_add(Di2,M,Ai2);
	    }

	    // write back accelerations
	    Ai0.store(&a[i]);
	    Ai1.store(&a[n+i]);
	    Ai2.store(&a[2*n+i]);
	  }

    // now we need to reduce the private accelerations
#pragma omp critical
    {
      for (int i=0; i<3*n; i++)
	aglobal[i] += a[i];
    }
  } // end parallel regions  
}

template<size_t simd_width>
void acceleration_blocked_vectorized_interleaved (int n, double* __restrict__ x, double* __restrict__ m, double* __restrict__ aglobal)
{
  const size_t W=simd_width; // SIMD width
  using VecWd = typename SIMDSelector<simd_width>::SIMDType; // SIMD type

#pragma omp parallel shared(aglobal), firstprivate(n,x,m)
  {
    // 16 registers!
    VecWd Xi0,Xi1,Xi2;
    VecWd Ai0,Ai1,Ai2;
    VecWd Di0[2],Di1[2],Di2[2];
    VecWd M[2],R3[2];
    
    double xj0[2];
    double xj1[2];
    double xj2[2];
    double mj[2];

    // make private acceleration vector to accumulate to
    std::vector<double> a(3*n,0.0);
    
#pragma omp for
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
	  
	    // prefetching of scalar(!) quantities
	    xj0[0] = x[J];
	    xj0[1] = x[J+1];
	    xj1[0] = x[n+J];
	    xj1[1] = x[n+J+1];
	    xj2[0] = x[2*n+J];
	    xj2[1] = x[2*n+J+1];
	    mj[0] = G*m[J];
	    mj[1] = G*m[J+1];

	    // loop over masses j
	    for (int j=J; j<J+B-2; j+=2)
	      {
		// now we compute the interaction of W masses with the mass j
		// distance vectors
		Di0[0] = VecWd(xj0[0]); // hope that these are loaded now
		Di0[1] = VecWd(xj0[1]); // hope that these are loaded now
		Di1[0] = VecWd(xj1[0]);
		Di1[1] = VecWd(xj1[1]);
		Di2[0] = VecWd(xj2[0]);
		Di2[1] = VecWd(xj2[1]);
		xj0[0] = x[j+2]; // prefetch
		xj0[1] = x[j+3]; // prefetch
		xj1[0] = x[n+j+2];
		xj1[1] = x[n+j+3];
		xj2[0] = x[2*n+j+2];
		xj2[1] = x[2*n+j+3];

		Di0[0] -= Xi0;
		Di0[1] -= Xi0;
		Di1[0] -= Xi1;
		Di1[1] -= Xi1;
		Di2[0] -= Xi2;
		Di2[1] -= Xi2;

		// compute W distances^3
		R3[0] = VecWd(epsilon2);
		R3[1] = VecWd(epsilon2);
		R3[0] = mul_add(Di0[0],Di0[0],R3[0]);
		R3[1] = mul_add(Di0[1],Di0[1],R3[1]);
		R3[0] = mul_add(Di1[0],Di1[0],R3[0]);
		R3[1] = mul_add(Di1[1],Di1[1],R3[1]);
		R3[0] = mul_add(Di2[0],Di2[0],R3[0]);
		R3[1] = mul_add(Di2[1],Di2[1],R3[1]);
		M[0] = sqrt(R3[0]);
		M[1] = sqrt(R3[1]);
		R3[0] *= M[0];
		R3[1] *= M[1];
	      
		// update acceleration
		M[0] = VecWd(mj[0]);
		M[1] = VecWd(mj[1]);
		mj[0] = G*m[j+2]; // prefetch
		mj[1] = G*m[j+3]; // prefetch
		M[0] /= R3[0];
		M[1] /= R3[1];
		Ai0 = mul_add(Di0[0],M[0],Ai0);
		Ai1 = mul_add(Di1[0],M[0],Ai1);
		Ai2 = mul_add(Di2[0],M[0],Ai2);
		Ai0 = mul_add(Di0[1],M[1],Ai0);
		Ai1 = mul_add(Di1[1],M[1],Ai1);
		Ai2 = mul_add(Di2[1],M[1],Ai2);
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
	      R3[0] = mul_add(Di0[0],Di0[0],R3[0]);
	      R3[1] = mul_add(Di0[1],Di0[1],R3[1]);
	      R3[0] = mul_add(Di1[0],Di1[0],R3[0]);
	      R3[1] = mul_add(Di1[1],Di1[1],R3[1]);
	      R3[0] = mul_add(Di2[0],Di2[0],R3[0]);
	      R3[1] = mul_add(Di2[1],Di2[1],R3[1]);
	      M[0] = sqrt(R3[0]);
	      M[1] = sqrt(R3[1]);
	      R3[0] *= M[0];
	      R3[1] *= M[1];
	      
	      // update acceleration
	      M[0] = VecWd(mj[0]);
	      M[1] = VecWd(mj[1]);
	      M[0] /= R3[0];
	      M[1] /= R3[1];
	      Ai0 = mul_add(Di0[0],M[0],Ai0);
	      Ai1 = mul_add(Di1[0],M[0],Ai1);
	      Ai2 = mul_add(Di2[0],M[0],Ai2);
	      Ai0 = mul_add(Di0[1],M[1],Ai0);
	      Ai1 = mul_add(Di1[1],M[1],Ai1);
	      Ai2 = mul_add(Di2[1],M[1],Ai2);
	    }

	    // write back accelerations
	    Ai0.store(&a[i]);
	    Ai1.store(&a[n+i]);
	    Ai2.store(&a[2*n+i]);
	  }

    // now we need to reduce the private accelerations
#pragma omp critical
    {
      for (int i=0; i<3*n; i++)
	aglobal[i] += a[i];
    }
  } // end parallel regions  
}

#ifdef __AVX512F__
void acceleration_blocked_vectorized_512 (int n, double* __restrict__ x, double* __restrict__ m, double* __restrict__ aglobal)
{
  const size_t W=8; // SIMD width
  using VecWd = typename SIMDSelector<W>::SIMDType; // SIMD type

#pragma omp parallel shared(aglobal), firstprivate(n,x,m)
  {

    // 26 registers!
    VecWd Xi0,Xi1,Xi2;
    VecWd Ai0,Ai1,Ai2;
    VecWd Di0[4],Di1[4],Di2[4];
    VecWd M[4],R3[4];

    Vec4d xj0;
    Vec4d xj1;
    Vec4d xj2;
    Vec4d mj;
    Vec4d g=Vec4d(G);

    // make private acceleration vector to accumulate to
    std::vector<double> a(3*n,0.0);
  
#pragma omp for
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
	  
	    // prefetching
	    xj0.load(&x[J]);
	    xj1.load(&x[n+J]);
	    xj2.load(&x[2*n+J]);
	    mj.load(&m[J]);

	    // loop over masses j
	    for (int j=J; j<J+B-4; j+=4)
	      {
		// now we compute the interaction of W masses with the mass j
		// distance vectors
		Di0[0] = VecWd(xj0[0]); // hope that these are loaded now
		Di0[1] = VecWd(xj0[1]); // hope that these are loaded now
		Di0[2] = VecWd(xj0[2]); // hope that these are loaded now
		Di0[3] = VecWd(xj0[3]); // hope that these are loaded now
		xj0.load(&x[j+4]);
		Di1[0] = VecWd(xj1[0]);
		Di1[1] = VecWd(xj1[1]);
		Di1[2] = VecWd(xj1[2]);
		Di1[3] = VecWd(xj1[3]);
		xj1.load(&x[n+j+4]);
		Di2[0] = VecWd(xj2[0]);
		Di2[1] = VecWd(xj2[1]);
		Di2[2] = VecWd(xj2[2]);
		Di2[3] = VecWd(xj2[3]);
		xj2.load(&x[2*n+j+4]);

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
		R3[0] = mul_add(Di0[0],Di0[0],R3[0]);
		R3[1] = mul_add(Di0[1],Di0[1],R3[1]);
		R3[2] = mul_add(Di0[2],Di0[2],R3[2]);
		R3[3] = mul_add(Di0[3],Di0[3],R3[3]);
		R3[0] = mul_add(Di1[0],Di1[0],R3[0]);
		R3[1] = mul_add(Di1[1],Di1[1],R3[1]);
		R3[2] = mul_add(Di1[2],Di1[2],R3[2]);
		R3[3] = mul_add(Di1[3],Di1[3],R3[3]);
		R3[0] = mul_add(Di2[0],Di2[0],R3[0]);
		R3[1] = mul_add(Di2[1],Di2[1],R3[1]);
		R3[2] = mul_add(Di2[2],Di2[2],R3[2]);
		R3[3] = mul_add(Di2[3],Di2[3],R3[3]);
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
		mj.load(&m[j+4]);
		M[0] /= R3[0];
		M[1] /= R3[1];
		M[2] /= R3[2];
		M[3] /= R3[3];
		Ai0 = mul_add(Di0[0],M[0],Ai0);
		Ai1 = mul_add(Di1[0],M[0],Ai1);
		Ai2 = mul_add(Di2[0],M[0],Ai2);
		Ai0 = mul_add(Di0[1],M[1],Ai0);
		Ai1 = mul_add(Di1[1],M[1],Ai1);
		Ai2 = mul_add(Di2[1],M[1],Ai2);
		Ai0 = mul_add(Di0[2],M[2],Ai0);
		Ai1 = mul_add(Di1[2],M[2],Ai1);
		Ai2 = mul_add(Di2[2],M[2],Ai2);
		Ai0 = mul_add(Di0[3],M[3],Ai0);
		Ai1 = mul_add(Di1[3],M[3],Ai1);
		Ai2 = mul_add(Di2[3],M[3],Ai2);
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
	      R3[0] = mul_add(Di0[0],Di0[0],R3[0]);
	      R3[1] = mul_add(Di0[1],Di0[1],R3[1]);
	      R3[2] = mul_add(Di0[2],Di0[2],R3[2]);
	      R3[3] = mul_add(Di0[3],Di0[3],R3[3]);
	      R3[0] = mul_add(Di1[0],Di1[0],R3[0]);
	      R3[1] = mul_add(Di1[1],Di1[1],R3[1]);
	      R3[2] = mul_add(Di1[2],Di1[2],R3[2]);
	      R3[3] = mul_add(Di1[3],Di1[3],R3[3]);
	      R3[0] = mul_add(Di2[0],Di2[0],R3[0]);
	      R3[1] = mul_add(Di2[1],Di2[1],R3[1]);
	      R3[2] = mul_add(Di2[2],Di2[2],R3[2]);
	      R3[3] = mul_add(Di2[3],Di2[3],R3[3]);
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
	      Ai0 = mul_add(Di0[0],M[0],Ai0);
	      Ai1 = mul_add(Di1[0],M[0],Ai1);
	      Ai2 = mul_add(Di2[0],M[0],Ai2);
	      Ai0 = mul_add(Di0[1],M[1],Ai0);
	      Ai1 = mul_add(Di1[1],M[1],Ai1);
	      Ai2 = mul_add(Di2[1],M[1],Ai2);
	      Ai0 = mul_add(Di0[2],M[2],Ai0);
	      Ai1 = mul_add(Di1[2],M[2],Ai1);
	      Ai2 = mul_add(Di2[2],M[2],Ai2);
	      Ai0 = mul_add(Di0[3],M[3],Ai0);
	      Ai1 = mul_add(Di1[3],M[3],Ai1);
	      Ai2 = mul_add(Di2[3],M[3],Ai2);
	    }

	    // write back accelerations
	    Ai0.store(&a[i]);
	    Ai1.store(&a[n+i]);
	    Ai2.store(&a[2*n+i]);
	  }

    // now we need to reduce the private accelerations
#pragma omp critical
    {
      for (int i=0; i<3*n; i++)
	aglobal[i] += a[i];
    }
  } // end parallel regions  

}
#endif

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
  acceleration_blocked_vectorized<2>(n,x,m,a);
  //acceleration_blocked_vectorized_interleaved<4>(n,x,m,a);
  //acceleration_blocked_vectorized_512(n,x,m,a);

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
  if (B%8!=0)
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

  std::cout << "memory in one tile acceleration kernel: " << (B*10)*8 << " bytes" << std::endl;
  std::cout << "memory in acceleration kernel: " << n*7*8 << " bytes" << std::endl;
  std::cout << "memory in leapfrog kernel: " << n*10*8 << " bytes" << std::endl;
  std::cout << "memory total: " << n*16*8 << " bytes" << std::endl;
  std::cout << "flops in one iteration: " << 19.0*n*n+12.0*n << std::endl;
  std::cout << "total intensity: " << (13.0*n*(n-1.0)+12.0*n)/(n*10*8) << " flops/byte" << std::endl;
  std::cout << "tile intensity: " << 19.0*B*B/(B*10*8) << " flops/byte" << std::endl;

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
          double flop2 = mod*(19.0*n*n+12.0*n);
          printf("%g seconds for %g ops = %g (%g) GFLOPS \n",elapsed,flop,flop/elapsed/1E9,flop2/elapsed/1E9);

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
