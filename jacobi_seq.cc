#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <stdio.h>
#include "vcl/vectorclass.h"
#include "time_experiment.hh"

#ifndef __GNUC__
#define __restrict__
#endif

const int B=128;
const int K=40;

struct GlobalContext
{
  // input data
  int n;     // nxn lattice of points
  int iterations; // number of iterations to do
  double* u0; // the initial guess
  double* u1; // temporary vector

  // output data

  GlobalContext (int n_)
    : n(n_)
  {}
};

void jacobi_vanilla_kernel (int n, int iterations, double* __restrict__ uold, double* __restrict__ unew)
{
  // do iterations
  for (int i=0; i<iterations; i++)
    {
      for (int i1=1; i1<n-1; i1++)
        for (int i0=1; i0<n-1; i0++)
          unew[i1*n+i0] = 0.25*(uold[i1*n+i0-n]+uold[i1*n+i0-1]
                                +uold[i1*n+i0+1]+uold[i1*n+i0+n]);
      std::swap(uold,unew);
    }
}

void jacobi_vanilla (std::shared_ptr<GlobalContext> context)
{
  jacobi_vanilla_kernel(context->n,context->iterations,context->u0,context->u1);
}

void jacobi_wave_kernel (int n, int iterations, double* __restrict__ uold, double* __restrict__ unew)
{
  double* u[2];
  u[0] = uold;
  u[1] = unew;
  
  // do iterations
  for (int kk=0; kk<iterations; kk+=K)
    for (int m=2; m<=n+K-2; m++)
      for (int k=std::max(1,m-n+2); k<=std::min(K,m-1); k++)
	{
	  int i1=m-k;
	  int dst=k%2;
	  int src=1-dst;
	  for (int i0=1; i0<n-1; i0++)
	    u[dst][i1*n+i0] = 0.25*(u[src][i1*n+i0-n]+u[src][i1*n+i0-1]
				    +u[src][i1*n+i0+1]+u[src][i1*n+i0+n]);
	}
}

void jacobi_wave (std::shared_ptr<GlobalContext> context)
{
  jacobi_wave_kernel(context->n,context->iterations,context->u0,context->u1);
}

void jacobi_blocked (std::shared_ptr<GlobalContext> context)
{
  int n = context->n;
  double* uold = context->u0;
  double* unew = context->u1;

  int blocksB=((n-2)/B)*B;
  int remainderB=(n-2)%B;
  // std::cout << "blocksB=" << blocksB << " remainderB=" << remainderB << std::endl;

  // do iterations
  for (int i=0; i<context->iterations; i++)
    {
      for (int I1=1; I1<1+blocksB; I1+=B)
	for (int I0=1; I0<1+blocksB; I0+=B)
	  for (int i1=I1; i1<I1+B; i1++)
	    for (int i0=I0; i0<I0+B; i0++)
	      unew[i1*n+i0] = 0.25*(uold[i1*n+i0-n]+uold[i1*n+i0-1]
				    +uold[i1*n+i0+1]+uold[i1*n+i0+n]);
      for (int I0=1; I0<1+blocksB; I0+=B)
	for (int i1=1+blocksB; i1<n-1; i1++)
	  for (int i0=I0; i0<I0+B; i0++)
	    unew[i1*n+i0] = 0.25*(uold[i1*n+i0-n]+uold[i1*n+i0-1]
				  +uold[i1*n+i0+1]+uold[i1*n+i0+n]);
      for (int i1=1+blocksB; i1<n-1; i1++)
	for (int i0=1+blocksB; i0<n-1; i0++)
	  unew[i1*n+i0] = 0.25*(uold[i1*n+i0-n]+uold[i1*n+i0-1]
				+uold[i1*n+i0+1]+uold[i1*n+i0+n]);
      std::swap(uold,unew);
    }

  // result should be in u1
  if (context->u1!=uold)
    std::swap(context->u0,context->u1);
}

void jacobi_vectorized (std::shared_ptr<GlobalContext> context)
{
  int n = context->n;
  double* uold = context->u0;
  double* unew = context->u1;

  int blocks4=((n-2)/4)*4;

  Vec4d QUARTER=Vec4d(0.25);
  Vec4d A,B,C,D,E;

  // do iterations
  for (int i=0; i<context->iterations; i++)
    {
      for (int i1=1; i1<n-1; i1++)
	{
	  int istart=i1*n+1;
	  int iend=istart+blocks4;
	  for (int index=istart; index<iend; index+=4)
	    {
	      A = Vec4d(0.0);
	      B.load(&uold[index-n]);
	      C.load(&uold[index-1]);
	      D.load(&uold[index+1]);
	      E.load(&uold[index+n]);
	      A = mul_add(QUARTER,B,A);
	      A = mul_add(QUARTER,C,A);
	      A = mul_add(QUARTER,D,A);
	      A = mul_add(QUARTER,E,A);
	      A.store(&unew[index]);
	    }
	  for (int index=iend; index<n-1; index++)
	    unew[index] = 0.25*(uold[index-n]+uold[index-1]+uold[index+1]+uold[index+n]);
	}
      std::swap(uold,unew);
    }
  // result should be in u1
  if (context->u1!=uold)
    std::swap(context->u0,context->u1);
}

void jacobi_vectorized_wave (std::shared_ptr<GlobalContext> context)
{
  int n = context->n;
  double* u[2];
  u[0] = context->u0;
  u[1] = context->u1;
  
  int blocks4=((n-2)/4)*4;

  Vec4d QUARTER=Vec4d(0.25);
  Vec4d A,B,C,D,E;

  // do iterations
  for (int kk=0; kk<context->iterations; kk+=K)
    for (int m=2; m<=n+K-2; m++)
      for (int k=std::max(1,m-n+2); k<=std::min(K,m-1); k++)
	{
	  int i1=m-k;
	  int dst=k%2;
	  int src=1-dst;
	  int istart=i1*n+1;
	  int iend=istart+blocks4;
	  for (int index=istart; index<iend; index+=4)
	    {
	      A = Vec4d(0.0);
	      B.load(&u[src][index-n]);
	      C.load(&u[src][index-1]);
	      D.load(&u[src][index+1]);
	      E.load(&u[src][index+n]);
	      A = mul_add(QUARTER,B,A);
	      A = mul_add(QUARTER,C,A);
	      A = mul_add(QUARTER,D,A);
	      A = mul_add(QUARTER,E,A);
	      A.store(&u[dst][index]);
	    }
	  for (int index=iend; index<n-1; index++)
	    u[dst][index] = 0.25*(u[src][index-n]+u[src][index-1]+u[src][index+1]+u[src][index+n]);
	}
}


// main function runs the experiments and outputs results as csv
int main (int argc, char** argv)
{
  // read parameters
  int n=1024;
  int iterations=1000;
  if (argc!=3)
    {
      std::cout << "usage: ./jacobi_vanilla <size> <iterations>"
                << std::endl;
      exit(1);
    }
  sscanf(argv[1],"%d",&n);
  sscanf(argv[2],"%d",&iterations);
  // std::cout << "jacobi_vanilla: n=" << n
  //        << " iterations=" << iterations
  //        << " memory (mbytes)=" << (n*n)*8.0*2.0/1024.0/1024.0
  //        << std::endl;

  // check sizes
  if (K%2==1)
    {
      std::cout << "K must be even" << std::endl;
      exit(1);
    }
  if (iterations%K!=0)
    {
      std::cout << "iterations must be a multiple of K" << std::endl;
      exit(1);
    }
  
  // get global context shared by aall threads
  auto context = std::make_shared<GlobalContext>(n);
  context->iterations=iterations;

  // allocate aligned arrays
  context->u0 = new (std::align_val_t(64)) double [n*n];
  context->u1 = new (std::align_val_t(64)) double [n*n];

  // fill boundary values and initial values
  auto g = [&](int i0, int i1) { return  (i0>0 && i0<n-1 && i1>0 && i1<n-1)
                                 ? 0.0 : ((double)(i0+i1))/n; };

  std::cout << "N,";
  std::cout << "vanilla,";
  std::cout << "blocked,";
  std::cout << "wave,";
  std::cout << "vectorized,";
  std::cout << "vectorized_wave";
  std::cout << std::endl;
  std::cout << n*n;

  // warmup
  for (int i1=0; i1<n; i1++)
    for (int i0=0; i0<n; i0++)
      context->u0[i1*n+i0] = context->u1[i1*n+i0] = g(i0,i1);
  auto start = get_time_stamp();
  jacobi_blocked(context);
  auto stop = get_time_stamp();
  double elapsed = get_duration_seconds(start,stop);
  double updates = context->iterations;
  updates *= (n-2)*(n-2);
  // std::cout << "," << updates/elapsed/1e9;

    // vanilla
  for (int i1=0; i1<n; i1++)
    for (int i0=0; i0<n; i0++)
      context->u0[i1*n+i0] = context->u1[i1*n+i0] = g(i0,i1);
  start = get_time_stamp();
  jacobi_vanilla(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start,stop);
  updates = context->iterations;
  updates *= (n-2)*(n-2);
  std::cout << "," << updates/elapsed/1e9;

  // blocked
  for (int i1=0; i1<n; i1++)
    for (int i0=0; i0<n; i0++)
      context->u0[i1*n+i0] = context->u1[i1*n+i0] = g(i0,i1);
  start = get_time_stamp();
  jacobi_blocked(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start,stop);
  std::cout << "," << updates/elapsed/1e9;
  
  // wave
  for (int i1=0; i1<n; i1++)
    for (int i0=0; i0<n; i0++)
      context->u0[i1*n+i0] = context->u1[i1*n+i0] = g(i0,i1);
  start = get_time_stamp();
  jacobi_wave(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start,stop);
  std::cout << "," << updates/elapsed/1e9;

  // vectorized
  for (int i1=0; i1<n; i1++)
    for (int i0=0; i0<n; i0++)
      context->u0[i1*n+i0] = context->u1[i1*n+i0] = g(i0,i1);
  start = get_time_stamp();
  jacobi_vectorized(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start,stop);
  std::cout << "," << updates/elapsed/1e9;

  // vectorized wave
  for (int i1=0; i1<n; i1++)
    for (int i0=0; i0<n; i0++)
      context->u0[i1*n+i0] = context->u1[i1*n+i0] = g(i0,i1);
  start = get_time_stamp();
  jacobi_vectorized_wave(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start,stop);
  std::cout << "," << updates/elapsed/1e9;

  std::cout << std::endl;

  // deallocate arrays
  delete[] context->u1;
  delete[] context->u0;
  
  return 0;
}
