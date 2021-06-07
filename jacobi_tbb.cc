#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <stdio.h>
#include "time_experiment.hh"
#include <oneapi/tbb.h>

#ifndef __GNUC__
#define __restrict__
#endif

// a global mutex for protecting output
oneapi::tbb::spin_mutex mutex{};

// structure for passing parameters
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

//=================================================
// the vanilla version 
//=================================================

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


//=================================================
// the task-based version
//=================================================

// standard function_node for Lambda
using Lambda_node = oneapi::tbb::flow::function_node<
  std::tuple<int,int>,int>;
template<int K>
class Lambda
{
  int n;
  double* u[2];
  int i;
  int i1start;
public:
  Lambda (int _n, double* u0, double* u1, int _i)
    : n(_n), i(_i)
  {
    u[0] = u0; u[1] = u1;
    i1start = 1+i*K;
  }
  int operator() (const std::tuple<int,int>& in)
  {
    for (int r=0; r<K/2; ++r)
      for (int k=0; k<=r; ++k)
        {
          int src = k%2;
          int dst = 1-src;
          int i1 = i1start+(r-k)*2+k;
          for (int i0=1; i0<n-1; i0++)
            u[dst][i1*n+i0] = 0.25*(u[src][i1*n+i0-n]
                                    +u[src][i1*n+i0-1]
                                    +u[src][i1*n+i0+1]
                                    +u[src][i1*n+i0+n]);
          i1++;
          for (int i0=1; i0<n-1; i0++)
            u[dst][i1*n+i0] = 0.25*(u[src][i1*n+i0-n]
                                    +u[src][i1*n+i0-1]
                                    +u[src][i1*n+i0+1]
                                    +u[src][i1*n+i0+n]);
        }
    if (K/2%2==0) std::swap(u[0],u[1]);
    return std::get<0>(in);
  }
};


// multifunction_node with two outputs for V, L, R
using V_node = oneapi::tbb::flow::multifunction_node<
  std::tuple<int,int>, std::tuple<int,int>>;
template<int K>
class V
{
  int n;
  double* u[2];
  int steps;
  int i;
  int i1start;
public:
  V (int _n, double* u0, double* u1, int _steps, int _i)
    : n(_n), steps(_steps), i(_i)
  {
    u[0] = u1; u[1] = u0;
    i1start = K/2+1+i*K;
  }
  void operator() (const std::tuple<int,int>& in,
                   V_node::output_ports_type &op)
  {
    for (int r=0; r<K/2; ++r)
      for (int s=0; s<K/2-r; ++s)
        {
          int k = r+s;
          int i1 = i1start+K/2-1+r-s;
          int src=k%2;
          int dst=1-src;
          for (int i0=1; i0<n-1; i0++)
            u[dst][i1*n+i0] = 0.25*(u[src][i1*n+i0-n]+u[src][i1*n+i0-1]
                                    +u[src][i1*n+i0+1]+u[src][i1*n+i0+n]);
          i1++;
          for (int i0=1; i0<n-1; i0++)
            u[dst][i1*n+i0] = 0.25*(u[src][i1*n+i0-n]+u[src][i1*n+i0-1]
                                    +u[src][i1*n+i0+1]+u[src][i1*n+i0+n]);        
        }
    if (K/2%2==0) std::swap(u[0],u[1]);
    int step = std::get<0>(in);
    if (step+1<=steps)
      {
        std::get<0>(op).try_put(step+1);
        std::get<1>(op).try_put(step+1);
      }
  }
};

using LR_node = oneapi::tbb::flow::multifunction_node<int, std::tuple<int,int>>;
template<int K>
class L
{
  int n;
  double* u[2];
  int steps;
  int i1start;
public:
  L (int _n, double* u0, double* u1, int _steps)
    : n(_n), steps(_steps)
  {
    u[0] = u1; u[1] = u0;
    i1start = 1;
  }
  void operator() (const int& step, LR_node::output_ports_type &op)
  {
    const int w = K/2-1;
    for (int k=0; k<K/2; ++k)
      {
        int src=k%2;
        int dst=1-src;
        for (int i1=i1start; i1<i1start+k+1; i1++)
          for (int i0=1; i0<n-1; i0++)
            u[dst][i1*n+i0] = 0.25*(u[src][i1*n+i0-n]+u[src][i1*n+i0-1]
                                    +u[src][i1*n+i0+1]+u[src][i1*n+i0+n]);
      }
    if (K/2%2==0) std::swap(u[0],u[1]);

    // oneapi::tbb::spin_mutex::scoped_lock lock{mutex};
    // std::cout << "Left: step " << step << std::endl;
    if (step+1<=steps)
      std::get<0>(op).try_put(step+1);
  }
};
template<int K>
class R
{
  int n;
  double* u[2];
  int steps;
  int i1end;
public:
  R (int _n, double* u0, double* u1, int _steps)
    : n(_n), steps(_steps)
  {
    u[0] = u1; u[1] = u0;
    i1end = n-1;
  }
  void operator() (const int& step, LR_node::output_ports_type &op)
  {
    const int w = K/2-1;
    for (int k=0; k<K/2; ++k)
      {
        int src=k%2;
        int dst=1-src;
        for (int i1=i1end-1-k; i1<i1end; i1++)
          for (int i0=1; i0<n-1; i0++)
            u[dst][i1*n+i0] = 0.25*(u[src][i1*n+i0-n]+u[src][i1*n+i0-1]
                                    +u[src][i1*n+i0+1]+u[src][i1*n+i0+n]);
      }
    if (K/2%2==0) std::swap(u[0],u[1]);

    // oneapi::tbb::spin_mutex::scoped_lock lock{mutex};
    // std::cout << "Right: step " << step << std::endl; 
    if (step+1<=steps)
      std::get<1>(op).try_put(step+1);
  }
};

template<int K>
void jacobi_tbb (std::shared_ptr<GlobalContext> context)
{
  int steps = context->iterations/(K/2+1);
  int n = context->n;
  double* uold = context->u0;
  double* unew = context->u1;
  const int tiles = (n-2)/K;

  // our graph
  oneapi::tbb::flow::graph g;

  // make nodes
  // std::cout << std::endl;
  // std::cout << "making nodes" << std::endl;
  std::vector<Lambda_node> lambda_nodes;
  for (int i=0; i<tiles; ++i)
    lambda_nodes.push_back(Lambda_node{g,1,Lambda<K>(n,uold,unew,i)});
  using two_join_node = oneapi::tbb::flow::join_node<std::tuple<int,int>,
						     oneapi::tbb::flow::queueing>;
  std::vector<two_join_node> lambda_join_nodes;
  for (int i=0; i<tiles; ++i)
    lambda_join_nodes.push_back(two_join_node{g});
  std::vector<V_node> v_nodes;
  for (int i=0; i<tiles-1; ++i)
    v_nodes.push_back(V_node{g,1,V<K>(n,uold,unew,steps,i)});
  std::vector<two_join_node> v_join_nodes;
  for (int i=0; i<tiles-1; ++i)
    v_join_nodes.push_back(two_join_node{g});
  LR_node left(g,1,L<K>(n,uold,unew,steps));
  LR_node right(g,1,R<K>(n,uold,unew,steps));
  
  // make edges
  // std::cout << "making edges" << std::endl;
  for (int i=0; i<tiles; ++i)
    oneapi::tbb::flow::make_edge(lambda_join_nodes[i],lambda_nodes[i]);
  for (int i=0; i<tiles-1; ++i)
    oneapi::tbb::flow::make_edge(v_join_nodes[i],v_nodes[i]);
  for (int i=0; i<tiles-1; ++i)
    oneapi::tbb::flow::make_edge(lambda_nodes[i],
                                 oneapi::tbb::flow::input_port<0>(v_join_nodes[i]));
  for (int i=0; i<tiles-1; ++i)
    oneapi::tbb::flow::make_edge(lambda_nodes[i+1],
                                 oneapi::tbb::flow::input_port<1>(v_join_nodes[i]));
  for (int i=0; i<tiles-1; ++i)
    oneapi::tbb::flow::make_edge(oneapi::tbb::flow::output_port<0>(v_nodes[i]),
                                 oneapi::tbb::flow::input_port<1>(lambda_join_nodes[i]));
  for (int i=0; i<tiles-1; ++i)
    oneapi::tbb::flow::make_edge(oneapi::tbb::flow::output_port<1>(v_nodes[i]),
                                 oneapi::tbb::flow::input_port<0>(lambda_join_nodes[i+1]));

  oneapi::tbb::flow::make_edge(lambda_nodes[0],left);
  oneapi::tbb::flow::make_edge(oneapi::tbb::flow::output_port<0>(left),
                               oneapi::tbb::flow::input_port<0>(lambda_join_nodes[0]));
  oneapi::tbb::flow::make_edge(lambda_nodes[tiles-1],right);
  oneapi::tbb::flow::make_edge(oneapi::tbb::flow::output_port<1>(right),
                               oneapi::tbb::flow::input_port<1>(lambda_join_nodes[tiles-1]));

  // // initial messages
  // std::cout << "making nodes" << std::endl;
  std::tuple<int,int> first_iter(1,1);
  for (int i=0; i<tiles; ++i)
    lambda_nodes[i].try_put(first_iter);

  // std::cout << "starting graph" << std::endl;
  g.wait_for_all();
}


// main function runs the experiments and outputs results as csv
int main (int argc, char** argv)
{  
  // compile time known parameters
  const int K=12; // parameter for wave front and TBB version; K/2 should be odd

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

  // check sizes
  if (K%2==1)
    {
      std::cout << "K must be even, K=" << K << std::endl;
      exit(1);
    }
  if ((n-2)%K!=0)
    {
      std::cout << "K must divide n-2, K=" << K << " n-2=" << n-2 << std::endl;
      exit(1);
    }
  if (iterations%(K/2+1)!=0)
    {
      std::cout << "iterations must be a multiple of (K/2+1), K=" << K << " iterations=" << iterations << std::endl;
      exit(1);
    }
  
  // get global context shared by aall threads
  auto context = std::make_shared<GlobalContext>(n);
  context->iterations=iterations;

  // allocate aligned arrays
  context->u0 = new (std::align_val_t(64)) double [n*n];
  context->u1 = new (std::align_val_t(64)) double [n*n];

  // fill boundary values and initial values
  auto bndcond = [&](int i0, int i1) { return  (i0>0 && i0<n-1 && i1>0 && i1<n-1)
                                 ? 0.0 : ((double)(i0+i1))/n; };

  std::cout << "N,";
  std::cout << "vanilla,";
  std::cout << "tbb";
  std::cout << std::endl;
  std::cout << n*n;
  
  // warmup
  for (int i1=0; i1<n; i1++)
    for (int i0=0; i0<n; i0++)
      context->u0[i1*n+i0] = context->u1[i1*n+i0] = bndcond(i0,i1);
  auto start = get_time_stamp();
  jacobi_vanilla(context);
  auto stop = get_time_stamp();
  double elapsed = get_duration_seconds(start,stop);
  double updates = context->iterations;
  updates *= (n-2)*(n-2);
  // std::cout << "," << updates/elapsed/1e9;

  // vanilla
  for (int i1=0; i1<n; i1++)
    for (int i0=0; i0<n; i0++)
      context->u0[i1*n+i0] = context->u1[i1*n+i0] = bndcond(i0,i1);
  start = get_time_stamp();
  jacobi_vanilla(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start,stop);
  updates = context->iterations;
  updates *= (n-2)*(n-2);
  std::cout << "," << updates/elapsed/1e9;

  // tbb
  for (int i1=0; i1<n; i1++)
    for (int i0=0; i0<n; i0++)
      context->u0[i1*n+i0] = context->u1[i1*n+i0] = bndcond(i0,i1);
  start = get_time_stamp();
  jacobi_tbb<K>(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start,stop);
  updates = context->iterations;
  updates *= (n-2)*(n-2);
  std::cout << "," << updates/elapsed/1e9;
  
  std::cout << std::endl;

  // deallocate arrays
  delete[] context->u1;
  delete[] context->u0;
  
  return 0;
}
