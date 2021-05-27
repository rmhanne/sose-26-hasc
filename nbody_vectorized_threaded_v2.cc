#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <array>
#include <memory>

#include "nbody_io.hh"
#include "nbody_generate.hh"
#include "time_experiment.hh"

#include "vcl/vectorclass.h"


#define VECTORIZED_VERSION_1

/*const double gamma = 6.674E-11;*/
const double G = 1.0;
const double epsilon2 = 1E-10;

// basic data type for position, velocity, acceleration
const int M=4;
typedef double double3[M]; // pad up for later use with SIMD
const int B=32; // block size for tiling


// Note: Since we want to load these points as Vec4d it makes sense to have an
// alignment of 64. Since C++-17 std::vector will use an alignment that fits to
// the value type so the data in a std::vector<PointPadded> will also have an
// alignment of 64!
struct alignas (64) PointPadded {
  // Store a vector of size 4. We only ever use the first three variables but
  // this way we can load a Vec4d and simply get a 0 in the last SIMD lane.
  std::array<double, 4> _x {};

  double& operator[](int i) {return _x[i];}

  const double& operator[](int i) const {return _x[i];}

  PointPadded& operator+=(const PointPadded& other) {
    for (std::size_t i=0; i<_x.size(); ++i){
      _x[i] += other._x[i];
    }
    return *this;
  }

  PointPadded operator+(const PointPadded& other) const {
    PointPadded output = *this;
    output += other;
    return output;
  }
};


template <int B>
void acceleration_vectorized_parallel_privatization_interleaved_kernel(const std::vector<PointPadded>& x,
                                                                       const std::vector<double>& m,
                                                                       std::vector<PointPadded>& a,
                                                                       const int rank,
                                                                       const int number_of_threads) {
  const int n = x.size();

#ifdef VECTORIZED_VERSION_1

  Vec4d A0, A1;
  Vec4d D0, D1, D2, D3;  // distances
  Vec4d E0, E1, E2, E3;  // distances
  Vec4d S0, S, U, T0, T1, T2;

  int thread_index = 0;
  for (int I = 0; I < n; I += B, thread_index=(thread_index+1)%number_of_threads) {

    if (rank == thread_index){

      // Diagonal block (I,I)
      //
      // Only look at upper right part of the block. This is not vectorized, as
      // the bulk of the work is happening in the other blocks and visiting only
      // the upper right parts makes things more fiddely.
      for (int i = I; i < I + B; i++) {
        for (int j = i + 1; j < I + B; j++) {
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
      // blocks J>I
      for (int J = I + B; J < n; J += B) {
        for (int i = I; i < I + B; i += 2) {
          for (int j = J; j < J + B; j += 4) {
            // distances 2x4 masses
            T0.load(&x[i][0]);
            T1.load(&x[i + 1][0]);
            D0.load(&x[j][0]);
            D1.load(&x[j + 1][0]);
            D2.load(&x[j + 2][0]);
            D3.load(&x[j + 3][0]);
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
            S = blend4<0, 4, 1, 5>(D0, D1);
            U = blend4<0, 4, 1, 5>(D2, D3);
            T0 = blend4<0, 1, 4, 5>(S, U);  // all 0 components
            T1 = blend4<2, 3, 6, 7>(S, U);  // all 1 components
            S = blend4<2, 6, V_DC, V_DC>(D0, D1);
            U = blend4<2, 6, V_DC, V_DC>(D2, D3);
            T2 = blend4<0, 1, 4, 5>(S, U);  // all 2 components

            // the norms first batch
            S0 = Vec4d(epsilon2);
            S0 = mul_add(T0, T0, S0);
            S0 = mul_add(T1, T1, S0);
            S0 = mul_add(T2, T2, S0);

            // transpose second batch
            S = blend4<0, 4, 1, 5>(E0, E1);
            U = blend4<0, 4, 1, 5>(E2, E3);
            T0 = blend4<0, 1, 4, 5>(S, U);  // all 0 components
            T1 = blend4<2, 3, 6, 7>(S, U);  // all 1 components
            S = blend4<2, 6, V_DC, V_DC>(E0, E1);
            U = blend4<2, 6, V_DC, V_DC>(E2, E3);
            T2 = blend4<0, 1, 4, 5>(S, U);  // all 2 components

            // the norms second batch
            S = Vec4d(epsilon2);
            S = mul_add(T0, T0, S);
            S = mul_add(T1, T1, S);
            S = mul_add(T2, T2, S);

            // sqrt first batch
            U = sqrt(S0);
            U *= S0;  // now U contains r^3
            T0 = Vec4d(G);
            S0 =
              T0 / U;  // now S is the inverse factor for four pairs first batch

            // sqrt second batch
            U = sqrt(S);
            U *= S;  // now U contains r^3
            S = T0 /
              U;  // now S is the inverse factor for four pairs second batch

            // update both rows from all columns
            A0.load(&a[i][0]);
            A1.load(&a[i + 1][0]);
            T2 = Vec4d(m[j]);              // mass col j
            U = permute4<0, 0, 0, 0>(S0);  // scalar factor column j
            T0 = T2 * U;
            A0 = mul_add(T0, D0, A0);
            U = permute4<0, 0, 0, 0>(S);  // scalar factor column j
            T1 = T2 * U;
            A1 = mul_add(T1, E0, A1);

            T2 = Vec4d(m[j + 1]);          // mass col j+1
            U = permute4<1, 1, 1, 1>(S0);  // scalar factor column j
            T0 = T2 * U;
            A0 = mul_add(T0, D1, A0);
            U = permute4<1, 1, 1, 1>(S);  // scalar factor column j
            T1 = T2 * U;
            A1 = mul_add(T1, E1, A1);

            T2 = Vec4d(m[j + 2]);          // mass col j+2
            U = permute4<2, 2, 2, 2>(S0);  // scalar factor column j
            T0 = T2 * U;
            A0 = mul_add(T0, D2, A0);
            U = permute4<2, 2, 2, 2>(S);  // scalar factor column j
            T1 = T2 * U;
            A1 = mul_add(T1, E2, A1);

            T2 = Vec4d(m[j + 3]);          // mass col j+3
            U = permute4<3, 3, 3, 3>(S0);  // scalar factor column j
            T0 = T2 * U;
            A0 = mul_add(T0, D3, A0);
            U = permute4<3, 3, 3, 3>(S);  // scalar factor column j
            T1 = T2 * U;
            A1 = mul_add(T1, E3, A1);
            A0.store(&a[i][0]);
            A1.store(&a[i + 1][0]);

            // now update all columns from both rows
            T0 = Vec4d(m[i]);      // row 0 mass
            T1 = Vec4d(m[i + 1]);  // row 1 mass
            A0.load(&a[j][0]);
            U = permute4<0, 0, 0, 0>(S0);  // scalar factor column j row 0
            T2 = T0 * U;
            A0 = nmul_add(T2, D0, A0);
            U = permute4<0, 0, 0, 0>(S);  // scalar factor column j row 1
            T2 = T1 * U;
            A0 = nmul_add(T2, E0, A0);
            A0.store(&a[j][0]);

            A0.load(&a[j + 1][0]);
            U = permute4<1, 1, 1, 1>(S0);  // scalar factor column j+1 row 0
            T2 = T0 * U;
            A0 = nmul_add(T2, D1, A0);
            U = permute4<1, 1, 1, 1>(S);  // scalar factor column j+1 row 1
            T2 = T1 * U;
            A0 = nmul_add(T2, E1, A0);
            A0.store(&a[j + 1][0]);

            A0.load(&a[j + 2][0]);
            U = permute4<2, 2, 2, 2>(S0);  // scalar factor column j+2 row 0
            T2 = T0 * U;
            A0 = nmul_add(T2, D2, A0);
            U = permute4<2, 2, 2, 2>(S);  // scalar factor column j+2 row 1
            T2 = T1 * U;
            A0 = nmul_add(T2, E2, A0);
            A0.store(&a[j + 2][0]);

            A0.load(&a[j + 3][0]);
            U = permute4<3, 3, 3, 3>(S0);  // scalar factor column j+3 row 0
            T2 = T0 * U;
            A0 = nmul_add(T2, D3, A0);
            U = permute4<3, 3, 3, 3>(S);  // scalar factor column j+3 row 1
            T2 = T1 * U;
            A0 = nmul_add(T2, E3, A0);
            A0.store(&a[j + 3][0]);
          }
        }
      }
    }
  }

#else

  Vec4d Ai; // acceleration row
  Vec4d D0,D1,D2,D3; // distances
  Vec4d A0,A1,A2,A3; // accelerations columns
  Vec4d S,U,T0,T1,T2; // temporaries

  int thread_index = 0;
  for (int I = 0; I < n; I += B, thread_index=(thread_index+1)%number_of_threads) {

    if (rank == thread_index){

      // Diagonal block (I,I)
      //
      // Only look at upper right part of the block. This is not vectorized, as
      // the bulk of the work is happening in the other blocks and visiting only
      // the upper right parts makes things more fiddely.
      for (int i = I; i < I + B; i++) {
        for (int j = i + 1; j < I + B; j++) {
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
      // blocks J>I
      for (int J = I + B; J < n; J += B) {
        for (int j = J; j < J + B; j += 4) {
          // update accelerations of four particles
          A0.load(&a[j][0]);
          A1.load(&a[j+1][0]);
          A2.load(&a[j+2][0]);
          A3.load(&a[j+3][0]);

          // loop over particles in row
          for (int i = I; i < I + B; i += 1) {
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
#endif
}

template <int B>
void leapfrog(const std::size_t number_of_threads,
              const double dt,
              std::vector<PointPadded>& x,
              std::vector<PointPadded>& v,
              const std::vector<double>& m,
              std::vector<PointPadded>& a
              ) {
  const std::size_t n = x.size();

  std::vector<std::thread> threads(number_of_threads);
  if (n % number_of_threads != 0) {
    std::cout << "n not multiple of number of threads" << std::endl;
    exit(1);
  }

  // update position: 6n flops
  // save and clear acceleration
  int current_index = 0;
  int elements = n / number_of_threads;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread([=, &x, &v, &a]() {
      for (int i = current_index; i < current_index + elements; i++) {
        x[i][0] += dt * v[i][0];
        x[i][1] += dt * v[i][1];
        x[i][2] += dt * v[i][2];
        a[i][0] = a[i][1] = a[i][2] = 0.0;
      }
    });
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();

  // Local variable for privatization
  std::vector<std::vector<PointPadded>> local_a(number_of_threads);
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    local_a[i].resize(n);
  }

  // compute new acceleration: n*(n-1)*13 flops
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread(acceleration_vectorized_parallel_privatization_interleaved_kernel<B>,
                             std::cref(x), std::cref(m), std::ref(local_a[i]), i, number_of_threads);
  }
  for (auto &entry : threads) entry.join();

  // Reduction
  const std::size_t dimension = 3;
  for (std::size_t i=0; i<number_of_threads; ++i){
    for (std::size_t j=0; j<n; ++j){
      for (std::size_t k=0; k<dimension; ++k){
        a[j][k] += local_a[i][j][k];
      }
    }
  }

  // update velocity: 6n flops
  current_index = 0;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread([=, &x, &v, &a]() {
      for (int i = current_index; i < current_index + elements; i++) {
        v[i][0] += dt * a[i][0];
        v[i][1] += dt * a[i][1];
        v[i][2] += dt * a[i][2];
      }
    });
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();
}


template <typename T, typename T2>
void initialize(std::vector<T>& x_vec,
                std::vector<T>& v_vec,
                std::vector<double>& m_vec,
                std::vector<T2>& a_vec
                ){

  const int n = x_vec.size();

  const int M = 4;
  typedef double double3[M];
  double3 *x = new (std::align_val_t{64}) double3[n]();
  double3 *v = new (std::align_val_t{64}) double3[n]();
  double *m = new (std::align_val_t{64}) double[n]();
  double3 *a = new (std::align_val_t{64}) double3[n]();

  // Should already be zero with the above. Just to be sure.
  for (int i = 0; i < n; i++) {
    m[i] = 0.0;
    for (int j = 0; j < M; j++)
      x[i][j] = v[i][j] = a[i][j] = 0.0;
  }

  two_plummer(n, 17, x, v, m, 0);
  for (int i=0; i<n; ++i) {
    m_vec[i] = m[i];
    for (std::size_t j=0; j<x_vec[i]._x.size(); ++j){
      x_vec[i][j] = x[i][j];
      v_vec[i][j] = v[i][j];
      a_vec[i][j] = a[i][j];
    }
  }

  delete[] x;
  delete[] v;
  delete[] m;
  delete[] a;
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
  // Barrier barrier;

  GlobalContext (int P)
    : nthreads(P) //, barrier(P)
  {}
};


void run_simulation(std::shared_ptr<GlobalContext> context) {
  std::vector<PointPadded> x(context->n);
  std::vector<PointPadded> v(context->n);
  std::vector<double> m(context->n);
  std::vector<PointPadded> a(context->n);

  for (std::size_t i=0; i<context->n; ++i){
      m[i] = context->m[i];
    for (std::size_t j=0; j<M; ++j){
      x[i][j] = context->x[i][j];
      v[i][j] = context->v[i][j];
      a[i][j] = context->a[i][j];
    }
  }

  auto start = get_time_stamp();

  // do time steps
  int k = context->k + 1;
  for (; k<=context->timesteps; k++)
    {
      leapfrog<B>(context->nthreads, context->dt, x, v, m, a);
      context->t += context->dt;

      if (k%context->mod==0)
        {
          auto stop = get_time_stamp();
          double elapsed = get_duration_seconds(start,stop);
          double flop = context->mod*(13.0*context->n*(context->n-1.0)+12.0*context->n);
          printf("%g seconds for %g ops = %g GFLOPS\n",elapsed,flop,flop/elapsed/1E9);
          printf("writing %s_%06d.vtk \n",context->basename,k/context->mod);
          sprintf(context->name,"%s_%06d.vtk",context->basename,k/context->mod);
          context->file = fopen(context->name,"w");
          write_vtk_file_double(context->file,context->n,context->x,context->v,context->m,context->t,context->dt);
          fclose(context->file);
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
      context->x = static_cast<double3*>(calloc(context->n,sizeof(double3)));
      context->v = static_cast<double3*>(calloc(context->n,sizeof(double3)));
      context->m = static_cast<double*>(calloc(context->n,sizeof(double)));
      read_vtk_file_double(context->file,context->n,context->x,context->v,context->m,&context->t,&context->dt);
      fclose(context->file);
      context->k *= context->mod; // adjust step number
      std::cout << "loaded " << context->n << "bodies from file " << std::string(context->basename) << std::endl;
    }
  // set up computation from initial condition
  if (argc==7)
    {
      // context->x = static_cast<double3*>(calloc(context->n,sizeof(double3)));
      // context->v = static_cast<double3*>(calloc(context->n,sizeof(double3)));
      // context->m = static_cast<double*>(calloc(context->n,sizeof(double)));
      context->x = new (std::align_val_t{64}) double3[context->n]();
      context->v = new (std::align_val_t{64}) double3[context->n]();
      context->m = new (std::align_val_t{64}) double[context->n]();

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
  // context->a = static_cast<double3*>(calloc(context->n,sizeof(double3)));
  context->a = new (std::align_val_t{64}) double3[context->n]();

  // explicitly fill/clear padded values
  for (int i=0; i<context->n; i++)
    {
      for (int j=3; j<M; j++)
	context->x[i][j] = context->v[i][j] = context->a[i][j] = 0.0;
    }

  // // report alignment
  // std::cout << "x aligned at " << alignment(context->x) << std::endl;
  // std::cout << "v aligned at " << alignment(context->v) << std::endl;
  // std::cout << "a aligned at " << alignment(context->a) << std::endl;
  // std::cout << "m aligned at " << alignment(context->m) << std::endl;

  // report simulation parameters
  std::cout << "step=" << context->k << " finalstep=" << context->timesteps << " time=" << context->t << " dt=" << context->dt << std::endl;

  run_simulation(context);
 
  return 0;
}
