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
#include <wasm_simd128.h>

#ifndef __GNUC__
#define __restrict__
#endif

// basic data type for position, velocity, acceleration
const int M = 3;
typedef double double3[M]; // pad up for later use with SIMD
const int B = 64;					 // block size for tiling

/*const double gamma = 6.674E-11;*/
const double G = 1.0;
const double epsilon2 = 1E-10;

/** \brief compute acceleration vector from position and masses
 *
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


void transpose(std::array<v128_t, 2>& A) {
  // I don't know how to shuffle in wasm :S so I'll do it in simple code, it's 2x2 anyways
  std::array<std::array<double, 2>, 2> B;
  wasm_v128_store(&B[0], A[0]);
  wasm_v128_store(&B[1], A[1]);
  B[0][1] = std::exchange(B[1][0], B[0][1]);
  A[0] = wasm_v128_load(&B[0]);
  A[1] = wasm_v128_load(&B[1]);
}

/** \brief compute acceleration vector from position and masses
 *
 * Vectorized version blocked: works on 2x2 masses; symmetry is exploited; transpose for scalar factors needed
 * Executes \sum_{i=0}^{n-1} (n-i-1)*26 = n(n-1)*13
 * flops including 1 division and one square root
 */
void
acceleration_wasm(int n,
                  double* __restrict__ x,
                  double* __restrict__ m,
                  double* __restrict__ a)
{
  const size_t W = 2;   // SIMD width
  using VecWd = v128_t; // SIMD type

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
          Xi[0] = wasm_v128_load(&x[i]);
          Xi[1] = wasm_v128_load(&x[n + i]);
          Xi[2] = wasm_v128_load(&x[2 * n + i]);
          for (int j = J; j < J + B; j += W) {
            // load acceleration for i-index
            std::array<VecWd, 3> A;
            A[0] = wasm_v128_load(&a[i]);
            A[1] = wasm_v128_load(&a[n + i]);
            A[2] = wasm_v128_load(&a[2 * n + i]);
            std::array<VecWd, W> N;
            for (int w = 0; w != W; w += 1) {
              std::array<VecWd, 3> Dj{ wasm_f64x2_splat(x[j + w]),
                                       wasm_f64x2_splat(x[n + j + w]),
                                       wasm_f64x2_splat(x[2 * n + j + w]) };
              Dj[0] = wasm_f64x2_sub(Dj[0], Xi[0]);
              Dj[1] = wasm_f64x2_sub(Dj[1], Xi[1]);
              Dj[2] = wasm_f64x2_sub(Dj[2], Xi[2]);
              N[w] = wasm_f64x2_splat(epsilon2);
              N[w] = wasm_f64x2_add(wasm_f64x2_mul(Dj[0], Dj[0]), N[w]);
              N[w] = wasm_f64x2_add(wasm_f64x2_mul(Dj[1], Dj[1]), N[w]);
              N[w] = wasm_f64x2_add(wasm_f64x2_mul(Dj[2], Dj[2]), N[w]);
              // calculate row of N := G/d_ji^3 (note that this is transposed -> ji vs ij!)
              N[w] =
                wasm_f64x2_div(wasm_f64x2_splat(G),
                               wasm_f64x2_mul(N[w], wasm_f64x2_sqrt(N[w])));
              // update acceleration for mass i (note that we take adventage of the transposed N)
              VecWd M = wasm_f64x2_splat(m[i + w]);
              A[0] = wasm_f64x2_add(
                wasm_f64x2_mul(wasm_f64x2_mul(M, Dj[0]), N[w]), A[0]);
              A[1] = wasm_f64x2_add(
                wasm_f64x2_mul(wasm_f64x2_mul(M, Dj[1]), N[w]), A[1]);
              A[2] = wasm_f64x2_add(
                wasm_f64x2_mul(wasm_f64x2_mul(M, Dj[2]), N[w]), A[2]);
            }
            // now we need to transpose N
            transpose(N); // N_ji -> N_ij
            // store acceleration for i-index
            wasm_v128_store(&a[i], A[0]);
            wasm_v128_store(&a[n + i], A[1]);
            wasm_v128_store(&a[2 * n + i], A[2]);
            // load acceleration for j-index
            A[0] = wasm_v128_load(&a[j]);
            A[1] = wasm_v128_load(&a[n + j]);
            A[2] = wasm_v128_load(&a[2 * n + j]);

            std::array<VecWd, 3> Xj;
            Xj[0] = wasm_v128_load(&x[j]);
            Xj[1] = wasm_v128_load(&x[n + j]);
            Xj[2] = wasm_v128_load(&x[2 * n + j]);
            // update acceleration for mass j (note that we need the original N non-transposed version)
            for (int w = 0; w != W; w += 1) {
              std::array<VecWd, 3> Di{ wasm_f64x2_splat(x[i + w]),
                                       wasm_f64x2_splat(x[n + i + w]),
                                       wasm_f64x2_splat(x[2 * n + i + w]) };
              Di[0] = wasm_f64x2_sub(Di[0], Xj[0]);
              Di[1] = wasm_f64x2_sub(Di[1], Xj[1]);
              Di[2] = wasm_f64x2_sub(Di[2], Xj[2]);
              VecWd M = wasm_f64x2_splat(m[j + w]);
              A[0] = wasm_f64x2_add(
                wasm_f64x2_mul(wasm_f64x2_mul(M, Di[0]), N[w]), A[0]);
              A[1] = wasm_f64x2_add(
                wasm_f64x2_mul(wasm_f64x2_mul(M, Di[1]), N[w]), A[1]);
              A[2] = wasm_f64x2_add(
                wasm_f64x2_mul(wasm_f64x2_mul(M, Di[2]), N[w]), A[2]);
            }
            // store acceleration for j-index
            wasm_v128_store(&a[j], A[0]);
            wasm_v128_store(&a[n + j], A[1]);
            wasm_v128_store(&a[2 * n + j], A[2]);
          }
        }
      }
    }
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
  // acceleration(n,x,m,a);
  // acceleration_blocked(n,x,m,a);
  acceleration_wasm(n, x, m, a);

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
  else // using default initial values so that the html version runs without arguments
  {
    std::cout << "running with following arguments: wasm 3200 1000 0.1 10 " << std::endl;
    strcpy(basename, "wasm");
    n = 3200;
    timesteps = 1000;
    dt = 0.1;
    mod = 10;
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
      // this hangs in wasm... or I am not patient enough ¯\_(ツ)_/¯
      // file = fopen(name, "w");
      // copy(x, X, n);
      // copy(v, V, n);
      // write_vtk_file_double(file, n, x, v, m, t, dt);
      // fclose(file);

      start = get_time_stamp();
    }
  }

  return 0;
}
