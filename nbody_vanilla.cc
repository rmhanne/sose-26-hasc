#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>

#include "nbody_io.hh"
#include "nbody_generate.hh"
#include "time_experiment.hh"

#ifndef __GNUC__
#define __restrict__
#endif

// basic data type for position, velocity, acceleration
typedef double double3[4]; // pad up for later use with SIMD
const int B = 32;					 // block size for tiling

/*const double gamma = 6.674E-11;*/
const double G = 1.0;
const double epsilon2 = 1E-10;

/** \brief compute acceleration vector from position and masses
 *
 * Executes \sum_{i=0}^{n-1} (n-i-1)*26 = n(n-1)*13
 * flops including 1 division and one square root
 */
void acceleration(int n, double3 *__restrict__ x, double *__restrict__ m, double3 *__restrict__ a)
{
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++)
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

/** \brief compute acceleration vector from position and masses
 *
 * Executes \sum_{i=0}^{n-1} (n-i-1)*26 = n(n-1)*13
 * flops including 1 division and one square root
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
    a[i][0] = a[i][1] = a[i][2] = 0.0;

  // compute new acceleration: n*(n-1)*13 flops
  acceleration_blocked(n, x, m, a);

  // update velocity: 6n flops
  for (int i = 0; i < n; i++)
  {
    v[i][0] += dt * a[i][0];
    v[i][1] += dt * a[i][1];
    v[i][2] += dt * a[i][2];
  }
}

int main(int argc, char **argv)
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
    x = new (std::align_val_t{64}) double3[n];
    v = new (std::align_val_t{64}) double3[n];
    m = new (std::align_val_t{64}) double[n];
    // x = static_cast<double3*>(calloc(n,sizeof(double3)));
    // v = static_cast<double3*>(calloc(n,sizeof(double3)));
    // m = static_cast<double*>(calloc(n,sizeof(double)));
    read_vtk_file_double(file, n, x, v, m, &t, &dt);
    fclose(file);
    k *= mod; // adjust step number
    std::cout << "loaded " << n << "bodies from file " << std::string(basename) << std::endl;
  }
  // set up computation from initial condition
  if (argc == 6)
  {
    x = new (std::align_val_t{64}) double3[n];
    v = new (std::align_val_t{64}) double3[n];
    m = new (std::align_val_t{64}) double[n];

    // x = static_cast<double3*>(calloc(n,sizeof(double3)));
    // v = static_cast<double3*>(calloc(n,sizeof(double3)));
    // m = static_cast<double*>(calloc(n,sizeof(double)));
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

  // allocate acceleration vector
  a = new (std::align_val_t{64}) double3[n];
  // a = static_cast<double3*>(calloc(n,sizeof(double3)));

  // initialize timestep and write first file
  std::cout << "step=" << k << " finalstep=" << timesteps << " time=" << t << " dt=" << dt << std::endl;
  double elapsed_total = 0.0;
  auto start = get_time_stamp();

  // do time steps
  k += 1;
  int cnt = 0;
  for (; k <= timesteps; k++)
  {
    leapfrog(n, dt, x, v, m, a);
    t += dt;
    cnt++;
    if (k % mod == 0)
    {
      auto stop = get_time_stamp();
      double elapsed = get_duration_seconds(start, stop);
      elapsed_total += elapsed;
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

  double flop = cnt * (13.0 * n * (n - 1.0) + 12.0 * n);
  printf("%g seconds for %g ops = %g GFLOPS\n", elapsed_total, flop, flop / elapsed_total / 1E9);

  delete[] x;
  delete[] v;
  delete[] m;
  delete[] a;

  return 0;
}
