#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <array>

#include <mpi.h>

#include "nbody_io.hh"
#include "nbody_generate.hh"
#include "time_experiment.hh"

#ifndef __GNUC__
#define __restrict__
#endif

// basic data type for position, velocity, acceleration
const int M = 4;
typedef double double3[M]; // pad up for later use with SIMD
const int B = 32;          // block size for tiling

// model parameters
const double G = 1.0;          // gravitational constant
const double epsilon2 = 1E-10; // softening parameter

// have the parallel configuration as global variables
int P;    // total number of MPI processes
int rank; // my number 0 <= rank < P

// data decomposition
int blocks_total;    // the total number of blocks of size B
int blocks_per_rank; // number of blocks of size B in each rank
int masses_per_rank; // number of masses in each rank

// #include "acceleration_async.cc"

// map local block number to global block number
int g(int i, int r)
{
  if (i % 2 == 0)
    return i * P + r;
  else
    return i * P + P - 1 - r;
}

// mpi parallel version based on blocked AoS
void acceleration_blocking(int n, double3 *__restrict__ x, double *__restrict__ m,
                           double3 *__restrict__ a)
{
  // start with interaction of masses in the SAME process
  // loop over rows of blocks
  for (int I = 0; I < masses_per_rank; I += B)
  {
    // block (I,I) diagonal block; requires only upper triangle
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
    // blocks J>I; offdiagonal block. Stars are all different
    for (int J = I + B; J < masses_per_rank; J += B) // loop over columns of blocks
      for (int i = I; i < I + B; i++)
        for (int j = J; j < J + B; j++)
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

  // data buffers
  std::vector<std::array<double, M>> xin(masses_per_rank);
  std::vector<std::array<double, M>> xout(masses_per_rank);
  std::vector<std::array<double, M>> ain(masses_per_rank);
  std::vector<std::array<double, M>> aout(masses_per_rank);
  std::vector<double> min(masses_per_rank);
  std::vector<double> mout(masses_per_rank);

  // prepare outgoing buffers
  for (int i = 0; i < masses_per_rank; i++)
    for (int j = 0; j < M; j++)
      xout[i][j] = x[i][j]; // send my own positions in first round
  for (int i = 0; i < masses_per_rank; i++)
    for (int j = 0; j < M; j++)
      aout[i][j] = 0.0; // other ranks will accumulate to that
  for (int i = 0; i < masses_per_rank; i++)
    mout[i] = m[i]; // send my own masses in first round

  // message tags
  int x_tag = 42;
  int m_tag = 43;
  int a_tag = 44;

  // now do the interactions with masses in other processes
  // Idea:
  //   - communicate in a ring structure, so we see the masses of all other processes
  //   - still use symmetry in evaluation
  //   - accumulate acceleration for the foreign rank and pass it on as well
  for (int round = 1; round < P; ++round)
  {
    // std::cout << rank << ": starting round " << round << std::endl;

    // exchange data with neighbors: send *out to left, rcv *in from right
    if (rank % 2 == 0) // to avoid deadlock; P must be even
    {
      MPI_Send(xout.data(), masses_per_rank * M, MPI_DOUBLE, (rank + P - 1) % P, x_tag, MPI_COMM_WORLD);
      MPI_Recv(xin.data(), masses_per_rank * M, MPI_DOUBLE, (rank + 1) % P, x_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(mout.data(), masses_per_rank, MPI_DOUBLE, (rank + P - 1) % P, m_tag, MPI_COMM_WORLD);
      MPI_Recv(min.data(), masses_per_rank, MPI_DOUBLE, (rank + 1) % P, m_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(aout.data(), masses_per_rank * M, MPI_DOUBLE, (rank + P - 1) % P, a_tag, MPI_COMM_WORLD);
      MPI_Recv(ain.data(), masses_per_rank * M, MPI_DOUBLE, (rank + 1) % P, a_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else
    {
      MPI_Recv(xin.data(), masses_per_rank * M, MPI_DOUBLE, (rank + 1) % P, x_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(xout.data(), masses_per_rank * M, MPI_DOUBLE, (rank + P - 1) % P, x_tag, MPI_COMM_WORLD);
      MPI_Recv(min.data(), masses_per_rank, MPI_DOUBLE, (rank + 1) % P, m_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(mout.data(), masses_per_rank, MPI_DOUBLE, (rank + P - 1) % P, m_tag, MPI_COMM_WORLD);
      MPI_Recv(ain.data(), masses_per_rank * M, MPI_DOUBLE, (rank + 1) % P, a_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(aout.data(), masses_per_rank * M, MPI_DOUBLE, (rank + P - 1) % P, a_tag, MPI_COMM_WORLD);
    }

    // the positions and masses we have in this round are from the following rank
    int s = (rank + round) % P;

    // compute block interactions
    for (int I = 0; I < masses_per_rank; I += B)
      for (int J = 0; J < masses_per_rank; J += B)
        if (g(J / B, s) > g(I / B, rank))
          for (int j = J; j < J + B; j++)
            for (int i = I; i < I + B; i++)
            {
              double d0 = xin[j][0] - x[i][0];
              double d1 = xin[j][1] - x[i][1];
              double d2 = xin[j][2] - x[i][2];
              double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
              double r = sqrt(r2);
              double invfact = G / (r * r2);
              double factori = m[i] * invfact;
              double factorj = m[j] * invfact;
              a[i][0] += factorj * d0;
              a[i][1] += factorj * d1;
              a[i][2] += factorj * d2;
              ain[j][0] -= factori * d0;
              ain[j][1] -= factori * d1;
              ain[j][2] -= factori * d2;
            }

    // now swap in and out; so we send off what we just worked on
    std::swap(xin, xout); // now the received positions are in xin
    std::swap(min, mout); // now the received masses are in min
    std::swap(ain, aout); // now the received accelerations are in min
  }

  // need to shift acceleration one more time, so we get the
  // computations of the others
  if (rank % 2 == 0) // to avoid deadlock; P must be even
  {
    MPI_Send(aout.data(), masses_per_rank * M, MPI_DOUBLE, (rank + P - 1) % P, a_tag, MPI_COMM_WORLD);
    MPI_Recv(ain.data(), masses_per_rank * M, MPI_DOUBLE, (rank + 1) % P, a_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  else
  {
    MPI_Recv(ain.data(), masses_per_rank * M, MPI_DOUBLE, (rank + 1) % P, a_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(aout.data(), masses_per_rank * M, MPI_DOUBLE, (rank + P - 1) % P, a_tag, MPI_COMM_WORLD);
  }

  // accumulate the contributions of others to our acceleration
  for (int i = 0; i < masses_per_rank; i++)
    for (int j = 0; j < 3; j++)
      a[i][j] += ain[i][j];
}

/** \brief do one time step with leapfrog
 *
 * n is the number of masses in this process
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
  acceleration_blocking(n, x, m, a);

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
  double *m_init;     // array for maasses generated
  double3 *x_init;    // array for positions generated
  double3 *v_init;    // array for velocites generated
  int timesteps;      // final time step number
  int k;              // time step number
  int mod;            // files are written when k is a multiple of mod
  char basename[256]; // common part of file name
  char name[256];     // filename with number
  FILE *file;         // C style file hande
  double t;           // current time
  double dt;          // time step

  // initialize mpi
  MPI_Init(&argc, &argv);

  // get rank and size and store it in global variables
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // strategy for the setup phase:
  // args are read by all processes
  // generation of initial value and I/O are only done by process 0
  // Process 0 allocates an array holding all data
  // other processes allocate only their chunk

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
    if (rank == 0)
    {
      std::cout << "usage: " << std::endl;
      std::cout << "nbody_vanilla <basename> <load step> <final step> <every>" << std::endl;
      std::cout << "nbody_vanilla <basename> <nbodies> <timesteps> <timestep> <every>" << std::endl;
    }
    MPI_Finalize();
    return 0;
  }
  // we do not know the number of masses yet in case of file restart

  // setup of masses is done by rank 0 only
  if (rank == 0)
  {
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
      x_init = new (std::align_val_t(64)) double3[n];
      v_init = new (std::align_val_t(64)) double3[n];
      m_init = new (std::align_val_t(64)) double[n];
      read_vtk_file_double(file, n, x_init, v_init, m_init, &t, &dt);
      fclose(file);
      k *= mod; // adjust step number
      std::cout << "loaded " << n << "bodies from file " << std::string(basename) << std::endl;
    }
    // set up computation from initial condition
    if (argc == 6)
    {
      x_init = new (std::align_val_t(64)) double3[n];
      v_init = new (std::align_val_t(64)) double3[n];
      m_init = new (std::align_val_t(64)) double[n];
      two_plummer(n, 17, x_init, v_init, m_init);
      std::cout << "initialized " << n << " bodies" << std::endl;
      k = 0;
      t = 0.0;
      printf("writing %s_%06d.vtk \n", basename, k);
      sprintf(name, "%s_%06d.vtk", basename, k);
      file = fopen(name, "w");
      write_vtk_file_double(file, n, x_init, v_init, m_init, t, dt);
      fclose(file);
    }
  }

  // now broadcast parameters
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&mod, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&timesteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // check sizes
  // number of masses must be a multiple of block size
  if (n % (P * B) != 0) // i.e. n = k*B*P
  {
    if (rank == 0)
      std::cout << n << " is not a multiple of B*P, B=" << B << " P=" << P << std::endl;
    MPI_Finalize();
    return 0;
  }
  if ((n / (P * B)) % 2 != 0)
  {
    if (rank == 0)
      std::cout << n << " divided by B*P is not even, B=" << B << " P=" << P << std::endl;
    MPI_Finalize();
    return 0;
  }
  if (B % 4 != 0)
  {
    if (rank == 0)
      std::cout << B << "=B is not a multiple of 4 " << std::endl;
    MPI_Finalize();
    return 0;
  }
  if (P % 2 != 0)
  {
    if (rank == 0)
      std::cout << P << "=P is not even" << std::endl;
    MPI_Finalize();
    return 0;
  }

  // now we know we can proceed
  // data decomposition and scatter of initial values
  // Approach:
  //   rank 0: - allocates memory for all masses as in sequential program
  //           - but is only reesponsibel for the first Nrank*B masses
  //   rank>0: - allocates only ist local part
  blocks_total = n / B;               // total number of blocks
  blocks_per_rank = blocks_total / P; // my number of blocks
  masses_per_rank = blocks_per_rank * B;
  std::cout << rank << ": has " << blocks_per_rank << " blocks and " << masses_per_rank << " masses" << std::endl;

  // do further memory allocations
  x = new (std::align_val_t(64)) double3[masses_per_rank];
  v = new (std::align_val_t(64)) double3[masses_per_rank];
  m = new (std::align_val_t(64)) double[masses_per_rank];
  a = new (std::align_val_t(64)) double3[masses_per_rank];
  // explicitly fill/clear padded values
  for (int i = 0; i < masses_per_rank; i++)
    for (int j = 3; j < M; j++)
      x[i][j] = v[i][j] = a[i][j] = 0.0;

  // scatter data
  MPI_Scatter(&x_init[0][0], masses_per_rank * M, MPI_DOUBLE,
              &x[0][0], masses_per_rank * M, MPI_DOUBLE,
              0, MPI_COMM_WORLD);
  if (rank == 0)
  {
    for (int i = 0; i < masses_per_rank; i++)
      for (int j = 0; j < 3; j++)
        x[i][j] = x_init[i][j];
  }

  MPI_Scatter(&v_init[0][0], masses_per_rank * M, MPI_DOUBLE,
              &v[0][0], masses_per_rank * M, MPI_DOUBLE,
              0, MPI_COMM_WORLD);
  if (rank == 0)
  {
    for (int i = 0; i < masses_per_rank; i++)
      for (int j = 0; j < 3; j++)
        v[i][j] = v_init[i][j];
  }

  MPI_Scatter(m_init, masses_per_rank, MPI_DOUBLE,
              m, masses_per_rank, MPI_DOUBLE,
              0, MPI_COMM_WORLD);
  if (rank == 0)
  {
    for (int i = 0; i < masses_per_rank; i++)
      m[i] = m_init[i];
  }

  // initialize timestep and write first file
  if (rank == 0)
    std::cout << "step=" << k << " finalstep=" << timesteps << " time=" << t << " dt=" << dt << std::endl;
  double elapsed_total = 0.0;
  auto start = get_time_stamp();

  // do time steps
  k += 1;
  int cnt = 0;
  for (; k <= timesteps; k++)
  {
    leapfrog(masses_per_rank, dt, x, v, m, a);
    t += dt;
    cnt++;
    if (k % mod == 0)
    {
      auto stop = get_time_stamp();
      double elapsed = get_duration_seconds(start, stop);
      elapsed_total += elapsed;
      double flop = mod * (13.0 * n * (n - 1.0) + 12.0 * n);
      if (rank == 0)
        printf("%d: %g seconds for %g ops = %g GFLOPS\n", rank, elapsed, flop, flop / elapsed / 1E9);

      // collect data
      MPI_Gather(&x[0][0], masses_per_rank * M, MPI_DOUBLE,
                 &x_init[0][0], masses_per_rank * M, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
      MPI_Gather(&v[0][0], masses_per_rank * M, MPI_DOUBLE,
                 &v_init[0][0], masses_per_rank * M, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

      // write file in rank 0
      if (rank == 0)
      {
        printf("writing %s_%06d.vtk \n", basename, k / mod);
        sprintf(name, "%s_%06d.vtk", basename, k / mod);
        file = fopen(name, "w");
        write_vtk_file_double(file, n, x_init, v_init, m_init, t, dt);
        fclose(file);
      }

      start = get_time_stamp();
    }
  }

  double flop = cnt * (13.0 * n * (n - 1.0) + 12.0 * n);
  if (rank == 0)
    printf("%g seconds for %g ops = %g GFLOPS\n", elapsed_total, flop, flop / elapsed_total / 1E9);

  // clean up mpi
  MPI_Finalize();

  return 0;
}
