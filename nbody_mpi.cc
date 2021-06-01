#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

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

// model parameters
const double G = 1.0; // gravitational constant
const double epsilon2 = 1E-10; // softening parameter

// have the parallel configuration as global variables
int P; // total number of MPI processes
int rank; // my number 0 <= rank < P

// data decomposition
int blocks_total; // the total number of blocks of size B
int blocks_per_rank; // number of blocks of size B in eaach rank
int masses_per_rank; // number of masses in each rank

// version 3: 1x4 interaction, column major
void acceleration (int n, double3* __restrict__ x, double* __restrict__ m,
                   double3* __restrict__ a)
{
  Vec4d Ai; // acceleration row
  Vec4d D0,D1,D2,D3; // distances
  Vec4d A0,A1,A2,A3; // accelerations columns
  Vec4d S,U,T0,T1,T2; // temporaries

  // start with interaction of blocks in the SAME process
  // loop over rows of blocks
  for (int I=0; I<masses_per_rank; I+=B)
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
      for (int J=I+B; J<masses_per_rank; J+=B) // loop over columns of blocks
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

  // data buffers
  std::vector<std::array<double,M>> xin(masses_per_rank);
  std::vector<std::array<double,M>> xout(masses_per_rank);
  std::vector<std::array<double,M>> ain(masses_per_rank);
  std::vector<std::array<double,M>> aout(masses_per_rank);
  std::vector<double> min(masses_per_rank);
  std::vector<double> mout(masses_per_rank);

  // prepare outgoing buffers
  for (int i=0; i<masses_per_rank; i++)
    for (int j=0; j<M; j++)
      xout[i][j] = x[i][j]; // send my own positions in first round
  for (int i=0; i<masses_per_rank; i++)
    for (int j=0; j<M; j++)
      aout[i][j] = 0.0; // other ranks will accumulate to that
  for (int i=0; i<masses_per_rank; i++)
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
  for (int round=1; round<P; ++round)
    {
      //std::cout << rank << ": starting round " << round << std::endl;
      
      // exchange data with neighbors: send *out to left, rcv *in from right
      if (rank%2==0) // to avoid deadlock; P must be even
        {
          MPI_Send(xout.data(),masses_per_rank*M,MPI_DOUBLE,(rank+P-1)%P,x_tag,MPI_COMM_WORLD);
          MPI_Recv(xin.data(),masses_per_rank*M,MPI_DOUBLE,(rank+1)%P,x_tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
          MPI_Send(mout.data(),masses_per_rank,MPI_DOUBLE,(rank+P-1)%P,m_tag,MPI_COMM_WORLD);
          MPI_Recv(min.data(),masses_per_rank,MPI_DOUBLE,(rank+1)%P,m_tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
          MPI_Send(aout.data(),masses_per_rank*M,MPI_DOUBLE,(rank+P-1)%P,a_tag,MPI_COMM_WORLD);
          MPI_Recv(ain.data(),masses_per_rank*M,MPI_DOUBLE,(rank+1)%P,a_tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
      else
        {
          MPI_Recv(xin.data(),masses_per_rank*M,MPI_DOUBLE,(rank+1)%P,x_tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
          MPI_Send(xout.data(),masses_per_rank*M,MPI_DOUBLE,(rank+P-1)%P,x_tag,MPI_COMM_WORLD);
          MPI_Recv(min.data(),masses_per_rank,MPI_DOUBLE,(rank+1)%P,m_tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
          MPI_Send(mout.data(),masses_per_rank,MPI_DOUBLE,(rank+P-1)%P,m_tag,MPI_COMM_WORLD);
          MPI_Recv(ain.data(),masses_per_rank*M,MPI_DOUBLE,(rank+1)%P,a_tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
          MPI_Send(aout.data(),masses_per_rank*M,MPI_DOUBLE,(rank+P-1)%P,a_tag,MPI_COMM_WORLD);
        }

      // the positions and masses we have in this round are from the following rank
      // in round P-1 we have seen all the data
      int s = (rank+round)%P;
      int O = (rank<s) ? 0 : B; // determine if diagonal needs to be computed

      // compute block interactions
      for (int I=0; I<masses_per_rank; I+=B)
        for (int J=I+O; J<masses_per_rank; J+=B) // loop over columns of blocks
          for (int j=J; j<J+B; j+=4) // loop over 4 particles
            {
              // update accelerations of four particles
              A0.load(&ain[j][0]);
              A1.load(&ain[j+1][0]);
              A2.load(&ain[j+2][0]);
              A3.load(&ain[j+3][0]);

              // loop over particles in row
              for (int i=I; i<I+B; i+=1)
                {
                  // distances 2x4 masses
                  T0.load(&x[i][0]); // position particle i
                  D0.load(&xin[j][0]); // positions of all particles j...j+3
                  D1.load(&xin[j+1][0]);
                  D2.load(&xin[j+2][0]);
                  D3.load(&xin[j+3][0]);
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
                  T2 = Vec4d(min[j]); // mass col j               
                  U = permute4<0,0,0,0>(S); // scalar factor column j
                  T0 = T2*U;
                  Ai = mul_add(T0,D0,Ai);

                  T2 = Vec4d(min[j+1]); // mass col j+1           
                  U = permute4<1,1,1,1>(S); // scalar factor column j
                  T0 = T2*U;
                  Ai = mul_add(T0,D1,Ai);

                  T2 = Vec4d(min[j+2]); // mass col j+2           
                  U = permute4<2,2,2,2>(S); // scalar factor column j
                  T0 = T2*U;
                  Ai = mul_add(T0,D2,Ai);

                  T2 = Vec4d(min[j+3]); // mass col j+3           
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
              A0.store(&ain[j][0]);
              A1.store(&ain[j+1][0]);
              A2.store(&ain[j+2][0]);
              A3.store(&ain[j+3][0]);
            }

      // now swap in and out; so we send off what we just worked on
      std::swap(xin,xout); // now the received positions are in xin
      std::swap(min,mout); // now the received masses are in min
      std::swap(ain,aout); // now the received accelerations are in min
    }

  // need to shift acceleration one more time, so we get the
  // computations of the others
  if (rank%2==0) // to avoid deadlock; P must be even
    {
      MPI_Send(aout.data(),masses_per_rank*M,MPI_DOUBLE,(rank+P-1)%P,a_tag,MPI_COMM_WORLD);
      MPI_Recv(ain.data(),masses_per_rank*M,MPI_DOUBLE,(rank+1)%P,a_tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
  else
    {
      MPI_Recv(ain.data(),masses_per_rank*M,MPI_DOUBLE,(rank+1)%P,a_tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Send(aout.data(),masses_per_rank*M,MPI_DOUBLE,(rank+P-1)%P,a_tag,MPI_COMM_WORLD);
    }

  // accumulate the contributions of others to our acceleration
  for (int i=0; i<masses_per_rank; i++)
    for (int j=0; j<3; j++)
      a[i][j] += ain[i][j];
}

/** \brief do one time step with leapfrog
 *
 * n is the number of masses in this process
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
  acceleration(n,x,m,a);

  // update velocity: 6n flops
  for (int i=0; i<n; i++)
    {
      v[i][0] += dt*a[i][0];
      v[i][1] += dt*a[i][1];
      v[i][2] += dt*a[i][2];
    }
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

  // initialize mpi
  MPI_Init(&argc,&argv);

  // get rank and size and store it in global variables
  MPI_Comm_size(MPI_COMM_WORLD,&P);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  // strategy for the setup phase:
  // args are read by all processes
  // generation of initial value and I/O are only done by process 0
  // Process 0 allocates an array holding all data
  // other processes allocate only their chunk
  
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
      if (rank==0)
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
  if (rank==0)
    {
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
    }
  
  // now broadcast parameters
  MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&k,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&mod,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&timesteps,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&t,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Bcast(&dt,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

  // check sizes
  // number of masses must be a multiple of block size
  if (n%(P*B)!=0) // i.e. n = k*B*P
    {
      if (rank==0)
        std::cout << n << " is not a multiple of B*P, B=" << B << " P=" << P << std::endl;
      MPI_Finalize();
      return 0;
    }
  if (B%4!=0)
    {
      if (rank==0)
        std::cout << B << "=B is not a multiple of 4 " << std::endl;
      MPI_Finalize();
      return 0;
    }
  if (P%2!=0)
    {
      if (rank==0)
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
  blocks_total = n/B; // total number of blocks
  blocks_per_rank = blocks_total/P; // my number of blocks
  masses_per_rank = blocks_per_rank*B;
  std::cout << rank << ": has " << blocks_per_rank << " blocks and " <<  masses_per_rank << " masses" << std::endl;

  // do further memory allocations
  if (rank==0)
    {
      // arrays for x,v,m are already allocated above
      // allocate acceleration vector
      a = new (std::align_val_t(64)) double3[n];
      // explicitly fill/clear padded values
      for (int i=0; i<n; i++)
        for (int j=3; j<M; j++)
          x[i][j] = v[i][j] = a[i][j] = 0.0;
    }
  else
    {
      x = new (std::align_val_t(64)) double3[masses_per_rank];
      v = new (std::align_val_t(64)) double3[masses_per_rank];
      m = new (std::align_val_t(64)) double[masses_per_rank];
      a = new (std::align_val_t(64)) double3[masses_per_rank];
      // explicitly fill/clear padded values
      for (int i=0; i<masses_per_rank; i++)
        for (int j=3; j<M; j++)
          x[i][j] = v[i][j] = a[i][j] = 0.0;
    }

  // scatter data to 
  MPI_Scatter(&x[0][0],masses_per_rank*M,MPI_DOUBLE,
              &x[0][0],masses_per_rank*M,MPI_DOUBLE,
              0,MPI_COMM_WORLD);
  MPI_Scatter(&v[0][0],masses_per_rank*M,MPI_DOUBLE,
              &v[0][0],masses_per_rank*M,MPI_DOUBLE,
              0,MPI_COMM_WORLD);
  MPI_Scatter(m,masses_per_rank,MPI_DOUBLE,
              m,masses_per_rank,MPI_DOUBLE,
              0,MPI_COMM_WORLD);
  
  // initialize timestep and write first file
  if (rank==0)
    std::cout << "step=" << k << " finalstep=" << timesteps << " time=" << t << " dt=" << dt << std::endl;
  auto start = get_time_stamp();

  // do time steps
  k += 1;
  for (; k<=timesteps; k++)
    {
      leapfrog(masses_per_rank,dt,x,v,m,a);
      t += dt;
      if (k%mod==0)
        {
          auto stop = get_time_stamp();
          double elapsed = get_duration_seconds(start,stop);
          double flop = mod*(13.0*n*(n-1.0)+12.0*n);
          if (rank==0)
            printf("%d: %g seconds for %g ops = %g GFLOPS\n",rank,elapsed,flop,flop/elapsed/1E9);

          // collect data
          MPI_Gather(&x[0][0],masses_per_rank*M,MPI_DOUBLE,
                     &x[0][0],masses_per_rank*M,MPI_DOUBLE,
                     0,MPI_COMM_WORLD);
          MPI_Gather(&v[0][0],masses_per_rank*M,MPI_DOUBLE,
                     &v[0][0],masses_per_rank*M,MPI_DOUBLE,
                     0,MPI_COMM_WORLD);


          // write file in rank 0
          if (rank==0)
            {
              printf("writing %s_%06d.vtk \n",basename,k/mod);                 
              sprintf(name,"%s_%06d.vtk",basename,k/mod);
              file = fopen(name,"w");
              write_vtk_file_double(file,n,x,v,m,t,dt);
              fclose(file);
            }
                  
          start = get_time_stamp();
        }
    }

  // clean up mpi
  MPI_Finalize();

  return 0;
}
