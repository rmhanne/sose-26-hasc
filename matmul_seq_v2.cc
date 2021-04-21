#include <iostream>
#include <vector>

#include "time_experiment.hh"

// This assumes that vector class library
// is available in the directory vcl
#include "vcl/vectorclass.h"

const int M = 48;
const int N = 128*M;
double A[N][N] __attribute__((aligned(64)));
double B[N][N] __attribute__((aligned(64)));
double C[N][N] __attribute__((aligned(64)));

// initialize
void initialize (double A[N][N], double B[N][N], double C[N][N])
{
  int i,j,k;

  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      {
        A[i][j] = (1.0*i*j)/(N*N);
        B[i][j] = (1.0+i+j)/N;
        C[i][j] = 0.0;
      }
}

// vanilla with SIMD vectorization of 4x4 matmul C = A*B + C
void matmul1 (int n, double A[N][N], double B[N][N], double C[N][N])
{
  Vec4d Brow[4], Crow[4], AXX;
  
  for (int i=0; i<n; i+=4)
    for (int j=0; j<n; j+=4)
      {
        Crow[0].load(&C[i][j]);
        Crow[1].load(&C[i+1][j]);
        Crow[2].load(&C[i+2][j]);
        Crow[3].load(&C[i+3][j]);
        for (int k=0; k<n; k+=4)
          {
            // now multiply C_ij += A_ik * B_kj (4x4 matrices)
            Brow[0].load(&B[k][j]);
            Brow[1].load(&B[k+1][j]);
            Brow[2].load(&B[k+2][j]);
            Brow[3].load(&B[k+3][j]);
            for (int ii=0; ii<4; ++ii)
              for (int kk=0; kk<4; ++kk)
                {
                  AXX = Vec4d(A[i+ii][k+kk]);
                  Crow[ii] = mul_add(AXX,Brow[kk],Crow[ii]);
                }
          }
        Crow[0].store(&C[i][j]);
        Crow[1].store(&C[i+1][j]);
        Crow[2].store(&C[i+2][j]);
        Crow[3].store(&C[i+3][j]);
      }
}

// now tiling with SIMD vectorization of 4x4 matmul C = A*B + C
void matmul2 (int n, double A[N][N], double B[N][N], double C[N][N])
{
  Vec4d Brow[4], Crow[4], AXX;
  
  for (int i=0; i<n; i+=M)
    for (int j=0; j<n; j+=M)
      for (int k=0; k<n; k+=M)
        for (int s=i; s<i+M; s+=4)
          for (int t=j; t<j+M; t+=4)
            {
              Crow[0].load(&C[s][t]);
              Crow[1].load(&C[s+1][t]);
              Crow[2].load(&C[s+2][t]);
              Crow[3].load(&C[s+3][t]);
              for (int u=k; u<k+M; u+=4)
                {
                  // now multiply C_st += A_su * B_ut (4x4 matrices)
                  Brow[0].load(&B[u][t]);
                  Brow[1].load(&B[u+1][t]);
                  Brow[2].load(&B[u+2][t]);
                  Brow[3].load(&B[u+3][t]);
                  for (int ii=0; ii<4; ++ii)
                    for (int kk=0; kk<4; ++kk)
                      {
                        AXX = Vec4d(A[s+ii][u+kk]);
                        Crow[ii] = mul_add(AXX,Brow[kk],Crow[ii]);
                      }
                }
              Crow[0].store(&C[s][t]);
              Crow[1].store(&C[s+1][t]);
              Crow[2].store(&C[s+2][t]);
              Crow[3].store(&C[s+3][t]);
            }
}

// tiling and SIMD with vectorization of 8x8 matmul C = A*B + C
void matmul3 (int n, double A[N][N], double B[N][N], double C[N][N])
{
  Vec4d CC[4][2], BB[2], AA[2];
  
  for (int i=0; i<n; i+=M) // loop over tiles
    for (int j=0; j<n; j+=M)
      for (int k=0; k<n; k+=M)
        for (int s=i; s<i+M; s+=8) // loop over 8x8 blocks (assume M is multiple of 8) for better cache line use
          for (int t=j; t<j+M; t+=8)
            for (int ii=0; ii<8; ii+=4) // do 4 rows of C at a time
              {
                for (int p=0; p<4; ++p)
                  {
                    // load store amortized over M/8 matrix multiplications
                    CC[p][0].load(&C[s+ii+p][t]); CC[p][1].load(&C[s+ii+p][t+4]); 
                  }
                for (int u=k; u<k+M; u+=8)
                  for (int q=0; q<8; q+=1)
                    {
                       // 2 loads of B now amortized over ... 8 fmas
                      AA[0] = Vec4d(A[s+ii+0][u+q]); // load-broadcast
                      BB[0].load(&B[u+q][t]); BB[1].load(&B[u+q][t+4]);

                      // now process one half column of A
                      AA[1] = Vec4d(A[s+ii+1][u+q]); // load-broadcast
                      CC[0][0] = mul_add(AA[0],BB[0],CC[0][0]);
                      CC[0][1] = mul_add(AA[0],BB[1],CC[0][1]);

                      AA[0] = Vec4d(A[s+ii+2][u+q]); // load-broadcast
                      CC[1][0] = mul_add(AA[1],BB[0],CC[1][0]);
                      CC[1][1] = mul_add(AA[1],BB[1],CC[1][1]);

                      AA[1] = Vec4d(A[s+ii+3][u+q]); // load-broadcast
                      CC[2][0] = mul_add(AA[0],BB[0],CC[2][0]);
                      CC[2][1] = mul_add(AA[0],BB[1],CC[2][1]);

                      CC[3][0] = mul_add(AA[1],BB[0],CC[3][0]);
                      CC[3][1] = mul_add(AA[1],BB[1],CC[3][1]);
                    }
                for (int p=0; p<4; ++p)
                  {
                    // load store amortized over M/8 matrix multiplications
                    CC[p][0].store(&C[s+ii+p][t]); CC[p][1].store(&C[s+ii+p][t+4]); 
                  }
              }
}

// tiling and SIMD with vectorization of 8x8 matmul C = A*B + C
void matmul3b (int n, double A[N][N], double B[N][N], double C[N][N])
{
  Vec4d CC[4][3], BB[3], AA; // fits exactly 16 registers
  
  for (int i=0; i<n; i+=M) // loop over tiles
    for (int j=0; j<n; j+=M)
      for (int k=0; k<n; k+=M)
        for (int s=i; s<i+M; s+=12) // loop over 12x12 blocks (assume M is multiple of 12) for better cache line use
          for (int t=j; t<j+M; t+=12)
            for (int ii=0; ii<12; ii+=4) // do 4 rows of C at a time
              {
                for (int p=0; p<4; ++p)
                  {
                    // load store amortized over M/8 matrix multiplications
                    CC[p][0].load(&C[s+ii+p][t]); CC[p][1].load(&C[s+ii+p][t+4]); CC[p][2].load(&C[s+ii+p][t+8]);
                  }
                for (int u=k; u<k+M; u+=12)
                  for (int q=0; q<12; q+=1)
                    {
                       // 3 loads of B now amortized over ... 12 fmas
                      BB[0].load(&B[u+q][t]); BB[1].load(&B[u+q][t+4]); BB[2].load(&B[u+q][t+8]);

                      // now process one half column of A
                      AA = Vec4d(A[s+ii+0][u+q]); // load-broadcast
                      CC[0][0] = mul_add(AA,BB[0],CC[0][0]);
                      CC[0][1] = mul_add(AA,BB[1],CC[0][1]);
                      CC[0][2] = mul_add(AA,BB[2],CC[0][2]);

                      AA = Vec4d(A[s+ii+1][u+q]); // load-broadcast
                      CC[1][0] = mul_add(AA,BB[0],CC[1][0]);
                      CC[1][1] = mul_add(AA,BB[1],CC[1][1]);
                      CC[1][2] = mul_add(AA,BB[2],CC[1][2]);

                      AA = Vec4d(A[s+ii+2][u+q]); // load-broadcast
                      CC[2][0] = mul_add(AA,BB[0],CC[2][0]);
                      CC[2][1] = mul_add(AA,BB[1],CC[2][1]);
                      CC[2][2] = mul_add(AA,BB[2],CC[2][2]);

                      AA = Vec4d(A[s+ii+3][u+q]); // load-broadcast
                      CC[3][0] = mul_add(AA,BB[0],CC[3][0]);
                      CC[3][1] = mul_add(AA,BB[1],CC[3][1]);
                      CC[3][2] = mul_add(AA,BB[2],CC[3][2]);
                    }
                for (int p=0; p<4; ++p)
                  {
                    // load store amortized over M/8 matrix multiplications
                    CC[p][0].store(&C[s+ii+p][t]); CC[p][1].store(&C[s+ii+p][t+4]); CC[p][2].store(&C[s+ii+p][t+8]);
                  }
              }
}

// package an experiment as a functor
class Experiment1 {
  int n;
public:
  // construct an experiment
  Experiment1 (int n_) : n(n_) {initialize(A,B,C);}
  // run an experiment; can be called several times
  void run () const {matmul1(n,A,B,C);}
  // report number of operations
  double operations () const
  {return 2.0*n*n*n;}
};

// package an experiment as a functor
class Experiment2 {
  int n;
public:
  // construct an experiment
  Experiment2 (int n_) : n(n_) {initialize(A,B,C);}
  // run an experiment; can be called several times
  void run () const {matmul2(n,A,B,C);}
  // report number of operations
  double operations () const
  {return 2.0*n*n*n;}
};

// package an experiment as a functor
class Experiment3 {
  int n;
public:
  // construct an experiment
  Experiment3 (int n_) : n(n_) {initialize(A,B,C);}
  // run an experiment; can be called several times
  void run () const {matmul3b(n,A,B,C);}
  // report number of operations
  double operations () const
  {return 2.0*n*n*n;}
};

int main (int argc, char** argv)
{
  std::vector<int> sizes;
  for (int i=M; i<=6500; i*=2) sizes.push_back(i);
  std::cout << "N, vanillavec, tiledvec4x4, tiledvec12x12" << std::endl;
  //std::cout << "N, vanillavec, tiledvec4x4" << std::endl;
  for (auto i : sizes)
    { 
      Experiment1 e1(i);
      Experiment2 e2(i);
      Experiment3 e3(i);
      auto d1 = time_experiment(e1,500000);
      auto d2 = time_experiment(e2,500000);
      auto d3 = time_experiment(e3,500000);
      double flops1 = d1.first*e1.operations()/d1.second*1e6/1e9;
      double flops2 = d2.first*e2.operations()/d2.second*1e6/1e9;
      double flops3 = d3.first*e3.operations()/d3.second*1e6/1e9;
      std::cout << i
                << ", " << flops1
                << ", " << flops2
                << ", " << flops3
                << std::endl;
    }
  return 0;
}
