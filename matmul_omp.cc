#include <iostream>
#include <vector>
#ifdef _OPENMP
#include<omp.h> // headers for runtime if available
#endif
#include "time_experiment.hh"

// This assumes that vector class library
// is available in the directory vcl
#include "vcl/vectorclass.h"

const int P = 24;   // basic block size is a multiple of 4, 8 and 12
const int Q = 4;    // multiplier
const int M = P*Q;  // tile size
const int N = P*256;// maximum problem size; 
double A1[N][N] __attribute__((aligned(64))); // input matrix 1
double B1[N][N] __attribute__((aligned(64))); // input matrix 2
double C1[N][N] __attribute__((aligned(64))); // output matrix 1
double C0[N][N] __attribute__((aligned(64))); // output matrix 2

// initialize all entries up to N
void initialize (double A[N][N], double B[N][N], double C[N][N])
{
  int i,j;

  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      {
        A[i][j] = (1.0*i*j)/(N*N);
        B[i][j] = (1.0+i+j)/N;
        C[i][j] = 0.0;
      }
}

// norm of difference of two matrices
double compare (int n, double A1[N][N], double A2[N][N])
{
  int i,j;
  double sum = 0.0;

  for (i=0; i<n; i++)
    for (j=0; j<n; j++)
      sum += std::abs(A1[i][j]-A2[i][j]);
  return sum;
}

// naive matmul C = A*B + C; this gives the right result for comparison
void matmul0 (int n, double A[N][N], double B[N][N], double C[N][N])
{
#pragma omp parallel for schedule (static,16) collapse (2) firstprivate (n,A,B,C)
  for (int i=0; i<n; i++)
    for (int j=0; j<n; j++)
      for (int k=0; k<n; k++)
        C[i][j] += A[i][k]*B[k][j];
}

// naive matmul C = A*B + C; this gives the right result for comparison
void matmul1 (int n, double A[N][N], double B[N][N], double C[N][N])
{
#pragma omp parallel for schedule (static) collapse (2) firstprivate (n,A,B,C)
  for (int i=0; i<n; i+=M)
    for (int j=0; j<n; j+=M)
      for (int k=0; k<n; k+=M)
        for (int s=i; s<i+M; s+=1)
          for (int t=j; t<j+M; t+=1)
#pragma omp simd simdlen(4)
	    for (int u=k; u<k+M; u+=1)
	      C[s][t] += A[s][u]*B[u][t];
}

// tiling and SIMD with vectorization of 4x12 blocks
void matmul4 (int n, double A[N][N], double B[N][N], double C[N][N])
{
  Vec4d CC[4][3], BB[3], AA; // fits exactly 16 registers
  
#pragma omp parallel for schedule (static) firstprivate(n,A,B,C) private(CC,BB,AA) collapse (2)
  for (int i=0; i<n; i+=M) // loop over tiles
    for (int j=0; j<n; j+=M)
      for (int k=0; k<n; k+=M)
        for (int s=i; s<i+M; s+=4) // loop over 4x12 blocks of C within the tiles
          for (int t=j; t<j+M; t+=12)
	    {
	      // C_st is aa 4x12 block which is loaded now
	      for (int p=0; p<4; ++p)
		{
		  // load store amortized over M/8 matrix multiplications
		  CC[p][0].load(&C[s+p][t]); CC[p][1].load(&C[s+p][t+4]); CC[p][2].load(&C[s+p][t+8]);
		}
	      for (int u=k; u<k+M; u+=12)
		// C_st += A_su*B_ut where now A_su is 4x12 and B_ut is 12x12
		for (int q=0; q<12; q+=1) // columns of A / rows of B
		  {
		    // 3 loads of B now amortized over ... 12 fmas
		    BB[0].load(&B[u+q][t]); BB[1].load(&B[u+q][t+4]); BB[2].load(&B[u+q][t+8]);

		    AA = Vec4d(A[s][u+q]); // load-broadcast
		    CC[0][0] = mul_add(AA,BB[0],CC[0][0]);
		    CC[0][1] = mul_add(AA,BB[1],CC[0][1]);
		    CC[0][2] = mul_add(AA,BB[2],CC[0][2]);

		    AA = Vec4d(A[s+1][u+q]); // load-broadcast
		    CC[1][0] = mul_add(AA,BB[0],CC[1][0]);
		    CC[1][1] = mul_add(AA,BB[1],CC[1][1]);
		    CC[1][2] = mul_add(AA,BB[2],CC[1][2]);

		    AA = Vec4d(A[s+2][u+q]); // load-broadcast
		    CC[2][0] = mul_add(AA,BB[0],CC[2][0]);
		    CC[2][1] = mul_add(AA,BB[1],CC[2][1]);
		    CC[2][2] = mul_add(AA,BB[2],CC[2][2]);

		    AA = Vec4d(A[s+3][u+q]); // load-broadcast
		    CC[3][0] = mul_add(AA,BB[0],CC[3][0]);
		    CC[3][1] = mul_add(AA,BB[1],CC[3][1]);
		    CC[3][2] = mul_add(AA,BB[2],CC[3][2]);

		    // compute intensity of inner loop w.r.t. register loads: 96 flops/128 bytes = 0.75
		  }
	      // write back C
	      for (int p=0; p<4; ++p)
		{
		  // load store amortized over M/8 matrix multiplications
		  CC[p][0].store(&C[s+p][t]); CC[p][1].store(&C[s+p][t+4]); CC[p][2].store(&C[s+p][t+8]);
		}
	    }
}

// package an experiment as a functor
class Experiment1 {
  int n;
public:
  // construct an experiment
  Experiment1 (int n_) : n(n_) {initialize(A1,B1,C1);}
  // run an experiment; can be called several times
  void run () const {matmul1(n,A1,B1,C1);}
  // report number of operations
  double operations () const
  {return 2.0*n*n*n;}
};

// package an experiment as a functor
class Experiment4 {
  int n;
public:
  // construct an experiment
  Experiment4 (int n_) : n(n_) {initialize(A1,B1,C1);}
  // run an experiment; can be called several times
  void run () const {matmul4(n,A1,B1,C1);}
  // report number of operations
  double operations () const
  {return 2.0*n*n*n;}
};

int main (int argc, char** argv)
{
  // test algorithms whether they produce the same result as the vanilla version
  if (0)
  {
    initialize(A1,B1,C0);
    int size = 10*M;
    matmul1(size,A1,B1,C0);
    initialize(A1,B1,C1);
    matmul4(size,A1,B1,C1);
    std::cout << "matmul4 N=" << size << " diff=" << compare(size,C0,C1) << std::endl;
  }
  int P=omp_get_max_threads();
  
  std::vector<int> sizes;
  for (int i=M; i<=6500; i*=2) sizes.push_back(i);
  std::cout << "N, autovec_tiled P=" << P << ", vectorized_tiled P=" << P << std::endl;
  for (auto i : sizes)
    { 
      Experiment1 e1(i);
      Experiment4 e4(i);
      auto d1 = time_experiment(e1,500000);
      auto d4 = time_experiment(e4,500000);
      double flops1 = d1.first*e1.operations()/d1.second*1e6/1e9;
      double flops4 = d4.first*e4.operations()/d4.second*1e6/1e9;
      std::cout << i
		<< ", " << flops1
                << ", " << flops4
                << std::endl;
    }
  return 0;
}
