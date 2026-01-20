#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#ifdef _OPENMP
#include<omp.h> // headers for runtime if available
#endif
#include "time_experiment.hh"
#include "matmul_experiment.hh"

// This assumes that vector class library is available in the directory vcl
#include "vcl/vectorclass.h"

const int P = 6;   // basic block size is a multiple of 4, 8 and 12
const int Q = 8;    // multiplier=SIMD width
const int M = P*Q;  // tile size
const int N = 16000;// maximum problem size; 

// row-major index mapping
#define INDEX(i,j,n) ((i)*n+(j))

// initialize all entries
void initialize(int n, double A[], double B[], double C[])
{
  int i, j;

#pragma omp parallel for schedule (static) firstprivate(n,A,B,C) collapse (2)
  for (int i = 0; i < n; i += M)
    for (int j = 0; j < n; j += M)
      for (int s = i; s < i + M; s += 1)
        for (int t = j; t < j + M; t += 1)
        {
          C[INDEX(i, j, n)] = 0.0;
          A[INDEX(i, j, n)] = 3.333;
          B[INDEX(i, j, n)] = 1.0/3.333;
        }
}

// norm of difference of two matrices
double compare (int n, double A1[], double A2[])
{
  int i,j;
  double sum = 0.0;

  for (i=0; i<n; i++)
    for (j=0; j<n; j++)
      sum += std::abs(A1[INDEX(i,j,n)]-A2[INDEX(i,j,n)]);
  return sum;
}

template<size_t simd_width>
struct SIMDSelector
{
};
template<>
struct SIMDSelector<2>
{
  static const size_t simd_width = 2;
  static const size_t simd_registers = 16;
  typedef Vec2d SIMDType;
};
template<>
struct SIMDSelector<4>
{
  static const size_t simd_width = 4;
  static const size_t simd_registers = 16;
  typedef Vec4d SIMDType;
};
template<>
struct SIMDSelector<8>
{
  static const size_t simd_width = 8;
  static const size_t simd_registers = 32;
  typedef Vec8d SIMDType;
};

// version with tile size and SIMD width as a parameter
// tiling and SIMD with vectorization of 4x3*W blocks
void matmul4 (int n, const double A[], const double B[], double C[])
{
  using VecWd = typename SIMDSelector<Q>::SIMDType;
  VecWd CC[8][3], BB[3], AA; // fits exactly 16 registers
 
  if (M%8!=0) {
     std::cout << "M must be a multiple of 8" << std::endl;
     exit(1);
  } 
  if (M%(3*Q)!=0) {
     std::cout << "M must be a multiple of 3*Q" << std::endl;
     exit(1);
  } 
  if (n%M!=0) {
     std::cout << "n must be a multiple of M" << std::endl;
     exit(1);
  } 
#pragma omp parallel for schedule (static) firstprivate(n,A,B,C) private(CC,BB,AA) collapse (2)
  for (int i=0; i<n; i+=M) // loop over tiles
    for (int j=0; j<n; j+=M)
      for (int k=0; k<n; k+=M)
	      // C_ij += A_ik*B_kj where all blocks are MxM
	      // now C_ij is again blocked into 4x(3*W) blocks
        for (int s=i; s<i+M; s+=8) // loop over 4x3*W blocks of C within the tiles
          for (int t=j; t<j+M; t+=3*Q)
	        {
	          // C_st is a 4x3*W block in 12 SIMD registers which is loaded now
	          for (int p=0; p<8; ++p)
		        {
		          // load store amortized over M/8 matrix multiplications
		          CC[p][0].load(&C[INDEX(s+p,t,n)]);
		          CC[p][1].load(&C[INDEX(s+p,t+Q,n)]);
		          CC[p][2].load(&C[INDEX(s+p,t+2*Q,n)]);
		        }
	          // C_st += A_sM*B_Mt where now A_sM is 4xM and B_Mt is Mx3*W
	          for (int u=k; u<k+M; u+=1) // columns of A / rows of B
		        {
		          // 3 loads of B now amortized over ... 24 fmas
		          BB[0].load(&B[INDEX(u,t,n)]);
		          BB[1].load(&B[INDEX(u,t+Q,n)]);
		          BB[2].load(&B[INDEX(u,t+2*Q,n)]);

		          AA = VecWd(A[INDEX(s,u,n)]); // load-broadcast
		          CC[0][0] = mul_add(AA,BB[0],CC[0][0]);
		          CC[0][1] = mul_add(AA,BB[1],CC[0][1]);
		          CC[0][2] = mul_add(AA,BB[2],CC[0][2]);
		  
		          AA = VecWd(A[INDEX(s+1,u,n)]); // load-broadcast
		          CC[1][0] = mul_add(AA,BB[0],CC[1][0]);
		          CC[1][1] = mul_add(AA,BB[1],CC[1][1]);
		          CC[1][2] = mul_add(AA,BB[2],CC[1][2]);

		          AA = VecWd(A[INDEX(s+2,u,n)]); // load-broadcast
		          CC[2][0] = mul_add(AA,BB[0],CC[2][0]);
		          CC[2][1] = mul_add(AA,BB[1],CC[2][1]);
		          CC[2][2] = mul_add(AA,BB[2],CC[2][2]);

		          AA = VecWd(A[INDEX(s+3,u,n)]); // load-broadcast
		          CC[3][0] = mul_add(AA,BB[0],CC[3][0]);
		          CC[3][1] = mul_add(AA,BB[1],CC[3][1]);
		          CC[3][2] = mul_add(AA,BB[2],CC[3][2]);

		          AA = VecWd(A[INDEX(s+4,u,n)]); // load-broadcast
		          CC[4][0] = mul_add(AA,BB[0],CC[3][0]);
		          CC[4][1] = mul_add(AA,BB[1],CC[3][1]);
		          CC[4][2] = mul_add(AA,BB[2],CC[3][2]);

              AA = VecWd(A[INDEX(s+5,u,n)]); // load-broadcast
		          CC[5][0] = mul_add(AA,BB[0],CC[3][0]);
		          CC[5][1] = mul_add(AA,BB[1],CC[3][1]);
		          CC[5][2] = mul_add(AA,BB[2],CC[3][2]);

              AA = VecWd(A[INDEX(s+6,u,n)]); // load-broadcast
		          CC[6][0] = mul_add(AA,BB[0],CC[3][0]);
		          CC[6][1] = mul_add(AA,BB[1],CC[3][1]);
		          CC[6][2] = mul_add(AA,BB[2],CC[3][2]);

              AA = VecWd(A[INDEX(s+7,u,n)]); // load-broadcast
		          CC[7][0] = mul_add(AA,BB[0],CC[3][0]);
		          CC[7][1] = mul_add(AA,BB[1],CC[3][1]);
		          CC[7][2] = mul_add(AA,BB[2],CC[3][2]);
            }
	          // write back C
	          for (int p=0; p<8; ++p)
		        {
		          // load store amortized over M/8 matrix multiplications
		          CC[p][0].store(&C[INDEX(s+p,t,n)]);
		          CC[p][1].store(&C[INDEX(s+p,t+Q,n)]);
		          CC[p][2].store(&C[INDEX(s+p,t+2*Q,n)]);
		        }
	        }
}

int main (int argc, char** argv)
{
  // get number of threads used by openMP
  int nP=omp_get_max_threads();

  // print cpu name; works only on linux
  //auto rv = std::system("cat /proc/cpuinfo | grep 'model name' | tail -1 > model.txt"); // executes the UNIX command "ls -l >test.txt"
  //std::cout << std::ifstream("model.txt").rdbuf();

  // determine sizes to be tested
  std::vector<int> sizes;
  for (int i=M; i<=N; i*=2) sizes.push_back(i);
  std::cout << "N, "
	    << " threads=" << nP 
	    << std::endl;

  // run experiments
  for (auto i : sizes)
    { 
      auto e = make_experiment(initialize, matmul4,i);
      auto d = time_experiment(e);
      double flops = d.first*e.operations()/d.second/1e9;
      std::cout << i
                << ", " << flops
                << std::endl;
    }
  return 0;
}
