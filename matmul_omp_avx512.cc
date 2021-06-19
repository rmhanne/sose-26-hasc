#include <iostream>
#include <vector>
#ifdef _OPENMP
#include<omp.h> // headers for runtime if available
#endif
#include "time_experiment.hh"

// This assumes that vector class library
// is available in the directory vcl
#include "vcl/vectorclass.h"

const int P = 6;   // basic block size is a multiple of 4, 8 and 12
const int Q = 8;    // multiplier=SIMD width
const int M = P*Q;  // tile size
const int N = M*128;// maximum problem size; 

// row-major index mapping
#define INDEX(i,j,n) ((i)*n+(j))

// initialize all entries up to N
void initialize (double A[], double B[], double C[])
{
  int i,j;

  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      {
        A[INDEX(i,j,N)] = (1.0*i*j)/(N*N);
        B[INDEX(i,j,N)] = (1.0+i+j)/N;
        C[INDEX(i,j,N)] = 0.0;
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

// naive matmul C = A*B + C; this gives the right result for comparison
void matmul0 (int n, double A[], double B[], double C[])
{
#pragma omp parallel for schedule (static,16) collapse (2) firstprivate (n,A,B,C)
  for (int i=0; i<n; i++)
    for (int j=0; j<n; j++)
      for (int k=0; k<n; k++)
        C[INDEX(i,j,n)] += A[INDEX(i,k,n)]*B[INDEX(k,j,n)];
}

// naive matmul C = A*B + C; this gives the right result for comparison
void matmul1 (int n, double A[], double B[], double C[])
{
#pragma omp parallel for schedule (static) collapse (2) firstprivate (n,A,B,C)
  for (int i=0; i<n; i+=M)
    for (int j=0; j<n; j+=M)
      for (int k=0; k<n; k+=M)
        for (int s=i; s<i+M; s+=1)
          for (int t=j; t<j+M; t+=1)
#pragma omp simd simdlen(Q)
	    for (int u=k; u<k+M; u+=1)
	      C[INDEX(s,t,n)] += A[INDEX(s,u,n)]*B[INDEX(u,t,n)];
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
template<size_t M, size_t W>
void matmul4 (int n, double A[], double B[], double C[])
{
  using VecWd = typename SIMDSelector<W>::SIMDType;
  VecWd CC[4][3], BB[3], AA; // fits exactly 16 registers
 
  if (M%4!=0) {
     std::cout << "M must be a multiple of 4" << std::endl;
     exit(1);
  } 
  if (M%(3*W)!=0) {
     std::cout << "M must be a multiple of 3*W" << std::endl;
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
        for (int s=i; s<i+M; s+=4) // loop over 4x3*W blocks of C within the tiles
          for (int t=j; t<j+M; t+=3*W)
	    {
	      // C_st is a 4x3*W block in 12 SIMD registers which is loaded now
	      for (int p=0; p<4; ++p)
		{
		  // load store amortized over M/8 matrix multiplications
		  CC[p][0].load(&C[INDEX(s+p,t,n)]);
		  CC[p][1].load(&C[INDEX(s+p,t+W,n)]);
		  CC[p][2].load(&C[INDEX(s+p,t+2*W,n)]);
		}
	      // C_st += A_sM*B_Mt where now A_sM is 4xM and B_Mt is Mx3*W
	      for (int u=k; u<k+M; u+=1) // columns of A / rows of B
		{
		  // 3 loads of B now amortized over ... 12 fmas
		  BB[0].load(&B[INDEX(u,t,n)]);
		  BB[1].load(&B[INDEX(u,t+W,n)]);
		  BB[2].load(&B[INDEX(u,t+2*W,n)]);

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
		}
	      // write back C
	      for (int p=0; p<4; ++p)
		{
		  // load store amortized over M/8 matrix multiplications
		  CC[p][0].store(&C[INDEX(s+p,t,n)]);
		  CC[p][1].store(&C[INDEX(s+p,t+W,n)]);
		  CC[p][2].store(&C[INDEX(s+p,t+2*W,n)]);
		}
	    }
}

// package an experiment as a functor
template<size_t M, size_t W>
class Experiment4 {
  int n;
  double *A, *B, *C;
public:
  // construct an experiment
  Experiment4 (int n_) : n(n_)
  {
    A = new (std::align_val_t(64)) double[n*n];
    B = new (std::align_val_t(64)) double[n*n];
    C = new (std::align_val_t(64)) double[n*n];
  }
  ~Experiment4 ()
  {
    delete[] C;
    delete[] B;
    delete[] A;
  }
  // run an experiment; can be called several times
  void run () const {matmul4<M,W>(n,A,B,C);}
  // report number of operations
  double operations () const
  {return 2.0*n*n*n;}
};

int main (int argc, char** argv)
{
  int P=omp_get_max_threads();
  
  std::vector<int> sizes;
  for (int i=M; i<=N; i*=2) sizes.push_back(i);
  const size_t W=4; // SIMD
  std::cout << "N, "
	    << "Vec" << W << "d_4_3_" << M << " P=" << P 
	    << std::endl;
  for (auto i : sizes)
    { 
      Experiment4<M,W> e4(i);
      auto d4 = time_experiment(e4,500000);
      double flops4 = d4.first*e4.operations()/d4.second*1e6/1e9;
      std::cout << i
                << ", " << flops4
                << std::endl;
    }
  return 0;
}
