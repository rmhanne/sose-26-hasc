#include <iostream>
#include <vector>
#ifdef _OPENMP
#include<omp.h> // headers for runtime if available
#endif
#include "time_experiment.hh"

// This assumes that vector class library
// is available in the directory vcl
#include "vcl/vectorclass.h"

const int P = 12;   // basic block size is a multiple of 4, 8 and 12
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

// tiling and SIMD with vectorization of 4x24 blocks
void matmul4_128 (int n, double A[], double B[], double C[])
{
  const int W=2;
  using VecWd = Vec2d;
  VecWd CC[4][3], BB[3], AA; // fits exactly 16 registers
 
  if (Q!=8) {
     std::cout << "Q must be 8" << std::endl;
     return;
  } 
  if (M%Q!=0) {
     std::cout << "M must be a multiple of Q" << std::endl;
     return;
  } 
  if (M%(3*Q)!=0) {
     std::cout << "M must be a multiple of 3*Q" << std::endl;
     return;
  } 
  if (n%M!=0) {
     std::cout << "n must be a multiple of M" << std::endl;
     return;
  } 
#pragma omp parallel for schedule (static) firstprivate(n,A,B,C) private(CC,BB,AA) collapse (2)
  for (int i=0; i<n; i+=M) // loop over tiles
    for (int j=0; j<n; j+=M)
      for (int k=0; k<n; k+=M)
	// C_ij += A_ik*B_kj where all blocks are MxM
	// now C_ij is again blocked int 4x(3*W) blocks
        for (int s=i; s<i+M; s+=4) // loop over 4x3*W blocks of C within the tiles
          for (int t=j; t<j+M; t+=3*W)
	    {
	      // C_st is a 4x24 block in 12 SIMD registers which is loaded now
	      for (int p=0; p<4; ++p)
		{
		  // load store amortized over M/8 matrix multiplications
		  CC[p][0].load(&C[INDEX(s+p,t,n)]); CC[p][1].load(&C[INDEX(s+p,t+W,n)]); CC[p][2].load(&C[INDEX(s+p,t+2*W,n)]);
		}
	      for (int u=k; u<k+M; u+=3*W)
		// C_st += A_su*B_ut where now A_su is 4x24 and B_ut is 24x24
		for (int q=0; q<3*W; q+=1) // columns of A / rows of B
		  {
		    // 3 loads of B now amortized over ... 12 fmas
		    BB[0].load(&B[INDEX(u+q,t,n)]); BB[1].load(&B[INDEX(u+q,t+W,n)]); BB[2].load(&B[INDEX(u+q,t+2*W,n)]);

		    AA = VecWd(A[INDEX(s,u+q,n)]); // load-broadcast
		    CC[0][0] = mul_add(AA,BB[0],CC[0][0]);
		    CC[0][1] = mul_add(AA,BB[1],CC[0][1]);
		    CC[0][2] = mul_add(AA,BB[2],CC[0][2]);

		    AA = VecWd(A[INDEX(s+1,u+q,n)]); // load-broadcast
		    CC[1][0] = mul_add(AA,BB[0],CC[1][0]);
		    CC[1][1] = mul_add(AA,BB[1],CC[1][1]);
		    CC[1][2] = mul_add(AA,BB[2],CC[1][2]);

		    AA = VecWd(A[INDEX(s+2,u+q,n)]); // load-broadcast
		    CC[2][0] = mul_add(AA,BB[0],CC[2][0]);
		    CC[2][1] = mul_add(AA,BB[1],CC[2][1]);
		    CC[2][2] = mul_add(AA,BB[2],CC[2][2]);

		    AA = VecWd(A[INDEX(s+3,u+q,n)]); // load-broadcast
		    CC[3][0] = mul_add(AA,BB[0],CC[3][0]);
		    CC[3][1] = mul_add(AA,BB[1],CC[3][1]);
		    CC[3][2] = mul_add(AA,BB[2],CC[3][2]);
		  }
	      // write back C
	      for (int p=0; p<4; ++p)
		{
		  // load store amortized over M/8 matrix multiplications
		  CC[p][0].store(&C[INDEX(s+p,t,n)]); CC[p][1].store(&C[INDEX(s+p,t+W,n)]); CC[p][2].store(&C[INDEX(s+p,t+2*W,n)]);
		}
	    }
}

// tiling and SIMD with vectorization of 4x24 blocks
void matmul4_256 (int n, double A[], double B[], double C[])
{
  const int W=4;
  using VecWd = Vec4d;
  VecWd CC[4][3], BB[3], AA; // fits exactly 16 registers
 
  if (Q!=8) {
     std::cout << "Q must be 8" << std::endl;
     return;
  } 
  if (M%Q!=0) {
     std::cout << "M must be a multiple of Q" << std::endl;
     return;
  } 
  if (M%(3*Q)!=0) {
     std::cout << "M must be a multiple of 3*Q" << std::endl;
     return;
  } 
  if (n%M!=0) {
     std::cout << "n must be a multiple of M" << std::endl;
     return;
  } 
#pragma omp parallel for schedule (static) firstprivate(n,A,B,C) private(CC,BB,AA) collapse (2)
  for (int i=0; i<n; i+=M) // loop over tiles
    for (int j=0; j<n; j+=M)
      for (int k=0; k<n; k+=M)
	// C_ij += A_ik*B_kj where all blocks are MxM
	// now C_ij is again blocked int 4x(3*W) blocks
        for (int s=i; s<i+M; s+=4) // loop over 4x3*W blocks of C within the tiles
          for (int t=j; t<j+M; t+=3*W)
	    {
	      // C_st is a 4x24 block in 12 SIMD registers which is loaded now
	      for (int p=0; p<4; ++p)
		{
		  // load store amortized over M/8 matrix multiplications
		  CC[p][0].load(&C[INDEX(s+p,t,n)]); CC[p][1].load(&C[INDEX(s+p,t+W,n)]); CC[p][2].load(&C[INDEX(s+p,t+2*W,n)]);
		}
	      for (int u=k; u<k+M; u+=3*W)
		// C_st += A_su*B_ut where now A_su is 4x24 and B_ut is 24x24
		for (int q=0; q<3*W; q+=1) // columns of A / rows of B
		  {
		    // 3 loads of B now amortized over ... 12 fmas
		    BB[0].load(&B[INDEX(u+q,t,n)]); BB[1].load(&B[INDEX(u+q,t+W,n)]); BB[2].load(&B[INDEX(u+q,t+2*W,n)]);

		    AA = VecWd(A[INDEX(s,u+q,n)]); // load-broadcast
		    CC[0][0] = mul_add(AA,BB[0],CC[0][0]);
		    CC[0][1] = mul_add(AA,BB[1],CC[0][1]);
		    CC[0][2] = mul_add(AA,BB[2],CC[0][2]);

		    AA = VecWd(A[INDEX(s+1,u+q,n)]); // load-broadcast
		    CC[1][0] = mul_add(AA,BB[0],CC[1][0]);
		    CC[1][1] = mul_add(AA,BB[1],CC[1][1]);
		    CC[1][2] = mul_add(AA,BB[2],CC[1][2]);

		    AA = VecWd(A[INDEX(s+2,u+q,n)]); // load-broadcast
		    CC[2][0] = mul_add(AA,BB[0],CC[2][0]);
		    CC[2][1] = mul_add(AA,BB[1],CC[2][1]);
		    CC[2][2] = mul_add(AA,BB[2],CC[2][2]);

		    AA = VecWd(A[INDEX(s+3,u+q,n)]); // load-broadcast
		    CC[3][0] = mul_add(AA,BB[0],CC[3][0]);
		    CC[3][1] = mul_add(AA,BB[1],CC[3][1]);
		    CC[3][2] = mul_add(AA,BB[2],CC[3][2]);
		  }
	      // write back C
	      for (int p=0; p<4; ++p)
		{
		  // load store amortized over M/8 matrix multiplications
		  CC[p][0].store(&C[INDEX(s+p,t,n)]); CC[p][1].store(&C[INDEX(s+p,t+W,n)]); CC[p][2].store(&C[INDEX(s+p,t+2*W,n)]);
		}
	    }
}

// tiling and SIMD with vectorization of 4x24 blocks
void matmul4_512 (int n, double A[], double B[], double C[])
{
  Vec8d CC[4][3], BB[3], AA; // fits exactly 16 registers
 
  if (Q!=8) {
     std::cout << "SIMD Width must be 8" << std::endl;
     return;
  } 
  if (M%Q!=0) {
     std::cout << "M must be a multiple of Q" << std::endl;
     return;
  } 
  if (M%(3*Q)!=0) {
     std::cout << "M must be a multiple of 3*Q" << std::endl;
     return;
  } 
  if (n%M!=0) {
     std::cout << "n must be a multiple of M" << std::endl;
     return;
  } 
#pragma omp parallel for schedule (static) firstprivate(n,A,B,C) private(CC,BB,AA) collapse (2)
  for (int i=0; i<n; i+=M) // loop over tiles
    for (int j=0; j<n; j+=M)
      for (int k=0; k<n; k+=M)
	// C_ij += A_ik*B_kj where all blocks are MxM
	// now C_ij is again blocked int 4x(3Q) blocks
        for (int s=i; s<i+M; s+=4) // loop over 4x24 blocks of C within the tiles
          for (int t=j; t<j+M; t+=3*Q)
	    {
	      // C_st is a 4x24 block in 12 SIMD registers which is loaded now
	      for (int p=0; p<4; ++p)
		{
		  // load store amortized over M/8 matrix multiplications
		  CC[p][0].load(&C[INDEX(s+p,t,n)]); CC[p][1].load(&C[INDEX(s+p,t+Q,n)]); CC[p][2].load(&C[INDEX(s+p,t+2*Q,n)]);
		}
	      for (int u=k; u<k+M; u+=3*Q)
		// C_st += A_su*B_ut where now A_su is 4x24 and B_ut is 24x24
		for (int q=0; q<3*Q; q+=1) // columns of A / rows of B
		  {
		    // 3 loads of B now amortized over ... 12 fmas
		    BB[0].load(&B[INDEX(u+q,t,n)]); BB[1].load(&B[INDEX(u+q,t+Q,n)]); BB[2].load(&B[INDEX(u+q,t+2*Q,n)]);

		    AA = Vec8d(A[INDEX(s,u+q,n)]); // load-broadcast
		    CC[0][0] = mul_add(AA,BB[0],CC[0][0]);
		    CC[0][1] = mul_add(AA,BB[1],CC[0][1]);
		    CC[0][2] = mul_add(AA,BB[2],CC[0][2]);

		    AA = Vec8d(A[INDEX(s+1,u+q,n)]); // load-broadcast
		    CC[1][0] = mul_add(AA,BB[0],CC[1][0]);
		    CC[1][1] = mul_add(AA,BB[1],CC[1][1]);
		    CC[1][2] = mul_add(AA,BB[2],CC[1][2]);

		    AA = Vec8d(A[INDEX(s+2,u+q,n)]); // load-broadcast
		    CC[2][0] = mul_add(AA,BB[0],CC[2][0]);
		    CC[2][1] = mul_add(AA,BB[1],CC[2][1]);
		    CC[2][2] = mul_add(AA,BB[2],CC[2][2]);

		    AA = Vec8d(A[INDEX(s+3,u+q,n)]); // load-broadcast
		    CC[3][0] = mul_add(AA,BB[0],CC[3][0]);
		    CC[3][1] = mul_add(AA,BB[1],CC[3][1]);
		    CC[3][2] = mul_add(AA,BB[2],CC[3][2]);
		  }
	      // write back C
	      for (int p=0; p<4; ++p)
		{
		  // load store amortized over M/8 matrix multiplications
		  CC[p][0].store(&C[INDEX(s+p,t,n)]); CC[p][1].store(&C[INDEX(s+p,t+Q,n)]); CC[p][2].store(&C[INDEX(s+p,t+2*Q,n)]);
		}
	    }
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

// fully parametric version
// actually this one is much slower than the one above
// tiling and SIMD with vectorization of products of IxK and KxJ*4 matrices
// W is the SIMD width
template<size_t I, size_t J, size_t K, size_t W>
void matmul5 (int n, double A[], double B[], double C[])
{
  using VecWd = typename SIMDSelector<W>::SIMDType;

  // check divisibilities
  if (n%M!=0) {std::cout << "matrix size and tile size " << n << " is not a multiple of " << M << std::endl; exit(1);}
  if (M%I!=0) {std::cout << "tile size and I blocks " << M << " is not a multiple of " << I << std::endl; exit(1);}
  if (M%K!=0) {std::cout << "tile size and K blocks " << M << " is not a multiple of " << K << std::endl; exit(1);}
  if (M%(J*W)!=0) {std::cout << "tile size and J blocks " << M << " is not a multiple of " << J*W << std::endl; exit(1);}

  // check number of registers needed
  int registers = I*J + J + 1;
  if (registers>SIMDSelector<W>::simd_registers) {
    std::cout << "too many registers: " << registers << " is greater than " << SIMDSelector<W>::simd_registers << std::endl;
    exit(1);
  }

  // data in registers
  VecWd CC[I][J], BB[J], AA;

  // loop over tiles of size MxM
#pragma omp parallel for schedule (static) firstprivate(n,A,B,C) private(CC,BB,AA) collapse (2)
  for (int i=0; i<n; i+=M)
    for (int j=0; j<n; j+=M)
      for (int k=0; k<n; k+=M)
	// update subblock C_ij += A_ik*B_kj
        for (int s=i; s<i+M; s+=I)
          for (int t=j; t<j+M; t+=J*W)
	    {
	      // C_st is a IxJ*W subblock of C_ij
	      for (int p=0; p<I; ++p)
		for (int q=0; q<J; ++q)
		  CC[p][q].load(&C[INDEX(s+p,t+q*W,n)]);
	      for (int u=k; u<k+M; u+=K)
		// C_st += A_su*B_ut where now A_su is IxK and B_ut is KxJ*W
		for (int r=0; r<K; r+=1) // columns of A / rows of B
		  {
		    for (int q=0; q<J; ++q)
		      BB[q].load(&B[INDEX(u+r,t+q*W,n)]);
		    for (int p=0; p<I; ++p)
		      {
			AA = VecWd(A[INDEX(s+p,u+r,n)]); // load-broadcast
			for (int q=0; q<J; ++q)
			  CC[p][q] = mul_add(AA,BB[q],CC[p][q]);
		      }
		  }
	      // write back C
	      for (int p=0; p<I; ++p)
		for (int q=0; q<J; ++q)
		  CC[p][q].store(&C[INDEX(s+p,t+q*W,n)]);
	    }
}

// package an experiment as a functor
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
  void run () const {matmul4_512(n,A,B,C);}
  // report number of operations
  double operations () const
  {return 2.0*n*n*n;}
};

// package an experiment as a functor
template<size_t I, size_t J, size_t K, size_t W>
class Experiment5 {
  int n;
  double *A, *B, *C;
public:
  // construct an experiment
  Experiment5 (int n_) : n(n_)
  {
    A = new (std::align_val_t(64)) double[n*n];
    B = new (std::align_val_t(64)) double[n*n];
    C = new (std::align_val_t(64)) double[n*n];
  }
  ~Experiment5 ()
  {
    delete[] C;
    delete[] B;
    delete[] A;
  }
  // run an experiment; can be called several times
  void run () const {matmul5<I,J,K,W>(n,A,B,C);}
  // report number of operations
  double operations () const
  {return 2.0*n*n*n;}
};

int main (int argc, char** argv)
{
  int P=omp_get_max_threads();
  
  std::vector<int> sizes;
  for (int i=M; i<=N; i*=2) sizes.push_back(i);
  const size_t I=4;
  const size_t J=3;
  const size_t K=24;
  const size_t W=8;
  std::cout << "N, "
	  << "Vec8d_4_3_24x P=" << P 
//	  << "Vec" << W << "d_" << I <<"_" << J << "_" << K << " P=" << P
	    << std::endl;
  for (auto i : sizes)
    { 
      Experiment4 e4(i);
      //Experiment5<4,3,12,4> e5(i);
      auto d4 = time_experiment(e4,500000);
      //auto d5 = time_experiment(e5,500000);
      double flops4 = d4.first*e4.operations()/d4.second*1e6/1e9;
      //double flops5 = d5.first*e5.operations()/d5.second*1e6/1e9;
      std::cout << i
                << ", " << flops4
      //          << ", " << flops5
                << std::endl;
    }
  return 0;
}
