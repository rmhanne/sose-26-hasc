#include <iostream>
#include <vector>
#include <math.h>
#include "time_experiment.hh"
#include "vcl/vectorclass.h"
#ifdef _OPENMP
#include<omp.h> // headers for runtime if available
#endif

// row-major index mapping
#define INDEX(i,j,n) ((i)*n+(j))
const int M=80;

// initialize all entries up to N
void ludecomp (int n, double A[])
{
  // transformation to upper triangular form
  for (std::size_t k=0; k<n-1; ++k)
    for (std::size_t i=k+1; i<n; ++i)
      {
        double q = A[INDEX(i,k,n)]/A[INDEX(k,k,n)];
        A[INDEX(i,k,n)] = q;
        for (std::size_t j=k+1; j<n; ++j)
          A[INDEX(i,j,n)] -= q * A[INDEX(k,j,n)];
      }
}

// initialize all entries up to N
void ludecomp_pivot (int n, double A[])
{
  // transformation to upper triangular form
  for (std::size_t k=0; k<n-1; ++k)
    {
      // column pivoting with row exchange
      double abspivot=std::abs(A[INDEX(k,k,n)]);
      int pivotrow=k;
      for (std::size_t i=k+1; i<n; ++i)
        if (std::abs(A[INDEX(i,k,n)])>abspivot)
          {
            abspivot = std::abs(A[INDEX(i,k,n)]);
            pivotrow = i;
          }
      if (pivotrow!=k)
        for (std::size_t j=0; j<n; ++j) // whole row!
          std::swap(A[INDEX(k,j,n)], A[INDEX(pivotrow,j,n)]);
      // elimination step
      for (std::size_t i=k+1; i<n; ++i)
        {
          double q = A[INDEX(i,k,n)]/A[INDEX(k,k,n)];
          A[INDEX(i,k,n)] = q;
          for (std::size_t j=k+1; j<n; ++j)
            A[INDEX(i,j,n)] -= q * A[INDEX(k,j,n)];
        }
    }
}

// initialize all entries up to N
void ludecomp_ijk (int n, double A[])
{
  // transformation to upper triangular form
  for (std::size_t i=1; i<n; ++i)
    for (std::size_t j=0; j<i; ++j)
      {
        double q = A[INDEX(i,j,n)]/A[INDEX(j,j,n)];
        A[INDEX(i,j,n)] = q;
        for (std::size_t k=j+1; k<n; ++k)
          A[INDEX(i,k,n)] -= q * A[INDEX(j,k,n)];
      }
}

// initialize all entries up to N
void ludecomp_blocked (int n, double A[])
{
  if (n%M!=0) exit(1);

  for (std::size_t K=0; K<n; K+=M)
    {
      // 1a) LU decomposition of upper left block
      for (std::size_t k=K; k<K+M-1; ++k)
        for (std::size_t i=k+1; i<K+M; ++i)
          {
            double lik = A[INDEX(i,k,n)]/A[INDEX(k,k,n)];
            A[INDEX(i,k,n)] = lik;
            for (std::size_t j=k+1; j<K+M; ++j)
              A[INDEX(i,j,n)] -= lik * A[INDEX(k,j,n)];
          }
      
      // 1b) remaining blocks in first column
      for (std::size_t k=K; k<K+M; ++k)
        for (std::size_t i=K+M; i<n; ++i)
          {
            double lik = A[INDEX(i,k,n)]/A[INDEX(k,k,n)];
            A[INDEX(i,k,n)] = lik;
            for (std::size_t j=k+1; j<K+M; ++j)
              A[INDEX(i,j,n)] -= lik * A[INDEX(k,j,n)];
          }

      // 2) Solve for U_KJ
      for (std::size_t J=K+M; J<n; J+=M)
        for (std::size_t i=0; i<M; ++i)
          for (std::size_t k=0; k<i; ++k)
            for (std::size_t j=0; j<M; ++j)
              A[INDEX(K+i,J+j,n)] -= A[INDEX(K+i,K+k,n)]*A[INDEX(K+k,J+j,n)];
                
      // 3) update S
      for (std::size_t I=K+M; I<n; I+=M)
        for (std::size_t J=K+M; J<n; J+=M)
          for (std::size_t i=0; i<M; ++i)
            for (std::size_t j=0; j<M; ++j)
              for (std::size_t k=0; k<M; ++k)
                A[INDEX(I+i,J+j,n)] -= A[INDEX(I+i,K+k,n)]*A[INDEX(K+k,J+j,n)];
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

// Vectorized C += A*B, where A,B,C are MxM submatrices
// of nxn matrices stored in row-major layout
// SIMD with vectorization of 4x3*W blocks
template<size_t M, size_t W>
void matmul_kernel_4x3 (int n, double A[], double B[], double C[])
{
  using VecWd = typename SIMDSelector<W>::SIMDType;
  VecWd CC[4][3], BB[3], AA; // fits exactly 16 registers
 
  // C is blocked into 4x(3*W) blocks
  for (int s=0; s<M; s+=4) // loop over 4x3*W blocks of C within the tiles
    for (int t=0; t<M; t+=3*W)
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
        for (int u=0; u<M; u+=1) // columns of A / rows of B
          {
            // 3 loads of B now amortized over ... 12 fmas
            BB[0].load(&B[INDEX(u,t,n)]);
            BB[1].load(&B[INDEX(u,t+W,n)]);
            BB[2].load(&B[INDEX(u,t+2*W,n)]);
            
            AA = VecWd(A[INDEX(s,u,n)]); // load-broadcast
            CC[0][0] = nmul_add(AA,BB[0],CC[0][0]);
            CC[0][1] = nmul_add(AA,BB[1],CC[0][1]);
            CC[0][2] = nmul_add(AA,BB[2],CC[0][2]);
            
            AA = VecWd(A[INDEX(s+1,u,n)]); // load-broadcast
            CC[1][0] = nmul_add(AA,BB[0],CC[1][0]);
            CC[1][1] = nmul_add(AA,BB[1],CC[1][1]);
            CC[1][2] = nmul_add(AA,BB[2],CC[1][2]);
            
            AA = VecWd(A[INDEX(s+2,u,n)]); // load-broadcast
            CC[2][0] = nmul_add(AA,BB[0],CC[2][0]);
            CC[2][1] = nmul_add(AA,BB[1],CC[2][1]);
            CC[2][2] = nmul_add(AA,BB[2],CC[2][2]);
            
            AA = VecWd(A[INDEX(s+3,u,n)]); // load-broadcast
            CC[3][0] = nmul_add(AA,BB[0],CC[3][0]);
            CC[3][1] = nmul_add(AA,BB[1],CC[3][1]);
            CC[3][2] = nmul_add(AA,BB[2],CC[3][2]);
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

// Vectorized C += A*B, where A,B,C are MxM submatrices
// of nxn matrices stored in row-major layout
// SIMD with vectorization of 4x3*W blocks
template<size_t M, size_t W>
void matmul_kernel_4x4 (int n, double A[], double B[], double C[])
{
  using VecWd = typename SIMDSelector<W>::SIMDType;
  VecWd CC[4][4], BB[4], AA; // fits 21 registers
 
  // C is blocked into 4x(4*W) blocks
  for (int t=0; t<M; t+=4*W)
    for (int s=0; s<M; s+=4) // loop over 4x3*W blocks of C within the tiles
      {
        // C_st is a 4x4*W block in 16 SIMD registers which is loaded now
        for (int p=0; p<4; ++p)
          {
            // load store amortized over M/8 matrix multiplications
            CC[p][0].load(&C[INDEX(s+p,t,n)]);
            CC[p][1].load(&C[INDEX(s+p,t+W,n)]);
            CC[p][2].load(&C[INDEX(s+p,t+2*W,n)]);
            CC[p][3].load(&C[INDEX(s+p,t+3*W,n)]);
          }
        // C_st += A_sM*B_Mt where now A_sM is 4xM and B_Mt is Mx4*W
        for (int u=0; u<M; u+=1) // columns of A / rows of B
          {
            // 4 loads of B now amortized over ... 16 fmas
            BB[0].load(&B[INDEX(u,t,n)]);
            BB[1].load(&B[INDEX(u,t+W,n)]);
            BB[2].load(&B[INDEX(u,t+2*W,n)]);
            BB[3].load(&B[INDEX(u,t+3*W,n)]);
            
            AA = VecWd(A[INDEX(s,u,n)]); // load-broadcast
            CC[0][0] = nmul_add(AA,BB[0],CC[0][0]);
            CC[0][1] = nmul_add(AA,BB[1],CC[0][1]);
            CC[0][2] = nmul_add(AA,BB[2],CC[0][2]);
            CC[0][3] = nmul_add(AA,BB[3],CC[0][3]);
            
            AA = VecWd(A[INDEX(s+1,u,n)]); // load-broadcast
            CC[1][0] = nmul_add(AA,BB[0],CC[1][0]);
            CC[1][1] = nmul_add(AA,BB[1],CC[1][1]);
            CC[1][2] = nmul_add(AA,BB[2],CC[1][2]);
            CC[1][3] = nmul_add(AA,BB[3],CC[1][3]);
            
            AA = VecWd(A[INDEX(s+2,u,n)]); // load-broadcast
            CC[2][0] = nmul_add(AA,BB[0],CC[2][0]);
            CC[2][1] = nmul_add(AA,BB[1],CC[2][1]);
            CC[2][2] = nmul_add(AA,BB[2],CC[2][2]);
            CC[2][3] = nmul_add(AA,BB[3],CC[2][3]);
            
            AA = VecWd(A[INDEX(s+3,u,n)]); // load-broadcast
            CC[3][0] = nmul_add(AA,BB[0],CC[3][0]);
            CC[3][1] = nmul_add(AA,BB[1],CC[3][1]);
            CC[3][2] = nmul_add(AA,BB[2],CC[3][2]);
            CC[3][3] = nmul_add(AA,BB[3],CC[3][3]);
          }
        // write back C
        for (int p=0; p<4; ++p)
          {
            // load store amortized over M/8 matrix multiplications
            CC[p][0].store(&C[INDEX(s+p,t,n)]);
            CC[p][1].store(&C[INDEX(s+p,t+W,n)]);
            CC[p][2].store(&C[INDEX(s+p,t+2*W,n)]);
            CC[p][3].store(&C[INDEX(s+p,t+3*W,n)]);
          }
      }
}

// Vectorized C += A*B, where A,B,C are MxM submatrices
// of nxn matrices stored in row-major layout
// SIMD with vectorization of 4x3*W blocks
template<size_t M, size_t W>
void matmul_kernel_4x5 (int n, double A[], double B[], double C[])
{
  using VecWd = typename SIMDSelector<W>::SIMDType;
  VecWd CC[4][5], BB[5], AA; // fits 26 registers
 
  // C is blocked into 4x(4*W) blocks
  for (int t=0; t<M; t+=5*W)
    for (int s=0; s<M; s+=4) // loop over 4x3*W blocks of C within the tiles
      {
        // C_st is a 4x4*W block in 16 SIMD registers which is loaded now
        for (int p=0; p<4; ++p)
          {
            // load store amortized over M/8 matrix multiplications
            CC[p][0].load(&C[INDEX(s+p,t+0*W,n)]);
            CC[p][1].load(&C[INDEX(s+p,t+1*W,n)]);
            CC[p][2].load(&C[INDEX(s+p,t+2*W,n)]);
            CC[p][3].load(&C[INDEX(s+p,t+3*W,n)]);
            CC[p][4].load(&C[INDEX(s+p,t+4*W,n)]);
          }
        // C_st += A_sM*B_Mt where now A_sM is 4xM and B_Mt is Mx4*W
        for (int u=0; u<M; u+=1) // columns of A / rows of B
          {
            // 5 loads of B now amortized over ... 20 fmas
            BB[0].load(&B[INDEX(u,t+0*W,n)]);
            BB[1].load(&B[INDEX(u,t+1*W,n)]);
            BB[2].load(&B[INDEX(u,t+2*W,n)]);
            BB[3].load(&B[INDEX(u,t+3*W,n)]);
            BB[4].load(&B[INDEX(u,t+4*W,n)]);
            
            AA = VecWd(A[INDEX(s,u,n)]); // load-broadcast
            CC[0][0] = nmul_add(AA,BB[0],CC[0][0]);
            CC[0][1] = nmul_add(AA,BB[1],CC[0][1]);
            CC[0][2] = nmul_add(AA,BB[2],CC[0][2]);
            CC[0][3] = nmul_add(AA,BB[3],CC[0][3]);
            CC[0][4] = nmul_add(AA,BB[4],CC[0][4]);
            
            AA = VecWd(A[INDEX(s+1,u,n)]); // load-broadcast
            CC[1][0] = nmul_add(AA,BB[0],CC[1][0]);
            CC[1][1] = nmul_add(AA,BB[1],CC[1][1]);
            CC[1][2] = nmul_add(AA,BB[2],CC[1][2]);
            CC[1][3] = nmul_add(AA,BB[3],CC[1][3]);
            CC[1][4] = nmul_add(AA,BB[4],CC[1][4]);
            
            AA = VecWd(A[INDEX(s+2,u,n)]); // load-broadcast
            CC[2][0] = nmul_add(AA,BB[0],CC[2][0]);
            CC[2][1] = nmul_add(AA,BB[1],CC[2][1]);
            CC[2][2] = nmul_add(AA,BB[2],CC[2][2]);
            CC[2][3] = nmul_add(AA,BB[3],CC[2][3]);
            CC[2][4] = nmul_add(AA,BB[4],CC[2][4]);
            
            AA = VecWd(A[INDEX(s+3,u,n)]); // load-broadcast
            CC[3][0] = nmul_add(AA,BB[0],CC[3][0]);
            CC[3][1] = nmul_add(AA,BB[1],CC[3][1]);
            CC[3][2] = nmul_add(AA,BB[2],CC[3][2]);
            CC[3][3] = nmul_add(AA,BB[3],CC[3][3]);
            CC[3][4] = nmul_add(AA,BB[4],CC[3][4]);
          }
        // write back C
        for (int p=0; p<4; ++p)
          {
            // load store amortized over M/8 matrix multiplications
            CC[p][0].store(&C[INDEX(s+p,t+0*W,n)]);
            CC[p][1].store(&C[INDEX(s+p,t+1*W,n)]);
            CC[p][2].store(&C[INDEX(s+p,t+2*W,n)]);
            CC[p][3].store(&C[INDEX(s+p,t+3*W,n)]);
            CC[p][4].store(&C[INDEX(s+p,t+4*W,n)]);
          }
      }
}

// Vectorized C += A*B, where A,B,C are MxM submatrices
// of nxn matrices stored in row-major layout
// SIMD with vectorization of 4x3*W blocks
template<size_t M, size_t W>
void matmul_kernel_4x2 (int n, double A[], double B[], double C[])
{
  using VecWd = typename SIMDSelector<W>::SIMDType;
  VecWd CC[4][2], BB[2], AA; // fits 11 registers
 
  // C is blocked into 4x(4*W) blocks
  for (int t=0; t<M; t+=2*W)
    for (int s=0; s<M; s+=4) // loop over 4x3*W blocks of C within the tiles
      {
        // C_st is a 4x4*W block in 16 SIMD registers which is loaded now
        for (int p=0; p<4; ++p)
          {
            // load store amortized over M/8 matrix multiplications
            CC[p][0].load(&C[INDEX(s+p,t,n)]);
            CC[p][1].load(&C[INDEX(s+p,t+W,n)]);
          }
        // C_st += A_sM*B_Mt where now A_sM is 4xM and B_Mt is Mx2*W
        for (int u=0; u<M; u+=1) // columns of A / rows of B
          {
            // 4 loads of B now amortized over ... 16 fmas
            BB[0].load(&B[INDEX(u,t,n)]);
            BB[1].load(&B[INDEX(u,t+W,n)]);
            
            AA = VecWd(A[INDEX(s,u,n)]); // load-broadcast
            CC[0][0] = nmul_add(AA,BB[0],CC[0][0]);
            CC[0][1] = nmul_add(AA,BB[1],CC[0][1]);
            
            AA = VecWd(A[INDEX(s+1,u,n)]); // load-broadcast
            CC[1][0] = nmul_add(AA,BB[0],CC[1][0]);
            CC[1][1] = nmul_add(AA,BB[1],CC[1][1]);
            
            AA = VecWd(A[INDEX(s+2,u,n)]); // load-broadcast
            CC[2][0] = nmul_add(AA,BB[0],CC[2][0]);
            CC[2][1] = nmul_add(AA,BB[1],CC[2][1]);
            
            AA = VecWd(A[INDEX(s+3,u,n)]); // load-broadcast
            CC[3][0] = nmul_add(AA,BB[0],CC[3][0]);
            CC[3][1] = nmul_add(AA,BB[1],CC[3][1]);
          }
        // write back C
        for (int p=0; p<4; ++p)
          {
            // load store amortized over M/8 matrix multiplications
            CC[p][0].store(&C[INDEX(s+p,t,n)]);
            CC[p][1].store(&C[INDEX(s+p,t+W,n)]);
          }
      }
}

// Vectorized C += A*B, where A,B,C are MxM submatrices
// of nxn matrices stored in row-major layout
// SIMD with vectorization of 8x2*W blocks
template<size_t M, size_t W>
void matmul_kernel8x2 (int n, double A[], double B[], double C[])
{
  using VecWd = typename SIMDSelector<W>::SIMDType;
  VecWd CC[8][2], BB[2], AA[8]; // fits 30 registers
 
  // C is blocked into 4x(4*W) blocks
  for (int t=0; t<M; t+=2*W)
    for (int s=0; s<M; s+=8) // loop over 8x2*W blocks of C within the tiles
      {
        // C_st is a 8x2*W block
        for (int p=0; p<8; ++p)
          {
            // load store amortized over M/8 matrix multiplications
            CC[p][0].load(&C[INDEX(s+p,t,n)]);
            CC[p][1].load(&C[INDEX(s+p,t+W,n)]);
          }
        // C_st += A_sM*B_Mt where now A_sM is 4xM and B_Mt is Mx2*W
        for (int u=0; u<M; u+=1) // columns of A / rows of B
          {
            // loads of B now amortized over ... 8*2 fmas
            BB[0].load(&B[INDEX(u,t,n)]);
            BB[1].load(&B[INDEX(u,t+W,n)]);
            
            AA[0] = VecWd(A[INDEX(s+0,u,n)]); // load-broadcast
            CC[0][0] = nmul_add(AA[0],BB[0],CC[0][0]);
            CC[0][1] = nmul_add(AA[0],BB[1],CC[0][1]);
            
            AA[1] = VecWd(A[INDEX(s+1,u,n)]); // load-broadcast
            CC[1][0] = nmul_add(AA[1],BB[0],CC[1][0]);
            CC[1][1] = nmul_add(AA[1],BB[1],CC[1][1]);
            
            AA[2] = VecWd(A[INDEX(s+2,u,n)]); // load-broadcast
            CC[2][0] = nmul_add(AA[2],BB[0],CC[2][0]);
            CC[2][1] = nmul_add(AA[2],BB[1],CC[2][1]);
            
            AA[3] = VecWd(A[INDEX(s+3,u,n)]); // load-broadcast
            CC[3][0] = nmul_add(AA[3],BB[0],CC[3][0]);
            CC[3][1] = nmul_add(AA[3],BB[1],CC[3][1]);

            AA[4] = VecWd(A[INDEX(s+4,u,n)]); // load-broadcast
            CC[4][0] = nmul_add(AA[4],BB[0],CC[4][0]);
            CC[4][1] = nmul_add(AA[4],BB[1],CC[4][1]);

	    AA[5] = VecWd(A[INDEX(s+5,u,n)]); // load-broadcast
            CC[5][0] = nmul_add(AA[5],BB[0],CC[5][0]);
            CC[5][1] = nmul_add(AA[5],BB[1],CC[5][1]);
	    
	    AA[6] = VecWd(A[INDEX(s+6,u,n)]); // load-broadcast
            CC[6][0] = nmul_add(AA[6],BB[0],CC[6][0]);
            CC[6][1] = nmul_add(AA[6],BB[1],CC[6][1]);
	    
	    AA[7] = VecWd(A[INDEX(s+7,u,n)]); // load-broadcast
            CC[7][0] = nmul_add(AA[7],BB[0],CC[7][0]);
            CC[7][1] = nmul_add(AA[7],BB[1],CC[7][1]);
          }
        // write back C
        for (int p=0; p<8; ++p)
          {
            // load store amortized over M/8 matrix multiplications
            CC[p][0].store(&C[INDEX(s+p,t,n)]);
            CC[p][1].store(&C[INDEX(s+p,t+W,n)]);
          }
      }
}

// initialize all entries up to N
template<size_t W>
void ludecomp_blocked_vectorized (int n, double A[])
{
  if (n%M!=0) exit(1);
  if (M%4!=0) exit(1);
  if (M%(3*W)!=0) exit(1);

  for (std::size_t K=0; K<n; K+=M)
    {
      // 1a) LU decomposition of upper left block
      for (std::size_t k=K; k<K+M-1; ++k)
        for (std::size_t i=k+1; i<K+M; ++i)
          {
            double lik = A[INDEX(i,k,n)]/A[INDEX(k,k,n)];
            A[INDEX(i,k,n)] = lik;
            for (std::size_t j=k+1; j<K+M; ++j)
              A[INDEX(i,j,n)] -= lik * A[INDEX(k,j,n)];
          }
      
      // 1b) remaining blocks in first column
      for (std::size_t k=K; k<K+M; ++k)
        for (std::size_t i=K+M; i<n; ++i)
          {
            double lik = A[INDEX(i,k,n)]/A[INDEX(k,k,n)];
            A[INDEX(i,k,n)] = lik;
            for (std::size_t j=k+1; j<K+M; ++j)
              A[INDEX(i,j,n)] -= lik * A[INDEX(k,j,n)];
          }

      // 2) Solve for U_KJ
      for (std::size_t J=K+M; J<n; J+=M)
        for (std::size_t i=0; i<M; ++i)
          for (std::size_t k=0; k<i; ++k)
            for (std::size_t j=0; j<M; ++j)
              A[INDEX(K+i,J+j,n)] -= A[INDEX(K+i,K+k,n)]*A[INDEX(K+k,J+j,n)];
                
      // 3) update S
      for (std::size_t I=K+M; I<n; I+=M)
        for (std::size_t J=K+M; J<n; J+=M)
          matmul_kernel_4x3<M,W>(n,&A[INDEX(I,K,n)],&A[INDEX(K,J,n)],&A[INDEX(I,J,n)]);
    }
}

// initialize all entries up to N
template<size_t W>
void ludecomp_blocked_vectorized_omp (int n, double A[])
{
  if (n%M!=0) exit(1);
  if (M%4!=0) exit(1);
  if (M%(3*W)!=0) exit(1);
  using VecWd = typename SIMDSelector<W>::SIMDType;

  for (std::size_t K=0; K<n; K+=M)
    {
      // 1a) LU decomposition of upper left block
      for (std::size_t k=K; k<K+M-1; ++k)
        for (std::size_t i=k+1; i<K+M; ++i)
          {
            double lik = A[INDEX(i,k,n)]/A[INDEX(k,k,n)];
            A[INDEX(i,k,n)] = lik;
            for (std::size_t j=k+1; j<K+M; ++j)
              A[INDEX(i,j,n)] -= lik * A[INDEX(k,j,n)];
          }
      
      // 1b) remaining blocks in first column
#pragma omp parallel for if ((n-K-M)/M>4) schedule (static) firstprivate(n,A)
      for (std::size_t I=K+M; I<n; I+=M)
        for (std::size_t i=I; i<I+M; ++i)
          for (std::size_t k=K; k<K+M; ++k)
          {
            double lik = A[INDEX(i,k,n)]/A[INDEX(k,k,n)];
            A[INDEX(i,k,n)] = lik;
            for (std::size_t j=k+1; j<K+M; ++j)
              A[INDEX(i,j,n)] -= lik * A[INDEX(k,j,n)];
          }

      // 2) Solve for U_KJ
#pragma omp parallel for if ((n-K-M)/M>4) schedule (static) firstprivate(n,A)
      for (std::size_t J=K+M; J<n; J+=M)
        for (std::size_t i=0; i<M; ++i)
          for (std::size_t k=0; k<i; ++k)
            {
              VecWd Lik,Uij,Akj;
              Lik = VecWd(A[INDEX(K+i,K+k,n)]);
              for (std::size_t j=0; j<M; j+=W)
                {
                  Uij.load(&A[INDEX(K+i,J+j,n)]);
                  Akj.load(&A[INDEX(K+k,J+j,n)]);
                  Uij = nmul_add(Lik,Akj,Uij);
                  Uij.store(&A[INDEX(K+i,J+j,n)]);
                }
            }
                
      // 3) update S
#pragma omp parallel for schedule (static) firstprivate(n,A) collapse (2)
      for (std::size_t I=K+M; I<n; I+=M)
        for (std::size_t J=K+M; J<n; J+=M)
          matmul_kernel_4x3<M,W>(n,&A[INDEX(I,K,n)],&A[INDEX(K,J,n)],&A[INDEX(I,J,n)]);
    }
}

// initialize all entries up to N
template<size_t W>
void ludecomp_blocked_vectorized_omp_pivot (int n, double A[])
{
  if (n%M!=0) exit(1);
  if (M%4!=0) exit(1);
  if (M%(3*W)!=0) exit(1);
  using VecWd = typename SIMDSelector<W>::SIMDType;

  for (std::size_t K=0; K<n; K+=M)
    {
      // 1a) LU decomposition of upper left block
      for (std::size_t k=K; k<K+M; ++k)
        {
          // column pivoting with row exchange
          double abspivot=std::abs(A[INDEX(k,k,n)]);
          int pivotrow=k;
          for (std::size_t i=k+1; i<n; ++i)
            if (std::abs(A[INDEX(i,k,n)])>abspivot)
              {
                abspivot = std::abs(A[INDEX(i,k,n)]);
                pivotrow = i;
              }
          if (pivotrow!=k)
            {
              VecWd Sk,Si;
              for (std::size_t j=0; j<n; j+=W) // whole row!
                {
                  // use SIMD for swap
                  Sk.load(&A[INDEX(k,j,n)]);
                  Si.load(&A[INDEX(pivotrow,j,n)]);
                  Sk.store(&A[INDEX(pivotrow,j,n)]);
                  Si.store(&A[INDEX(k,j,n)]);
                }
            }
          // elimination
          for (std::size_t i=k+1; i<K+M; ++i)
            {
              double lik = A[INDEX(i,k,n)]/A[INDEX(k,k,n)];
              A[INDEX(i,k,n)] = lik;
              for (std::size_t j=k+1; j<K+M; ++j)
                A[INDEX(i,j,n)] -= lik * A[INDEX(k,j,n)];
            }
        }
      
      // 1b) remaining blocks in first column
#pragma omp parallel for if ((n-K-M)/M>4) schedule (static) firstprivate(n,A)
      for (std::size_t I=K+M; I<n; I+=M)
        for (std::size_t i=I; i<I+M; ++i)
          for (std::size_t k=K; k<K+M; ++k)
          {
            double lik = A[INDEX(i,k,n)]/A[INDEX(k,k,n)];
            A[INDEX(i,k,n)] = lik;
            for (std::size_t j=k+1; j<K+M; ++j)
              A[INDEX(i,j,n)] -= lik * A[INDEX(k,j,n)];
          }

      // 2) Solve for U_KJ
#pragma omp parallel for if ((n-K-M)/M>4) schedule (static) firstprivate(n,A)
      for (std::size_t J=K+M; J<n; J+=M)
        for (std::size_t i=0; i<M; ++i)
          for (std::size_t k=0; k<i; ++k)
            {
              VecWd Lik,Uij,Akj;
              Lik = VecWd(A[INDEX(K+i,K+k,n)]);
              for (std::size_t j=0; j<M; j+=W)
                {
                  Uij.load(&A[INDEX(K+i,J+j,n)]);
                  Akj.load(&A[INDEX(K+k,J+j,n)]);
                  Uij = nmul_add(Lik,Akj,Uij);
                  Uij.store(&A[INDEX(K+i,J+j,n)]);
                }
            }
      
      // 3) update S
#pragma omp parallel for if ((n-K-M)/M>4) schedule (static) firstprivate(n,A) collapse (2)
      for (std::size_t I=K+M; I<n; I+=M)
        for (std::size_t J=K+M; J<n; J+=M)
          matmul_kernel_4x3<M,W>(n,&A[INDEX(I,K,n)],&A[INDEX(K,J,n)],&A[INDEX(I,J,n)]);
    }
}


// initialize all entries up to N
template<size_t W, size_t blockM>
void ludecomp_blocked2_vectorized_omp (int n, double A[])
{
  if (n%M!=0) exit(1);
  if (M%4!=0) exit(1);
  if (M%(2*W)!=0) exit(1);
  using VecWd = typename SIMDSelector<W>::SIMDType;

  for (std::size_t K=0; K<n; K+=M)
    {
      // 1a) LU decomposition of upper left block
      for (std::size_t k=K; k<K+M-1; ++k)
        for (std::size_t i=k+1; i<K+M; ++i)
          {
            double lik = A[INDEX(i,k,n)]/A[INDEX(k,k,n)];
            A[INDEX(i,k,n)] = lik;
            for (std::size_t j=k+1; j<K+M; ++j)
              A[INDEX(i,j,n)] -= lik * A[INDEX(k,j,n)];
          }
      
      // 1b) remaining blocks in first column
#pragma omp parallel for if ((n-K-M)/M>4) schedule (static) firstprivate(n,A)
      for (std::size_t I=K+M; I<n; I+=M)
        for (std::size_t i=I; i<I+M; ++i)
          for (std::size_t k=K; k<K+M; ++k)
          {
            double lik = A[INDEX(i,k,n)]/A[INDEX(k,k,n)];
            A[INDEX(i,k,n)] = lik;
            for (std::size_t j=k+1; j<K+M; ++j)
              A[INDEX(i,j,n)] -= lik * A[INDEX(k,j,n)];
          }

      // 2) Solve for U_KJ
#pragma omp parallel for if ((n-K-M)/M>4) schedule (static) firstprivate(n,A)
      for (std::size_t J=K+M; J<n; J+=M)
        for (std::size_t i=0; i<M; ++i)
          for (std::size_t k=0; k<i; ++k)
            {
              VecWd Lik,Uij,Akj;
              Lik = VecWd(A[INDEX(K+i,K+k,n)]);
              for (std::size_t j=0; j<M; j+=W)
                {
                  Uij.load(&A[INDEX(K+i,J+j,n)]);
                  Akj.load(&A[INDEX(K+k,J+j,n)]);
                  Uij = nmul_add(Lik,Akj,Uij);
                  Uij.store(&A[INDEX(K+i,J+j,n)]);
                }
            }
                
      // 3) update S
      std::size_t remaining_blocks = (n-(K+M))/M; // should be divisible
      std::size_t superblocks = remaining_blocks/blockM;
      // superblocks
#pragma omp parallel for schedule (static) firstprivate(n,A,superblocks)
      for (std::size_t superblock=0; superblock<superblocks*superblocks; ++superblock)
	{
	  std::size_t superblocki = superblock/superblocks;
	  std::size_t superblockj = superblock%superblocks;
	  std::size_t II = K+M+superblocki*blockM*M;
	  std::size_t JJ = K+M+superblockj*blockM*M;
	  VecWd CC[4][3], BB[3], AA; // fits exactly 16 registers
	  for (std::size_t I=II; I<II+blockM*M; I+=M)
	    for (std::size_t J=JJ; J<JJ+blockM*M; J+=M)
	      matmul_kernel_4x2<M,W>(n,&A[INDEX(I,K,n)],&A[INDEX(K,J,n)],&A[INDEX(I,J,n)]);
	}
      // tail loops
      std::size_t n_end = K+M+superblocks*blockM*M;
#pragma omp parallel for if ((n_end-K-M)/M>4) schedule (static) firstprivate(n,A,n_end)
      for (std::size_t I=K+M; I<n_end; I+=M)
        for (std::size_t J=n_end; J<n; J+=M)
          matmul_kernel_4x2<M,W>(n,&A[INDEX(I,K,n)],&A[INDEX(K,J,n)],&A[INDEX(I,J,n)]);
#pragma omp parallel for if ((n_end-K-M)/M>4) schedule (static) firstprivate(n,A,n_end)
      for (std::size_t I=n_end; I<n; I+=M)
        for (std::size_t J=K+M; J<n; J+=M)
          matmul_kernel_4x2<M,W>(n,&A[INDEX(I,K,n)],&A[INDEX(K,J,n)],&A[INDEX(I,J,n)]);
    }
}

// initialize all entries up to N
template<size_t W, size_t blockM>
void ludecomp_blocked2_vectorized_omp_avx512 (int n, double A[])
{
  if (n%M!=0) exit(1);
  if (M%8!=0) exit(1);
  if (M%(2*W)!=0) exit(1);
  using VecWd = typename SIMDSelector<W>::SIMDType;

  for (std::size_t K=0; K<n; K+=M)
    {
      // 1a) LU decomposition of upper left block
      for (std::size_t k=K; k<K+M-1; ++k)
        for (std::size_t i=k+1; i<K+M; ++i)
          {
            double lik = A[INDEX(i,k,n)]/A[INDEX(k,k,n)];
            A[INDEX(i,k,n)] = lik;
            for (std::size_t j=k+1; j<K+M; ++j)
              A[INDEX(i,j,n)] -= lik * A[INDEX(k,j,n)];
          }
      
      // 1b) remaining blocks in first column
#pragma omp parallel for if ((n-K-M)/M>4) schedule (static) firstprivate(n,A)
      for (std::size_t I=K+M; I<n; I+=M)
        for (std::size_t i=I; i<I+M; ++i)
          for (std::size_t k=K; k<K+M; ++k)
          {
            double lik = A[INDEX(i,k,n)]/A[INDEX(k,k,n)];
            A[INDEX(i,k,n)] = lik;
            for (std::size_t j=k+1; j<K+M; ++j)
              A[INDEX(i,j,n)] -= lik * A[INDEX(k,j,n)];
          }

      // 2) Solve for U_KJ
#pragma omp parallel for if ((n-K-M)/M>4) schedule (static) firstprivate(n,A)
      for (std::size_t J=K+M; J<n; J+=M)
        for (std::size_t i=0; i<M; ++i)
          for (std::size_t k=0; k<i; ++k)
            {
              VecWd Lik,Uij,Akj;
              Lik = VecWd(A[INDEX(K+i,K+k,n)]);
              for (std::size_t j=0; j<M; j+=W)
                {
                  Uij.load(&A[INDEX(K+i,J+j,n)]);
                  Akj.load(&A[INDEX(K+k,J+j,n)]);
                  Uij = nmul_add(Lik,Akj,Uij);
                  Uij.store(&A[INDEX(K+i,J+j,n)]);
                }
            }
                
      // 3) update S
      std::size_t remaining_blocks = (n-(K+M))/M; // should be divisible
      std::size_t superblocks = remaining_blocks/blockM;
      // superblocks
#pragma omp parallel for schedule (static,1) firstprivate(n,A,superblocks)
      for (std::size_t superblock=0; superblock<superblocks*superblocks; ++superblock)
	{
	  std::size_t superblocki = superblock/superblocks;
	  std::size_t superblockj = superblock%superblocks;
	  std::size_t II = K+M+superblocki*blockM*M;
	  std::size_t JJ = K+M+superblockj*blockM*M;
	  for (std::size_t I=II; I<II+blockM*M; I+=M)
	    for (std::size_t J=JJ; J<JJ+blockM*M; J+=M)
	      matmul_kernel_8x2<M,W>(n,&A[INDEX(I,K,n)],&A[INDEX(K,J,n)],&A[INDEX(I,J,n)]);
	}
      // tail loops
      std::size_t n_end = K+M+superblocks*blockM*M;
#pragma omp parallel for if ((n_end-K-M)/M>4) schedule (static,1) firstprivate(n,A,n_end)
      for (std::size_t I=K+M; I<n_end; I+=M)
        for (std::size_t J=n_end; J<n; J+=M)
          matmul_kernel_8x2<M,W>(n,&A[INDEX(I,K,n)],&A[INDEX(K,J,n)],&A[INDEX(I,J,n)]);
#pragma omp parallel for if ((n_end-K-M)/M>4) schedule (static,1) firstprivate(n,A,n_end)
      for (std::size_t I=n_end; I<n; I+=M)
        for (std::size_t J=K+M; J<n; J+=M)
          matmul_kernel_8x2<M,W>(n,&A[INDEX(I,K,n)],&A[INDEX(K,J,n)],&A[INDEX(I,J,n)]);
    }
}

// package an experiment as a functor
class Experiment {
  int n;
  double *A;
  double *B;
public:
  // construct an experiment
  Experiment (int n_, double* A_) : n(n_), A(A_)
  {
    B = new (std::align_val_t(64)) double[n*n];
  }
  ~Experiment ()
  {
    delete[] B;
  }
  // run an experiment; can be called several times
  void run () const {
    for (std::size_t i=0; i<n*n; ++i) B[i] = A[i];
    //ludecomp(n,B);
    //ludecomp_pivot(n,B);
    //ludecomp_ijk(n,B);
    //ludecomp_blocked(n,B);
    //ludecomp_blocked_vectorized<8>(n,B);
    //ludecomp_blocked_vectorized_omp<4>(n,B);
    //ludecomp_blocked2_vectorized_omp<8,4>(n,B);
    ludecomp_blocked2_vectorized_omp_avx512<8,4>(n,B);
    //ludecomp_blocked_vectorized_omp_pivot<4>(n,B);
  }
  // report number of operations
  double operations () const
  {return 0.6666*n*n*n;}
};

template<typename T>
void setupA (const T& integrateK, double alpha, double beta, int n, double A[])
{
  double h = (beta-alpha)/n;
  for (int i=0; i<n; i++)
    {
      double ci = alpha+(i+0.5)*h;
      A[INDEX(i,i,n)] = integrateK(ci,ci,0.5*h);
      for (int j=i+1; j<n; j++)
        A[INDEX(j,i,n)] = A[INDEX(i,j,n)] = integrateK(ci,ci+(j-i)*h,0.5*h);
    }
}

void test (int n, double A[])
{
  double* A1 = new (std::align_val_t(64)) double[n*n];
  double* A2 = new (std::align_val_t(64)) double[n*n];

  // make reference lu decomp in A1
  for (std::size_t i=0; i<n*n; ++i) A1[i] = A[i];
  ludecomp_pivot(n,A1);

  // // test ijk
  // for (std::size_t i=0; i<n*n; ++i) A2[i] = A[i];
  // ludecomp_ijk(n,A2);
  double error = 0.0;
  int ie=-1; int je=-1;
  // for (int i=0; i<n; ++i)
  //   for (int j=0; j<n; j++)
  //     if (std::abs(A2[INDEX(i,j,n)]-A1[INDEX(i,j,n)])>error)
  //       {
  //         error = std::abs(A2[INDEX(i,j,n)]-A1[INDEX(i,j,n)]);
  //         ie = i;
  //         je = j;
  //       }
  // std::cout << "error in ludecomp_ijk is " << error << " at " << ie << "," << je << std::endl;
  
  // // test blocked
  // for (std::size_t i=0; i<n*n; ++i) A2[i] = A[i];
  // ludecomp_blocked(n,A2);
  // error = 0.0;
  // ie=-1; je=-1;
  // for (int i=0; i<n; ++i)
  //   for (int j=0; j<n; j++)
  //     if (std::abs(A2[INDEX(i,j,n)]-A1[INDEX(i,j,n)])>error)
  //       {
  //         error = std::abs(A2[INDEX(i,j,n)]-A1[INDEX(i,j,n)]);
  //         ie = i;
  //         je = j;
  //       }
  // std::cout << "error in ludecomp_blocked is " << error << " at " << ie << "," << je << std::endl;

  // // test blocked vectorized
  for (std::size_t i=0; i<n*n; ++i) A2[i] = A[i];
  ludecomp_blocked_vectorized_omp_pivot<4>(n,A2);
  error = 0.0;
  ie=-1; je=-1;
  for (int i=0; i<n; ++i)
    for (int j=0; j<n; j++)
      if (std::abs(A2[INDEX(i,j,n)]-A1[INDEX(i,j,n)])>error)
        {
          error = std::abs(A2[INDEX(i,j,n)]-A1[INDEX(i,j,n)]);
          ie = i;
          je = j;
        }
  std::cout << "error in ludecomp_blocked_vectorized is " << error << " at " << ie << "," << je << std::endl;

  // print error matrix
  // for (int i=0; i<n; ++i)
  //   {
  //     for (int j=0; j<n; j++)
  //    std::cout << "[" << i << "," << j << "," << std::abs(A2[INDEX(i,j,n)]-A1[INDEX(i,j,n)]) << "]";
  //     std::cout << std::endl;
  //   }

  delete[] A2;
  delete[] A1;
}


int main (int argc, char** argv)
{
  int P=omp_get_max_threads();

  // read discretization parameter
  int n = 240;
  if (argc!=2)
    {
      std::cout << "usage: ./lu <size>" << std::endl;
      exit(1);
    }
  sscanf(argv[1],"%d",&n);

  // Interval
  double alpha = 0.0;
  double beta = 1.0;

  // integration kernel
  double gamma = 0.5;
  auto integrateK = [&] (double ci, double cj, double h2) {
    double d = std::abs(ci-cj);
    if (d<h2) return 2.0/(1-gamma)*pow(h2,1-gamma);
    return 1.0/(1-gamma)*(pow(d+h2,1-gamma)-pow(d-h2,1-gamma)); 
  };

  // test
  {
    // double *A = new (std::align_val_t(64)) double[n*n];
    // setupA(integrateK,alpha,beta,n,A);
    // test(n,A);
    // delete[] A;
    // exit(0);
  }
  
  // measure
  std::cout << "N, P=" << P << std::endl;
  while (n<16000)
    {
      double *A = new (std::align_val_t(64)) double[n*n];
      setupA(integrateK,alpha,beta,n,A);
      Experiment e(n,A);
      auto d = time_experiment(e);
      double flops = d.first*e.operations()/d.second*1e6/1e9;
      std::cout << n << ", " << flops << std::endl;
      delete[] A;
      n *= 2;
    }
  
  return 0;
}
