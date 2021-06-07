#include <iostream>
#include <vector>

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
      sum += (A1[i][j]-A2[i][j])*(A1[i][j]-A2[i][j]);
  return sum;
}

// naive matmul C = A*B + C; this gives the right result for comparison
void matmul0 (int n, double A[N][N], double B[N][N], double C[N][N])
{
  int i,j,k;

  for (i=0; i<n; i++)
    for (j=0; j<n; j++)
      for (k=0; k<n; k++)
        C[i][j] += A[i][k]*B[k][j];
}

// non-tiled version with SIMD vectorization of 4x4 matmul C = A*B + C
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
	    // compute intensity of inner loop w.r.t. register loads: 128 flops / 256 bytes loaded = 0.5
          }
        Crow[0].store(&C[i][j]);
        Crow[1].store(&C[i+1][j]);
        Crow[2].store(&C[i+2][j]);
        Crow[3].store(&C[i+3][j]);
      }
}

// tiled version with SIMD vectorization of 4x4 matmul C = A*B + C
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

// tiling and SIMD with vectorization of 4x8 blocks
void matmul3 (int n, double A[N][N], double B[N][N], double C[N][N])
{
  Vec4d CC[4][2], BB[2], AA[2]; // use two A registers for prefetching
  
  for (int i=0; i<n; i+=M) // loop over tiles
    for (int j=0; j<n; j+=M)
      for (int k=0; k<n; k+=M)
        for (int s=i; s<i+M; s+=4) // loop over 4x8 blocks
          for (int t=j; t<j+M; t+=8)
	    // C_st is a 4x8 block which is loaded now
	    {
	      for (int p=0; p<4; ++p)
		{
		  // load store amortized over M/8 matrix multiplications
		  CC[p][0].load(&C[s+p][t]); CC[p][1].load(&C[s+p][t+4]); 
		}
	      for (int u=k; u<k+M; u+=8)
		for (int q=0; q<8; q+=1)
		  {
		    // 2 loads of B now amortized over ... 8 fmas
		    AA[0] = Vec4d(A[s+0][u+q]); // load-broadcast prefetch
		    BB[0].load(&B[u+q][t]); BB[1].load(&B[u+q][t+4]);
		    
		    // now process one half column of A
		    AA[1] = Vec4d(A[s+1][u+q]); // load-broadcast prefetch
		    CC[0][0] = mul_add(AA[0],BB[0],CC[0][0]);
		    CC[0][1] = mul_add(AA[0],BB[1],CC[0][1]);
		    
		    AA[0] = Vec4d(A[s+2][u+q]); // load-broadcast prefetch
		    CC[1][0] = mul_add(AA[1],BB[0],CC[1][0]);
		    CC[1][1] = mul_add(AA[1],BB[1],CC[1][1]);
		    
		    AA[1] = Vec4d(A[s+3][u+q]); // load-broadcast prefetch
		    CC[2][0] = mul_add(AA[0],BB[0],CC[2][0]);
		    CC[2][1] = mul_add(AA[0],BB[1],CC[2][1]);
		    
		    CC[3][0] = mul_add(AA[1],BB[0],CC[3][0]);
		    CC[3][1] = mul_add(AA[1],BB[1],CC[3][1]);
		  }
	      for (int p=0; p<4; ++p)
		{
		  CC[p][0].store(&C[s+p][t]); CC[p][1].store(&C[s+p][t+4]); 
		}
	    }
}

// tiling and SIMD with vectorization of 4x12 blocks
void matmul4 (int n, double A[N][N], double B[N][N], double C[N][N])
{
  Vec4d CC[4][3], BB[3], AA; // fits exactly 16 registers
  
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


// tiling and SIMD with vectorization of products of IxK and KxJ*4 matrices
template<int I, int J, int K>
void matmul5 (int n, double A[N][N], double B[N][N], double C[N][N])
{
  // check size
  if (n>N) {std::cout << "matrix too large " << n << " > " << N << std::endl; exit(1);}

  // check divisibilities
  if (n%M!=0) {std::cout << "matrix size and tile size " << n << " is not a multiple of " << M << std::endl; exit(1);}
  if (M%I!=0) {std::cout << "tile size and I blocks " << M << " is not a multiple of " << I << std::endl; exit(1);}
  if (M%K!=0) {std::cout << "tile size and K blocks " << M << " is not a multiple of " << K << std::endl; exit(1);}
  if (M%(J*4)!=0) {std::cout << "tile size and J blocks " << M << " is not a multiple of " << J*4 << std::endl; exit(1);}

  // check number of registers needed
  int registers = I*J + J + 1;
  if (registers>16)  {std::cout << "too many registers: " << registers << " is greater than 16" << std::endl; exit(1);}

  // data in registers
  Vec4d CC[I][J], BB[J], AA;

  // loop over tiles of size MxM
  for (int i=0; i<n; i+=M)
    for (int j=0; j<n; j+=M)
      for (int k=0; k<n; k+=M)
	// update subblock C_ij += A_ik*B_kj
        for (int s=i; s<i+M; s+=I)
          for (int t=j; t<j+M; t+=J*4)
	    {
	      // C_st is a IxJ*4 subblock of C_ij
	      for (int p=0; p<I; ++p)
		for (int q=0; q<J; ++q)
		  CC[p][q].load(&C[s+p][t+q*4]);
	      for (int u=k; u<k+M; u+=K)
		// C_st += A_su*B_ut where now A_su is IxK and B_ut is KxJ*4
		for (int r=0; r<K; r+=1) // columns of A / rows of B
		  {
		    for (int q=0; q<J; ++q)
		      BB[q].load(&B[u+r][t+q*4]);
		    for (int p=0; p<I; ++p)
		      {
			AA = Vec4d(A[s+p][u+r]); // load-broadcast
			for (int q=0; q<J; ++q)
			  CC[p][q] = mul_add(AA,BB[q],CC[p][q]);
		      }
		  }
	      // write back C
	      for (int p=0; p<I; ++p)
		for (int q=0; q<J; ++q)
		  CC[p][q].store(&C[s+p][t+q*4]);
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
class Experiment2 {
  int n;
public:
  // construct an experiment
  Experiment2 (int n_) : n(n_) {initialize(A1,B1,C1);}
  // run an experiment; can be called several times
  void run () const {matmul2(n,A1,B1,C1);}
  // report number of operations
  double operations () const
  {return 2.0*n*n*n;}
};

// package an experiment as a functor
class Experiment3 {
  int n;
public:
  // construct an experiment
  Experiment3 (int n_) : n(n_) {initialize(A1,B1,C1);}
  // run an experiment; can be called several times
  void run () const {matmul3(n,A1,B1,C1);}
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

// package an experiment as a functor
template<int I, int J, int K>
class Experiment5 {
  int n;
public:
  // construct an experiment
  Experiment5 (int n_) : n(n_) {initialize(A1,B1,C1);}
  // run an experiment; can be called several times
  void run () const {matmul5<I,J,K>(n,A1,B1,C1);}
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
    matmul0(size,A1,B1,C0);
    initialize(A1,B1,C1);
    // matmul2(size,A1,B1,C1);
    // std::cout << "matmul2 N=" << size << " diff=" << compare(size,C0,C1) << std::endl;
    // initialize(A1,B1,C1);
    // matmul3(size,A1,B1,C1);
    // std::cout << "matmul3 N=" << size << " diff=" << compare(size,C0,C1) << std::endl;
    // initialize(A1,B1,C1);
    // matmul4(size,A1,B1,C1);
    // std::cout << "matmul4 N=" << size << " diff=" << compare(size,C0,C1) << std::endl;
    matmul5<6,2,24>(size,A1,B1,C1);
    std::cout << "matmul5 N=" << size << " diff=" << compare(size,C0,C1) << std::endl;
  }
  std::vector<int> sizes;
  for (int i=M; i<=6500; i*=2) sizes.push_back(i);
  std::cout << "N, Exp4, Exp5_6_2_24" << std::endl;
  for (auto i : sizes)
    { 
      // Experiment1 e1(i);
      // Experiment2 e2(i);
      // Experiment3 e3(i);
      Experiment4 e4(i);
      Experiment5<6,2,24> e5(i);
      // auto d1 = time_experiment(e1,500000);
      // auto d2 = time_experiment(e2,500000);
      // auto d3 = time_experiment(e3,500000);
      auto d4 = time_experiment(e4,500000);
      auto d5 = time_experiment(e5,500000);
      // double flops1 = d1.first*e1.operations()/d1.second*1e6/1e9;
      // double flops2 = d2.first*e2.operations()/d2.second*1e6/1e9;
      // double flops3 = d3.first*e3.operations()/d3.second*1e6/1e9;
      double flops4 = d4.first*e4.operations()/d4.second*1e6/1e9;
      double flops5 = d5.first*e5.operations()/d5.second*1e6/1e9;
      std::cout << i
                // << ", " << flops1
		// << ", " << flops2
		// << ", " << flops3
                << ", " << flops4
                << ", " << flops5
                << std::endl;
    }
  return 0;
}
