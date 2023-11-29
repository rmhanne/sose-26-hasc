#include <iostream>
#include <vector>
#include "time_experiment.hh"

#ifdef _OPENMP
#include <omp.h> // headers for runtime if available
#endif

#define HAVE_NEON 1

#ifdef HAVE_VCL
#include "vcl/vectorclass.h"
#endif

#ifdef HAVE_NEON
#include <arm_neon.h>
#endif

#ifndef __GNUC__
#define __restrict__
#endif

const int P = 48;      // basic block size is a multiple of 4, 8 and 12
const int M = P;       // tile size
const int N = M * 256; // maximum problem size;

#define INDEX(i, j, n) ((i) * n + (j))

// initialize matrix entries
// use same partitioning to facilitate first touch policy for C
void initialize_vanilla(int n, double A[], double B[], double C[])
{
  // initialize A, B by thread 0
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
    {
      A[INDEX(i, j, n)] = (1.0 * i * j) / (n * n);
      B[INDEX(i, j, n)] = (1.0 + i + j) / n;
    }
    // initialize C in parallel
#pragma omp parallel for schedule(static) collapse(2) firstprivate(n, A, B, C)
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
    {
      C[INDEX(i, j, n)] = 0.0;
    }
}

// initialize matrix entries in blocks
// use same partitioning to facilitate first touch policy
void initialize_blocked(int n, double A[], double B[], double C[])
{
  // initialize A, B by thread 0
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
    {
    }
#pragma omp parallel for schedule(static) collapse(2) firstprivate(n, A, B, C)
  for (int i = 0; i < n; i += M)
    for (int j = 0; j < n; j += M)
    {
//      int rank = omp_get_thread_num();
#pragma omp critical
      //      std::cout << rank << ": i=" << i << " j=" << j << std::endl;
      for (int s = i; s < i + M; s += 1)
        for (int t = j; t < j + M; t += 1)
        {
          A[INDEX(s, t, n)] = (1.0 * i * j) / (n * n);
          B[INDEX(s, t, n)] = (1.0 + i + j) / n;
          C[INDEX(s, t, n)] = 0.0;
        }
    }
}

// norm of difference of two matrices
double compare(int n, double A1[], double A2[])
{
  int i, j;
  double sum = 0.0;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      sum += std::abs(A1[INDEX(i, j, n)] - A2[INDEX(i, j, n)]);
  return sum;
}

// naive matmul C = A*B + C; this gives the right result for comparison
void matmul0(int n, const double A[], const double B[], double C[])
{
#pragma omp parallel for schedule(static) collapse(2) firstprivate(n, A, B, C)
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        C[INDEX(i, j, n)] += A[INDEX(i, k, n)] * B[INDEX(k, j, n)];
}

// naive matmul C = A*B + C; this gives the right result for comparison
void matmul1(int n, const double A[], const double B[], double C[])
{
#pragma omp parallel for schedule(static) collapse(2) firstprivate(n, A, B, C)
  for (int i = 0; i < n; i += M)
    for (int j = 0; j < n; j += M)
      for (int k = 0; k < n; k += M)
        for (int s = i; s < i + M; s += 1)
          for (int u = k; u < k + M; u += 1)
#pragma omp simd simdlen(4)
            for (int t = j; t < j + M; t += 1)
              C[INDEX(s, t, n)] += A[INDEX(s, u, n)] * B[INDEX(u, t, n)];
}

// naive matmul C = A*B + C; this gives the right result for comparison
void matmul1_buffered(int n, const double A[], const double B[], double C[])
{
#pragma omp parallel
  {
    double *clocal;
    clocal = new (std::align_val_t(64)) double[M * M];
#pragma omp for schedule(static) collapse(2) firstprivate(n, A, B, C)
    for (int i = 0; i < n; i += M)
      for (int j = 0; j < n; j += M)
      {
        for (int s = 0; s < M; s += 1)
          for (int t = 0; t < M; t += 1)
            clocal[s * M + t] = 0.0;

        for (int k = 0; k < n; k += M)
          for (int s = i; s < i + M; s += 1)
            for (int u = k; u < k + M; u += 1)
#pragma omp simd simdlen(4)
              for (int t = j; t < j + M; t += 1)
                clocal[(s - i) * M + (t - j)] += A[INDEX(s, u, n)] * B[INDEX(u, t, n)];

        for (int s = 0; s < M; s += 1)
          for (int t = 0; t < M; t += 1)
            C[INDEX(i + s, j + t, n)] += clocal[s * M + t];
      }
    delete[] clocal;
  }
}

#ifdef HAVE_VCL
// tiling and SIMD with vectorization of 4x12 blocks
void matmul4(int n, const double A[], const double B[], double C[])
{
  Vec4d CC[4][3], BB[3], AA; // fits exactly 16 registers

#pragma omp parallel for schedule(static) firstprivate(n, A, B, C) private(CC, BB, AA) collapse(2)
  for (int i = 0; i < n; i += M) // loop over tiles
    for (int j = 0; j < n; j += M)
      for (int k = 0; k < n; k += M)
        for (int s = i; s < i + M; s += 4) // loop over 4x12 blocks of C within the tiles
          for (int t = j; t < j + M; t += 12)
          {
            // C_st is aa 4x12 block which is loaded now
            for (int p = 0; p < 4; ++p)
            {
              // load store amortized over M/8 matrix multiplications
              CC[p][0].load(&C[INDEX(s + p, t, n)]);
              CC[p][1].load(&C[INDEX(s + p, t + 4, n)]);
              CC[p][2].load(&C[INDEX(s + p, t + 8, n)]);
            }
            for (int u = k; u < k + M; u += 12)
              // C_st += A_su*B_ut where now A_su is 4x12 and B_ut is 12x12
              for (int q = 0; q < 12; q += 1) // columns of A / rows of B
              {
                // 3 loads of B now amortized over ... 12 fmas
                BB[0].load(&B[INDEX(u + q, t, n)]);
                BB[1].load(&B[INDEX(u + q, t + 4, n)]);
                BB[2].load(&B[INDEX(u + q, t + 8, n)]);

                AA = Vec4d(A[INDEX(s, u + q, n)]); // load-broadcast
                CC[0][0] = mul_add(AA, BB[0], CC[0][0]);
                CC[0][1] = mul_add(AA, BB[1], CC[0][1]);
                CC[0][2] = mul_add(AA, BB[2], CC[0][2]);

                AA = Vec4d(A[INDEX(s + 1, u + q, n)]); // load-broadcast
                CC[1][0] = mul_add(AA, BB[0], CC[1][0]);
                CC[1][1] = mul_add(AA, BB[1], CC[1][1]);
                CC[1][2] = mul_add(AA, BB[2], CC[1][2]);

                AA = Vec4d(A[INDEX(s + 2, u + q, n)]); // load-broadcast
                CC[2][0] = mul_add(AA, BB[0], CC[2][0]);
                CC[2][1] = mul_add(AA, BB[1], CC[2][1]);
                CC[2][2] = mul_add(AA, BB[2], CC[2][2]);

                AA = Vec4d(A[INDEX(s + 3, u + q, n)]); // load-broadcast
                CC[3][0] = mul_add(AA, BB[0], CC[3][0]);
                CC[3][1] = mul_add(AA, BB[1], CC[3][1]);
                CC[3][2] = mul_add(AA, BB[2], CC[3][2]);

                // compute intensity of inner loop w.r.t. register loads: 96 flops/128 bytes = 0.75
              }
            // write back C
            for (int p = 0; p < 4; ++p)
            {
              // load store amortized over M/8 matrix multiplications
              CC[p][0].store(&C[INDEX(s + p, t, n)]);
              CC[p][1].store(&C[INDEX(s + p, t + 4, n)]);
              CC[p][2].store(&C[INDEX(s + p, t + 8, n)]);
            }
          }
}
#endif

#ifdef HAVE_NEON
void matmul4(int n, const double A[], const double B[], double C[])
{

#pragma omp parallel
  {
    float64x2_t CC[4][3], BB[3], AA; // fits exactly 16 registers
    double *clocal;
    clocal = new (std::align_val_t(64)) double[M * M];
    // loop over the block matrices
#pragma omp for schedule(static) firstprivate(n, A, B, C) private(CC, BB, AA) collapse(2)
    for (int i = 0; i < n; i += M)
      for (int j = 0; j < n; j += M)
      {
        // clear block write buffer
        for (int s = 0; s < M; s += 1)
          for (int t = 0; t < M; t += 1)
            clocal[s * M + t] = 0.0;

        for (int k = 0; k < n; k += M)
          // multiplication of two block matrices C_ij = A_ik*B_kj
          for (int s = i; s < i + M; s += 4)   // process four rows together
            for (int t = j; t < j + M; t += 6) // process 3*2=6 columns together
            {
              // update a block of 4x6 elements of C_ij
              CC[0][0] = vld1q_f64(&clocal[INDEX(s - i, t - j, M)]);
              CC[0][1] = vld1q_f64(&clocal[INDEX(s - i, t - j + 2, M)]);
              CC[0][2] = vld1q_f64(&clocal[INDEX(s - i, t - j + 4, M)]);
              CC[1][0] = vld1q_f64(&clocal[INDEX(s - i + 1, t - j, M)]);
              CC[1][1] = vld1q_f64(&clocal[INDEX(s - i + 1, t - j + 2, M)]);
              CC[1][2] = vld1q_f64(&clocal[INDEX(s - i + 1, t - j + 4, M)]);
              CC[2][0] = vld1q_f64(&clocal[INDEX(s - i + 2, t - j, M)]);
              CC[2][1] = vld1q_f64(&clocal[INDEX(s - i + 2, t - j + 2, M)]);
              CC[2][2] = vld1q_f64(&clocal[INDEX(s - i + 2, t - j + 4, M)]);
              CC[3][0] = vld1q_f64(&clocal[INDEX(s - i + 3, t - j, M)]);
              CC[3][1] = vld1q_f64(&clocal[INDEX(s - i + 3, t - j + 2, M)]);
              CC[3][2] = vld1q_f64(&clocal[INDEX(s - i + 3, t - j + 4, M)]);

              for (int u = k; u < k + M; u += 1) // four rows times six columns
              {
                // load elements of B
                BB[0] = vld1q_f64(&B[INDEX(u, t, n)]);
                BB[0] = vld1q_f64(&B[INDEX(u, t + 2, n)]);
                BB[0] = vld1q_f64(&B[INDEX(u, t + 4, n)]);

                AA = vmovq_n_f64(A[INDEX(s, u, n)]); // load-broadcast
                CC[0][0] = vfmaq_f64(CC[0][0], AA, BB[0]);
                CC[0][1] = vfmaq_f64(CC[0][1], AA, BB[1]);
                CC[0][2] = vfmaq_f64(CC[0][2], AA, BB[2]);

                AA = vmovq_n_f64(A[INDEX(s + 1, u, n)]); // load-broadcast
                CC[1][0] = vfmaq_f64(CC[1][0], AA, BB[0]);
                CC[1][1] = vfmaq_f64(CC[1][1], AA, BB[1]);
                CC[1][2] = vfmaq_f64(CC[1][2], AA, BB[2]);

                AA = vmovq_n_f64(A[INDEX(s + 2, u, n)]); // load-broadcast
                CC[2][0] = vfmaq_f64(CC[2][0], AA, BB[0]);
                CC[2][1] = vfmaq_f64(CC[2][1], AA, BB[1]);
                CC[2][2] = vfmaq_f64(CC[2][2], AA, BB[2]);

                AA = vmovq_n_f64(A[INDEX(s + 3, u, n)]); // load-broadcast
                CC[3][0] = vfmaq_f64(CC[3][0], AA, BB[0]);
                CC[3][1] = vfmaq_f64(CC[3][1], AA, BB[1]);
                CC[3][2] = vfmaq_f64(CC[3][2], AA, BB[2]);
              }

              // write back the update block of 4x6 elements from C_ij to memory
              vst1q_f64(&clocal[INDEX(s - i, t - j, M)], CC[0][0]);
              vst1q_f64(&clocal[INDEX(s - i, t - j + 2, M)], CC[0][1]);
              vst1q_f64(&clocal[INDEX(s - i, t - j + 4, M)], CC[0][2]);
              vst1q_f64(&clocal[INDEX(s - i + 1, t - j, M)], CC[1][0]);
              vst1q_f64(&clocal[INDEX(s - i + 1, t - j + 2, M)], CC[1][1]);
              vst1q_f64(&clocal[INDEX(s - i + 1, t - j + 4, M)], CC[1][2]);
              vst1q_f64(&clocal[INDEX(s - i + 2, t - j, M)], CC[2][0]);
              vst1q_f64(&clocal[INDEX(s - i + 2, t - j + 2, M)], CC[2][1]);
              vst1q_f64(&clocal[INDEX(s - i + 2, t - j + 4, M)], CC[2][2]);
              vst1q_f64(&clocal[INDEX(s - i + 3, t - j, M)], CC[3][0]);
              vst1q_f64(&clocal[INDEX(s - i + 3, t - j + 2, M)], CC[3][1]);
              vst1q_f64(&clocal[INDEX(s - i + 3, t - j + 4, M)], CC[3][2]);
            }

        // write back buffer
        for (int s = 0; s < M; s += 1)
          for (int t = 0; t < M; t += 1)
            C[INDEX(i + s, j + t, n)] += clocal[s * M + t];
      }
  }
}
#endif

// package an experiment as a functor
class Experiment0
{
  int n;
  double *A, *B, *C;

public:
  // construct an experiment
  Experiment0(int n_) : n(n_)
  {
    A = new (std::align_val_t(64)) double[n * n];
    B = new (std::align_val_t(64)) double[n * n];
    C = new (std::align_val_t(64)) double[n * n];
    initialize_vanilla(n, A, B, C);
  }
  ~Experiment0()
  {
    delete[] C;
    delete[] B;
    delete[] A;
  }
  // run an experiment; can be called several times
  void run() const { matmul0(n, A, B, C); }
  // report number of operations
  double operations() const
  {
    return 2.0 * n * n * n;
  }
};

// package an experiment as a functor
class Experiment1
{
  int n;
  double *A, *B, *C;

public:
  // construct an experiment
  Experiment1(int n_) : n(n_)
  {
    A = new (std::align_val_t(64)) double[n * n];
    B = new (std::align_val_t(64)) double[n * n];
    C = new (std::align_val_t(64)) double[n * n];
    initialize_blocked(n, A, B, C);
  }
  ~Experiment1()
  {
    delete[] C;
    delete[] B;
    delete[] A;
  }
  // run an experiment; can be called several times
  void run() const { matmul1(n, A, B, C); }
  // report number of operations
  double operations() const
  {
    return 2.0 * n * n * n;
  }
};

// package an experiment as a functor
class Experiment4
{
  int n;
  double *A, *B, *C;

public:
  // construct an experiment
  Experiment4(int n_) : n(n_)
  {
    A = new (std::align_val_t(64)) double[n * n];
    B = new (std::align_val_t(64)) double[n * n];
    C = new (std::align_val_t(64)) double[n * n];
    initialize_blocked(n, A, B, C);
  }
  ~Experiment4()
  {
    delete[] C;
    delete[] B;
    delete[] A;
  }
  // run an experiment; can be called several times
  void run() const { matmul4(n, A, B, C); }
  // report number of operations
  double operations() const
  {
    return 2.0 * n * n * n;
  }
};

int main(int argc, char **argv)
{
  int P = omp_get_max_threads();

  //  std::cout << "memory for 3 matrices in GByte: " << 3.0 * N * N * 8 / 1024 / 1024 / 1024 << std::endl;

  std::vector<int> sizes;
  for (int i = M * 4; i <= N; i *= 2)
    sizes.push_back(i);
#ifdef HAVE_NEON
  std::cout << "N, neon_P=" << P << std::endl;
#endif
#ifdef HAVE_VCL
  std::cout << "N, avx2_P=" << P << std::endl;
#endif
  for (auto i : sizes)
  {
    Experiment4 e4(i);
    auto d4 = time_experiment(e4, 500000);
    double flops4 = d4.first * e4.operations() / d4.second * 1e6 / 1e9;
    std::cout << i
              << ", " << flops4
              << std::endl;
  }
  return 0;
}
