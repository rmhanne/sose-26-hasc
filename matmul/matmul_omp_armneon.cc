#include <iostream>
#include <vector>
#ifdef _OPENMP
#include<omp.h> // headers for runtime if available
#endif
#include "time_experiment.hh"
#include "matmul_experiment.hh"

// for ARM intrinsics
#include <arm_neon.h>

const int P = 24;     // basic block size is a multiple of 4, 8 and 12
const int Q = 4;      // multiplier
const int M = P * Q;  // tile size
const int N = 16000;  // maximum problem size; 

#define INDEX(i, j, n) ((i)*n + (j)) // row major mapping
#define CMINDEX(i, j, n) ((j)*n + (i)) // column major

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
double compare(int n, const double A1[], const double A2[])
{
  int i, j;
  double sum = 0.0;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      sum += std::abs(A1[INDEX(i, j, n)] - A2[INDEX(i, j, n)]);
  return sum;
}

void matmul5(int n, const double A[], const double B[], double C[])
{
  float64x2_t CC[4][3], BB[3], AA; // fits exactly 16 registers

  if (M%4!=0) {
     std::cout << "M must be a multiple of 4" << std::endl;
     exit(1);
  } 
  if (M%6!=0) {
     std::cout << "M must be a multiple of 6" << std::endl;
     exit(1);
  } 
  if (n%M!=0) {
     std::cout << "n must be a multiple of M" << std::endl;
     exit(1);
  } 

  // loop over the block matrices
#pragma omp parallel for schedule (static) firstprivate(n,A,B,C) private(CC,BB,AA) collapse (2)
  for (int i = 0; i < n; i += M)
    for (int j = 0; j < n; j += M)
      for (int k = 0; k < n; k += M)
        // multiplication of two block matrices C_ij = A_ik*B_kj
        for (int s = i; s < i + M; s += 4)   // process four rows together
          for (int t = j; t < j + M; t += 6) // process 3*2=6 columns together
          {
            // update a block of 4x6 elements of C_ij
            CC[0][0] = vld1q_f64(&C[INDEX(s, t, n)]);
            CC[0][1] = vld1q_f64(&C[INDEX(s, t + 2, n)]);
            CC[0][2] = vld1q_f64(&C[INDEX(s, t + 4, n)]);
            CC[1][0] = vld1q_f64(&C[INDEX(s + 1, t, n)]);
            CC[1][1] = vld1q_f64(&C[INDEX(s + 1, t + 2, n)]);
            CC[1][2] = vld1q_f64(&C[INDEX(s + 1, t + 4, n)]);
            CC[2][0] = vld1q_f64(&C[INDEX(s + 2, t, n)]);
            CC[2][1] = vld1q_f64(&C[INDEX(s + 2, t + 2, n)]);
            CC[2][2] = vld1q_f64(&C[INDEX(s + 2, t + 4, n)]);
            CC[3][0] = vld1q_f64(&C[INDEX(s + 3, t, n)]);
            CC[3][1] = vld1q_f64(&C[INDEX(s + 3, t + 2, n)]);
            CC[3][2] = vld1q_f64(&C[INDEX(s + 3, t + 4, n)]);

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
            vst1q_f64(&C[INDEX(s, t, n)], CC[0][0]);
            vst1q_f64(&C[INDEX(s, t + 2, n)], CC[0][1]);
            vst1q_f64(&C[INDEX(s, t + 4, n)], CC[0][2]);
            vst1q_f64(&C[INDEX(s + 1, t, n)], CC[1][0]);
            vst1q_f64(&C[INDEX(s + 1, t + 2, n)], CC[1][1]);
            vst1q_f64(&C[INDEX(s + 1, t + 4, n)], CC[1][2]);
            vst1q_f64(&C[INDEX(s + 2, t, n)], CC[2][0]);
            vst1q_f64(&C[INDEX(s + 2, t + 2, n)], CC[2][1]);
            vst1q_f64(&C[INDEX(s + 2, t + 4, n)], CC[2][2]);
            vst1q_f64(&C[INDEX(s + 3, t, n)], CC[3][0]);
            vst1q_f64(&C[INDEX(s + 3, t + 2, n)], CC[3][1]);
            vst1q_f64(&C[INDEX(s + 3, t + 4, n)], CC[3][2]);
          }
}

int main(int argc, char **argv)
{
  // get number of threads used by openMP
  int nP=omp_get_max_threads();
  std::vector<int> sizes;
  for (int i = M; i <= N; i *= 2)
    sizes.push_back(i);
  std::cout << "N=" << N << " M=" << M << " threads=" << nP << std::endl;
  for (auto i : sizes)
  {
    auto e = make_experiment(initialize, matmul5,i);
    auto d = time_experiment(e);
    double flops = d.first*e.operations()/d.second/1e9;
    std::cout << i
              << ", " << flops
              << std::endl;
  }
  return 0;
}
