#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#ifdef _OPENMP
#include<omp.h> // headers for runtime if available
#endif
#include "time_experiment.hh"
#include "matmul_experiment.hh"

#include <experimental/simd>
namespace stdx = std::experimental;

const int P = 12;   // basic block size is a multiple of 4, 8 and 12
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
  static const size_t num_rows = 8;
  static const size_t num_cols = 3;
};
template<>
struct SIMDSelector<4>
{
  static const size_t simd_width = 4;
  static const size_t num_rows = 4;
  static const size_t num_cols = 3;
};
template<>
struct SIMDSelector<8>
{
  static const size_t simd_width = 8;
  static const size_t num_rows = 8;
  static const size_t num_cols = 3;
};

void matmul_simd(int n, const double A[], const double B[], double C[])
{
  using simd_t = stdx::native_simd<double>;
  constexpr int simd_width = simd_t::size();
  constexpr int num_rows = SIMDSelector<simd_width>::num_rows;
  constexpr int num_cols = SIMDSelector<simd_width>::num_cols;
  constexpr int col_step = num_cols * simd_width;

  if (M%num_rows!=0) {
     std::cout << "M must be a multiple of num_rows" << std::endl;
     exit(1);
  } 
  if (M%col_step!=0) {
     std::cout << "M must be a multiple of num_cols*simd_width" << std::endl;
     exit(1);
  } 
  if (n%M!=0) {
     std::cout << "n must be a multiple of M" << std::endl;
     exit(1);
  } 

  simd_t CC[num_rows][num_cols], BB[num_cols], AA;
#pragma omp parallel for schedule (static) firstprivate(n,A,B,C) private(CC,BB,AA) collapse (2)
  for (int i = 0; i < n; i += M)
    for (int j = 0; j < n; j += M)
      for (int k = 0; k < n; k += M)
        for (int s = i; s < i + M; s += num_rows)
          for (int t = j; t < j + M; t += col_step)
          {
            for (int r = 0; r < num_rows; ++r)
              for (int c = 0; c < num_cols; ++c)
                CC[r][c].copy_from(&C[INDEX(s + r, t + c * simd_width, n)], stdx::vector_aligned);

            for (int u = k; u < k + M; u += 1)
            {
              for (int c = 0; c < num_cols; ++c)
                BB[c].copy_from(&B[INDEX(u, t + c * simd_width, n)], stdx::vector_aligned);

              for (int r = 0; r < num_rows; ++r)
              {
                AA = A[INDEX(s + r, u, n)];
                for (int c = 0; c < num_cols; ++c)
                  CC[r][c] += AA * BB[c];
              }
            }

            for (int r = 0; r < num_rows; ++r)
              for (int c = 0; c < num_cols; ++c)
                CC[r][c].copy_to(&C[INDEX(s + r, t + c * simd_width, n)], stdx::vector_aligned);
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
  std::cout << "std::simd vectorization with simd_width " << stdx::native_simd<double>::size() << std::endl;
  for (auto i : sizes)
    { 
      auto e = make_experiment(initialize, matmul_simd,i);
      auto d = time_experiment(e);
      double flops = d.first*e.operations()/d.second/1e9;
      std::cout << i
                << ", " << flops
                << std::endl;
    }
  return 0;
}
