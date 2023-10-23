#include <iostream>
#include <vector>
#include "time_experiment.hh"

const int P = 24;       // basic block size is a multiple of 4, 8 and 12
const int Q = 4;       // multiplier
const int M = P * Q;   // tile size
const int N = M * 64; // maximum problem size;

#define INDEX(i, j, n) ((i)*n + (j)) // row major
// #define INDEX(i, j, n) ((j)*n + (i)) // column major

// initialize all entries up to N
void initialize(int n, double A[], double B[], double C[])
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      A[INDEX(i, j, n)] = (1.0 * i * j) / (n * n);
      B[INDEX(i, j, n)] = (1.0 + i + j) / n;
      C[INDEX(i, j, n)] = 0.0;
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

void matmul0(int n, const double A[], const double B[], double C[])
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        C[INDEX(i, j, n)] += A[INDEX(i, k, n)] * B[INDEX(k, j, n)];
}

void matmul1(int n, const double A[], const double B[], double C[])
{
  for (int i = 0; i < n; i += M)
    for (int j = 0; j < n; j += M)
      for (int k = 0; k < n; k += M)
        for (int s = i; s < i + M; s += 1)
          for (int u = k; u < k + M; u += 1)
            for (int t = j; t < j + M; t += 1)
              C[INDEX(s, t, n)] += A[INDEX(s, u, n)] * B[INDEX(u, t, n)];
}

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
    initialize(n, A, B, C);
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
    initialize(n, A, B, C);
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

int main(int argc, char **argv)
{
  std::cout << "memory for 3 matrices in GByte: " << 3.0 * N * N * 8 / 1024 / 1024 / 1024 << std::endl;

  std::vector<int> sizes;
  for (int i = M; i <= N; i *= 2)
    sizes.push_back(i);
  std::cout << "N=" << N << " M=" << M << std::endl;
  for (auto i : sizes)
  {
    Experiment0 e(i);
    auto d = time_experiment(e, 500000);
    double flops = d.first * e.operations() / d.second * 1e6 / 1e9;
    std::cout << i
              << ", " << flops
              << std::endl;
  }
  return 0;
}
