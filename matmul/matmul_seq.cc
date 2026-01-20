#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include "time_experiment.hh"
#include "matmul_experiment.hh"

const int P = 24;       // basic block size is a multiple of 4, 8 and 12
const int Q = 4;       // multiplier
const int M = P * Q;   // tile size
const int N = M * 64; // maximum problem size;

#define INDEX(i, j, n) ((i)*n + (j)) // row major layout
#define CMINDEX(i, j, n) ((j)*n + (i)) // column major layout

// initialize all entries
void initialize(int n, double A[], double B[], double C[])
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      A[INDEX(i, j, n)] = 3.333;
      B[INDEX(i, j, n)] = 1.0/3.333;
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

// textbook matmul in row major layout
void matmul0(int n, const double A[], const double B[], double C[])
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        C[INDEX(i, j, n)] += A[INDEX(i, k, n)] * B[INDEX(k, j, n)];
}

// flip j and k loops in row major layout
void matmul1(int n, const double A[], const double B[], double C[])
{
  for (int i = 0; i < n; i++)
    for (int k = 0; k < n; k++)
      for (int j = 0; j < n; j++)
        C[INDEX(i, j, n)] += A[INDEX(i, k, n)] * B[INDEX(k, j, n)];
}

// tiling in MxM blocks and flipped last two loops in row major layout
void matmul2(int n, const double A[], const double B[], double C[])
{
  for (int i = 0; i < n; i += M)
    for (int j = 0; j < n; j += M)
      for (int k = 0; k < n; k += M)
        for (int s = i; s < i + M; s += 1)
          for (int u = k; u < k + M; u += 1)
            for (int t = j; t < j + M; t += 1)
              C[INDEX(s, t, n)] += A[INDEX(s, u, n)] * B[INDEX(u, t, n)];
}

// textbook matmul in column major layout with loop flip
void matmul3(int n, const double A[], const double B[], double C[])
{
  for (int j = 0; j < n; j++)
    for (int k = 0; k < n; k++)
      for (int i = 0; i < n; i++)
        C[CMINDEX(i, j, n)] += A[CMINDEX(i, k, n)] * B[CMINDEX(k, j, n)];
}


int main(int argc, char **argv)
{
  // print cpu name; works only on linux
  //auto rv = std::system("cat /proc/cpuinfo | grep 'model name' | tail -1 > model.txt"); // executes the UNIX command "ls -l >test.txt"
  //std::cout << std::ifstream("model.txt").rdbuf();

  // determine sizes to be tested
  std::vector<int> sizes;
  for (int i = M; i <= N; i *= 2)
    sizes.push_back(i);

  // run experiments
  for (auto i : sizes)
  {
    auto e0 = make_experiment(initialize,matmul0,i);
    auto d0 = time_experiment(e0);
    double flops0 = d0.first * e0.operations() / d0.second / 1e9;

    auto e1 = make_experiment(initialize,matmul1,i);
    auto d1 = time_experiment(e1);
    double flops1 = d1.first * e1.operations() / d1.second / 1e9;

    auto e2 = make_experiment(initialize,matmul2,i);
    auto d2 = time_experiment(e2);
    double flops2 = d2.first * e2.operations() / d2.second / 1e9;

    auto e3 = make_experiment(initialize,matmul3,i);
    auto d3 = time_experiment(e3);
    double flops3 = d3.first * e3.operations() / d3.second / 1e9;

    std::cout << i
              << ", " << flops0
              << ", " << flops1
              << ", " << flops2
              << ", " << flops3
              << std::endl;
  }
  return 0;
}
