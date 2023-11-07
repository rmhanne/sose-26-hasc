#include <iostream>
#include <vector>
#include <string>
#include "time_experiment.hh"
#include <arm_neon.h>

// B = A^T,
// A,B are nxn matrices stored row-major in a 1d array,
// assume A and B are NOT the same matrix

// vectorized transpose with 4x4 blocks and consecutive write
void transposeb4(int n, double *A, double *B)
{
  float64x2_t A00, A01, A10, A11, A20, A21, A30, A31;
  float64x2_t B00, B01, B10, B11, B20, B21, B30, B31;

  for (int i = 0; i < n; i += 4)   // split i loop in blocks of size M
    for (int j = 0; j < n; j += 4) // split j loop in blocks of size M
    {
      // load A
      A00 = vld1q_f64(&A[(j)*n + i]);
      A01 = vld1q_f64(&A[(j)*n + i + 2]);
      A10 = vld1q_f64(&A[(j + 1) * n + i]);
      A11 = vld1q_f64(&A[(j + 1) * n + i + 2]);
      A20 = vld1q_f64(&A[(j + 2) * n + i]);
      A21 = vld1q_f64(&A[(j + 2) * n + i + 2]);
      A30 = vld1q_f64(&A[(j + 3) * n + i]);
      A31 = vld1q_f64(&A[(j + 3) * n + i + 2]);

      // transpose
      B00 = A00;
      B00 = vcopyq_laneq_f64(B00, 1, A10, 0);
      B10 = A10;
      B10 = vcopyq_laneq_f64(B10, 0, A00, 1);

      B20 = A01;
      B20 = vcopyq_laneq_f64(B20, 1, A11, 0);
      B30 = A11;
      B30 = vcopyq_laneq_f64(B30, 0, A01, 1);

      B01 = A20;
      B01 = vcopyq_laneq_f64(B01, 1, A30, 0);
      B11 = A30;
      B11 = vcopyq_laneq_f64(B11, 0, A20, 1);

      B21 = A21;
      B21 = vcopyq_laneq_f64(B21, 1, A31, 0);
      B31 = A31;
      B31 = vcopyq_laneq_f64(B31, 0, A21, 1);
      // store B
      vst1q_f64(&B[(i)*n + j], B00);
      vst1q_f64(&B[(i)*n + j + 2], B01);
      vst1q_f64(&B[(i + 1) * n + j], B10);
      vst1q_f64(&B[(i + 1) * n + j + 2], B11);
      vst1q_f64(&B[(i + 1) * n + j], B20);
      vst1q_f64(&B[(i + 2) * n + j + 2], B21);
      vst1q_f64(&B[(i + 1) * n + j], B30);
      vst1q_f64(&B[(i + 2) * n + j + 2], B31);
    }
}

// vectorized transpose with 4x4 blocks and consecutive write & TLB blocking
void transposeb4_x(int n, double *A, double *B)
{
  float64x2_t A00, A01, A10, A11, A20, A21, A30, A31;
  float64x2_t B00, B01, B10, B11, B20, B21, B30, B31;

  const int M = 48;

  for (int I = 0; I < n; I += M)
    for (int J = 0; J < n; J += M)
      for (int i = I; i < I + M; i += 4)
        for (int j = J; j < J + M; j += 4)
        {
          // load A
          A00 = vld1q_f64(&A[(j)*n + i]);
          A01 = vld1q_f64(&A[(j)*n + i + 2]);
          A10 = vld1q_f64(&A[(j + 1) * n + i]);
          A11 = vld1q_f64(&A[(j + 1) * n + i + 2]);
          A20 = vld1q_f64(&A[(j + 2) * n + i]);
          A21 = vld1q_f64(&A[(j + 2) * n + i + 2]);
          A30 = vld1q_f64(&A[(j + 3) * n + i]);
          A31 = vld1q_f64(&A[(j + 3) * n + i + 2]);

          // transpose
          B00 = A00;
          B00 = vcopyq_laneq_f64(B00, 1, A10, 0);
          B10 = A10;
          B10 = vcopyq_laneq_f64(B10, 0, A00, 1);

          B20 = A01;
          B20 = vcopyq_laneq_f64(B20, 1, A11, 0);
          B30 = A11;
          B30 = vcopyq_laneq_f64(B30, 0, A01, 1);

          B01 = A20;
          B01 = vcopyq_laneq_f64(B01, 1, A30, 0);
          B11 = A30;
          B11 = vcopyq_laneq_f64(B11, 0, A20, 1);

          B21 = A21;
          B21 = vcopyq_laneq_f64(B21, 1, A31, 0);
          B31 = A31;
          B31 = vcopyq_laneq_f64(B31, 0, A21, 1);
          // store B
          vst1q_f64(&B[(i)*n + j], B00);
          vst1q_f64(&B[(i)*n + j + 2], B01);
          vst1q_f64(&B[(i + 1) * n + j], B10);
          vst1q_f64(&B[(i + 1) * n + j + 2], B11);
          vst1q_f64(&B[(i + 1) * n + j], B20);
          vst1q_f64(&B[(i + 2) * n + j + 2], B21);
          vst1q_f64(&B[(i + 1) * n + j], B30);
          vst1q_f64(&B[(i + 2) * n + j + 2], B31);
        }
}

void test()
{
  float64x2_t A00, A01,
      A10, A11,
      A20, A21,
      A30, A31;
  float64x2_t B00, B01,
      B10, B11,
      B20, B21,
      B30, B31;

  double A[16], B[16];

  for (int i = 0; i < 16; ++i)
    A[i] = (double)i;

  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
      std::cout << A[i * 4 + j] << " ";
    std::cout << std::endl;
  }

  A00 = vld1q_f64(&A[0]);
  A01 = vld1q_f64(&A[2]);
  A10 = vld1q_f64(&A[4]);
  A11 = vld1q_f64(&A[6]);
  A20 = vld1q_f64(&A[8]);
  A21 = vld1q_f64(&A[10]);
  A30 = vld1q_f64(&A[12]);
  A31 = vld1q_f64(&A[14]);

  B00 = vmovq_n_f64(1.0);
  B01 = vmovq_n_f64(1.0);
  B01 = vmovq_n_f64(10.0);
  B11 = vmovq_n_f64(10.0);
  B20 = vmovq_n_f64(100.0);
  B30 = vmovq_n_f64(100.0);
  B21 = vmovq_n_f64(1000.0);
  B31 = vmovq_n_f64(1000.0);

  // transpose
  B00 = A00;
  B10 = A10;
  B00 = vcopyq_laneq_f64(B00, 1, A10, 0);
  B10 = vcopyq_laneq_f64(B10, 0, A00, 1);

  B20 = A01;
  B30 = A11;
  B20 = vcopyq_laneq_f64(B20, 1, A11, 0);
  B30 = vcopyq_laneq_f64(B30, 0, A01, 1);

  B01 = A20;
  B11 = A30;
  B01 = vcopyq_laneq_f64(B01, 1, A30, 0);
  B11 = vcopyq_laneq_f64(B11, 0, A20, 1);

  B21 = A21;
  B31 = A31;
  B21 = vcopyq_laneq_f64(B21, 1, A31, 0);
  B31 = vcopyq_laneq_f64(B31, 0, A21, 1);

  vst1q_f64(&B[0], B00);
  vst1q_f64(&B[2], B01);
  vst1q_f64(&B[4], B10);
  vst1q_f64(&B[6], B11);
  vst1q_f64(&B[8], B20);
  vst1q_f64(&B[10], B21);
  vst1q_f64(&B[12], B30);
  vst1q_f64(&B[14], B31);

  std::cout << std::endl;
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
      std::cout << B[i * 4 + j] << " ";
    std::cout << std::endl;
  }
}

void test2()
{
  float64x2_t A00, A10;
  float64x2_t B00, B10;

  double A[4], B[4];

  for (int i = 0; i < 4; ++i)
    A[i] = (double)i;

  for (int i = 0; i < 2; ++i)
  {
    for (int j = 0; j < 2; ++j)
      std::cout << A[i * 2 + j] << " ";
    std::cout << std::endl;
  }

  A00 = vld1q_f64(&A[0]);
  A10 = vld1q_f64(&A[2]);

  B00 = vmovq_n_f64(111.0);
  B10 = vmovq_n_f64(111.0);

  // transpose
  B00 = A00;
  B00 = vcopyq_laneq_f64(B00, 1, A10, 0);
  B10 = A10;
  B10 = vcopyq_laneq_f64(B10, 0, A00, 1);

  vst1q_f64(&B[0], B00);
  vst1q_f64(&B[2], B10);

  std::cout << std::endl;
  for (int i = 0; i < 2; ++i)
  {
    for (int j = 0; j < 2; ++j)
      std::cout << B[i * 2 + j] << " ";
    std::cout << std::endl;
  }

  std::cout << std::endl;
  for (int i = 0; i < 2; ++i)
  {
    for (int j = 0; j < 2; ++j)
      std::cout << A[i * 2 + j] << " ";
    std::cout << std::endl;
  }
}

// initialize square matrix
void initialize(int n, double *A)
{
  for (int i = 0; i < n * n; i++)
    A[i] = i;
}

class Experimentb4
{
  int n;
  double *A, *B;

public:
  // construct an experiment
  Experimentb4(int n_) : n(n_)
  {
    std::cout << "Exp1: " << n << std::endl;
    A = new (std::align_val_t{64}) double[n * n];
    B = new (std::align_val_t{64}) double[n * n];
    initialize(n, A);
    initialize(n, B);
    if (((size_t)A) % 64 != 0)
    {
      std::cout << "Exp1: A not aligned to 64 " << std::endl;
    }
    if (((size_t)B) % 64 != 0)
    {
      std::cout << "Exp1: B not aligned to 64 " << std::endl;
    }
  }
  ~Experimentb4()
  {
    delete[] A;
    delete[] B;
  }
  // run an experiment; can be called several times
  void run() const
  {
    transposeb4_x(n, A, B);
  }
  // report number of operations for one run
  double operations() const
  {
    return n * n;
  }
};

// main function runs the experiments and outputs results as csv
int main(int argc, char **argv)
{
  std::vector<int> sizes; // vector with problem sizes to try
  // for (int i=16; i<=16384; i*=2) sizes.push_back(i);
  for (int i = 48; i <= 25000; i *= 2)
    sizes.push_back(i);

  std::vector<std::string> expnames; // name of experiment

  // experiment 1
  expnames.push_back("arm-neon-b4-M48");
  std::cout << expnames.back() << std::endl;
  std::vector<double> bandwidth1;
  for (auto n : sizes)
  {
    Experimentb4 e(n);
    auto d = time_experiment(e, 1000000);
    double result = d.first * e.operations() * 2 * sizeof(double) / d.second * 1e6 / 1e9;
    bandwidth1.push_back(result);
    std::cout << result << std::endl;
  }

  // output results
  // Note: size of TLB mentioned in https://www.realworldtech.com/haswell-cpu/5/
  std::cout << "N";
  for (std::string s : expnames)
    std::cout << ", " << s;
  std::cout << std::endl;
  for (int i = 0; i < sizes.size(); i++)
  {
    std::cout << sizes[i];
    std::cout << ", " << bandwidth1[i];
    std::cout << std::endl;
  }

  return 0;
}
