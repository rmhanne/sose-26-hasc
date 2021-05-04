#include <iostream>
#include <vector>
#include <random>

#include <Eigen/Dense>

#include "mkl.h"

#include "time_experiment.hh"

double dummy(0.0);

void initialize(const int n, double* a){
  std::uniform_real_distribution<double> unif(-1.0,1.0);
  std::default_random_engine re;
  for (int i=0; i<n; ++i){
    for (int j=0; j<n; ++j){
      a[i*n+j] = unif(re);
    }
  }
}

template<typename T>
void initialize_eigen(T& t){
  t = T::Random(t.rows(), t.cols());
}

// ====================
// Naive ipmlementation
// ====================
//
void matmul_naive(const int n, const double *a, const double *b, double *c){
  for (int i=0; i<n; ++i)
    for (int j=0; j<n; ++j)
      for (int k=0; k<n; ++k)
        c[i*n+j] = a[i*n+k] * b[k*n+j];
}

class ExperimentNaive{
  mutable double *_a, *_b, *_c;
  const int _n;
public:
  ExperimentNaive(int n) : _n(n){
    _a = new double[_n*_n];
    _b = new double[_n*_n];
    _c = new double[_n*_n];
    initialize(_n, _a);
    initialize(_n, _b);
    initialize(_n, _c);
  }
  ~ExperimentNaive(){
    delete[] _a;
    delete[] _b;
    delete[] _c;
  }
  void run() const{
    matmul_naive(_n, _a, _b, _c);

    // Prevent the compiler from optimizing away the loop.
    // The same results can be achieved by:
    //
    // dummy += _c[random_int]
    //
    // where randam_int is a random integer within [0,n-1]. I kept this version
    // as it is shorter.
    asm volatile("" : : "r,m"(_a) : "memory");
    asm volatile("" : : "r,m"(_c) : "memory");
    dummy += _c[0];
  }
  double operations() const{
    return 2.0*_n*_n*_n;
  }
};

// =====
// Eigen
// =====

template<typename T>
void matmul_eigen(const T& a, const T& b, T& c){
  c = a * b;
}

class ExperimentEigen{
  Eigen::MatrixXd _a, _b;
  mutable Eigen::MatrixXd _c;
  const int _n;
public:
  ExperimentEigen(int n) : _a(n,n), _b(n,n), _c(n,n), _n(n){
    initialize_eigen(_a);
    initialize_eigen(_b);
    initialize_eigen(_c);
  }
  void run() const{
    matmul_eigen(_a, _b, _c);
  }
  double operations() const{
    return 2.0*_n*_n*_n;
  }
};

// ===
// mkl
// ===

class ExperimentMkl{
  mutable double *_a, *_b, *_c;
  const int _n;
public:
  ExperimentMkl(int n) : _n(n){
    _a = (double *) mkl_malloc(_n*_n*sizeof(double), 64);
    _b = (double *) mkl_malloc(_n*_n*sizeof(double), 64);
    _c = (double *) mkl_malloc(_n*_n*sizeof(double), 64);
    initialize(_n, _a);
    initialize(_n, _b);
    initialize(_n, _c);
  }
  ~ExperimentMkl(){
    mkl_free(_a);
    mkl_free(_b);
    mkl_free(_c);
  }
  void run() const{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                _n, _n, _n,
                1.0,
                _a, _n,
                _b, _n,
                0.0,
                _c, _n);
  }
  double operations() const{
    return 2.0*_n*_n*_n;
  }
};

int main()
{
  std::vector<int> sizes;
  for (int i=8; i<1e4; i*=2)
    sizes.push_back(i);

  std::cout << "# Naive:" << std::endl;
  for (auto n: sizes){
    if (n>1e3){
      std::cout << "Aborted naive implementation as it gets too slow." << std::endl;
      break;
    }
    ExperimentNaive e(n);
    auto d = time_experiment(e);
    auto flops = d.first*e.operations()/d.second*1e6/1e9;
    std::cout << "n: " << n
              << "   GFlops/s: " << flops
              << std::endl;
  }
  std::cout << std::endl << "# Eigen:" << std::endl;
  for (auto n: sizes){
    ExperimentEigen e(n);
    auto d = time_experiment(e);
    auto flops = d.first*e.operations()/d.second*1e6/1e9;
    std::cout << "n: " << n
              << "   GFlops/s: " << flops
              << std::endl;
  }
  std::cout << std::endl << "# mkl" << std::endl;
  for (auto n: sizes){
    ExperimentMkl e(n);
    auto d = time_experiment(e);
    auto flops = d.first*e.operations()/d.second*1e6/1e9;
    std::cout << "n: " << n
              << "   GFlops/s: " << flops
              << std::endl;
  }

  return 0;
}
