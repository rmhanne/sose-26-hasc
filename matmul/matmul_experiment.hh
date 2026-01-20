#ifndef HASC_MATMUL_EXPERIMENT_HH
#define HASC_MATMUL_EXPERIMENT_HH

// package an experiment as a functor
template<typename T>
class ExperimentMatMul
{
  using PtrTypeI = void (*)(int, T*, T*, T*);
  using PtrTypeF = void (*)(int, const T*, const T*, T*);
  int n;
  PtrTypeI pI;
  PtrTypeF pF;
  T *A, *B, *C;

public:
  // construct an experiment
  ExperimentMatMul(PtrTypeI pI_, PtrTypeF pF_, int n_) : pI(pI_), pF(pF_), n(n_)
  {
    A = new (std::align_val_t(64)) T[n * n];
    B = new (std::align_val_t(64)) T[n * n];
    C = new (std::align_val_t(64)) T[n * n];
    (*pI)(n, A, B, C);
  }
  ~ExperimentMatMul()
  {
    delete[] C;
    delete[] B;
    delete[] A;
  }
  // run an experiment; can be called several times
  void operator() () const 
  { 
    (*pF)(n, A, B, C); 
  }
  // report number of operations
  double operations() const
  {
    return 2.0 * n * n * n;
  }
};

template<typename T>
ExperimentMatMul<T> make_experiment (void (*pI)(int, T*, T*, T*), void (*pF)(int, const T*, const T*, T*), int n)
{
  return ExperimentMatMul<T>(pI,pF,n);
}

#endif