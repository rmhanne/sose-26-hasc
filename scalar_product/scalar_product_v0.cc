#include <iostream>
#include <vector>
#include <thread>

#include "time_experiment.hh"

using NUMBER=double;

const int N=32*1024*1024; // problem size
std::vector<NUMBER> x(N,1.0); // first vector
std::vector<NUMBER> y(N,1.0); // second vector
NUMBER sum=0.0;    // result

// the scalarproduct
void f (int n) // for simplicity arguments and result are global
{
  for (int i=0; i<n; ++i)
    sum += x[i]*y[i];
}

// package an experiment as a functor
class Experiment {
  int n;
public:
  // construct an experiment
  Experiment (int n_) : n(n_) {}
  // run an experiment; can be called several times
  void operator() () const { sum=0; f(n); }
  // report number of operations
  double operations () const {return 2.0*n;}
};

int main ()
{
  std::cout << N*sizeof(NUMBER)/1024/1024 << " MByte per vector" << std::endl;
  std::vector<int> sizes = {256,1024,4096,16384,65536,262144,1048576,4*1048576,16*1048576,32*1048576};
  for (auto i : sizes) { 
    Experiment e(i);
    auto d = time_experiment(e);
    double flops = d.first*e.operations()/d.second/1e9;
    std::cout << "n=" << i << " took " << d.second << " us for " << d.first << " repetitions"
	      << " " << flops << " Gflops/s"
	      << " " << flops*sizeof(NUMBER) << " GByte/s" << std::endl;
  }
  return 0;
}
