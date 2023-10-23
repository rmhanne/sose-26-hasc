#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

#include "time_experiment.hh"

using NUMBER=double;

const int P = 4;
const int N=32*1024*1024;     // problem size
std::vector<NUMBER> x(N,1.0); // first vector
std::vector<NUMBER> y(N,1.0); // second vector
std::vector<NUMBER> sums(P);  // sum of each thread (access to seperate memory locations is OK, §41.2.1)
std::vector<int> flags(P,0);   // flag that result is ready

// scalar product with lock-free sumation
void f (int rank, int n)
{
  // compute local sum
  NUMBER mysum=0.0;
  for (int i=(n*rank)/P; i<(n*(rank+1))/P; ++i)
    mysum += x[i]*y[i];
  sums[rank] = mysum;

  // parallel algorithm for global sum
  for (int stride=1; stride<P; stride*=2)
    if (rank%(2*stride)==0)
      {
        // add result from partner
        auto other = rank + stride;
        if (other<P)
          {
            while (flags[other]==0) // wait for result
              flags[0] = 0; // dummy write access to flag to prevent optimization!!
            sums[rank] += sums[other];
            flags[other] = 0; // reset flag
          }
      }
    else
      {
        // notify that result is ready
        flags[rank] = 1;
        while (flags[rank]==1) // wait until result has been read
	        flags[0] = 0; // dummy access
        break;
      }
}

// package an experiment as a functor
class Experiment
{
  int n;
public:
  // construct an experiment
  Experiment (int n_) : n(n_) {}
  // run an experiment; can be called several times
  void run () const
  {
    for (int rank=0; rank<P; ++rank) sums[rank] = 0;
    std::vector<std::thread> threads;
    for (int rank=0; rank<P; ++rank)
      threads.push_back(std::thread{f,rank,n});
    for (int rank=0; rank<P; ++rank)
      threads[rank].join();
  }
  // report number of operations
  double operations () const
  {
    return 2.0*n;
  }
};

int main ()
{
  std::cout << N*sizeof(NUMBER)/1024/1024 << " MByte per vector" << std::endl;
  std::vector<int> sizes = {256,1024,4096,16384,65536,262144,1048576,4*1048576,16*1048576,32*1048576};
  for (auto i : sizes)
    { 
      Experiment e(i);
      auto d = time_experiment(e);
      double flops = d.first*e.operations()/d.second*1e6/1e9;
      std::cout << "n=" << i << " took " << d.second << " us for " << d.first << " repetitions"
                << " " << flops << " Gflops/s"
                << " " << flops*8 << " GByte/s"
                << std::endl;
    }
  return 0;
}
