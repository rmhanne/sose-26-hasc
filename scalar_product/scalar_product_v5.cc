#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "time_experiment.hh"

using NUMBER=double;

const int P = 8;
const int N=16*1024*1024;     // problem size
std::vector<NUMBER> x(N,1.0); // first vector
std::vector<NUMBER> y(N,1.0); // second vector
std::vector<NUMBER> sums(P);  // sum of each thread
std::vector<int> flags(P,0);   // flags[i]==1 signals that thread i provides the result
std::vector<std::mutex> ms(P); // mutexes
std::vector<std::condition_variable> cvs(P); // and condition variables

// scalar product with lock-free sumation
void f (int rank)
{
  // compute local sum
  NUMBER mysum=0.0;
  for (int i=(N*rank)/P; i<(N*(rank+1))/P; ++i)
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
            std::unique_lock<std::mutex> lock{ms[other]};
            cvs[other].wait(lock,[other]{return flags[other]==1;});
            sums[rank] += sums[other];
            flags[other] = 0; // reset flag
          }
      }
    else
      {
        // notify that result is ready
        std::unique_lock<std::mutex> lock{ms[rank]};
        flags[rank] = 1;
        cvs[rank].notify_one();
        break;
      }
}

int main ()
{
  std::vector<std::thread> threads;
  for (int rank=0; rank<P; ++rank)
    threads.push_back(std::thread{f,rank});
  for (int rank=0; rank<P; ++rank)
    threads[rank].join();
  std::cout << "main sum = " << sums[0] << std::endl;
}
