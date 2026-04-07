#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

const int P = std::thread::hardware_concurrency(); // number of threads
const int N=25600;            // problem size
std::vector<double> x(N,1.0); // first vector
std::vector<double> y(N,1.0); // second vector
double sum=0.0;    // result
std::mutex m;      // ensure exclusive access to sum

// SPMD style programming with single lock
void f (int rank)
{
  double mysum=0.0;
  for (int i=(N*rank)/P; i<(N*(rank+1))/P; ++i) // beware of overflow
    mysum += x[i]*y[i];
  std::lock_guard<std::mutex> lock{m};
  sum += mysum;
}

int main ()
{
  std::vector<std::thread> threads;
  for (int rank=0; rank<P; ++rank)
    threads.push_back(std::thread{f,rank});
  for (int rank=0; rank<P; ++rank)
    threads[rank].join();
  std::cout << "main sum = " << sum << std::endl;
}
