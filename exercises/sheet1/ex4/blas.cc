#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

const int P = std::thread::hardware_concurrency(); // number of threads
const int N=P*1'000'000;            // problem size
const int loops = 100;
std::vector<double> x(N,1.0); // first vector
std::vector<double> y(N,2.0); // second vector
double alpha=4.2;
std::vector<double> ynew(N,0.0); // new y

// SPMD style programming
void f (int rank)
{
  for (int cnt=0; cnt<loops; ++cnt)
  {
    for (int i=(N*rank)/P; i<(N*(rank+1))/P; ++i) // no overflow because P|N
      ynew[i] = alpha*x[i]+y[i];
  }
}

int main ()
{
  std::cout << "using " << P << " threads" << std::endl;
  std::vector<std::thread> threads;
  for (int rank=0; rank<P; ++rank)
    threads.push_back(std::thread{f,rank}); // passing an argument !
  for (int rank=0; rank<P; ++rank)
    threads[rank].join();
  std::cout << "ynew[0] = " << ynew[0] << std::endl;
  std::cout << "ynew[N-1] = " << ynew[N-1] << std::endl;
}
