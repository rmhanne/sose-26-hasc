#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

const int N=256;              // problem size
std::vector<double> x(N,1.0); // first vector
std::vector<double> y(N,1.0); // second vector
double sum=0.0;    // result
std::mutex m;      // ensure exclusive access to sum

void f1 ()
{
  double mysum=0.0;
  for (int i=0; i<N/2; ++i)
    mysum += x[i]*y[i];
  m.lock();
  sum += mysum;
  m.unlock();
}

void f2 ()
{
  double mysum=0.0;
  for (int i=N/2; i<N; ++i)
    mysum += x[i]*y[i];
  m.lock();
  sum += mysum;
  m.unlock();
}

int main ()
{
  std::thread t1{f1};
  std::thread t2{f2};
  t1.join();
  t2.join();
  std::cout << "main sum = " << sum << std::endl;
}
