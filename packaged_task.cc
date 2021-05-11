#include <iostream>
#include <thread>
#include <future>
#include <cmath>
#include <algorithm>

// a vector of doubles
const int N=1000; 
double v[N]={1.0};

// computes Euclidean norm squared of a part of a vector
double norm2 (double* begin, double* end, double init)
{
  double sum=init;
  for (double* p=begin; p!=end; p++)
    sum += (*p)*(*p);
  return sum;
}

// compute norm in parallel using packaged tasks
double parallel_norm_pt (double* v, int N)
{
  // determine type of the function we want to
  // execute as a task
  using TaskType = decltype(norm2);

  // packed_task extracts arguments and return type
  // it holds a future/promise pair for the result
  // we do not see the promise explicitly because
  // the return value of norm2 is automagically put in the promise
  std::packaged_task<TaskType> pt0{norm2};
  std::packaged_task<TaskType> pt1{norm2};

  // get the future
  std::future<double> f0{pt0.get_future()};
  std::future<double> f1{pt1.get_future()};

  // now start the tasks
  std::thread t0{move(pt0),v,v+N/2,0.0};
  std::thread t1{move(pt1),v+N/2,v+N,0.0};

  // may do something else here

  // get the result
  double result = std::sqrt(f0.get()+f1.get());

  // join threads
  t0.join();
  t1.join();

  return result;
}

// compute norm in parallel using async
double parallel_norm_async (double* v, int N)
{
  // do yourself for small vectors
  if (N<100)
    return std::sqrt(norm2(v,v+N,0.0));

  // compute with up to four threads
  auto f0 = std::async(norm2,v,v+N/4,0.0);
  auto f1 = std::async(norm2,v+N/4,v+N/2,0.0);
  auto f2 = std::async(norm2,v+N/2,v+3*N/4,0.0);
  auto f3 = std::async(norm2,v+3*N/4,v+N,0.0);

  // may do something else here

  // get the result
  return std::sqrt(f0.get()+f1.get()+f2.get()+f3.get());
}

int main ()
{
  std::fill_n(v,N,-1.0);
  std::cout << "norm=" << parallel_norm_pt(v,N) << std::endl;
  std::cout << "norm=" << parallel_norm_async(v,N) << std::endl;
}
