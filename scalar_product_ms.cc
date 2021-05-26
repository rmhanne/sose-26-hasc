#include <iostream>
#include <thread>

#include "MessageSystem.hh"

double initializerx (int index)
{
  return 1.0/(1.0+index);
}

double initializery (int index)
{
  return 1.0+index;
}

void scalar_product_thread (const int rank, const int P, const int n)
{
  register_thread(rank);

  // data decomposition and initialization
  int istart = rank*n/P;
  int iend = (rank+1)*n/P;
  std::vector<double> x(iend-istart);
  std::vector<double> y(iend-istart);
  for (int i=0; i<x.size(); i++)
    {
      x[i] = initializerx(istart+i);
      y[i] = initializery(istart+i);
    }

  // compute the scalar product
  double sum = 0.0;
  for (int i=0; i<x.size(); i++)
    sum += x[i]*y[i];

  // now the reduction
  // get result from higher rank
  if (rank+1<P) {
    double s;
    recv(rank+1,s);
    sum += s;
  }
  // pass on result to lower rank
  if (rank-1>=0) send(rank-1,sum);

  // now rank 0 knows the result

  // broadcast the result
  if (rank-1>=0) recv(rank-1,sum);
  if (rank+1<P) send(rank+1,sum);
  
  // now all ranks have the result
  if (rank==P-1)
    std::cout << rank << ": result=" << sum << std::endl;
}

int main (int argc, char** argv)
{
  const int P=100;
  const int n=1000;
  initialize_message_system(P);
  std::vector<std::thread> threads;
  for (int i=0; i<P; ++i)
    threads.push_back(std::thread{scalar_product_thread,i,P,n});
  for (int i=0; i<P; ++i)
    threads[i].join();
}
