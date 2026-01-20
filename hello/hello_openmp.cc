#include <iostream>

#ifdef _OPENMP
#include<omp.h> // headers for runtime if available
#endif

int main (int argc, char** argv)
{
  // start sequential as usual
#pragma omp parallel // execute the following block in parallel
  { // number of parallel threads conrolled in various ways
    int id = omp_get_thread_num(); // call library function
#pragma omp critical // execute following block exclusive
    std::cout << "I am " << id << std::endl;
  } // join parallel threads at end of parallel regions

  // Example: parallel scalar product
  double x[100];
  double y[100];
  int n=100;
  double result=0.0;
  for (int i=0; i<n; i++) x[i]=1.0/(1.0+i);
  for (int i=0; i<n; i++) y[i]=(1.0+i);

#pragma omp parallel if (n>10) \
  num_threads (4) \
  shared (x,n) \
  reduction (+: result)
  {
    int P = omp_get_num_threads(); 
    int rank = omp_get_thread_num(); 
    for (int i= rank*n/P; i<(rank+1)*n/P; i++)
      result += x[i]*y[i];
  }
  std::cout << "scalar product is "
            << result << std::endl;

  // equivalent parallel for loop
  result = 0.0;
#pragma omp parallel if (n>10) \
  num_threads (4) \
  shared (x,y,n) \
  reduction (+: result)
  {
#pragma omp for schedule (static)
    for (int i=0; i<n; i++)
      result += x[i]*y[i];
  }
  std::cout << "scalar product is "
            << result << std::endl;

  // and a the combined form
  result = 0.0;
#pragma omp parallel for \
  num_threads (4) \
  schedule (static) \
  shared (x,y,n) \
  reduction (+: result)
  for (int i=0; i<n; i++) result += x[i]*y[i];
  std::cout << "scalar product is "
            << result << std::endl;


  result = 0.0;
#pragma omp parallel				\
  shared (x,y,n)				\
  reduction (+: result)
  {
#pragma omp sections
    {
#pragma omp section
      for (int i=0; i<25; i++) result += x[i]*y[i];
#pragma omp section
      for (int i=25;i<50; i++) result += x[i]*y[i];
#pragma omp section
      for (int i=50;i<75; i++) result += x[i]*y[i];
#pragma omp section
      for (int i=75;i<100; i++) result += x[i]*y[i];
    }
  }
  std::cout << "scalar product is "
            << result << std::endl;

  return 0;
}
