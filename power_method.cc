#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#include <thread>
#include <stdio.h>
#include <stdlib.h>
#include "time_experiment.hh"
#include "Barrier.hh"

// get alignment of a pointer in bytes
size_t alignment(const void *p)
{
  for (size_t m = 64; m > 1; m /= 2)
    if (((size_t)p) % m == 0)
      return m;
  return 1;
}

// initialize matrix in column major order
void initialize_matrix_cm(int n, int ibegin, int iend, double *A)
{
  int w = 10;
  for (int j = 0; j < n; j++)
    for (int i = ibegin; i < iend; i++)
    {
      if (std::abs(i - j) == 0)
        A[j * n + i] = 3.0;
      else if (std::abs(i - j) < w)
        A[j * n + i] = -1.0 / std::abs(i - j);
      else
        A[j * n + i] = 1e-10;
    }
}

// initialize matrix in row major order
void initialize_matrix(int n, int ibegin, int iend, double *A)
{
  int w = 10;
  for (int i = ibegin; i < iend; i++)
    for (int j = 0; j < n; j++)
    {
      if (std::abs(i - j) == 0)
        A[i * n + j] = 3.0;
      else if (std::abs(i - j) < w)
        A[i * n + j] = -1.0 / std::abs(i - j);
      else
        A[i * n + j] = 1e-10;
    }
}

// y = Ax, matrix assumed to be column major layout
// does matrix times vector for rows i \in [ibegin,iend)
void matvec_cm(int n, int ibegin, int iend, const double *A, const double *x, double *y)
{
  for (int i = ibegin; i < iend; i++)
    y[i] = 0.0;
  for (int j = 0; j < n; j++)
    for (int i = ibegin; i < iend; ++i)
      y[i] += A[j * n + i] * x[j];
}

// y = Ax, matrix assumed to be row major layout
// does matrix times vector for rows i \in [ibegin,iend)
void matvec(int n, int ibegin, int iend, const double *A, const double *x, double *y)
{
  for (int i = ibegin; i < iend; ++i)
    y[i] = 0.0;
  for (int i = ibegin; i < iend; ++i)
    for (int j = 0; j < n; j++)
      y[i] += A[i * n + j] * x[j];
}

// compute scalar product of two vectors
double scalar_product(int n, const double *x, const double *y)
{
  double sum = 0.0;
  for (int i = 0; i < n; i++)
    sum += x[i] * y[i];
  return sum;
}

// scale all entries of vector x by
void scale(int n, double a, double *x)
{
  for (int i = 0; i < n; i++)
    x[i] *= a;
}

// scale all entries of vector x by
void copy(int n, const double *x, double *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i];
}

// scale all entries of vector x by
void copy_scale(int n, double a, const double *x, double *y)
{
  for (int i = 0; i < n; i++)
    y[i] = a * x[i];
}

// sequential version of power method
double lambda_max(int n, double *A)
{
  // initialize matrix
  initialize_matrix(n, 0, n, A);

  // allocate and initialize vectors
  double *x = new (std::align_val_t{64}) double[n];
  double *y = new (std::align_val_t{64}) double[n];

  // prepare initial vector
  for (int i = 0; i < n; i++)
    x[i] = rand();
  double norm = std::sqrt(scalar_product(n, x, x));
  // std::cout << "Iteration " << 0 << " norm=" << norm << std::endl;
  scale(n, 1.0 / norm, x);
  double muold = 1.0;

  // power iteration
  for (int i = 1; i <= 1000; i++)
  {
    matvec(n, 0, n, A, x, y);            // y=Ax
    double mu = scalar_product(n, y, x); // Raleigh quotient
    // std::cout << "Iteration " << i << " mu=" << mu << " rate=" << std::abs(mu-muold)/std::abs(muold) << std::endl;
    if (std::abs(mu - muold) / std::abs(muold) < 1e-6)
    {
      muold = mu;
      break;
    }
    muold = mu;
    double norm = std::sqrt(scalar_product(n, y, y));
    copy_scale(n, 1.0 / norm, y, x);
  }

  delete[] y;
  delete[] x;
  return muold;
}

struct GlobalContext
{
  int nthreads; // number of threads involved

  // input data
  int n;                 // nxn matrix
  /* const */ double *A; // the matrix
  double *x;             // the initial guess
  double *y;             // temporary vector

  // shared state
  Barrier barrier;
  bool finished;

  // output data
  double lambda_max; // for the result
  int iterations;    // number of iterations needed

  // timing
  std::vector<double> elapsed;

  GlobalContext(int P)
      : nthreads(P), barrier(P), finished(false), elapsed(P, 0.0)
  {
  }
};

// threads executing the power method in parallel
// SPMD style; all threads execute the same code
void lambda_max_par_thread(std::shared_ptr<GlobalContext> context, int rank)
{
  int P = context->nthreads;
  int n = context->n;
  double muold = 1.0;
  int ibegin = rank * n / P;
  int iend = (rank + 1) * n / P;

  // initialize matrix
  initialize_matrix(n, ibegin, iend, context->A);

  // prepare summation vector
  for (int i = ibegin; i < iend; i++)
    context->y[i] = 0.0;

  // start timer
  auto start = get_time_stamp();

  // power iteration
  int i = 1;
  while (i <= 1000)
  {
    matvec(n, ibegin, iend, context->A, context->x, context->y); // parallel matvec y=Ax
    context->barrier.wait2(rank);
    if (rank == 0)
    {
      double mu = scalar_product(n, context->y, context->x); // Raleigh quotient
      double rate = std::abs(mu - muold) / std::abs(muold);
      if (rate < 1e-6)
        context->finished = true;
      muold = mu;
      double norm = std::sqrt(scalar_product(n, context->y, context->y));
      copy_scale(n, 1.0 / norm, context->y, context->x);
    }
    context->barrier.wait2(rank);
    if (context->finished)
      break;
    i++;
  }

  // stop timer
  auto stop = get_time_stamp();
  double elapsed = get_duration_seconds(start, stop);
  context->elapsed[rank] = elapsed;

  // store results
  if (rank == 0)
  {
    context->lambda_max = muold;
    context->iterations = i;
  }
}

// a wrapper function that starts all the threads
// executing the parallel power method
// how many threads we have can be deduced from flag.size()
double lambda_max_par(int P, int n, double *A)
{
  // get global context shared by all threads
  auto context = std::make_shared<GlobalContext>(P);

  // prepare input data
  context->n = n;
  context->A = A;
  context->x = new (std::align_val_t{64}) double[n];
  context->y = new (std::align_val_t{64}) double[n];

  // prepare initial vector
  for (int i = 0; i < n; i++)
    context->x[i] = rand();

  // now start the threads and wait for
  // the computation to finish
  std::vector<std::thread> th;
  for (int i = 0; i < P - 1; ++i)
    th.push_back(std::thread{lambda_max_par_thread, context, i + 1});
  lambda_max_par_thread(context, 0);
  for (int i = 0; i < P - 1; ++i)
    th[i].join();

  // get maximum time of all the threads
  auto itmax = std::max_element(context->elapsed.begin(), context->elapsed.end());
  double elapsed = *itmax;

  // release memory
  delete[] context->y;
  delete[] context->x;

  // compute bandwidth
  double bytes = (1.0 * n * n + n * 7) * 8 * context->iterations;
  std::cout << "duration=" << elapsed << " bandwidth=" << bytes / 1e9 / elapsed << " GBytes/second" << std::endl;
  std::cout << "iterations=" << context->iterations << " time_per_iteration=" << elapsed / context->iterations << std::endl;

  // and return the result
  return context->lambda_max;
}

void time_barrier_thread(std::shared_ptr<GlobalContext> context, int rank)
{
  int P = context->nthreads;
  int n = context->n;
  int count = 0;

  for (int i = 0; i < n; i++)
  {
    context->barrier.wait2(rank);
    count += 1;
  }

  if (rank == 0)
  {
    context->lambda_max = count;
    context->iterations = count;
  }
}

void time_barrier(int P, int n)
{
  // get global context shared by all threads
  auto context = std::make_shared<GlobalContext>(P);

  // prepare input data
  context->n = n;

  // start timer
  auto start = get_time_stamp();

  // now start the threads and wait for
  // the computation to finish
  std::vector<std::thread> th;
  for (int i = 0; i < P; ++i)
    th.push_back(std::thread{time_barrier_thread, context, i});
  for (int i = 0; i < P; ++i)
    th[i].join();

  // stop timer
  auto stop = get_time_stamp();
  double elapsed = get_duration_seconds(start, stop);

  // print result
  std::cout << "time barrier P=" << P << " duration=" << elapsed << " time per round=" << elapsed / n << " result=" << context->lambda_max << std::endl;
}

// thread function for timing matmul
void time_matmul_thread(std::shared_ptr<GlobalContext> context, int rank)
{
  int P = context->nthreads;
  int n = context->n;
  int ibegin = rank * n / P; // partition matrix rows
  int iend = (rank + 1) * n / P;
  double *A = context->A;

  // initialize matrix in row major order
  initialize_matrix(n, ibegin, iend, A);

  // prepare initial vector
  for (int i = ibegin; i < iend; i++)
    context->x[i] = rand();
  for (int i = ibegin; i < iend; i++)
    context->y[i] = 0.0;

  // start timer
  auto start = get_time_stamp();

  int i = 0;
  while (i < 1000)
  {
    matvec(n, ibegin, iend, context->A, context->x, context->y); // parallel matvec y=Ax
    i++;
  }

  // stop timer
  auto stop = get_time_stamp();
  double elapsed = get_duration_seconds(start, stop);
  context->elapsed[rank] = elapsed;

  // store results
  if (rank == 0)
  {
    context->lambda_max = 1.0;
    context->iterations = i;
  }
}

// starts parallel threads timing the parallel matmul
double time_matmul(int P, int n, double *A)
{
  // get global context shared by all threads
  auto context = std::make_shared<GlobalContext>(P);

  // prepare input data
  context->n = n;
  context->A = A;
  context->x = new (std::align_val_t{64}) double[n];
  context->y = new (std::align_val_t{64}) double[n];

  // start timer
  auto start = get_time_stamp();

  // now start the threads and wait for
  // the computation to finish
  std::vector<std::thread> th;
  for (int i = 0; i < P - 1; ++i)
    th.push_back(std::thread{time_matmul_thread, context, i + 1});
  time_matmul_thread(context, 0);
  for (int i = 0; i < P - 1; ++i)
    th[i].join();

  auto stop = get_time_stamp();
  double elapsed = get_duration_seconds(start, stop);

  // release memory
  delete[] context->y;
  delete[] context->x;

  // print result
  std::cout << "matmul duration=" << elapsed << " iterations=" << context->iterations << " time_per_iteration=" << elapsed / context->iterations << std::endl;
  double bytes = (1.0 * n * n + 2.0 * n) * 8 * context->iterations;
  std::cout << "bandwidth=" << bytes / 1e9 / elapsed << " GBytes/second" << std::endl;
  for (int i = 0; i < P; ++i)
    std::cout << i << ": duration=" << context->elapsed[i] << std::endl;

  // and return the result
  return context->lambda_max;
}

// main function runs the experiments and outputs results as csv
int main(int argc, char **argv)
{
  int n = 1024;
  int P = 1;

  if (argc != 3)
  {
    std::cout << "usage: ./power_method <size> <nthreads>" << std::endl;
    exit(1);
  }
  sscanf(argv[1], "%d", &n);
  sscanf(argv[2], "%d", &P);

  // time_barrier(P, n);
  // return 0;

  std::cout << "power_method: n=" << n << " P=" << P << std::endl;
  double *A = new (std::align_val_t{64}) double[n * n];

  // time_matmul(P, n, A);
  // delete[] A;
  // return 0;

  auto lambda = lambda_max_par(P, n, A);
  std::cout << "lambda_max=" << lambda << std::endl;
  delete[] A;

  return 0;
}
