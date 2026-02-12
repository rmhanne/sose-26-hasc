#include <iostream>
#include <array>
#include <vector>
#include "time_experiment.hh"
#include <sycl/sycl.hpp>

// initialize all entries
template <typename T>
void initialize(int n, T A[], T B[], T C[])
{
  for (int i = 0; i < n; i += 1)
    for (int j = 0; j < n; j += 1)
    {
      A[i * n + j] = 3.333;
      B[i * n + j] = 1.0 / 3.333;
      C[i * n + j] = 0.0;
    }
}

// norm of difference of two matrices
template <typename T>
double compare(int n, T A1[], T A2[])
{
  double sum = 0.0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      sum += std::abs(A1[i * n + j] - A2[i * n + j]);
  return sum;
}

// sequential function for comparison
template <typename T>
void matmul(int n, const T A[], const T B[], T C[])
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      C[i*n+j] = 0.0;
  for (int i = 0; i < n; i++)
    for (int k = 0; k < n; k++)
      for (int j = 0; j < n; j++)
        C[i * n + j] += A[i * n + k] * B[k * n + j];
}

int main(int argc, char **argv)
{
  // define number type to be used in matrices
  using T = double;

  // define block size and matrix size
  const int M = 48;
  const int n = 64 * M;

  // create a queue on a cpu device
  sycl::queue q{sycl::cpu_selector_v};
  std::cout << "default device is "
            << q.get_device().get_info<sycl::info::device::name>()
            << std::endl;
  
  // print info
  auto dev = q.get_device();
  std::cout << "max_work_group_size=" << dev.get_info<sycl::info::device::max_work_group_size>() << std::endl;
  std::cout << "local_mem_size=" << dev.get_info<sycl::info::device::local_mem_size>() << std::endl;
  auto sgs = dev.get_info<sycl::info::device::sub_group_sizes>();
  for (int i=0; i<sgs.size(); i++)
    std::cout << "sub_group_sizes=" << sgs[i] << std::endl;

  // we use the explicit data movement strategy here
  // allocate three matrices on host
  T *host_A = new (std::align_val_t(64)) T[n * n];
  T *host_B = new (std::align_val_t(64)) T[n * n];
  T *host_C = new (std::align_val_t(64)) T[n * n];
  T *host_D = new (std::align_val_t(64)) T[n * n];

  // allocate three arrays on the device
  T *device_A = sycl::malloc_device<T>(n * n, q);
  T *device_B = sycl::malloc_device<T>(n * n, q);
  T *device_C = sycl::malloc_device<T>(n * n, q);

  // initialize data on host
  initialize(n, host_A, host_B, host_C);

  // transfer data to device
  q.submit([&](sycl::handler &h)
           { h.memcpy(device_A, host_A, n * n * sizeof(T)); });
  q.submit([&](sycl::handler &h)
           { h.memcpy(device_B, host_B, n * n * sizeof(T)); });
  q.submit([&](sycl::handler &h)
           { h.memcpy(device_C, host_C, n * n * sizeof(T)); });
  q.wait(); // wait for data to be transfered

  // do something on device
  auto kernel = [=](sycl::nd_item<2> item)
  {
    int i = item.get_global_id(0);
    int j = item.get_global_id(1);
    for (int k = 0; k < n; ++k)
      device_C[i * n + j] += device_A[i * n + k] * device_B[k * n + j];
  };
  sycl::range global{n,n}; // matrix size
  sycl::range local{M,M};  // tile size 
  std::cout << "n=" << n << ", M=" << M << std::endl;
  auto start = get_time_stamp();
  q.submit([&](sycl::handler &h)
           {
    h.parallel_for(sycl::nd_range{global,local}, kernel); });
  q.wait();
  auto stop = get_time_stamp();
  double elapsed = get_duration_seconds(start, stop);
  double flops = 2.0 * n * n * n;
  std::cout << elapsed << " seconds for " << flops << " flops -> " << flops / elapsed / 1e9 << " GFLOP/s" << std::endl;

  // copy data back from device
  q.submit([&](sycl::handler &h)
           { h.memcpy(host_A, device_A, n * n * sizeof(T)); });
  q.submit([&](sycl::handler &h)
           { h.memcpy(host_B, device_B, n * n * sizeof(T)); });
  q.submit([&](sycl::handler &h)
           { h.memcpy(host_C, device_C, n * n * sizeof(T)); });
  q.wait();

  // check result
//  matmul(n, host_A, host_B, host_D);
//  auto error = compare(n,host_C,host_D);
//  std::cout << "error=" << error << std::endl;

  // free memory on device
  sycl::free(device_C, q);
  sycl::free(device_B, q);
  sycl::free(device_A, q);

  // free memory on host
  delete[] host_D;
  delete[] host_C;
  delete[] host_B;
  delete[] host_A;

  return 0;
}
