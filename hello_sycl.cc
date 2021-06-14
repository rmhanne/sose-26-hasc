#include <iostream>
#include <array>
#include <CL/sycl.hpp>

// custom device selector
// prints all devices available to the application
class MySelector : public sycl::device_selector {
public:
  int operator() (const sycl::device& dev) const
  {
    int score = -1;
    if (dev.is_host()) score += 100;
    if (dev.is_cpu()) score += 200;
    std::cout << "device: " 
              << dev.get_info<sycl::info::device::name>() 
              << std::endl;
    return score;
  }
};


int main (int argc, char** argv)
{
  // create a queue on the default device
  // queues take tasks for execution
  // tasks are running code or transferring data, etc
  // One queue is bound to a specific device
  sycl::queue Q{ MySelector{} };
  std::cout << "default device is " 
            << Q.get_device().get_info<sycl::info::device::name>() 
            << std::endl;

  // make some data on the host
  const int N=256;
  std::array<int,N> data;

  // create a buffer that transfers data between host and device
  // Buffers abstract 1,2 or 3-dimensional arrays of a type T
  // The type T must be trivially copyable (consecutive in memory)
  // buffers provide implicit data transfer
  // you can also manage the data transfers explicitly
  // Class template argument deduction is used to infer
  // template arguments of the buffer class
  // Here, an std:array is used to infer all template args for buffer
  // the runtime given size and initial values
  sycl::buffer B{data};

  // submit a task
  // Queues can handle task graphs given by a DAG
  // Here we have only one task
  Q.submit([&](sycl::handler& h) // lambda defines code for a node in task graph
    {
      // We now define a command group for this task
      //
      // the task needs to access data in the buffer
      // access is only possible through accessors
      // The accessor cares for implicit data movement to make
      // the data available in the device
      // A acts like a 1d array since our buffer is 1d
      sycl::accessor A{B,h};
      h.parallel_for(N,       // kernel invocation on device
        [=](auto& i){A[i]=i;} // this is the code executed on device
      );
    });

  // access the buffer now on host
  // this will imply that data is copied back from device
  // (if it is not in a shared memory)
  sycl::host_accessor A{B};
  for (int i=0; i<N; ++i)
    if (A[i]!=i)
      std::cout << "A[" << i << "]=" << A[i] << std::endl;

  return 0;
}
