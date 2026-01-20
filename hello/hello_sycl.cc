#include <iostream>
#include <array>
#include <vector>
#include <sycl/sycl.hpp>

// custom device selector
// prints all devices available to the application
// class MySelector : public sycl::device_selector {
// public:
//   int operator() (const sycl::device& dev) const
//   {
//     int score = -1;
//     if (dev.is_cpu()) score += 200;
//     if (dev.is_gpu() || dev.is_accelerator()) score += 300;
//     std::cout << "device: " 
//               << dev.get_info<sycl::info::device::name>() 
//               << std::endl;
//     return score;
//   }
// };


int main (int argc, char** argv)
{
  // create a queue on the default device
  // queues take tasks for execution
  // tasks are running code or transferring data, etc
  // One queue is bound to a specific device
  sycl::queue Q{ sycl::cpu_selector_v };
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

  // a basic data parallel kernel with a 2d execution range
  Q.submit([&](sycl::handler& h)
    {
      h.parallel_for(
        sycl::range{10,20}, // index range to loop over
        [=](sycl::id<2> idx){int m = idx[0]+idx[1];} // one invoke per index
      );
    });

  // basic data parallel kernel with item instead of index
  Q.submit([&](sycl::handler& h)
    {
      h.parallel_for(
        sycl::range{10,20}, // index range to loop over
        [=](sycl::item<2> itm)
        {
          sycl::id<2> idx=itm.get_id(); 
          int m = idx[0]+idx[1];
        }
      );
    });

  // explicit ND-range kernel
  sycl::range global{64,64}; // 64x64 items
  sycl::range local{4,4};    // subdivided into 4x4 groups
  Q.submit([&](sycl::handler& h)
    {
      h.parallel_for(
        sycl::nd_range{global,local}, // index range to loop over
        [=](sycl::nd_item<2> itm)
        {
          sycl::id<2> idx = itm.get_global_id(); // id in whole range
          sycl::id<2> lidx = itm.get_local_id(); // id in group
          sycl::group<2> grp = itm.get_group(); // my work group
          sycl::id<2> grp_id = grp.get_group_id(); // group of this item
          sycl::range<2> grp_range = grp.get_local_range(); // size of group
          sycl::sub_group sgrp = itm.get_sub_group(); // my sub_group
          int m = idx[0]+idx[1];
        }
      );
    });

  // wait for all tasks submitted until now
  Q.wait();

  // more data; now with explicit transfer
  std::vector<double> xhost(256,0.0); // allocation in host
  double* xdevice = sycl::malloc_device<double>(xhost.size(),Q);
  Q.submit([&](sycl::handler& h){h.memcpy(xdevice,xhost.data(),xhost.size()*sizeof(double));});
  Q.wait();
  Q.submit([&](sycl::handler& h){h.parallel_for(xhost.size(), [=](sycl::id<1> i){xdevice[i]+=1.0;} );});
  Q.wait();
  Q.submit([&](sycl::handler& h){h.memcpy(xhost.data(),xdevice,xhost.size()*sizeof(double));});
  Q.wait();
  sycl::free(xdevice,Q);

  // access the buffer now on host
  // this will imply that data is copied back from device
  // (if it is not in a shared memory)
  sycl::host_accessor A{B};
  for (int i=0; i<N; ++i)
    if (A[i]!=i)
      std::cout << "A[" << i << "]=" << A[i] << std::endl;

  return 0;
}
