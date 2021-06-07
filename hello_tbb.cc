#include <iostream>
#include <vector>
#include <oneapi/tbb.h>

class VectorSumKernel
{
  std::vector<double> &x,&y,&z;
public:
  VectorSumWorker (std::vector<double>& _x, std::vector<double>& _y,
                   std::vector<double>& _z)
    : x(_x), y(_y), z(_z)
  {}
  void operator() (const oneapi::tbb::blocked_range<size_t>& r) const
  {
    for (size_t i=r.begin(); i<r.end(); ++i) z[i]=x[i]+y[i];
  }
};

int main (int argc, char** argv)
{
  std::vector<double> x(1000,1.0), y(1000,2.0), z(1000);

  // first version: pass kernel object
  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0,x.size()),
                            VectorSumKernel(x,y,z));

  // second version: pass lambda
  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0,x.size()),
                            [&](const oneapi::tbb::blocked_range<size_t>& r)
                            {
                              for (size_t i=r.begin(); i<r.end(); ++i)
                                z[i]=x[i]+y[i];
                            });
  return 0;
}
