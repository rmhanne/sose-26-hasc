#include <iostream>
#include <vector>

#include "time_experiment.hh"

const int N = 64 * 1024 * 1024; // max size
std::vector<int> x(N, 1.0);     // data vector
int rglobal = 0;

class Experiment
{
  int n, s;

public:
  // construct an experiment
  Experiment(int n_, int s_) : n(n_), s(s_)
  {
    // initialize a closed loop
    for (int i = 0; i < n; i += s)
      x[i] = (i + s) % n;
  }
  // run an experiment; can be called several times
  void run() const
  {
    for (int k = 0; k < s; k++)
    {
      int i = x[0];
      while (i != 0)
        i = x[i];
    }
  }
  // report number of operations
  double operations() const
  {
    return n;
  }
};

class EmptyExperiment
{
  int n, s, l;

public:
  // construct an experiment
  EmptyExperiment(int n_, int s_) : n(n_), s(s_)
  {
    // initialize a closed loop
    for (int i = 0; i < n; i += s)
      x[i] = (i + s) % n;
    int l = n/s;
  }
  // run an experiment; can be called several times
  void run() const
  {
    for (int k = 0; k < s; k++)
    {
      int i = 1;
      while (i < l)
        ++i;
    }
  }
  // report number of operations
  double operations() const
  {
    return n;
  }
};

int main()
{
  std::vector<int> sizes;
  for (int i = 256; i < N; i *= 2)
    sizes.push_back(i);
  std::cout << "N, ";
  for (int stride = 1; stride < N; stride *= 2)
  {
    std::string sep;
    if (stride < N / 2)
      sep = ", ";
    std::cout << stride * sizeof(int) << sep;
  }
  std::cout << std::endl;
  for (auto vecsize : sizes)
  {
    std::cout << vecsize * sizeof(int) << ", ";
    std::vector<double> readtimes;
    for (int stride = 1; stride < vecsize; stride *= 2)
    {
      Experiment e(vecsize, stride);
      auto d = time_experiment(e, 500000);
      EmptyExperiment ee(vecsize, stride);
      auto dd = time_experiment(ee, 50000);
      double time_per_experiment = d.second * 1.0e-6 / d.first;
      double time_per_emptyexperiment = dd.second * 1e-6 / dd.first;
      double read_time = (time_per_experiment - time_per_emptyexperiment) / e.operations() * 1e9;
      // std::cout << "XXX " << stride*sizeof(int) << " " << d.first << " " << d.second << " " << time_per_experiment << std::endl;
      // std::cout << "YYY " << stride*sizeof(int) << " " << dd.first << " " << dd.second << " " << time_per_emptyexperiment << std::endl;
      readtimes.push_back(read_time);
    }
    for (int i = 0; i < readtimes.size() - 1; i++)
      std::cout << readtimes[i] << ", ";
    std::cout << readtimes.back() << std::endl;
  }
  std::cout << rglobal << std::endl;
  return 0;
}
