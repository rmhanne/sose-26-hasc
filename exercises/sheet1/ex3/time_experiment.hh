#ifndef HASC_TIME_EXPERIMENT_HH
#define HASC_TIME_EXPERIMENT_HH

#include <chrono>
#include <utility>

//! a function getting current value of high resolution clock
auto get_time_stamp()
{
  return std::chrono::high_resolution_clock::now();
}

// compute difference of two time stamps in seconds
// the data type T is that returned by get_time_stamp
template <typename T>
double get_duration_seconds(T start, T stop)
{
  auto duration = stop - start;
  auto dcast = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  return dcast / 1e6;
}

/*! \brief time an experiment
 * This function template takes an experiment, which is a class with a method
 * void run () {} and measures the time that it takes to execute that function.
 * To make timings accurate it repeats the experiment by running run() several
 * times such that the total execution time takes at least a given time. This is
 * the second, optional parameter.
 * The result of the function is a std::pair where the first component is an integer
 * holding the number of repetitions of the experiment and the second component is the
 * total execution time for these repetitions in microseconds.
 *
 * \param experiment an experiment with a run method
 * \param mintime minimum total runtime (given in microseconds)
 *
 */
template <typename T>
auto time_experiment(const T &experiment, double mintime = 0.5)
{
  auto start = get_time_stamp();
  auto stop = start;
  auto duration = get_duration_seconds(start,stop);
  std::pair<long, double> rv;
  long rep = 1;
  while (duration < mintime && rep < 1000000000)
  {
    start = get_time_stamp();
    for (long k = 0; k < rep; k++)
      experiment();
    stop = get_time_stamp();
    duration = get_duration_seconds(start,stop);
    // std::cout << " rep=" << rep << " d=" << dcast << std::endl;
    rv.first = rep;
    rv.second = duration;
    rep *= 2;
  }
  return rv;
}

#endif
