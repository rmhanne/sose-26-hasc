#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <thread>
#include <mutex>
#include <atomic>

#include "vcl/vectorclass.h"

#include "nbody_generate.hh"
#include "nbody_io.hh"
#include "time_experiment.hh"

// #define VECTORIZED_VERSION_1

/*const double gamma = 6.674E-11;*/
const double G = 1.0;
const double epsilon2 = 1E-10;


// Note: The main reason to have this class can be found in the TBB
// implementation, where we needed the + operator. Without additinoal operators
// we could of course just use an std::array directly.
struct Point {
  std::array<double, 3> _x {};

  // Point() :_x{0.0} {};
  // Point(const Point &point) {_x = point._x; }

  double& operator[](int i) {return _x[i];}

  const double& operator[](int i) const {return _x[i];}
};

// Note: Since we want to load these points as Vec4d it makes sense to have an
// alignment of 64. Since C++-17 std::vector will use an alignment that fits to
// the value type so the data in a std::vector<PointPadded> will also have an
// alignment of 64!
struct alignas (64) PointPadded {
  // Store a vector of size 4. We only ever use the first three variables but
  // this way we can load a Vec4d and simply get a 0 in the last SIMD lane.
  std::array<double, 4> _x {};

  double& operator[](int i) {return _x[i];}

  const double& operator[](int i) const {return _x[i];}

  PointPadded& operator+=(const PointPadded& other) {
    for (std::size_t i=0; i<_x.size(); ++i){
      _x[i] += other._x[i];
    }
    return *this;
  }

  PointPadded operator+(const PointPadded& other) const {
    PointPadded output = *this;
    output += other;
    return output;
  }
};


void acceleration_vanilla(const std::vector<Point>& x,
                          const std::vector<double>& m,
                          std::vector<Point>& a) {
  const int n = x.size();

  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++) {
      double d0 = x[j][0] - x[i][0];
      double d1 = x[j][1] - x[i][1];
      double d2 = x[j][2] - x[i][2];
      double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
      double r = sqrt(r2);
      double invfact = G / (r * r2);
      double factori = m[i] * invfact;
      double factorj = m[j] * invfact;
      a[i][0] += factorj * d0;
      a[i][1] += factorj * d1;
      a[i][2] += factorj * d2;
      a[j][0] -= factori * d0;
      a[j][1] -= factori * d1;
      a[j][2] -= factori * d2;
    }
}
void leapfrog_vanilla(const double dt,
                      std::vector<Point>& x,
                      std::vector<Point>& v,
                      const std::vector<double>& m,
                      std::vector<Point>& a,
                      const std::size_t number_of_threads=1) {
  const int n = x.size();

  // update position: 6n flops
  for (int i = 0; i < n; i++) {
    x[i][0] += dt * v[i][0];
    x[i][1] += dt * v[i][1];
    x[i][2] += dt * v[i][2];
  }

  // save and clear acceleration
  for (int i = 0; i < n; i++) a[i][0] = a[i][1] = a[i][2] = 0.0;

  // compute new acceleration: n*(n-1)*13 flops
  acceleration_vanilla(x, m, a);

  // update velocity: 6n flops
  for (int i = 0; i < n; i++) {
    v[i][0] += dt * a[i][0];
    v[i][1] += dt * a[i][1];
    v[i][2] += dt * a[i][2];
  }
}

template <int B>
void acceleration_blocked(const std::vector<Point>& x,
                          const std::vector<double>& m,
                          std::vector<Point>& a) {
  const int n = x.size();

  for (int I = 0; I < n; I += B) {
    // Diagonal block (I,I)
    //
    // Only look at upper right part of the block
    for (int i = I; i < I + B; i++) {
      for (int j = i + 1; j < I + B; j++) {
        double d0 = x[j][0] - x[i][0];
        double d1 = x[j][1] - x[i][1];
        double d2 = x[j][2] - x[i][2];
        double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
        double r = sqrt(r2);
        double invfact = G / (r * r2);
        double factori = m[i] * invfact;
        double factorj = m[j] * invfact;
        a[i][0] += factorj * d0;
        a[i][1] += factorj * d1;
        a[i][2] += factorj * d2;
        a[j][0] -= factori * d0;
        a[j][1] -= factori * d1;
        a[j][2] -= factori * d2;
      }
    }
    // blocks J>I
    //
    // Look at full block
    for (int J = I + B; J < n; J += B) {
      for (int i = I; i < I + B; i += 1) {
        for (int j = J; j < J + B; j += 1) {
          double d0 = x[j][0] - x[i][0];
          double d1 = x[j][1] - x[i][1];
          double d2 = x[j][2] - x[i][2];
          double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
          double r = sqrt(r2);
          double invfact = G / (r * r2);
          double factori = m[i] * invfact;
          double factorj = m[j] * invfact;
          a[i][0] += factorj * d0;
          a[i][1] += factorj * d1;
          a[i][2] += factorj * d2;
          a[j][0] -= factori * d0;
          a[j][1] -= factori * d1;
          a[j][2] -= factori * d2;
        }
      }
    }
  }
}
template <int B>
void leapfrog_blocked(const double dt,
                      std::vector<Point>& x,
                      std::vector<Point>& v,
                      const std::vector<double>& m,
                      std::vector<Point>& a,
                      const std::size_t number_of_threads=1) {
  const int n = x.size();

  // update position: 6n flops
  for (int i = 0; i < n; i++) {
    x[i][0] += dt * v[i][0];
    x[i][1] += dt * v[i][1];
    x[i][2] += dt * v[i][2];
  }

  // save and clear acceleration
  for (int i = 0; i < n; i++) a[i][0] = a[i][1] = a[i][2] = 0.0;

  // compute new acceleration: n*(n-1)*13 flops
  acceleration_blocked<B>(x, m, a);

  // update velocity: 6n flops
  for (int i = 0; i < n; i++) {
    v[i][0] += dt * a[i][0];
    v[i][1] += dt * a[i][1];
    v[i][2] += dt * a[i][2];
  }
}



void acceleration_parallel_coarse_lock_kernel(const std::vector<Point>& x,
                                              const std::vector<double>& m,
                                              std::vector<Point>& a,
                                              int i_start,
                                              int i_end,
                                              std::mutex& mutex) {
  const int n = x.size();
  for (int i = i_start; i < i_end; i++)
    for (int j = i + 1; j < n; j++) {
      double d0 = x[j][0] - x[i][0];
      double d1 = x[j][1] - x[i][1];
      double d2 = x[j][2] - x[i][2];
      double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
      double r = sqrt(r2);
      double invfact = G / (r * r2);
      double factori = m[i] * invfact;
      double factorj = m[j] * invfact;
      std::lock_guard<std::mutex> lock_guard {mutex};
      a[i][0] += factorj * d0;
      a[i][1] += factorj * d1;
      a[i][2] += factorj * d2;
      a[j][0] -= factori * d0;
      a[j][1] -= factori * d1;
      a[j][2] -= factori * d2;
    }
}
void leapfrog_parallel_coarse_lock(const double dt,
                                   std::vector<Point>& x,
                                   std::vector<Point>& v,
                                   const std::vector<double>& m,
                                   std::vector<Point>& a,
                                   const std::size_t number_of_threads) {
  const int n = x.size();

  std::vector<std::thread> threads(number_of_threads);
  if (n % number_of_threads != 0) {
    std::cout << "n not multiple of number of threads" << std::endl;
    exit(1);
  }

  // update position: 6n flops
  // save and clear acceleration
  int current_index = 0;
  int elements = n / number_of_threads;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread([=, &x, &v, &a]() {
      for (int i = current_index; i < current_index + elements; i++) {
        x[i][0] += dt * v[i][0];
        x[i][1] += dt * v[i][1];
        x[i][2] += dt * v[i][2];
        a[i][0] = a[i][1] = a[i][2] = 0.0;
      }
    });
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();

  // compute new acceleration: n*(n-1)*13 flops
  current_index = 0;
  std::mutex mutex;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread(acceleration_parallel_coarse_lock_kernel, std::cref(x), std::cref(m), std::ref(a),
                             current_index, current_index + elements, std::ref(mutex));
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();

  // update velocity: 6n flops
  current_index = 0;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread([=, &x, &v, &a]() {
      for (int i = current_index; i < current_index + elements; i++) {
        v[i][0] += dt * a[i][0];
        v[i][1] += dt * a[i][1];
        v[i][2] += dt * a[i][2];
      }
    });
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();
}

void acceleration_parallel_fine_lock_kernel(const std::vector<Point>& x,
                                            const std::vector<double>& m,
                                            std::vector<Point>& a,
                                            int i_start,
                                            int i_end,
                                            std::vector<std::mutex>& mutex) {
  const int n = x.size();
  for (int i = i_start; i < i_end; i++)
    for (int j = i + 1; j < n; j++) {
      double d0 = x[j][0] - x[i][0];
      double d1 = x[j][1] - x[i][1];
      double d2 = x[j][2] - x[i][2];
      double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
      double r = sqrt(r2);
      double invfact = G / (r * r2);
      double factori = m[i] * invfact;
      double factorj = m[j] * invfact;
      std::unique_lock<std::mutex> lock_0 {mutex[i]};
      a[i][0] += factorj * d0;
      a[i][1] += factorj * d1;
      a[i][2] += factorj * d2;
      lock_0.unlock();
      std::unique_lock<std::mutex> lock_1 {mutex[j]};
      a[j][0] -= factori * d0;
      a[j][1] -= factori * d1;
      a[j][2] -= factori * d2;
      lock_1.unlock();
    }
}
void leapfrog_parallel_fine_lock(const double dt,
                                   std::vector<Point>& x,
                                   std::vector<Point>& v,
                                   const std::vector<double>& m,
                                   std::vector<Point>& a,
                                   const std::size_t number_of_threads) {
  const int n = x.size();

  std::vector<std::thread> threads(number_of_threads);
  if (n % number_of_threads != 0) {
    std::cout << "n not multiple of number of threads" << std::endl;
    exit(1);
  }

  // update position: 6n flops
  // save and clear acceleration
  int current_index = 0;
  int elements = n / number_of_threads;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread([=, &x, &v, &a]() {
      for (int i = current_index; i < current_index + elements; i++) {
        x[i][0] += dt * v[i][0];
        x[i][1] += dt * v[i][1];
        x[i][2] += dt * v[i][2];
        a[i][0] = a[i][1] = a[i][2] = 0.0;
      }
    });
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();

  // compute new acceleration: n*(n-1)*13 flops
  current_index = 0;
  std::vector<std::mutex> mutex(n);
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread(acceleration_parallel_fine_lock_kernel, std::cref(x), std::cref(m), std::ref(a),
                             current_index, current_index + elements, std::ref(mutex));
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();

  // update velocity: 6n flops
  current_index = 0;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread([=, &x, &v, &a]() {
      for (int i = current_index; i < current_index + elements; i++) {
        v[i][0] += dt * a[i][0];
        v[i][1] += dt * a[i][1];
        v[i][2] += dt * a[i][2];
      }
    });
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();
}

template<typename T>
void acceleration_parallel_atomic_update_kernel(const std::vector<Point>& x,
                                                const std::vector<double>& m,
                                                std::vector<T>& a,
                                                int i_start,
                                                int i_end) {
  const int n = x.size();
  for (int i = i_start; i < i_end; i++)
    for (int j = i + 1; j < n; j++) {
      double d0 = x[j][0] - x[i][0];
      double d1 = x[j][1] - x[i][1];
      double d2 = x[j][2] - x[i][2];
      double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
      double r = sqrt(r2);
      double invfact = G / (r * r2);
      double factori = m[i] * invfact;
      double factorj = m[j] * invfact;

      // Note: This requires C++20. Could be done with compare_exchange trickery
      // but I decided to just go for newer C++
      a[i][0] += factorj * d0;
      a[i][1] += factorj * d1;
      a[i][2] += factorj * d2;
      a[j][0] -= factori * d0;
      a[j][1] -= factori * d1;
      a[j][2] -= factori * d2;
    }
}
template <typename T>
void leapfrog_parallel_atomic_update(const double dt,
                                     std::vector<Point>& x,
                                     std::vector<Point>& v,
                                     const std::vector<double>& m,
                                     std::vector<T>& a,
                                     const std::size_t number_of_threads) {
  const int n = x.size();

  std::vector<std::thread> threads(number_of_threads);
  if (n % number_of_threads != 0) {
    std::cout << "n not multiple of number of threads" << std::endl;
    exit(1);
  }

  // update position: 6n flops
  for (int i = 0; i < n; i++) {
    x[i][0] += dt * v[i][0];
    x[i][1] += dt * v[i][1];
    x[i][2] += dt * v[i][2];
  }

  // save and clear acceleration
  for (int i = 0; i < n; i++) a[i][0] = a[i][1] = a[i][2] = 0.0;


  // compute new acceleration: n*(n-1)*13 flops
  int current_index = 0;
  int elements = n / number_of_threads;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread(acceleration_parallel_atomic_update_kernel<std::array<std::atomic<double>,3>>, std::cref(x), std::cref(m), std::ref(a),
                             current_index, current_index + elements);
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();


  // update velocity: 6n flops
  for (int i = 0; i < n; i++) {
    v[i][0] += dt * a[i][0];
    v[i][1] += dt * a[i][1];
    v[i][2] += dt * a[i][2];
  }
}


void acceleration_parallel_no_symmetry_kernel(const std::vector<Point>& x,
                                              const std::vector<double>& m,
                                              std::vector<Point>& a,
                                              int i_start,
                                              int i_end) {
  const int n = x.size();
  for (int i = i_start; i < i_end; i++)
    for (int j = 0; j < n; j++) {
      double d0 = x[j][0] - x[i][0];
      double d1 = x[j][1] - x[i][1];
      double d2 = x[j][2] - x[i][2];
      double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
      double r = sqrt(r2);
      double invfact = G / (r * r2);
      // double factori = m[i] * invfact;
      double factorj = m[j] * invfact;
      a[i][0] += factorj * d0;
      a[i][1] += factorj * d1;
      a[i][2] += factorj * d2;
      // a[j][0] -= factori * d0;
      // a[j][1] -= factori * d1;
      // a[j][2] -= factori * d2;
    }
}
void leapfrog_parallel_no_symmetry(const double dt,
                                   std::vector<Point>& x,
                                   std::vector<Point>& v,
                                   const std::vector<double>& m,
                                   std::vector<Point>& a,
                                   const std::size_t number_of_threads) {
  const int n = x.size();

  std::vector<std::thread> threads(number_of_threads);
  if (n % number_of_threads != 0) {
    std::cout << "n not multiple of number of threads" << std::endl;
    exit(1);
  }

  // update position: 6n flops
  // save and clear acceleration
  int current_index = 0;
  int elements = n / number_of_threads;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread([=, &x, &v, &a]() {
      for (int i = current_index; i < current_index + elements; i++) {
        x[i][0] += dt * v[i][0];
        x[i][1] += dt * v[i][1];
        x[i][2] += dt * v[i][2];
        a[i][0] = a[i][1] = a[i][2] = 0.0;
      }
    });
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();

  // compute new acceleration: n*(n-1)*13 flops
  current_index = 0;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread(acceleration_parallel_no_symmetry_kernel, std::cref(x), std::cref(m), std::ref(a),
                             current_index, current_index + elements);
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();

  // update velocity: 6n flops
  current_index = 0;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread([=, &x, &v, &a]() {
      for (int i = current_index; i < current_index + elements; i++) {
        v[i][0] += dt * a[i][0];
        v[i][1] += dt * a[i][1];
        v[i][2] += dt * a[i][2];
      }
    });
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();
}


void acceleration_parallel_privatization_kernel(const std::vector<Point>& x,
                                                const std::vector<double>& m,
                                                std::vector<std::array<double,3>>& a,
                                                const int i_start,
                                                const int i_end) {
  const int n = x.size();
  for (int i = i_start; i < i_end; i++)
    for (int j = i + 1; j < n; j++) {
      double d0 = x[j][0] - x[i][0];
      double d1 = x[j][1] - x[i][1];
      double d2 = x[j][2] - x[i][2];
      double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
      double r = sqrt(r2);
      double invfact = G / (r * r2);
      double factori = m[i] * invfact;
      double factorj = m[j] * invfact;
      a[i][0] += factorj * d0;
      a[i][1] += factorj * d1;
      a[i][2] += factorj * d2;
      a[j][0] -= factori * d0;
      a[j][1] -= factori * d1;
      a[j][2] -= factori * d2;
    }
}
void leapfrog_parallel_privatization(const double dt,
                                     std::vector<Point>& x,
                                     std::vector<Point>& v,
                                     const std::vector<double>& m,
                                     std::vector<Point>& a,
                                     const std::size_t number_of_threads) {
  const std::size_t n = x.size();

  std::vector<std::thread> threads(number_of_threads);
  if (n % number_of_threads != 0) {
    std::cout << "n not multiple of number of threads" << std::endl;
    exit(1);
  }

  // update position: 6n flops
  // save and clear acceleration
  int current_index = 0;
  int elements = n / number_of_threads;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread([=, &x, &v, &a]() {
      for (int i = current_index; i < current_index + elements; i++) {
        x[i][0] += dt * v[i][0];
        x[i][1] += dt * v[i][1];
        x[i][2] += dt * v[i][2];
        a[i][0] = a[i][1] = a[i][2] = 0.0;
      }
    });
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();


  const std::size_t array_size = 3;
  std::vector<std::vector<std::array<double, array_size>>> local_a(number_of_threads);
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    local_a[i].resize(n);
  }

  // compute new acceleration: n*(n-1)*13 flops
  current_index = 0;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread(acceleration_parallel_privatization_kernel, std::cref(x), std::cref(m), std::ref(local_a[i]),
                             current_index, current_index + elements);
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();

  // Reduction
  for (std::size_t i=0; i<number_of_threads; ++i){
    for (std::size_t j=0; j<n; ++j){
      for (std::size_t k=0; k<array_size; ++k){
        a[j][k] += local_a[i][j][k];
      }
    }
  }

  // update velocity: 6n flops
  current_index = 0;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread([=, &x, &v, &a]() {
      for (int i = current_index; i < current_index + elements; i++) {
        v[i][0] += dt * a[i][0];
        v[i][1] += dt * a[i][1];
        v[i][2] += dt * a[i][2];
      }
    });
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();
}



void acceleration_parallel_privatization_interleaved_splitting_kernel(const std::vector<Point>& x,
                                                                      const std::vector<double>& m,
                                                                      std::vector<std::array<double,3>>& a,
                                                                      int thread_index, int thread_numbers) {
  const int n = x.size();
  for (int i = thread_index; i < n; i+=thread_numbers)
    for (int j = i + 1; j < n; j++) {
      double d0 = x[j][0] - x[i][0];
      double d1 = x[j][1] - x[i][1];
      double d2 = x[j][2] - x[i][2];
      double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
      double r = sqrt(r2);
      double invfact = G / (r * r2);
      double factori = m[i] * invfact;
      double factorj = m[j] * invfact;
      a[i][0] += factorj * d0;
      a[i][1] += factorj * d1;
      a[i][2] += factorj * d2;
      a[j][0] -= factori * d0;
      a[j][1] -= factori * d1;
      a[j][2] -= factori * d2;
    }
}
void leapfrog_parallel_privatization_interleaved_splitting(const double dt,
                                                           std::vector<Point>& x,
                                                           std::vector<Point>& v,
                                                           const std::vector<double>& m,
                                                           std::vector<Point>& a,
                                                           const std::size_t number_of_threads) {
  const std::size_t n = x.size();

  std::vector<std::thread> threads(number_of_threads);
  if (n % number_of_threads != 0) {
    std::cout << "n not multiple of number of threads" << std::endl;
    exit(1);
  }

  // update position: 6n flops
  // save and clear acceleration
  int current_index = 0;
  int elements = n / number_of_threads;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread([=, &x, &v, &a]() {
      for (int i = current_index; i < current_index + elements; i++) {
        x[i][0] += dt * v[i][0];
        x[i][1] += dt * v[i][1];
        x[i][2] += dt * v[i][2];
        a[i][0] = a[i][1] = a[i][2] = 0.0;
      }
    });
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();


  const std::size_t array_size = 3;
  std::vector<std::vector<std::array<double, array_size>>> local_a(number_of_threads);
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    local_a[i].resize(n);
  }

  // compute new acceleration: n*(n-1)*13 flops
  current_index = 0;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread(acceleration_parallel_privatization_interleaved_splitting_kernel,
                             std::cref(x), std::cref(m), std::ref(local_a[i]),
                             i, number_of_threads);
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();

  // Reduction
  for (std::size_t i=0; i<number_of_threads; ++i){
    for (std::size_t j=0; j<n; ++j){
      for (std::size_t k=0; k<array_size; ++k){
        a[j][k] += local_a[i][j][k];
      }
    }
  }

  // update velocity: 6n flops
  current_index = 0;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread([=, &x, &v, &a]() {
      for (int i = current_index; i < current_index + elements; i++) {
        v[i][0] += dt * a[i][0];
        v[i][1] += dt * a[i][1];
        v[i][2] += dt * a[i][2];
      }
    });
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();
}


void leapfrog_parallel_privatization_workload_splitting(const double dt,
                                                        std::vector<Point>& x,
                                                        std::vector<Point>& v,
                                                        const std::vector<double>& m,
                                                        std::vector<Point>& a,
                                                        const std::size_t number_of_threads) {
  const std::size_t n = x.size();

  std::vector<std::thread> threads(number_of_threads);
  if (n % number_of_threads != 0) {
    std::cout << "n not multiple of number of threads" << std::endl;
    exit(1);
  }

  // update position: 6n flops
  // save and clear acceleration
  int current_index = 0;
  int elements = n / number_of_threads;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread([=, &x, &v, &a]() {
      for (int i = current_index; i < current_index + elements; i++) {
        x[i][0] += dt * v[i][0];
        x[i][1] += dt * v[i][1];
        x[i][2] += dt * v[i][2];
        a[i][0] = a[i][1] = a[i][2] = 0.0;
      }
    });
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();


  std::vector<std::pair<int, int>> indices;
  std::size_t start (0.0);
  std::size_t workload(0);
  int target_load = n*(n-1)/2/number_of_threads;
  for (std::size_t i=0; i<n; ++i){
    workload += n-i-1;
    if (workload > (indices.size()+1)*target_load){
      indices.push_back({start, i+1});
      start = i+1;
    }
  }
  indices.push_back({start, n});

  // Local variable for privatization
  const std::size_t array_size = 3;
  std::vector<std::vector<std::array<double, array_size>>> local_a(number_of_threads);
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    local_a[i].resize(n);
  }

  // compute new acceleration: n*(n-1)*13 flops
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread(acceleration_parallel_privatization_kernel,
                             std::cref(x), std::cref(m), std::ref(local_a[i]),
                             indices[i].first, indices[i].second);
  }
  for (auto &entry : threads) entry.join();

  // Reduction
  for (std::size_t i=0; i<number_of_threads; ++i){
    for (std::size_t j=0; j<n; ++j){
      for (std::size_t k=0; k<array_size; ++k){
        a[j][k] += local_a[i][j][k];
      }
    }
  }

  // update velocity: 6n flops
  current_index = 0;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread([=, &x, &v, &a]() {
      for (int i = current_index; i < current_index + elements; i++) {
        v[i][0] += dt * a[i][0];
        v[i][1] += dt * a[i][1];
        v[i][2] += dt * a[i][2];
      }
    });
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();
}

template <int B>
void acceleration_vectorized(const std::vector<PointPadded>& x,
                             const std::vector<double>& m,
                             std::vector<PointPadded>& a) {
  const int n = x.size();

  // Exactly 16 Vec4d for 16 registers
  Vec4d A0, A1;
  Vec4d D0, D1, D2, D3;  // distances
  Vec4d E0, E1, E2, E3;  // distances
  Vec4d S0, S, U, T0, T1, T2;

  for (int I = 0; I < n; I += B) {
    // Diagonal block (I,I)
    //
    // Only look at upper right part of the block. This is not vectorized, as
    // the bulk of the work is happening in the other blocks and visiting only
    // the upper right parts makes things more fiddely.
    for (int i = I; i < I + B; i++) {
      for (int j = i + 1; j < I + B; j++) {
        double d0 = x[j][0] - x[i][0];
        double d1 = x[j][1] - x[i][1];
        double d2 = x[j][2] - x[i][2];
        double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
        double r = sqrt(r2);
        double invfact = G / (r * r2);
        double factori = m[i] * invfact;
        double factorj = m[j] * invfact;
        a[i][0] += factorj * d0;
        a[i][1] += factorj * d1;
        a[i][2] += factorj * d2;
        a[j][0] -= factori * d0;
        a[j][1] -= factori * d1;
        a[j][2] -= factori * d2;
      }
    }
    // blocks J>I
    for (int J = I + B; J < n; J += B) {
      for (int i = I; i < I + B; i += 2) {
        for (int j = J; j < J + B; j += 4) {
          // distances 2x4 masses
          T0.load(&x[i][0]);
          T1.load(&x[i + 1][0]);
          D0.load(&x[j][0]);
          D1.load(&x[j + 1][0]);
          D2.load(&x[j + 2][0]);
          D3.load(&x[j + 3][0]);
          E0 = D0;
          E1 = D1;
          E2 = D2;
          E3 = D3;
          D0 -= T0;
          D1 -= T0;
          D2 -= T0;
          D3 -= T0;
          E0 -= T1;
          E1 -= T1;
          E2 -= T1;
          E3 -= T1;

          // transpose first batch
          S = blend4<0, 4, 1, 5>(D0, D1);
          U = blend4<0, 4, 1, 5>(D2, D3);
          T0 = blend4<0, 1, 4, 5>(S, U);  // all 0 components
          T1 = blend4<2, 3, 6, 7>(S, U);  // all 1 components
          S = blend4<2, 6, V_DC, V_DC>(D0, D1);
          U = blend4<2, 6, V_DC, V_DC>(D2, D3);
          T2 = blend4<0, 1, 4, 5>(S, U);  // all 2 components

          // the norms first batch
          S0 = Vec4d(epsilon2);
          S0 = mul_add(T0, T0, S0);
          S0 = mul_add(T1, T1, S0);
          S0 = mul_add(T2, T2, S0);

          // transpose second batch
          S = blend4<0, 4, 1, 5>(E0, E1);
          U = blend4<0, 4, 1, 5>(E2, E3);
          T0 = blend4<0, 1, 4, 5>(S, U);  // all 0 components
          T1 = blend4<2, 3, 6, 7>(S, U);  // all 1 components
          S = blend4<2, 6, V_DC, V_DC>(E0, E1);
          U = blend4<2, 6, V_DC, V_DC>(E2, E3);
          T2 = blend4<0, 1, 4, 5>(S, U);  // all 2 components

          // the norms second batch
          S = Vec4d(epsilon2);
          S = mul_add(T0, T0, S);
          S = mul_add(T1, T1, S);
          S = mul_add(T2, T2, S);

          // sqrt first batch
          U = sqrt(S0);
          U *= S0;  // now U contains r^3
          T0 = Vec4d(G);
          S0 =
              T0 / U;  // now S is the inverse factor for four pairs first batch

          // sqrt second batch
          U = sqrt(S);
          U *= S;  // now U contains r^3
          S = T0 /
              U;  // now S is the inverse factor for four pairs second batch

          // update both rows from all columns
          A0.load(&a[i][0]);
          A1.load(&a[i + 1][0]);
          T2 = Vec4d(m[j]);              // mass col j
          U = permute4<0, 0, 0, 0>(S0);  // scalar factor column j
          T0 = T2 * U;
          A0 = mul_add(T0, D0, A0);
          U = permute4<0, 0, 0, 0>(S);  // scalar factor column j
          T1 = T2 * U;
          A1 = mul_add(T1, E0, A1);

          T2 = Vec4d(m[j + 1]);          // mass col j+1
          U = permute4<1, 1, 1, 1>(S0);  // scalar factor column j
          T0 = T2 * U;
          A0 = mul_add(T0, D1, A0);
          U = permute4<1, 1, 1, 1>(S);  // scalar factor column j
          T1 = T2 * U;
          A1 = mul_add(T1, E1, A1);

          T2 = Vec4d(m[j + 2]);          // mass col j+2
          U = permute4<2, 2, 2, 2>(S0);  // scalar factor column j
          T0 = T2 * U;
          A0 = mul_add(T0, D2, A0);
          U = permute4<2, 2, 2, 2>(S);  // scalar factor column j
          T1 = T2 * U;
          A1 = mul_add(T1, E2, A1);

          T2 = Vec4d(m[j + 3]);          // mass col j+3
          U = permute4<3, 3, 3, 3>(S0);  // scalar factor column j
          T0 = T2 * U;
          A0 = mul_add(T0, D3, A0);
          U = permute4<3, 3, 3, 3>(S);  // scalar factor column j
          T1 = T2 * U;
          A1 = mul_add(T1, E3, A1);
          A0.store(&a[i][0]);
          A1.store(&a[i + 1][0]);

          // now update all columns from both rows
          T0 = Vec4d(m[i]);      // row 0 mass
          T1 = Vec4d(m[i + 1]);  // row 1 mass
          A0.load(&a[j][0]);
          U = permute4<0, 0, 0, 0>(S0);  // scalar factor column j row 0
          T2 = T0 * U;
          A0 = nmul_add(T2, D0, A0);
          U = permute4<0, 0, 0, 0>(S);  // scalar factor column j row 1
          T2 = T1 * U;
          A0 = nmul_add(T2, E0, A0);
          A0.store(&a[j][0]);

          A0.load(&a[j + 1][0]);
          U = permute4<1, 1, 1, 1>(S0);  // scalar factor column j+1 row 0
          T2 = T0 * U;
          A0 = nmul_add(T2, D1, A0);
          U = permute4<1, 1, 1, 1>(S);  // scalar factor column j+1 row 1
          T2 = T1 * U;
          A0 = nmul_add(T2, E1, A0);
          A0.store(&a[j + 1][0]);

          A0.load(&a[j + 2][0]);
          U = permute4<2, 2, 2, 2>(S0);  // scalar factor column j+2 row 0
          T2 = T0 * U;
          A0 = nmul_add(T2, D2, A0);
          U = permute4<2, 2, 2, 2>(S);  // scalar factor column j+2 row 1
          T2 = T1 * U;
          A0 = nmul_add(T2, E2, A0);
          A0.store(&a[j + 2][0]);

          A0.load(&a[j + 3][0]);
          U = permute4<3, 3, 3, 3>(S0);  // scalar factor column j+3 row 0
          T2 = T0 * U;
          A0 = nmul_add(T2, D3, A0);
          U = permute4<3, 3, 3, 3>(S);  // scalar factor column j+3 row 1
          T2 = T1 * U;
          A0 = nmul_add(T2, E3, A0);
          A0.store(&a[j + 3][0]);
        }
      }
    }
  }
}
template <int B>
void leapfrog_vectorized(const double dt,
                         std::vector<PointPadded>& x,
                         std::vector<PointPadded>& v,
                         const std::vector<double>& m,
                         std::vector<PointPadded>& a,
                         const std::size_t number_of_threads=1) {
  const int n = x.size();

  // update position: 6n flops
  for (int i = 0; i < n; i++) {
    x[i][0] += dt * v[i][0];
    x[i][1] += dt * v[i][1];
    x[i][2] += dt * v[i][2];
  }

  // save and clear acceleration
  for (int i = 0; i < n; i++) a[i][0] = a[i][1] = a[i][2] = 0.0;

  // compute new acceleration: n*(n-1)*13 flops
  acceleration_vectorized<B>(x, m, a);

  // update velocity: 6n flops
  for (int i = 0; i < n; i++) {
    v[i][0] += dt * a[i][0];
    v[i][1] += dt * a[i][1];
    v[i][2] += dt * a[i][2];
  }
}


template <int B>
void acceleration_vectorized_parallel_privatization_kernel(const std::vector<PointPadded>& x,
                                                           const std::vector<double>& m,
                                                           std::vector<PointPadded>& a,
                                                           const int i_start,
                                                           const int i_end) {
  const int n = x.size();

  // Exactly 16 Vec4d for 16 registers
  Vec4d A0, A1;
  Vec4d D0, D1, D2, D3;  // distances
  Vec4d E0, E1, E2, E3;  // distances
  Vec4d S0, S, U, T0, T1, T2;

  for (int I = i_start; I < i_end; I += B) {
    // Diagonal block (I,I)
    //
    // Only look at upper right part of the block. This is not vectorized, as
    // the bulk of the work is happening in the other blocks and visiting only
    // the upper right parts makes things more fiddely.
    for (int i = I; i < I + B; i++) {
      for (int j = i + 1; j < I + B; j++) {
        double d0 = x[j][0] - x[i][0];
        double d1 = x[j][1] - x[i][1];
        double d2 = x[j][2] - x[i][2];
        double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
        double r = sqrt(r2);
        double invfact = G / (r * r2);
        double factori = m[i] * invfact;
        double factorj = m[j] * invfact;
        a[i][0] += factorj * d0;
        a[i][1] += factorj * d1;
        a[i][2] += factorj * d2;
        a[j][0] -= factori * d0;
        a[j][1] -= factori * d1;
        a[j][2] -= factori * d2;
      }
    }
    // blocks J>I
    for (int J = I + B; J < n; J += B) {
      for (int i = I; i < I + B; i += 2) {
        for (int j = J; j < J + B; j += 4) {
          // distances 2x4 masses
          T0.load(&x[i][0]);
          T1.load(&x[i + 1][0]);
          D0.load(&x[j][0]);
          D1.load(&x[j + 1][0]);
          D2.load(&x[j + 2][0]);
          D3.load(&x[j + 3][0]);
          E0 = D0;
          E1 = D1;
          E2 = D2;
          E3 = D3;
          D0 -= T0;
          D1 -= T0;
          D2 -= T0;
          D3 -= T0;
          E0 -= T1;
          E1 -= T1;
          E2 -= T1;
          E3 -= T1;

          // transpose first batch
          S = blend4<0, 4, 1, 5>(D0, D1);
          U = blend4<0, 4, 1, 5>(D2, D3);
          T0 = blend4<0, 1, 4, 5>(S, U);  // all 0 components
          T1 = blend4<2, 3, 6, 7>(S, U);  // all 1 components
          S = blend4<2, 6, V_DC, V_DC>(D0, D1);
          U = blend4<2, 6, V_DC, V_DC>(D2, D3);
          T2 = blend4<0, 1, 4, 5>(S, U);  // all 2 components

          // the norms first batch
          S0 = Vec4d(epsilon2);
          S0 = mul_add(T0, T0, S0);
          S0 = mul_add(T1, T1, S0);
          S0 = mul_add(T2, T2, S0);

          // transpose second batch
          S = blend4<0, 4, 1, 5>(E0, E1);
          U = blend4<0, 4, 1, 5>(E2, E3);
          T0 = blend4<0, 1, 4, 5>(S, U);  // all 0 components
          T1 = blend4<2, 3, 6, 7>(S, U);  // all 1 components
          S = blend4<2, 6, V_DC, V_DC>(E0, E1);
          U = blend4<2, 6, V_DC, V_DC>(E2, E3);
          T2 = blend4<0, 1, 4, 5>(S, U);  // all 2 components

          // the norms second batch
          S = Vec4d(epsilon2);
          S = mul_add(T0, T0, S);
          S = mul_add(T1, T1, S);
          S = mul_add(T2, T2, S);

          // sqrt first batch
          U = sqrt(S0);
          U *= S0;  // now U contains r^3
          T0 = Vec4d(G);
          S0 =
              T0 / U;  // now S is the inverse factor for four pairs first batch

          // sqrt second batch
          U = sqrt(S);
          U *= S;  // now U contains r^3
          S = T0 /
              U;  // now S is the inverse factor for four pairs second batch

          // update both rows from all columns
          A0.load(&a[i][0]);
          A1.load(&a[i + 1][0]);
          T2 = Vec4d(m[j]);              // mass col j
          U = permute4<0, 0, 0, 0>(S0);  // scalar factor column j
          T0 = T2 * U;
          A0 = mul_add(T0, D0, A0);
          U = permute4<0, 0, 0, 0>(S);  // scalar factor column j
          T1 = T2 * U;
          A1 = mul_add(T1, E0, A1);

          T2 = Vec4d(m[j + 1]);          // mass col j+1
          U = permute4<1, 1, 1, 1>(S0);  // scalar factor column j
          T0 = T2 * U;
          A0 = mul_add(T0, D1, A0);
          U = permute4<1, 1, 1, 1>(S);  // scalar factor column j
          T1 = T2 * U;
          A1 = mul_add(T1, E1, A1);

          T2 = Vec4d(m[j + 2]);          // mass col j+2
          U = permute4<2, 2, 2, 2>(S0);  // scalar factor column j
          T0 = T2 * U;
          A0 = mul_add(T0, D2, A0);
          U = permute4<2, 2, 2, 2>(S);  // scalar factor column j
          T1 = T2 * U;
          A1 = mul_add(T1, E2, A1);

          T2 = Vec4d(m[j + 3]);          // mass col j+3
          U = permute4<3, 3, 3, 3>(S0);  // scalar factor column j
          T0 = T2 * U;
          A0 = mul_add(T0, D3, A0);
          U = permute4<3, 3, 3, 3>(S);  // scalar factor column j
          T1 = T2 * U;
          A1 = mul_add(T1, E3, A1);
          A0.store(&a[i][0]);
          A1.store(&a[i + 1][0]);

          // now update all columns from both rows
          T0 = Vec4d(m[i]);      // row 0 mass
          T1 = Vec4d(m[i + 1]);  // row 1 mass
          A0.load(&a[j][0]);
          U = permute4<0, 0, 0, 0>(S0);  // scalar factor column j row 0
          T2 = T0 * U;
          A0 = nmul_add(T2, D0, A0);
          U = permute4<0, 0, 0, 0>(S);  // scalar factor column j row 1
          T2 = T1 * U;
          A0 = nmul_add(T2, E0, A0);
          A0.store(&a[j][0]);

          A0.load(&a[j + 1][0]);
          U = permute4<1, 1, 1, 1>(S0);  // scalar factor column j+1 row 0
          T2 = T0 * U;
          A0 = nmul_add(T2, D1, A0);
          U = permute4<1, 1, 1, 1>(S);  // scalar factor column j+1 row 1
          T2 = T1 * U;
          A0 = nmul_add(T2, E1, A0);
          A0.store(&a[j + 1][0]);

          A0.load(&a[j + 2][0]);
          U = permute4<2, 2, 2, 2>(S0);  // scalar factor column j+2 row 0
          T2 = T0 * U;
          A0 = nmul_add(T2, D2, A0);
          U = permute4<2, 2, 2, 2>(S);  // scalar factor column j+2 row 1
          T2 = T1 * U;
          A0 = nmul_add(T2, E2, A0);
          A0.store(&a[j + 2][0]);

          A0.load(&a[j + 3][0]);
          U = permute4<3, 3, 3, 3>(S0);  // scalar factor column j+3 row 0
          T2 = T0 * U;
          A0 = nmul_add(T2, D3, A0);
          U = permute4<3, 3, 3, 3>(S);  // scalar factor column j+3 row 1
          T2 = T1 * U;
          A0 = nmul_add(T2, E3, A0);
          A0.store(&a[j + 3][0]);
        }
      }
    }
  }
}
template <int B>
void leapfrog_vectorized_parallel_privatization_workload_splitting(const double dt,
                                                                   std::vector<PointPadded>& x,
                                                                   std::vector<PointPadded>& v,
                                                                   const std::vector<double>& m,
                                                                   std::vector<PointPadded>& a,
                                                                   const std::size_t number_of_threads) {
  const std::size_t n = x.size();

  std::vector<std::thread> threads(number_of_threads);
  if (n % number_of_threads != 0) {
    std::cout << "n not multiple of number of threads" << std::endl;
    exit(1);
  }


  // update position: 6n flops
  // save and clear acceleration
  int current_index = 0;
  int elements = n / number_of_threads;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread([=, &x, &v, &a]() {
      for (int i = current_index; i < current_index + elements; i++) {
        x[i][0] += dt * v[i][0];
        x[i][1] += dt * v[i][1];
        x[i][2] += dt * v[i][2];
        a[i][0] = a[i][1] = a[i][2] = 0.0;
      }
    });
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();


  std::vector<std::pair<int, int>> indices;
  std::size_t start (0.0);
  std::size_t workload(0);
  int target_load = n*(n-1)/2/number_of_threads;
  for (std::size_t i=0; i<n; i+=B){
    for (std::size_t j=0; j<B; ++j){
      workload += n-(i+j)-1;
    }
    if (workload > (indices.size()+1)*target_load){
      indices.push_back({start, i+B});
      start = i+B;
    }
  }
  indices.push_back({start, n});

  // Local variable for privatization
  std::vector<std::vector<PointPadded>> local_a(number_of_threads);
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    local_a[i].resize(n);
  }

  // compute new acceleration: n*(n-1)*13 flops
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread(acceleration_vectorized_parallel_privatization_kernel<B>,
                             std::cref(x), std::cref(m), std::ref(local_a[i]),
                             indices[i].first, indices[i].second);
  }
  for (auto &entry : threads) entry.join();

  // Reduction
  const std::size_t dimension = 3;
  for (std::size_t i=0; i<number_of_threads; ++i){
    for (std::size_t j=0; j<n; ++j){
      for (std::size_t k=0; k<dimension; ++k){
        a[j][k] += local_a[i][j][k];
      }
    }
  }

  // update velocity: 6n flops
  current_index = 0;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread([=, &x, &v, &a]() {
      for (int i = current_index; i < current_index + elements; i++) {
        v[i][0] += dt * a[i][0];
        v[i][1] += dt * a[i][1];
        v[i][2] += dt * a[i][2];
      }
    });
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();
}


template <int B>
void acceleration_vectorized_parallel_privatization_interleaved_kernel(const std::vector<PointPadded>& x,
                                                                       const std::vector<double>& m,
                                                                       std::vector<PointPadded>& a,
                                                                       const int rank,
                                                                       const int number_of_threads) {
  const int n = x.size();

#ifdef VECTORIZED_VERSION_1

  Vec4d A0, A1;
  Vec4d D0, D1, D2, D3;  // distances
  Vec4d E0, E1, E2, E3;  // distances
  Vec4d S0, S, U, T0, T1, T2;

  int thread_index = 0;
  for (int I = 0; I < n; I += B, thread_index=(thread_index+1)%number_of_threads) {

    if (rank == thread_index){

      // Diagonal block (I,I)
      //
      // Only look at upper right part of the block. This is not vectorized, as
      // the bulk of the work is happening in the other blocks and visiting only
      // the upper right parts makes things more fiddely.
      for (int i = I; i < I + B; i++) {
        for (int j = i + 1; j < I + B; j++) {
          double d0 = x[j][0] - x[i][0];
          double d1 = x[j][1] - x[i][1];
          double d2 = x[j][2] - x[i][2];
          double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
          double r = sqrt(r2);
          double invfact = G / (r * r2);
          double factori = m[i] * invfact;
          double factorj = m[j] * invfact;
          a[i][0] += factorj * d0;
          a[i][1] += factorj * d1;
          a[i][2] += factorj * d2;
          a[j][0] -= factori * d0;
          a[j][1] -= factori * d1;
          a[j][2] -= factori * d2;
        }
      }
      // blocks J>I
      for (int J = I + B; J < n; J += B) {
        for (int i = I; i < I + B; i += 2) {
          for (int j = J; j < J + B; j += 4) {
            // distances 2x4 masses
            T0.load(&x[i][0]);
            T1.load(&x[i + 1][0]);
            D0.load(&x[j][0]);
            D1.load(&x[j + 1][0]);
            D2.load(&x[j + 2][0]);
            D3.load(&x[j + 3][0]);
            E0 = D0;
            E1 = D1;
            E2 = D2;
            E3 = D3;
            D0 -= T0;
            D1 -= T0;
            D2 -= T0;
            D3 -= T0;
            E0 -= T1;
            E1 -= T1;
            E2 -= T1;
            E3 -= T1;

            // transpose first batch
            S = blend4<0, 4, 1, 5>(D0, D1);
            U = blend4<0, 4, 1, 5>(D2, D3);
            T0 = blend4<0, 1, 4, 5>(S, U);  // all 0 components
            T1 = blend4<2, 3, 6, 7>(S, U);  // all 1 components
            S = blend4<2, 6, V_DC, V_DC>(D0, D1);
            U = blend4<2, 6, V_DC, V_DC>(D2, D3);
            T2 = blend4<0, 1, 4, 5>(S, U);  // all 2 components

            // the norms first batch
            S0 = Vec4d(epsilon2);
            S0 = mul_add(T0, T0, S0);
            S0 = mul_add(T1, T1, S0);
            S0 = mul_add(T2, T2, S0);

            // transpose second batch
            S = blend4<0, 4, 1, 5>(E0, E1);
            U = blend4<0, 4, 1, 5>(E2, E3);
            T0 = blend4<0, 1, 4, 5>(S, U);  // all 0 components
            T1 = blend4<2, 3, 6, 7>(S, U);  // all 1 components
            S = blend4<2, 6, V_DC, V_DC>(E0, E1);
            U = blend4<2, 6, V_DC, V_DC>(E2, E3);
            T2 = blend4<0, 1, 4, 5>(S, U);  // all 2 components

            // the norms second batch
            S = Vec4d(epsilon2);
            S = mul_add(T0, T0, S);
            S = mul_add(T1, T1, S);
            S = mul_add(T2, T2, S);

            // sqrt first batch
            U = sqrt(S0);
            U *= S0;  // now U contains r^3
            T0 = Vec4d(G);
            S0 =
              T0 / U;  // now S is the inverse factor for four pairs first batch

            // sqrt second batch
            U = sqrt(S);
            U *= S;  // now U contains r^3
            S = T0 /
              U;  // now S is the inverse factor for four pairs second batch

            // update both rows from all columns
            A0.load(&a[i][0]);
            A1.load(&a[i + 1][0]);
            T2 = Vec4d(m[j]);              // mass col j
            U = permute4<0, 0, 0, 0>(S0);  // scalar factor column j
            T0 = T2 * U;
            A0 = mul_add(T0, D0, A0);
            U = permute4<0, 0, 0, 0>(S);  // scalar factor column j
            T1 = T2 * U;
            A1 = mul_add(T1, E0, A1);

            T2 = Vec4d(m[j + 1]);          // mass col j+1
            U = permute4<1, 1, 1, 1>(S0);  // scalar factor column j
            T0 = T2 * U;
            A0 = mul_add(T0, D1, A0);
            U = permute4<1, 1, 1, 1>(S);  // scalar factor column j
            T1 = T2 * U;
            A1 = mul_add(T1, E1, A1);

            T2 = Vec4d(m[j + 2]);          // mass col j+2
            U = permute4<2, 2, 2, 2>(S0);  // scalar factor column j
            T0 = T2 * U;
            A0 = mul_add(T0, D2, A0);
            U = permute4<2, 2, 2, 2>(S);  // scalar factor column j
            T1 = T2 * U;
            A1 = mul_add(T1, E2, A1);

            T2 = Vec4d(m[j + 3]);          // mass col j+3
            U = permute4<3, 3, 3, 3>(S0);  // scalar factor column j
            T0 = T2 * U;
            A0 = mul_add(T0, D3, A0);
            U = permute4<3, 3, 3, 3>(S);  // scalar factor column j
            T1 = T2 * U;
            A1 = mul_add(T1, E3, A1);
            A0.store(&a[i][0]);
            A1.store(&a[i + 1][0]);

            // now update all columns from both rows
            T0 = Vec4d(m[i]);      // row 0 mass
            T1 = Vec4d(m[i + 1]);  // row 1 mass
            A0.load(&a[j][0]);
            U = permute4<0, 0, 0, 0>(S0);  // scalar factor column j row 0
            T2 = T0 * U;
            A0 = nmul_add(T2, D0, A0);
            U = permute4<0, 0, 0, 0>(S);  // scalar factor column j row 1
            T2 = T1 * U;
            A0 = nmul_add(T2, E0, A0);
            A0.store(&a[j][0]);

            A0.load(&a[j + 1][0]);
            U = permute4<1, 1, 1, 1>(S0);  // scalar factor column j+1 row 0
            T2 = T0 * U;
            A0 = nmul_add(T2, D1, A0);
            U = permute4<1, 1, 1, 1>(S);  // scalar factor column j+1 row 1
            T2 = T1 * U;
            A0 = nmul_add(T2, E1, A0);
            A0.store(&a[j + 1][0]);

            A0.load(&a[j + 2][0]);
            U = permute4<2, 2, 2, 2>(S0);  // scalar factor column j+2 row 0
            T2 = T0 * U;
            A0 = nmul_add(T2, D2, A0);
            U = permute4<2, 2, 2, 2>(S);  // scalar factor column j+2 row 1
            T2 = T1 * U;
            A0 = nmul_add(T2, E2, A0);
            A0.store(&a[j + 2][0]);

            A0.load(&a[j + 3][0]);
            U = permute4<3, 3, 3, 3>(S0);  // scalar factor column j+3 row 0
            T2 = T0 * U;
            A0 = nmul_add(T2, D3, A0);
            U = permute4<3, 3, 3, 3>(S);  // scalar factor column j+3 row 1
            T2 = T1 * U;
            A0 = nmul_add(T2, E3, A0);
            A0.store(&a[j + 3][0]);
          }
        }
      }
    }
  }

#else

  Vec4d Ai; // acceleration row
  Vec4d D0,D1,D2,D3; // distances
  Vec4d A0,A1,A2,A3; // accelerations columns
  Vec4d S,U,T0,T1,T2; // temporaries

  int thread_index = 0;
  for (int I = 0; I < n; I += B, thread_index=(thread_index+1)%number_of_threads) {

    if (rank == thread_index){

      // Diagonal block (I,I)
      //
      // Only look at upper right part of the block. This is not vectorized, as
      // the bulk of the work is happening in the other blocks and visiting only
      // the upper right parts makes things more fiddely.
      for (int i = I; i < I + B; i++) {
        for (int j = i + 1; j < I + B; j++) {
          double d0 = x[j][0] - x[i][0];
          double d1 = x[j][1] - x[i][1];
          double d2 = x[j][2] - x[i][2];
          double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
          double r = sqrt(r2);
          double invfact = G / (r * r2);
          double factori = m[i] * invfact;
          double factorj = m[j] * invfact;
          a[i][0] += factorj * d0;
          a[i][1] += factorj * d1;
          a[i][2] += factorj * d2;
          a[j][0] -= factori * d0;
          a[j][1] -= factori * d1;
          a[j][2] -= factori * d2;
        }
      }
      // blocks J>I
      for (int J = I + B; J < n; J += B) {
        for (int j = J; j < J + B; j += 4) {
          // update accelerations of four particles
          A0.load(&a[j][0]);
          A1.load(&a[j+1][0]);
          A2.load(&a[j+2][0]);
          A3.load(&a[j+3][0]);

          // loop over particles in row
          for (int i = I; i < I + B; i += 1) {
            // distances 2x4 masses
            T0.load(&x[i][0]); // position particle i
            D0.load(&x[j][0]); // positions of all particles j...j+3
            D1.load(&x[j+1][0]);
            D2.load(&x[j+2][0]);
            D3.load(&x[j+3][0]);
            D0 -= T0; // distance vector; is needed later
            D1 -= T0;
            D2 -= T0;
            D3 -= T0;

            // transpose distances
            S = blend4<0,4,1,5>(D0,D1);
            U = blend4<0,4,1,5>(D2,D3);
            T0 = blend4<0,1,4,5>(S,U); // all 0 components
            T1 = blend4<2,3,6,7>(S,U); // all 1 components
            S = blend4<2,6,V_DC,V_DC>(D0,D1);
            U = blend4<2,6,V_DC,V_DC>(D2,D3);
            T2 = blend4<0,1,4,5>(S,U); // all 2 components

            // the norms first batch
            S = Vec4d(epsilon2);
            S = mul_add(T0,T0,S);
            S = mul_add(T1,T1,S);
            S = mul_add(T2,T2,S); // now S contains the norms

            // determine inverse factors
            U = sqrt(S);
            U *= S; // now U contains r^3
            T0 = Vec4d(G);
            S =T0/U; // now S is the inverse factor for four particle pairs

            // now update accelerations
            Ai.load(&a[i][0]); // particle i, and we have particles j,...,j+3 in A0,..., A3

            // update Ai with all j
            T2 = Vec4d(m[j]); // mass col j
            U = permute4<0,0,0,0>(S); // scalar factor column j
            T0 = T2*U;
            Ai = mul_add(T0,D0,Ai);

            T2 = Vec4d(m[j+1]); // mass col j+1
            U = permute4<1,1,1,1>(S); // scalar factor column j
            T0 = T2*U;
            Ai = mul_add(T0,D1,Ai);

            T2 = Vec4d(m[j+2]); // mass col j+2
            U = permute4<2,2,2,2>(S); // scalar factor column j
            T0 = T2*U;
            Ai = mul_add(T0,D2,Ai);

            T2 = Vec4d(m[j+3]); // mass col j+3
            U = permute4<3,3,3,3>(S); // scalar factor column j
            T0 = T2*U;
            Ai = mul_add(T0,D3,Ai);

            Ai.store(&a[i][0]);

            // now update all columns from row i
            T0 = Vec4d(m[i]); // row 0 mass
            U = permute4<0,0,0,0>(S); // scalar factor column j row 0
            T2 = T0*U;
            A0 = nmul_add(T2,D0,A0);

            U = permute4<1,1,1,1>(S); // scalar factor column j+1 row 0
            T2 = T0*U;
            A1 = nmul_add(T2,D1,A1);

            U = permute4<2,2,2,2>(S); // scalar factor column j+2 row 0
            T2 = T0*U;
            A2 = nmul_add(T2,D2,A2);

            U = permute4<3,3,3,3>(S); // scalar factor column j+3 row 0
            T2 = T0*U;
            A3 = nmul_add(T2,D3,A3);
          }

          // update accelerations of four particles
          A0.store(&a[j][0]);
          A1.store(&a[j+1][0]);
          A2.store(&a[j+2][0]);
          A3.store(&a[j+3][0]);
        }
      }
    }
  }
#endif
}



// template <int B>
// void acceleration_vectorized_parallel_privatization_interleaved_kernel(const std::vector<PointPadded>& x,
//                                                                        const std::vector<double>& m,
//                                                                        std::vector<PointPadded>& a,
//                                                                        const int rank,
//                                                                        const int number_of_threads) {
//   const int n = x.size();

//   Vec4d A0, A1;
//   Vec4d D0, D1, D2, D3;  // distances
//   Vec4d E0, E1, E2, E3;  // distances
//   Vec4d S0, S, U, T0, T1, T2;

//   int thread_index = 0;
//   for (int I = 0; I < n; I += B, thread_index=(thread_index+1)%number_of_threads) {

//     if (rank == thread_index){

//       // Diagonal block (I,I)
//       //
//       // Only look at upper right part of the block. This is not vectorized, as
//       // the bulk of the work is happening in the other blocks and visiting only
//       // the upper right parts makes things more fiddely.
//       for (int i = I; i < I + B; i++) {
//         for (int j = i + 1; j < I + B; j++) {
//           double d0 = x[j][0] - x[i][0];
//           double d1 = x[j][1] - x[i][1];
//           double d2 = x[j][2] - x[i][2];
//           double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
//           double r = sqrt(r2);
//           double invfact = G / (r * r2);
//           double factori = m[i] * invfact;
//           double factorj = m[j] * invfact;
//           a[i][0] += factorj * d0;
//           a[i][1] += factorj * d1;
//           a[i][2] += factorj * d2;
//           a[j][0] -= factori * d0;
//           a[j][1] -= factori * d1;
//           a[j][2] -= factori * d2;
//         }
//       }
//       // blocks J>I
//       for (int J = I + B; J < n; J += B) {
//         for (int i = I; i < I + B; i += 2) {
//           for (int j = J; j < J + B; j += 4) {
//             // distances 2x4 masses
//             T0.load(&x[i][0]);
//             T1.load(&x[i + 1][0]);
//             D0.load(&x[j][0]);
//             D1.load(&x[j + 1][0]);
//             D2.load(&x[j + 2][0]);
//             D3.load(&x[j + 3][0]);
//             E0 = D0;
//             E1 = D1;
//             E2 = D2;
//             E3 = D3;
//             D0 -= T0;
//             D1 -= T0;
//             D2 -= T0;
//             D3 -= T0;
//             E0 -= T1;
//             E1 -= T1;
//             E2 -= T1;
//             E3 -= T1;

//             // transpose first batch
//             S = blend4<0, 4, 1, 5>(D0, D1);
//             U = blend4<0, 4, 1, 5>(D2, D3);
//             T0 = blend4<0, 1, 4, 5>(S, U);  // all 0 components
//             T1 = blend4<2, 3, 6, 7>(S, U);  // all 1 components
//             S = blend4<2, 6, V_DC, V_DC>(D0, D1);
//             U = blend4<2, 6, V_DC, V_DC>(D2, D3);
//             T2 = blend4<0, 1, 4, 5>(S, U);  // all 2 components

//             // the norms first batch
//             S0 = Vec4d(epsilon2);
//             S0 = mul_add(T0, T0, S0);
//             S0 = mul_add(T1, T1, S0);
//             S0 = mul_add(T2, T2, S0);

//             // transpose second batch
//             S = blend4<0, 4, 1, 5>(E0, E1);
//             U = blend4<0, 4, 1, 5>(E2, E3);
//             T0 = blend4<0, 1, 4, 5>(S, U);  // all 0 components
//             T1 = blend4<2, 3, 6, 7>(S, U);  // all 1 components
//             S = blend4<2, 6, V_DC, V_DC>(E0, E1);
//             U = blend4<2, 6, V_DC, V_DC>(E2, E3);
//             T2 = blend4<0, 1, 4, 5>(S, U);  // all 2 components

//             // the norms second batch
//             S = Vec4d(epsilon2);
//             S = mul_add(T0, T0, S);
//             S = mul_add(T1, T1, S);
//             S = mul_add(T2, T2, S);

//             // sqrt first batch
//             U = sqrt(S0);
//             U *= S0;  // now U contains r^3
//             T0 = Vec4d(G);
//             S0 =
//               T0 / U;  // now S is the inverse factor for four pairs first batch

//             // sqrt second batch
//             U = sqrt(S);
//             U *= S;  // now U contains r^3
//             S = T0 /
//               U;  // now S is the inverse factor for four pairs second batch

//             // update both rows from all columns
//             A0.load(&a[i][0]);
//             A1.load(&a[i + 1][0]);
//             T2 = Vec4d(m[j]);              // mass col j
//             U = permute4<0, 0, 0, 0>(S0);  // scalar factor column j
//             T0 = T2 * U;
//             A0 = mul_add(T0, D0, A0);
//             U = permute4<0, 0, 0, 0>(S);  // scalar factor column j
//             T1 = T2 * U;
//             A1 = mul_add(T1, E0, A1);

//             T2 = Vec4d(m[j + 1]);          // mass col j+1
//             U = permute4<1, 1, 1, 1>(S0);  // scalar factor column j
//             T0 = T2 * U;
//             A0 = mul_add(T0, D1, A0);
//             U = permute4<1, 1, 1, 1>(S);  // scalar factor column j
//             T1 = T2 * U;
//             A1 = mul_add(T1, E1, A1);

//             T2 = Vec4d(m[j + 2]);          // mass col j+2
//             U = permute4<2, 2, 2, 2>(S0);  // scalar factor column j
//             T0 = T2 * U;
//             A0 = mul_add(T0, D2, A0);
//             U = permute4<2, 2, 2, 2>(S);  // scalar factor column j
//             T1 = T2 * U;
//             A1 = mul_add(T1, E2, A1);

//             T2 = Vec4d(m[j + 3]);          // mass col j+3
//             U = permute4<3, 3, 3, 3>(S0);  // scalar factor column j
//             T0 = T2 * U;
//             A0 = mul_add(T0, D3, A0);
//             U = permute4<3, 3, 3, 3>(S);  // scalar factor column j
//             T1 = T2 * U;
//             A1 = mul_add(T1, E3, A1);
//             A0.store(&a[i][0]);
//             A1.store(&a[i + 1][0]);

//             // now update all columns from both rows
//             T0 = Vec4d(m[i]);      // row 0 mass
//             T1 = Vec4d(m[i + 1]);  // row 1 mass
//             A0.load(&a[j][0]);
//             U = permute4<0, 0, 0, 0>(S0);  // scalar factor column j row 0
//             T2 = T0 * U;
//             A0 = nmul_add(T2, D0, A0);
//             U = permute4<0, 0, 0, 0>(S);  // scalar factor column j row 1
//             T2 = T1 * U;
//             A0 = nmul_add(T2, E0, A0);
//             A0.store(&a[j][0]);

//             A0.load(&a[j + 1][0]);
//             U = permute4<1, 1, 1, 1>(S0);  // scalar factor column j+1 row 0
//             T2 = T0 * U;
//             A0 = nmul_add(T2, D1, A0);
//             U = permute4<1, 1, 1, 1>(S);  // scalar factor column j+1 row 1
//             T2 = T1 * U;
//             A0 = nmul_add(T2, E1, A0);
//             A0.store(&a[j + 1][0]);

//             A0.load(&a[j + 2][0]);
//             U = permute4<2, 2, 2, 2>(S0);  // scalar factor column j+2 row 0
//             T2 = T0 * U;
//             A0 = nmul_add(T2, D2, A0);
//             U = permute4<2, 2, 2, 2>(S);  // scalar factor column j+2 row 1
//             T2 = T1 * U;
//             A0 = nmul_add(T2, E2, A0);
//             A0.store(&a[j + 2][0]);

//             A0.load(&a[j + 3][0]);
//             U = permute4<3, 3, 3, 3>(S0);  // scalar factor column j+3 row 0
//             T2 = T0 * U;
//             A0 = nmul_add(T2, D3, A0);
//             U = permute4<3, 3, 3, 3>(S);  // scalar factor column j+3 row 1
//             T2 = T1 * U;
//             A0 = nmul_add(T2, E3, A0);
//             A0.store(&a[j + 3][0]);
//           }
//         }
//       }
//     }
//   }
// }


template <int B>
void leapfrog_vectorized_parallel_privatization_interleaved_splitting(const double dt,
                                                                   std::vector<PointPadded>& x,
                                                                   std::vector<PointPadded>& v,
                                                                   const std::vector<double>& m,
                                                                   std::vector<PointPadded>& a,
                                                                   const std::size_t number_of_threads) {
  const std::size_t n = x.size();

  std::vector<std::thread> threads(number_of_threads);
  if (n % number_of_threads != 0) {
    std::cout << "n not multiple of number of threads" << std::endl;
    exit(1);
  }


  // update position: 6n flops
  // save and clear acceleration
  int current_index = 0;
  int elements = n / number_of_threads;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread([=, &x, &v, &a]() {
      for (int i = current_index; i < current_index + elements; i++) {
        x[i][0] += dt * v[i][0];
        x[i][1] += dt * v[i][1];
        x[i][2] += dt * v[i][2];
        a[i][0] = a[i][1] = a[i][2] = 0.0;
      }
    });
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();

  // Local variable for privatization
  std::vector<std::vector<PointPadded>> local_a(number_of_threads);
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    local_a[i].resize(n);
  }

  // compute new acceleration: n*(n-1)*13 flops
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread(acceleration_vectorized_parallel_privatization_interleaved_kernel<B>,
                             std::cref(x), std::cref(m), std::ref(local_a[i]), i, number_of_threads);
  }
  for (auto &entry : threads) entry.join();

  // Reduction
  const std::size_t dimension = 3;
  for (std::size_t i=0; i<number_of_threads; ++i){
    for (std::size_t j=0; j<n; ++j){
      for (std::size_t k=0; k<dimension; ++k){
        a[j][k] += local_a[i][j][k];
      }
    }
  }

  // update velocity: 6n flops
  current_index = 0;
  for (std::size_t i = 0; i < number_of_threads; ++i) {
    threads[i] = std::thread([=, &x, &v, &a]() {
      for (int i = current_index; i < current_index + elements; i++) {
        v[i][0] += dt * a[i][0];
        v[i][1] += dt * a[i][1];
        v[i][2] += dt * a[i][2];
      }
    });
    current_index += elements;
  }
  for (auto &entry : threads) entry.join();
}


// ===================
// Utility and Testing
// ===================
template <typename T>
size_t alignment(const T *p) {
  for (size_t m = 64; m > 1; m /= 2)
    if (((size_t)p) % m == 0) return m;
  return 1;
}
template <typename T>
double norm(const T& t){
  double value = 0.0;
  for (auto v: t._x){
    value += v*v;
  }
  return std::sqrt(value);
}
template <typename T>
double sum_of_norms(std::vector<T> x, std::vector<T> v) {
  double sum = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    sum += norm(x[i]);
    sum += norm(v[i]);
  }
  return sum;
}
template <typename T, typename T2>
void initialize(std::vector<T>& x_vec,
                std::vector<T>& v_vec,
                std::vector<double>& m_vec,
                std::vector<T2>& a_vec
                ){

  const int n = x_vec.size();

  const int M = 4;
  typedef double double3[M];
  double3 *x = new (std::align_val_t{64}) double3[n]();
  double3 *v = new (std::align_val_t{64}) double3[n]();
  double *m = new (std::align_val_t{64}) double[n]();
  double3 *a = new (std::align_val_t{64}) double3[n]();

  // Should already be zero with the above. Just to be sure.
  for (int i = 0; i < n; i++) {
    m[i] = 0.0;
    for (int j = 0; j < M; j++)
      x[i][j] = v[i][j] = a[i][j] = 0.0;
  }

  two_plummer(n, 17, x, v, m, 0);
  for (int i=0; i<n; ++i) {
    m_vec[i] = m[i];
    for (std::size_t j=0; j<x_vec[i]._x.size(); ++j){
      x_vec[i][j] = x[i][j];
      v_vec[i][j] = v[i][j];
      a_vec[i][j] = a[i][j];
    }
  }

  delete[] x;
  delete[] v;
  delete[] m;
  delete[] a;
}

bool test_nbody() {
  const int n = 64 * 64;
  const int blocksize = 16;
  const int number_of_threads = 4;
  std::cout << "Tests run with n=" << n
            << "  blocksize=" << blocksize
            << "  and number_of_threads=" << number_of_threads
            << std::endl;

  // const int n = 8;
  // const int blocksize = 2;

  std::vector<Point> x(n);
  std::vector<Point> v(n);
  std::vector<double> m(n);
  std::vector<Point> a(n);
  initialize(x, v, m, a);

  const int dt = 1;
  leapfrog_vanilla(dt, x, v, m, a);
  double reference_sum = sum_of_norms(x, v);

  const double max_error = 1e-5;
  double error = 0.0;
  double sum = 0.0;

  initialize(x, v, m, a);
  leapfrog_vanilla(dt, x, v, m, a);
  sum = sum_of_norms(x, v);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_vanilla: " << error << std::endl;
  if (error > max_error) {
    return false;
  }

  initialize(x, v, m, a);
  leapfrog_blocked<blocksize>(dt, x, v, m, a);
  sum = sum_of_norms(x, v);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_blocked: " << error << std::endl;
  if (error > max_error) {
    return false;
  }

  initialize(x, v, m, a);
  leapfrog_parallel_coarse_lock(dt, x, v, m, a, number_of_threads);
  sum = sum_of_norms(x, v);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_parallel_coarse_lock: " << error << std::endl;
  if (error > max_error) {
    return false;
  }

  initialize(x, v, m, a);
  leapfrog_parallel_fine_lock(dt, x, v, m, a, number_of_threads);
  sum = sum_of_norms(x, v);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_parallel_fine_lock: " << error << std::endl;
  if (error > max_error) {
    return false;
  }

  std::vector<std::array<std::atomic<double>, 3>> a_atomic(n);
  initialize(x, v, m, a_atomic);
  leapfrog_parallel_atomic_update(dt, x, v, m, a_atomic, number_of_threads);
  sum = sum_of_norms(x, v);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_parallel_atomic_update: " << error << std::endl;
  if (error > max_error) {
    return false;
  }

  initialize(x, v, m, a);
  leapfrog_parallel_no_symmetry(dt, x, v, m, a, number_of_threads);
  sum = sum_of_norms(x, v);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_parallel_no_symmetry: " << error << std::endl;
  if (error > max_error) {
    return false;
  }

  initialize(x, v, m, a);
  leapfrog_parallel_privatization(dt, x, v, m, a, number_of_threads);
  sum = sum_of_norms(x, v);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_parallel_privatization: " << error << std::endl;
  if (error > max_error) {
    return false;
  }

  initialize(x, v, m, a);
  leapfrog_parallel_privatization_interleaved_splitting(dt, x, v, m, a, number_of_threads);
  sum = sum_of_norms(x, v);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_parallel_privatization_interleaved_splitting: " << error << std::endl;
  if (error > max_error) {
    return false;
  }

  initialize(x, v, m, a);
  leapfrog_parallel_privatization_workload_splitting(dt, x, v, m, a, number_of_threads);
  sum = sum_of_norms(x, v);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_parallel_privatization_workload_splitting: " << error << std::endl;
  if (error > max_error) {
    return false;
  }

  std::vector<PointPadded> x_pad(n);
  std::vector<PointPadded> v_pad(n);
  std::vector<PointPadded> a_pad(n);
  initialize(x_pad, v_pad, m, a_pad);
  leapfrog_vectorized<blocksize>(dt, x_pad, v_pad, m, a_pad);
  sum = sum_of_norms(x_pad, v_pad);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_vectorized: " << error << std::endl;
  if (error > max_error) {
    return false;
  }

  initialize(x_pad, v_pad, m, a_pad);
  leapfrog_vectorized_parallel_privatization_workload_splitting<blocksize>(dt, x_pad, v_pad, m, a_pad, number_of_threads);
  sum = sum_of_norms(x_pad, v_pad);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_vectorized_parallel_privatization_workload_splitting: " << error << std::endl;
  if (error > max_error) {
    return false;
  }

  initialize(x_pad, v_pad, m, a_pad);
  leapfrog_vectorized_parallel_privatization_interleaved_splitting<blocksize>(dt, x_pad, v_pad, m, a_pad, number_of_threads);
  sum = sum_of_norms(x_pad, v_pad);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_vectorized_parallel_privatization_interleaved_splitting: " << error << std::endl;
  if (error > max_error) {
    return false;
  }

  return true;
}

template <typename F, typename T1, typename T2>
double single_performance_test(const int number_of_threads, const T1& _1, const T2& _2, const int n, const int timesteps, const double dt, const F&& f) {
  // _1 and _2 are just dummy objects passed to deduce the types T1 and T2. This
  // is horrible design, but it was a quick hack to make things run
  std::vector<T1> x(n);
  std::vector<T1> v(n);
  std::vector<double> m(n);
  std::vector<T2> a(n);
  initialize(x, v, m, a);

  auto start = get_time_stamp();
  auto stop = start;
  for (int k = 0; k < timesteps; ++k) {
    f(dt, x, v, m, a, number_of_threads);
  }
  stop = get_time_stamp();
  double elapsed = get_duration_seconds(start, stop);
  double flops = timesteps * (13.0 * n * (n - 1.0) + 12.0 * n);
  return flops / elapsed / 1e9;
}
void run_performance_test(const int number_of_threads, const int n, const int timesteps, const double dt) {
  // Note: For some tests we need different data types for some or all the
  // vectors. It would be possible to specify the template parameter to the
  // single_performance_test call, but since it takes a function pointer we woul
  // need to add the type (using decltype) everywhere.
  //
  // Instead we pass some dummy objects to automatically deduce the template
  // parameter. This is of course not a good design but it was the easiest way
  // out since I already had all the testing infrastructure written.
  Point _point;
  PointPadded _point_padded;

  std::cout << "== Sequential (maybe auto vectorized, not thread parallel)" << std::endl;
  std::cout << "leapfrog_vanilla " << single_performance_test(number_of_threads, _point, _point, n, timesteps, dt, leapfrog_vanilla) << std::endl;
  std::cout << "leapfrog_blocked<2> " << single_performance_test(number_of_threads, _point, _point, n, timesteps, dt, leapfrog_blocked<2>) << std::endl;
  std::cout << "leapfrog_blocked<4> " << single_performance_test(number_of_threads, _point, _point, n, timesteps, dt, leapfrog_blocked<4>) << std::endl;
  std::cout << "leapfrog_blocked<16> " << single_performance_test(number_of_threads, _point, _point, n, timesteps, dt, leapfrog_blocked<16>) << std::endl;
  std::cout << "leapfrog_blocked<64> " << single_performance_test(number_of_threads, _point, _point, n, timesteps, dt, leapfrog_blocked<64>) << std::endl;
  std::cout << "leapfrog_blocked<256> " << single_performance_test(number_of_threads, _point, _point, n, timesteps, dt, leapfrog_blocked<256>) << std::endl;

  std::cout << "== Parallel (thread parallel, not vectorized)" << std::endl;
  std::cout << "leapfrog_parallel_coarse_lock " << single_performance_test(number_of_threads, _point, _point, n, timesteps, dt, leapfrog_parallel_coarse_lock) << std::endl;
  std::cout << "leapfrog_parallel_fine_lock " << single_performance_test(number_of_threads, _point, _point, n, timesteps, dt, leapfrog_parallel_fine_lock) << std::endl;

  using PointAtomic = std::array<std::atomic<double>,3>;
  PointAtomic _point_atomic;
  std::cout << "leapfrog_parallel_atomic_update " << single_performance_test(number_of_threads, _point, _point_atomic, n, timesteps, dt, leapfrog_parallel_atomic_update<PointAtomic>) << std::endl;

  std::cout << "leapfrog_parallel_no_symmetry " << single_performance_test(number_of_threads, _point, _point, n, timesteps, dt, leapfrog_parallel_no_symmetry) << std::endl;
  std::cout << "leapfrog_parallel_privatization " << single_performance_test(number_of_threads, _point, _point, n, timesteps, dt, leapfrog_parallel_privatization) << std::endl;
  std::cout << "leapfrog_parallel_privatization_interleaved_splitting " << single_performance_test(number_of_threads, _point, _point, n, timesteps, dt, leapfrog_parallel_privatization_interleaved_splitting) << std::endl;
  std::cout << "leapfrog_parallel_privatization_workload_splitting " << single_performance_test(number_of_threads, _point, _point, n, timesteps, dt, leapfrog_parallel_privatization_workload_splitting) << std::endl;

  std::cout << "== Unsequential (vectorized, not thread parallel)" << std::endl;
  std::cout << "leapfrog_vectorized<4> " << single_performance_test(number_of_threads, _point_padded, _point_padded, n, timesteps, dt, leapfrog_vectorized<4>) << std::endl;
  std::cout << "leapfrog_vectorized<16> " << single_performance_test(number_of_threads, _point_padded, _point_padded, n, timesteps, dt, leapfrog_vectorized<16>) << std::endl;
  std::cout << "leapfrog_vectorized<64> " << single_performance_test(number_of_threads, _point_padded, _point_padded, n, timesteps, dt, leapfrog_vectorized<64>) << std::endl;
  std::cout << "leapfrog_vectorized<128> " << single_performance_test(number_of_threads, _point_padded, _point_padded, n, timesteps, dt, leapfrog_vectorized<128>) << std::endl;
  std::cout << "leapfrog_vectorized<256> " << single_performance_test(number_of_threads, _point_padded, _point_padded, n, timesteps, dt, leapfrog_vectorized<256>) << std::endl;

  std::cout << "== Unsequential Parallel (vectorized and thread parallel)" << std::endl;
  std::cout << "leapfrog_vectorized_parallel_privatization_workload_splitting<64> " << single_performance_test(number_of_threads, _point_padded, _point_padded, n, timesteps, dt, leapfrog_vectorized_parallel_privatization_workload_splitting<64>) << std::endl;
  std::cout << "leapfrog_vectorized_parallel_privatization_interleaved_splitting<64> " << single_performance_test(number_of_threads, _point_padded, _point_padded, n, timesteps, dt, leapfrog_vectorized_parallel_privatization_interleaved_splitting<64>) << std::endl;
}

// void run_simulation(const int n, const int timesteps, const double dt){
//   std::vector<Point> x(n);
//   std::vector<Point> v(n);
//   std::vector<double> m(n);
//   std::vector<Point> a(n);
//   initialize(x, v, m, a);

//   int mod = 10;
//   char basename[256] = "nbody";  // common part of file name
//   char name[256];      // filename with number
//   FILE *file;          // C style file hande

//   double t = 0.0;
//   for (int k = 0; k < timesteps; ++k) {
//     leapfrog_vanilla(dt, x, v, m, a);
//     t += dt;
//     if (k % mod == 0) {
//       printf("writing %s_%06d.vtk \n", basename, k / mod);
//       sprintf(name, "%s_%06d.vtk", basename, k / mod);
//       file = fopen(name, "w");
//       write_vtk_file_double(file, n, x, v, m, t, dt);
//       fclose(file);
//     }
//   }
// }

int main(int argc, char **argv) {
  int n;               // number of bodies in the system
  int timesteps;       // number of timesteps
  double dt;           // time step
  unsigned int number_of_threads;

  // command line for restarting
  if (argc == 5) {
    sscanf(argv[1], "%d", &n);
    sscanf(argv[2], "%d", &timesteps);
    sscanf(argv[3], "%lg", &dt);
    sscanf(argv[4], "%d", &number_of_threads);
  } else  // invalid command line, print usage
  {
    std::cout << "usage: " << std::endl;
    std::cout
        << "nbody_vanilla <nbodies> <timesteps> <timestep> <number_of_threads>"
        << std::endl;
    return 1;
  }

  std::cout << std::endl << "Run a small test problem" << std::endl;
  if (!test_nbody()) {
    std::cout << "Test failed!" << std::endl;
    return 1;
  }
  else {
    std::cout << "Test succesfull" << std::endl;
  }


  std::cout << std::endl << "Run performance tests:" << std::endl;
  run_performance_test(number_of_threads, n, timesteps, dt);

  // run_simulation(n, timesteps, dt);

  return 0;
}
