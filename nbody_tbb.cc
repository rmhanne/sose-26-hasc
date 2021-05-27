#include <math.h>
#include <stdio.h>
#include <stdlib.h>


#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "nbody_generate.hh"
#include "nbody_io.hh"
#include "time_experiment.hh"
#include "vcl/vectorclass.h"


#include <tbb/tbb.h>


/*const double gamma = 6.674E-11;*/
const double G = 1.0;
const double epsilon2 = 1E-10;


struct Point {
  std::array<double, 3> _x {};

  double& operator[](int i) {return _x[i];}

  const double& operator[](int i) const {return _x[i];}

  Point& operator+=(const Point& other) {
    for (std::size_t i=0; i<_x.size(); ++i){
      _x[i] += other._x[i];
    }
    return *this;
  }

  Point operator+(const Point& other) const {
    Point output = *this;
    output += other;
    return output;
  }
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

template <typename T>
void acceleration_vanilla(const std::vector<T>& x,
                          const std::vector<double>& m,
                          std::vector<T>& a) {
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
template <typename T>
void leapfrog_vanilla(const double dt,
                      std::vector<T>& x,
                      std::vector<T>& v,
                      const std::vector<double>& m,
                      std::vector<T>& a) {
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


template <typename T, int B>
void acceleration_blocked(const std::vector<T>& x,
                          const std::vector<double>& m,
                          std::vector<T>& a) {
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
template <typename T, int B>
void leapfrog_blocked(const double dt,
                      std::vector<T>& x,
                      std::vector<T>& v,
                      const std::vector<double>& m,
                      std::vector<T>& a) {
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
  acceleration_blocked<T, B>(x, m, a);

  // update velocity: 6n flops
  for (int i = 0; i < n; i++) {
    v[i][0] += dt * a[i][0];
    v[i][1] += dt * a[i][1];
    v[i][2] += dt * a[i][2];
  }
}

template <typename T, int B>
void acceleration_blocked_parallel_no_symmetry_tbb(const std::vector<T>& x,
                                                   const std::vector<double>& m,
                                                   std::vector<T>& a) {
  const int n = x.size();

  tbb::parallel_for(0, n, B, [n, &x, &m, &a](int I) {
    for (int J = 0; J<n; J+= B){
      for (int i=I; i<I+B; ++i){
        for (int j=J; j<J+B; ++j){
          double d0 = x[j][0] - x[i][0];
          double d1 = x[j][1] - x[i][1];
          double d2 = x[j][2] - x[i][2];
          double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
          double r = sqrt(r2);
          double invfact = G / (r * r2);
          double factorj = m[j] * invfact;
          a[i][0] += factorj * d0;
          a[i][1] += factorj * d1;
          a[i][2] += factorj * d2;
        }
      }
    }
  });
}
template <typename T, int B>
void leapfrog_blocked_parallel_no_symmetry_tbb(const double dt,
                                               std::vector<T>& x,
                                               std::vector<T>& v,
                                               const std::vector<double>& m,
                                               std::vector<T>& a) {
  const int n = x.size();

  // update position: 6n flops
  tbb::parallel_for(0, n, 1, [&x, &v, &a, dt](int i) {
    x[i][0] += dt * v[i][0];
    x[i][1] += dt * v[i][1];
    x[i][2] += dt * v[i][2];
    a[i][0] = a[i][1] = a[i][2] = 0.0;
  });

  // compute new acceleration: n*(n-1)*13 flops
  acceleration_blocked_parallel_no_symmetry_tbb<T, B>(x, m, a);

  // update velocity: 6n flops
  tbb::parallel_for(0, n, 1, [&v, &a, dt](int i) {
    v[i][0] += dt * a[i][0];
    v[i][1] += dt * a[i][1];
    v[i][2] += dt * a[i][2];
  });
}


template <typename T, int B>
void acceleration_blocked_parallel_tbb(const std::vector<T>& x,
                                       const std::vector<double>& m,
                                       std::vector<T>& a) {
  const int n = x.size();
  using vector_t = std::vector<T>;

  // ===========================================================================
  // Use combinable as thread local storage to store local acceleration updates.
  // Afterwards we do a sequential reduction of these local vectors.
  //
  // Note: This was faster than the parallel reduction version below. Maybe the
  // blocked_range below is not optimal for the triangular loop domain?
  // ===========================================================================

  tbb::combinable<vector_t> priv_a{[n](){return vector_t(n);}};

  tbb::parallel_for(0, n, B, [n, &x, &m, &priv_a](int I) {
    vector_t& a = priv_a.local();
    // block (I,I)
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
      for (int i = I; i < I + B; i++) {
        for (int j = J; j < J + B; j++) {
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

  });

  // Do a sequential reduction
  priv_a.combine_each([&](vector_t b){
    std::transform(a.begin(), // source 1 begin
                   a.end(),   // source 1 end
                   b.begin(), // source 2 begin
                   a.begin(), // destination begin
                   std::plus<T>());
  });
}
template <typename T, int B>
void leapfrog_blocked_parallel_tbb(const double dt,
                                   std::vector<T>& x,
                                   std::vector<T>& v,
                                   const std::vector<double>& m,
                                   std::vector<T>& a) {
  const int n = x.size();

  // update position: 6n flops
  tbb::parallel_for(0, n, 1, [&x, &v, &a, dt](int i) {
    x[i][0] += dt * v[i][0];
    x[i][1] += dt * v[i][1];
    x[i][2] += dt * v[i][2];
    a[i][0] = a[i][1] = a[i][2] = 0.0;
  });

  // compute new acceleration: n*(n-1)*13 flops
  acceleration_blocked_parallel_tbb<T, B>(x, m, a);

  // update velocity: 6n flops
  tbb::parallel_for(0, n, 1, [&v, &a, dt](int i) {
    v[i][0] += dt * a[i][0];
    v[i][1] += dt * a[i][1];
    v[i][2] += dt * a[i][2];
  });
}


template <typename T, int B>
void acceleration_vectorized_parallel_tbb(const std::vector<T>& x,
                                          const std::vector<double>& m,
                                          std::vector<T>& a) {
  const int n = x.size();
  using vector_t = std::vector<T>;

  tbb::combinable<vector_t> priv_a{[n](){return vector_t(n);}};

  tbb::parallel_for(0, n, B, [n, &x, &m, &priv_a](int I) {
    vector_t& a = priv_a.local();

    // Exactly 16 Vec4d for 16 registers
    Vec4d A0, A1;
    Vec4d D0, D1, D2, D3;  // distances
    Vec4d E0, E1, E2, E3;  // distances
    Vec4d S0, S, U, T0, T1, T2;

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
  });

  // Do a sequential reduction
  priv_a.combine_each([&](vector_t b){
    std::transform(a.begin(), // source 1 begin
                   a.end(),   // source 1 end
                   b.begin(), // source 2 begin
                   a.begin(), // destination begin
                   std::plus<T>());
  });
}
template <typename T, int B>
void leapfrog_vectorized_parallel_tbb(const double dt,
                                      std::vector<T>& x,
                                      std::vector<T>& v,
                                      const std::vector<double>& m,
                                      std::vector<T>& a) {
  const int n = x.size();

  // update position: 6n flops
  tbb::parallel_for(0, n, 1, [&x, &v, &a, dt](int i) {
    x[i][0] += dt * v[i][0];
    x[i][1] += dt * v[i][1];
    x[i][2] += dt * v[i][2];
    a[i][0] = a[i][1] = a[i][2] = 0.0;
  });

  // compute new acceleration: n*(n-1)*13 flops
  acceleration_vectorized_parallel_tbb<T, B>(x, m, a);

  // update velocity: 6n flops
  tbb::parallel_for(0, n, 1, [&v, &a, dt](int i) {
    v[i][0] += dt * a[i][0];
    v[i][1] += dt * a[i][1];
    v[i][2] += dt * a[i][2];
  });
}


template <typename T>
void acceleration_vanilla_parallel_tbb(const std::vector<T>& x,
                                       const std::vector<double>& m,
                                       std::vector<T>& a) {
  const int n = x.size();
  using vector_t = std::vector<T>;

  // ===========================================================================
  // Use combinable as thread local storage to store local acceleration updates.
  // Afterwards we do a sequential reduction of these local vectors.
  //
  // Note: This was faster than the parallel reduction version below. Maybe the
  // blocked_range below is not optimal for the triangular loop domain? Things
  // might of course look different on a machine with more cores.
  // ===========================================================================

  tbb::combinable<vector_t> priv_a{[n](){return vector_t(n);}};

  tbb::parallel_for(0, n, 1, [n, &x, &m, &priv_a](int i) {
    vector_t& a = priv_a.local();
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
  });

  // Do a sequential reduction
  priv_a.combine_each([&](vector_t b){
    std::transform(a.begin(), // source 1 begin
                   a.end(),   // source 1 end
                   b.begin(), // source 2 begin
                   a.begin(), // destination begin
                   std::plus<T>());
  });

  // // ==========================
  // // Parallel reduction version
  // // ==========================

  // a = tbb::parallel_reduce(tbb::blocked_range<std::size_t>(0, n),
  //                          vector_t(n),
  //                          [&](const tbb::blocked_range<std::size_t>& r, vector_t a_local) -> vector_t{
  //                            for (std::size_t i=r.begin(); i!=r.end(); ++i){
  //                              for (int j = i + 1; j < n; j++) {
  //                                double d0 = x[j][0] - x[i][0];
  //                                double d1 = x[j][1] - x[i][1];
  //                                double d2 = x[j][2] - x[i][2];
  //                                double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
  //                                double r = sqrt(r2);
  //                                double invfact = G / (r * r2);
  //                                double factori = m[i] * invfact;
  //                                double factorj = m[j] * invfact;
  //                                a_local[i][0] += factorj * d0;
  //                                a_local[i][1] += factorj * d1;
  //                                a_local[i][2] += factorj * d2;
  //                                a_local[j][0] -= factori * d0;
  //                                a_local[j][1] -= factori * d1;
  //                                a_local[j][2] -= factori * d2;
  //                              }
  //                            }
  //                            return a_local;
  //                          },
  //                          [](vector_t first, const vector_t& second) -> vector_t{
  //                            std::transform(first.begin(),
  //                                           first.end(),
  //                                           second.begin(),
  //                                           first.begin(),
  //                                           std::plus<T>());
  //                            return first;
  //                          }
  //                          );

}
template <typename T>
void leapfrog_vanilla_parallel_tbb(const double dt,
                                   std::vector<T>& x,
                                   std::vector<T>& v,
                                   const std::vector<double>& m,
                                   std::vector<T>& a) {
  const int n = x.size();

  // update position: 6n flops
  tbb::parallel_for(0, n, 1, [&x, &v, &a, dt](int i) {
    x[i][0] += dt * v[i][0];
    x[i][1] += dt * v[i][1];
    x[i][2] += dt * v[i][2];
    a[i][0] = a[i][1] = a[i][2] = 0.0;
  });

  // compute new acceleration: n*(n-1)*13 flops
  acceleration_vanilla_parallel_tbb<T>(x, m, a);

  // update velocity: 6n flops
  tbb::parallel_for(0, n, 1, [&v, &a, dt](int i) {
    v[i][0] += dt * a[i][0];
    v[i][1] += dt * a[i][1];
    v[i][2] += dt * a[i][2];
  });
}


template <typename T>
void acceleration_vanilla_parallel_no_symmetry_tbb(const std::vector<T>& x,
                                                   const std::vector<double>& m,
                                                   std::vector<T>& a) {
  const int n = x.size();
  tbb::parallel_for(0, n, 1, [n, &x, &m, &a](int i) {
    for (int j = 0; j < n; j++) {
      double d0 = x[j][0] - x[i][0];
      double d1 = x[j][1] - x[i][1];
      double d2 = x[j][2] - x[i][2];
      double r2 = d0 * d0 + d1 * d1 + d2 * d2 + epsilon2;
      double r = sqrt(r2);
      double invfact = G / (r * r2);
      double factorj = m[j] * invfact;
      a[i][0] += factorj * d0;
      a[i][1] += factorj * d1;
      a[i][2] += factorj * d2;
    }
  });
}
template <typename T>
void leapfrog_vanilla_parallel_no_symmetry_tbb(const double dt,
                                               std::vector<T>& x,
                                               std::vector<T>& v,
                                               const std::vector<double>& m,
                                               std::vector<T>& a) {
  const int n = x.size();

  // update position: 6n flops
  tbb::parallel_for(0, n, 1, [&x, &v, &a, dt](int i) {
    x[i][0] += dt * v[i][0];
    x[i][1] += dt * v[i][1];
    x[i][2] += dt * v[i][2];
    a[i][0] = a[i][1] = a[i][2] = 0.0;
  });

  // compute new acceleration: n*(n-1)*13 flops
  acceleration_vanilla_parallel_tbb<T>(x, m, a);

  // update velocity: 6n flops
  tbb::parallel_for(0, n, 1, [&v, &a, dt](int i) {
    v[i][0] += dt * a[i][0];
    v[i][1] += dt * a[i][1];
    v[i][2] += dt * a[i][2];
  });
}




template <typename T, int B>
void acceleration_vectorized(const std::vector<T>& x,
                             const std::vector<double>& m,
                             std::vector<T>& a) {
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
template <typename T, int B>
void leapfrog_vectorized(const double dt,
                         std::vector<T>& x,
                         std::vector<T>& v,
                         const std::vector<double>& m,
                         std::vector<T>& a) {
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
  acceleration_vectorized<T, B>(x, m, a);

  // update velocity: 6n flops
  for (int i = 0; i < n; i++) {
    v[i][0] += dt * a[i][0];
    v[i][1] += dt * a[i][1];
    v[i][2] += dt * a[i][2];
  }
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
// Note: Use template parameters here as the acceleration vector will be of type
// std::vector<std::atomic<double>> for the atomic update tests!
template <typename T1, typename T2, typename T3>
void initialize(T1& x_vec,
                T1& v_vec,
                T2& m_vec,
                T3& a_vec
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
    std::vector<double> tmp (3);
    x_vec[i] = {x[i][0], x[i][1], x[i][2]};
    v_vec[i] = {v[i][0], v[i][1], v[i][2]};
    a_vec[i] = {a[i][0], a[i][1], a[i][2]};
  }

  delete[] x;
  delete[] v;
  delete[] m;
  delete[] a;
}

bool test_nbody() {
  const int n = 64 * 64;
  const int B = 64;

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
  leapfrog_vanilla_parallel_tbb(dt, x, v, m, a);
  sum = sum_of_norms(x, v);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_vanilla_parallel_tbb: " << error << std::endl;
  if (error > max_error) {
    return false;
  }

  initialize(x, v, m, a);
  leapfrog_vanilla_parallel_no_symmetry_tbb(dt, x, v, m, a);
  sum = sum_of_norms(x, v);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_vanilla_parallel_no_symmetry_tbb: " << error << std::endl;
  if (error > max_error) {
    return false;
  }

  initialize(x, v, m, a);
  leapfrog_blocked<Point, B>(dt, x, v, m, a);
  sum = sum_of_norms(x, v);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_blocked: " << error << std::endl;
  if (error > max_error) {
    return false;
  }

  initialize(x, v, m, a);
  leapfrog_blocked_parallel_tbb<Point, B>(dt, x, v, m, a);
  sum = sum_of_norms(x, v);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_blocked_parallel_tbb: " << error << std::endl;
  if (error > max_error) {
    return false;
  }

  initialize(x, v, m, a);
  leapfrog_blocked_parallel_no_symmetry_tbb<Point, B>(dt, x, v, m, a);
  sum = sum_of_norms(x, v);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_blocked_parallel_no_symmetry_tbb: " << error << std::endl;
  if (error > max_error) {
    return false;
  }

  // The vectorized version only works with padded points!
  std::vector<PointPadded> x_p(n);
  std::vector<PointPadded> v_p(n);
  std::vector<PointPadded> a_p(n);

  initialize(x_p, v_p, m, a_p);
  leapfrog_vectorized<PointPadded, B>(dt, x_p, v_p, m, a_p);
  sum = sum_of_norms(x_p, v_p);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_vectorized: " << error << std::endl;
  if (error > max_error) {
    return false;
  }

  initialize(x_p, v_p, m, a_p);
  leapfrog_vectorized_parallel_tbb<PointPadded, B>(dt, x_p, v_p, m, a_p);
  sum = sum_of_norms(x_p, v_p);
  error = std::abs(sum - reference_sum);
  std::cout << "leapfrog_vectorized_parallel_tbb: " << error << std::endl;
  if (error > max_error) {
    return false;
  }



  return true;
}

template <typename T, typename F>
double single_performance_test(const T& _, const int n, const int timesteps, const double dt, const F&& f) {
  std::vector<T> x(n);
  std::vector<T> v(n);
  std::vector<double> m(n);
  std::vector<T> a(n);
  initialize(x, v, m, a);

  if constexpr (std::is_same_v<T, PointPadded>){
    if (alignment(&x[0]) != 64 or alignment(&v[0]) != 64 or alignment(&a[0]) != 64){
      std::cout << "Warning: Your std::vector<PointPadded> does not have alignment 64" << std::endl;
    }
  }

  auto start = get_time_stamp();
  auto stop = start;
  for (int k = 0; k < timesteps; ++k) {
    f(dt, x, v, m, a);
  }
  stop = get_time_stamp();
  double elapsed = get_duration_seconds(start, stop);
  double flops = timesteps * (13.0 * n * (n - 1.0) + 12.0 * n);
  return flops / elapsed / 1e9;
}
template <typename T>
void run_performance_test(const int n, const int timesteps, const double dt) {
  // This is not good C++ style but the easiest way to avoid writing out the
  // template parameters for single_performance_test...
  T _dummy;

  std::cout << "== Sequential (maybe auto vectorized, not thread parallel)" << std::endl;
  std::cout << "leapfrog_vanilla " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_vanilla<T>) << std::endl;
  // std::cout << "leapfrog_blocked<T, 2> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked<T, 2>) << std::endl;
  // std::cout << "leapfrog_blocked<T, 4> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked<T, 4>) << std::endl;
  // std::cout << "leapfrog_blocked<T, 8> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked<T, 8>) << std::endl;
  // std::cout << "leapfrog_blocked<T, 16> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked<T, 16>) << std::endl;
  // std::cout << "leapfrog_blocked<T, 32> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked<T, 32>) << std::endl;
  // std::cout << "leapfrog_blocked<T, 64> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked<T, 64>) << std::endl;
  // std::cout << "leapfrog_blocked<T, 128> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked<T, 128>) << std::endl;
  // std::cout << "leapfrog_blocked<T, 256> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked<T, 256>) << std::endl;
  // std::cout << "leapfrog_blocked<T, 512> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked<T, 512>) << std::endl;

  std::cout << "== Parallel (thread parallel, not vectorized)" << std::endl;
  std::cout << "leapfrog_vanilla_parallel_tbb " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_vanilla_parallel_tbb<T>) << std::endl;
  // std::cout << "leapfrog_blocked_parallel_tbb<T, 2> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked_parallel_tbb<T, 2>) << std::endl;
  // std::cout << "leapfrog_blocked_parallel_tbb<T, 4> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked_parallel_tbb<T, 4>) << std::endl;
  // std::cout << "leapfrog_blocked_parallel_tbb<T, 8> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked_parallel_tbb<T, 8>) << std::endl;
  // std::cout << "leapfrog_blocked_parallel_tbb<T, 16> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked_parallel_tbb<T, 16>) << std::endl;
  // std::cout << "leapfrog_blocked_parallel_tbb<T, 32> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked_parallel_tbb<T, 32>) << std::endl;
  // std::cout << "leapfrog_blocked_parallel_tbb<T, 64> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked_parallel_tbb<T, 64>) << std::endl;
  // std::cout << "leapfrog_blocked_parallel_tbb<T, 128> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked_parallel_tbb<T, 128>) << std::endl;
  // std::cout << "leapfrog_blocked_parallel_tbb<T, 256> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked_parallel_tbb<T, 256>) << std::endl;
  // std::cout << "leapfrog_blocked_parallel_tbb<T, 512> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked_parallel_tbb<T, 512>) << std::endl;
  std::cout << "leapfrog_vanilla_parallel_no_symmetry_tbb " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_vanilla_parallel_no_symmetry_tbb<T>) << std::endl;
  std::cout << "leapfrog_blocked_parallel_no_symmetry_tbb<T, 2> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked_parallel_no_symmetry_tbb<T, 2>) << std::endl;
  std::cout << "leapfrog_blocked_parallel_no_symmetry_tbb<T, 4> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked_parallel_no_symmetry_tbb<T, 4>) << std::endl;
  std::cout << "leapfrog_blocked_parallel_no_symmetry_tbb<T, 8> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked_parallel_no_symmetry_tbb<T, 8>) << std::endl;
  std::cout << "leapfrog_blocked_parallel_no_symmetry_tbb<T, 16> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked_parallel_no_symmetry_tbb<T, 16>) << std::endl;
  std::cout << "leapfrog_blocked_parallel_no_symmetry_tbb<T, 32> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked_parallel_no_symmetry_tbb<T, 32>) << std::endl;
  std::cout << "leapfrog_blocked_parallel_no_symmetry_tbb<T, 64> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked_parallel_no_symmetry_tbb<T, 64>) << std::endl;
  std::cout << "leapfrog_blocked_parallel_no_symmetry_tbb<T, 128> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked_parallel_no_symmetry_tbb<T, 128>) << std::endl;
  std::cout << "leapfrog_blocked_parallel_no_symmetry_tbb<T, 256> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_blocked_parallel_no_symmetry_tbb<T, 256>) << std::endl;


  if constexpr (std::is_same_v<T, PointPadded>){
    // std::cout << "== Unsequential (vectorized, not thread parallel)" << std::endl;
    // std::cout << "leapfrog_vectorized<4> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_vectorized<T, 4>) << std::endl;
    // std::cout << "leapfrog_vectorized<8> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_vectorized<T, 8>) << std::endl;
    // std::cout << "leapfrog_vectorized<16> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_vectorized<T, 16>) << std::endl;
    // std::cout << "leapfrog_vectorized<32> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_vectorized<T, 32>) << std::endl;
    // std::cout << "leapfrog_vectorized<64> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_vectorized<T, 64>) << std::endl;
    // std::cout << "leapfrog_vectorized<128> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_vectorized<T, 128>) << std::endl;

    // std::cout << "== Unsequential Parallel (vectorized and thread parallel)" << std::endl;
    // std::cout << "leapfrog_vectorized_parallel_tbb<4> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_vectorized_parallel_tbb<T, 4>) << std::endl;
    // std::cout << "leapfrog_vectorized_parallel_tbb<8> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_vectorized_parallel_tbb<T, 8>) << std::endl;
    // std::cout << "leapfrog_vectorized_parallel_tbb<16> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_vectorized_parallel_tbb<T, 16>) << std::endl;
    // std::cout << "leapfrog_vectorized_parallel_tbb<32> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_vectorized_parallel_tbb<T, 32>) << std::endl;
    // std::cout << "leapfrog_vectorized_parallel_tbb<64> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_vectorized_parallel_tbb<T, 64>) << std::endl;
    // std::cout << "leapfrog_vectorized_parallel_tbb<128> " << single_performance_test(_dummy, n, timesteps, dt, leapfrog_vectorized_parallel_tbb<T, 128>) << std::endl;
  }

}

int main(int argc, char **argv) {
  int n;               // number of bodies in the system
  int timesteps;       // number of timesteps
  double dt;           // time step

  // command line for restarting
  if (argc == 4) {
    sscanf(argv[1], "%d", &n);
    sscanf(argv[2], "%d", &timesteps);
    sscanf(argv[3], "%lg", &dt);
  } else  // invalid command line, print usage
  {
    std::cout << "usage: " << std::endl;
    std::cout
        << "nbody_vanilla <nbodies> <timesteps> <timestep>"
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


  std::cout << std::endl << "Run performance tests for regular Point class" << std::endl;
  run_performance_test<Point>(n, timesteps, dt);

  std::cout << std::endl << "Run performance tests for padded Point class" << std::endl;
  run_performance_test<PointPadded>(n, timesteps, dt);

  // run_simulation(n, timesteps, dt);

  return 0;
}
