#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <stdio.h>
#include "time_experiment.hh"

#ifdef HAVE_VCL
#include "vcl/vectorclass.h"
#endif

#define HAVE_NEON 1

#ifdef HAVE_NEON
#include <arm_neon.h>
#endif

#ifndef __GNUC__
#define __restrict__
#endif

const int B = 20;
const int K = 4;

struct GlobalContext
{
  // input data
  int n;          // nxn lattice of points
  int iterations; // number of iterations to do
  double *u0;     // the initial guess
  double *u1;     // temporary vector

  // output data

  GlobalContext(int n_)
      : n(n_)
  {
  }
};

void jacobi_vanilla_kernel(int n, int iterations, double *__restrict__ uold, double *__restrict__ unew)
{
  // do iterations
  for (int i = 0; i < iterations; i++)
  {
    for (int i1 = 1; i1 < n - 1; i1++)
      for (int i0 = 1; i0 < n - 1; i0++)
        unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 - 1] + uold[i1 * n + i0 + 1] + uold[i1 * n + i0 + n]);
    std::swap(uold, unew);
  }
}

void jacobi_vanilla(std::shared_ptr<GlobalContext> context)
{
  jacobi_vanilla_kernel(context->n, context->iterations, context->u0, context->u1);
}

void jacobi_wave_kernel(int n, int iterations, double *__restrict__ uold, double *__restrict__ unew)
{
  double *u[2];
  u[0] = uold;
  u[1] = unew;

  // do iterations
  for (int kk = 0; kk < iterations; kk += K)
    for (int m = 2; m <= n + K - 2; m++)
      for (int k = std::max(1, m - n + 2); k <= std::min(K, m - 1); k++)
      {
        int i1 = m - k;
        int dst = k % 2;
        int src = 1 - dst;
        for (int i0 = 1; i0 < n - 1; i0++)
          u[dst][i1 * n + i0] = 0.25 * (u[src][i1 * n + i0 - n] + u[src][i1 * n + i0 - 1] + u[src][i1 * n + i0 + 1] + u[src][i1 * n + i0 + n]);
      }
}

void jacobi_wave(std::shared_ptr<GlobalContext> context)
{
  jacobi_wave_kernel(context->n, context->iterations, context->u0, context->u1);
}

void jacobi_blocked(std::shared_ptr<GlobalContext> context)
{
  int n = context->n;
  double *uold = context->u0;
  double *unew = context->u1;

  int blocksB = ((n - 2) / B) * B;
  int remainderB = (n - 2) % B;
  // std::cout << "blocksB=" << blocksB << " remainderB=" << remainderB << std::endl;

  // do iterations
  for (int i = 0; i < context->iterations; i++)
  {
    for (int I1 = 1; I1 < 1 + blocksB; I1 += B)
      for (int I0 = 1; I0 < 1 + blocksB; I0 += B)
        for (int i1 = I1; i1 < I1 + B; i1++)
          for (int i0 = I0; i0 < I0 + B; i0++)
            unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 - 1] + uold[i1 * n + i0 + 1] + uold[i1 * n + i0 + n]);
    for (int I0 = 1; I0 < 1 + blocksB; I0 += B)
      for (int i1 = 1 + blocksB; i1 < n - 1; i1++)
        for (int i0 = I0; i0 < I0 + B; i0++)
          unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 - 1] + uold[i1 * n + i0 + 1] + uold[i1 * n + i0 + n]);
    for (int i1 = 1 + blocksB; i1 < n - 1; i1++)
      for (int i0 = 1 + blocksB; i0 < n - 1; i0++)
        unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 - 1] + uold[i1 * n + i0 + 1] + uold[i1 * n + i0 + n]);
    std::swap(uold, unew);
  }
  for (int I1 = 1; I1 < 1 + blocksB; I1 += B)
    for (int i1 = I1; i1 < I1 + B; i1++)
      for (int i0 = 1 + blocksB; i0 < n - 1; i0++)
        unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 - 1] + uold[i1 * n + i0 + 1] + uold[i1 * n + i0 + n]);

  // result should be in u1
  if (context->u1 != uold)
    std::swap(context->u0, context->u1);
}

#ifdef HAVE_VCL
void jacobi_vectorized(std::shared_ptr<GlobalContext> context)
{
  int n = context->n;
  double *uold = context->u0;
  double *unew = context->u1;

  int blocks4 = ((n - 2) / 4) * 4;

  Vec4d QUARTER = Vec4d(0.25);
  Vec4d A, B, C, D, E;

  // do iterations
  for (int i = 0; i < context->iterations; i++)
  {
    for (int i1 = 1; i1 < n - 1; i1++)
    {
      int istart = i1 * n + 1;
      int iend = istart + blocks4;
      for (int index = istart; index < iend; index += 4)
      {
        A = Vec4d(0.0);
        B.load(&uold[index - n]);
        C.load(&uold[index - 1]);
        D.load(&uold[index + 1]);
        E.load(&uold[index + n]);
        A = mul_add(QUARTER, B, A);
        A = mul_add(QUARTER, C, A);
        A = mul_add(QUARTER, D, A);
        A = mul_add(QUARTER, E, A);
        A.store(&unew[index]);
      }
      for (int index = iend; index < n - 1; index++)
        unew[index] = 0.25 * (uold[index - n] + uold[index - 1] + uold[index + 1] + uold[index + n]);
    }
    std::swap(uold, unew);
  }
  // result should be in u1
  if (context->u1 != uold)
    std::swap(context->u0, context->u1);
}
#endif

#ifdef HAVE_VCL
void jacobi_vectorized_wave(std::shared_ptr<GlobalContext> context)
{
  int n = context->n;
  double *u[2];
  u[0] = context->u0;
  u[1] = context->u1;

  int blocks4 = ((n - 2) / 4) * 4;

  Vec4d QUARTER = Vec4d(0.25);
  Vec4d A, B, C, D, E;

  // do iterations
  for (int kk = 0; kk < context->iterations; kk += K)
    for (int m = 2; m <= n + K - 2; m++)
      for (int k = std::max(1, m - n + 2); k <= std::min(K, m - 1); k++)
      {
        int i1 = m - k;
        int dst = k % 2;
        int src = 1 - dst;
        int istart = i1 * n + 1;
        int iend = istart + blocks4;
        for (int index = istart; index < iend; index += 4)
        {
          A = Vec4d(0.0);
          B.load(&u[src][index - n]);
          C.load(&u[src][index - 1]);
          D.load(&u[src][index + 1]);
          E.load(&u[src][index + n]);
          A = mul_add(QUARTER, B, A);
          A = mul_add(QUARTER, C, A);
          A = mul_add(QUARTER, D, A);
          A = mul_add(QUARTER, E, A);
          A.store(&u[dst][index]);
        }
        for (int index = iend; index < n - 1; index++)
          u[dst][index] = 0.25 * (u[src][index - n] + u[src][index - 1] + u[src][index + 1] + u[src][index + n]);
      }
}
#endif

#ifdef HAVE_NEON
void jacobi_vectorized_neon(std::shared_ptr<GlobalContext> context)
{
  int n = context->n;
  double *uold = context->u0;
  double *unew = context->u1;

  int blocks4 = ((n - 2) / 4) * 4;

  float64x2_t QUARTER = vmovq_n_f64(0.25);
  float64x2_t S0, S1, N0, N1, C0, C1, M0, M1, M2;

  // do iterations
  for (int i = 0; i < context->iterations; i++)
  {
    for (int i1 = 1; i1 < n - 1; i1++)
    {
      int istart = i1 * n + 1;
      int iend = istart + blocks4;
      for (int index = istart; index < iend; index += 4)
      {
        C0 = vmovq_n_f64(0.0);
        C1 = vmovq_n_f64(0.0);
        S0 = vld1q_f64(&uold[index - n]);
        S1 = vld1q_f64(&uold[index + 2 - n]);
        M0 = vld1q_f64(&uold[index - 1]);
        M1 = vld1q_f64(&uold[index + 1]);
        M2 = vld1q_f64(&uold[index + 3]);
        N0 = vld1q_f64(&uold[index + n]);
        N1 = vld1q_f64(&uold[index + 2 + n]);
        C0 = vfmaq_f64(C0, QUARTER, S0);
        C1 = vfmaq_f64(C1, QUARTER, S1);
        C0 = vfmaq_f64(C0, QUARTER, M0);
        C1 = vfmaq_f64(C1, QUARTER, M1);
        C0 = vfmaq_f64(C0, QUARTER, M1);
        C1 = vfmaq_f64(C1, QUARTER, M2);
        C0 = vfmaq_f64(C0, QUARTER, N0);
        C1 = vfmaq_f64(C1, QUARTER, N1);
        vst1q_f64(&unew[index], C0);
        vst1q_f64(&unew[index + 2], C1);
      }
      for (int index = iend; index < n - 1; index++)
        unew[index] = 0.25 * (uold[index - n] + uold[index - 1] + uold[index + 1] + uold[index + n]);
    }
    std::swap(uold, unew);
  }
  // result should be in u1
  if (context->u1 != uold)
    std::swap(context->u0, context->u1);
}
#endif

#ifdef HAVE_NEON
// version without non-aligned loads
// using pointers
void jacobi_vectorized_neon_v2(std::shared_ptr<GlobalContext> context)
{
  std::size_t n = context->n;
  double *uold = context->u0;
  double *unew = context->u1;

  if (n % 2 != 0)
  {
    std::cout << "n=" << n << " must be even" << std::endl;
    return;
  }

  if (n < 6)
  {
    std::cout << "n=" << n << " must be greater or equal 6" << std::endl;
    return;
  }

  float64x2_t Q = vmovq_n_f64(0.25); // all 1/4
  float64x2_t Z = vmovq_n_f64(0.0);  // all 0
  float64x2_t C, N, E, S, W;         // 5x2 two numbers from uold  in stencil arrangement
  float64x2_t L, R;                  // left and right neighbor values constructed from W,C,E
  float64x2_t U;                     // the computed value to be written
  double *pS, *pC, *pN, *pU;         // pointers to avoid all that index computation

  // iteration loop
  for (std::size_t i = 0; i < context->iterations; i++)
  {
    // loop over rows
    for (std::size_t i1 = 1; i1 < n - 1; i1++)
    {
      std::size_t rowstart = i1 * n; // index of first entry in row i1
      pS = &uold[rowstart - n];
      pC = &uold[rowstart];
      pN = &uold[rowstart + n];
      S = vld1q_f64(pS); // load south
      pS += 2;           // increment immediately after use
      C = vld1q_f64(pC); // load center
      pC += 2;           // increment to second block
      E = vld1q_f64(pC); // load east
      pC += 2;           // increment immediately after use
      N = vld1q_f64(pN); // load north
      pN += 2;           // increment immediately after use

      S = vcopyq_laneq_f64(S, 0, Z, 0); // zero out lane 0 = left boundary
      L = vmovq_n_f64(0.0);
      L = vcopyq_laneq_f64(L, 1, C, 0); // left neighbors
      R = vmovq_n_f64(0.0);
      R = vcopyq_laneq_f64(R, 1, E, 0); // left neighbors
      N = vcopyq_laneq_f64(N, 0, Z, 0); // zero out lane 0 = left boundary

      pU = &unew[rowstart];
      U = vmovq_n_f64(0.0); // U computes first block including boundary
      U = vfmaq_f64(U, Q, S);
      U = vfmaq_f64(U, Q, L);
      U = vfmaq_f64(U, Q, R);
      U = vfmaq_f64(U, Q, N);
      vst1q_f64(pU, U);
      pU += 2;

      std::size_t rowend = i1 * n + n; // one after last entry in this row
      // loop over inner blocks in this row
      for (std::size_t index = rowstart + 2; index < rowend - 2; index += 2)
      {
        S = vld1q_f64(pS); // load south
        pS += 2;           // increment immediately after use
        W = C;
        N = vld1q_f64(pN); // load north
        pN += 2;           // increment immediately after use
        C = E;
        E = vld1q_f64(pC); // load east
        pC += 2;           // increment immediately after use

        L = vextq_f64(W, C, 1);
        R = vextq_f64(C, E, 1);
        U = vmovq_n_f64(0.0); // U computes first block including boundary
        U = vfmaq_f64(U, Q, S);
        U = vfmaq_f64(U, Q, L);
        U = vfmaq_f64(U, Q, R);
        U = vfmaq_f64(U, Q, N);
        vst1q_f64(pU, U);
        pU += 2;
      }

      // last block in this row including boundary
      S = vld1q_f64(pS); // load south
      W = C;
      N = vld1q_f64(pN); // load north
      C = E;

      S = vcopyq_laneq_f64(S, 1, Z, 1); // zero out lane 1 = right boundary
      L = vmovq_n_f64(0.0);
      L = vcopyq_laneq_f64(L, 0, W, 1); // left neighbors
      R = vmovq_n_f64(0.0);
      R = vcopyq_laneq_f64(R, 0, C, 1); // left neighbors
      N = vcopyq_laneq_f64(N, 1, Z, 1); // zero out lane 1 = right boundary

      U = vmovq_n_f64(0.0); // U computes first block including boundary
      U = vfmaq_f64(U, Q, S);
      U = vfmaq_f64(U, Q, L);
      U = vfmaq_f64(U, Q, R);
      U = vfmaq_f64(U, Q, N);
      vst1q_f64(pU, U);
    }
    std::swap(uold, unew);
  }
  // result should be in u1
  if (context->u1 != uold)
    std::swap(context->u0, context->u1);
}
#endif

#ifdef HAVE_NEON
// version without non-aligned loads
// and two rows in parallel
void jacobi_vectorized_neon_v3(std::shared_ptr<GlobalContext> context)
{
  std::size_t n = context->n;
  double *uold = context->u0;
  double *unew = context->u1;

  if (n % 2 != 0)
  {
    std::cout << "n=" << n << " must be even" << std::endl;
    return;
  }

  if (n < 6)
  {
    std::cout << "n=" << n << " must be greater or equal 6" << std::endl;
    return;
  }

  // needs exactly 16 registers
  float64x2_t Q = vmovq_n_f64(0.25);
  float64x2_t Z = vmovq_n_f64(0.0);
  float64x2_t N, S;
  float64x2_t C0, E0, W0, U0, L0, R0;
  float64x2_t C1, E1, W1, U1, L1, R1;

  // iteration loop
  for (std::size_t i = 0; i < context->iterations; i++)
  {
    // loop over rows; process two rows in parallel
    for (std::size_t i1 = 1; i1 < n - 1; i1 += 2)
    {
      std::size_t rowstart = i1 * n;           // first entry in row i1
      S = vld1q_f64(&uold[rowstart - n]);      // load south
      C0 = vld1q_f64(&uold[rowstart]);         // load center
      E0 = vld1q_f64(&uold[rowstart + 2]);     // load east
      C1 = vld1q_f64(&uold[rowstart + n]);     // load center
      E1 = vld1q_f64(&uold[rowstart + n + 2]); // load east
      N = vld1q_f64(&uold[rowstart + 2 * n]);  // load north

      S = vcopyq_laneq_f64(S, 0, Z, 0); // zero out lane 0 = left boundary
      L0 = vmovq_n_f64(0.0);
      L0 = vcopyq_laneq_f64(L0, 1, C0, 0); // left neighbors
      L1 = vmovq_n_f64(0.0);
      L1 = vcopyq_laneq_f64(L1, 1, C1, 0); // left neighbors
      R0 = vmovq_n_f64(0.0);
      R0 = vcopyq_laneq_f64(R0, 1, E0, 0); // left neighbors
      R1 = vmovq_n_f64(0.0);
      R1 = vcopyq_laneq_f64(R1, 1, E1, 0); // left neighbors
      C0 = vcopyq_laneq_f64(C0, 0, Z, 0);  // zero out lane 0 = left boundary
      C1 = vcopyq_laneq_f64(C1, 0, Z, 0);  // zero out lane 0 = left boundary
      N = vcopyq_laneq_f64(N, 0, Z, 0);    // zero out lane 0 = left boundary

      U0 = vmovq_n_f64(0.0); // U computes first block including boundary
      U0 = vfmaq_f64(U0, Q, S);
      U0 = vfmaq_f64(U0, Q, L0);
      U0 = vfmaq_f64(U0, Q, R0);
      U0 = vfmaq_f64(U0, Q, C1);
      vst1q_f64(&unew[rowstart], U0);
      U1 = vmovq_n_f64(0.0); // U computes first block including boundary
      U1 = vfmaq_f64(U1, Q, C0);
      U1 = vfmaq_f64(U1, Q, L1);
      U1 = vfmaq_f64(U1, Q, R1);
      U1 = vfmaq_f64(U1, Q, N);
      vst1q_f64(&unew[rowstart + n], U1);

      std::size_t rowend = i1 * n + n; // one after last entry in this row
      // loop over inner blocks in this row
      for (std::size_t index = rowstart + 2; index < rowend - 2; index += 2)
      {
        S = vld1q_f64(&uold[index - n]); // load south
        W0 = C0;
        C0 = E0;
        E0 = vld1q_f64(&uold[index + 2]); // load east
        W1 = C1;
        C1 = E1;
        E1 = vld1q_f64(&uold[index + n + 2]); // load east
        N = vld1q_f64(&uold[index + 2 * n]);  // load north

        L0 = vextq_f64(W0, C0, 1);
        R0 = vextq_f64(C0, E0, 1);
        L1 = vextq_f64(W1, C1, 1);
        R1 = vextq_f64(C1, E1, 1);

        U0 = vmovq_n_f64(0.0); // U computes first block including boundary
        U0 = vfmaq_f64(U0, Q, S);
        U0 = vfmaq_f64(U0, Q, L0);
        U0 = vfmaq_f64(U0, Q, R0);
        U0 = vfmaq_f64(U0, Q, C1);
        vst1q_f64(&unew[index], U0);
        U1 = vmovq_n_f64(0.0); // U computes first block including boundary
        U1 = vfmaq_f64(U1, Q, C0);
        U1 = vfmaq_f64(U1, Q, L1);
        U1 = vfmaq_f64(U1, Q, R1);
        U1 = vfmaq_f64(U1, Q, N);
        vst1q_f64(&unew[index + n], U1);
      }

      // last block in this row including boundary
      S = vld1q_f64(&uold[rowend - 2 - n]); // load south
      W0 = C0;
      C0 = E0;
      W1 = C1;
      C1 = E1;
      N = vld1q_f64(&uold[rowend - 2 + 2 * n]); // load north

      S = vcopyq_laneq_f64(S, 1, Z, 1); // zero out lane 1 = right boundary
      L0 = vmovq_n_f64(0.0);
      L0 = vcopyq_laneq_f64(L0, 0, W0, 1); // left neighbors
      L1 = vmovq_n_f64(0.0);
      L1 = vcopyq_laneq_f64(L1, 0, W1, 1); // left neighbors
      R0 = vmovq_n_f64(0.0);
      R0 = vcopyq_laneq_f64(R0, 0, C0, 1); // left neighbors
      R1 = vmovq_n_f64(0.0);
      R1 = vcopyq_laneq_f64(R1, 0, C1, 1); // left neighbors
      C0 = vcopyq_laneq_f64(C0, 1, Z, 1);  // zero out lane 0 = left boundary
      C1 = vcopyq_laneq_f64(C1, 1, Z, 1);  // zero out lane 0 = left boundary
      N = vcopyq_laneq_f64(N, 1, Z, 1);    // zero out lane 1 = right boundary

      U0 = vmovq_n_f64(0.0); // U computes first block including boundary
      U0 = vfmaq_f64(U0, Q, S);
      U0 = vfmaq_f64(U0, Q, L0);
      U0 = vfmaq_f64(U0, Q, R0);
      U0 = vfmaq_f64(U0, Q, C1);
      vst1q_f64(&unew[rowend - 2], U0);
      U1 = vmovq_n_f64(0.0); // U computes first block including boundary
      U1 = vfmaq_f64(U1, Q, C0);
      U1 = vfmaq_f64(U1, Q, L1);
      U1 = vfmaq_f64(U1, Q, R1);
      U1 = vfmaq_f64(U1, Q, N);
      vst1q_f64(&unew[rowend - 2 + n], U1);
    }
    std::swap(uold, unew);
  }
  // result should be in u1
  if (context->u1 != uold)
    std::swap(context->u0, context->u1);
}
#endif

#ifdef HAVE_NEON
void jacobi_vectorized_wave_neon(std::shared_ptr<GlobalContext> context)
{
  int n = context->n;
  double *u[2];
  u[0] = context->u0;
  u[1] = context->u1;

  float64x2_t Q = vmovq_n_f64(0.25); // all 1/4
  float64x2_t Z = vmovq_n_f64(0.0);  // all 0
  float64x2_t C, N, E, S, W;         // 5x2 two numbers from uold  in stencil arrangement
  float64x2_t L, R;                  // left and right neighbor values constructed from W,C,E
  float64x2_t U;                     // the computed value to be written
  double *pS, *pC, *pN, *pU;         // pointers to avoid all that index computation

  // do iterations
  for (int kk = 0; kk < context->iterations; kk += K)
    for (int m = 2; m <= n + K - 2; m++)
      for (int k = std::max(1, m - n + 2); k <= std::min(K, m - 1); k++)
      {
        int dst = k % 2;
        int src = 1 - dst;
        int i1 = m - k;

        std::size_t rowstart = i1 * n; // index of first entry in row i1
        pS = &u[src][rowstart - n];
        pC = &u[src][rowstart];
        pN = &u[src][rowstart + n];
        S = vld1q_f64(pS); // load south
        pS += 2;           // increment immediately after use
        C = vld1q_f64(pC); // load center
        pC += 2;           // increment to second block
        E = vld1q_f64(pC); // load east
        pC += 2;           // increment immediately after use
        N = vld1q_f64(pN); // load north
        pN += 2;           // increment immediately after use

        S = vcopyq_laneq_f64(S, 0, Z, 0); // zero out lane 0 = left boundary
        L = vmovq_n_f64(0.0);
        L = vcopyq_laneq_f64(L, 1, C, 0); // left neighbors
        R = vmovq_n_f64(0.0);
        R = vcopyq_laneq_f64(R, 1, E, 0); // left neighbors
        N = vcopyq_laneq_f64(N, 0, Z, 0); // zero out lane 0 = left boundary

        pU = &u[dst][rowstart];
        U = vmovq_n_f64(0.0); // U computes first block including boundary
        U = vfmaq_f64(U, Q, S);
        U = vfmaq_f64(U, Q, L);
        U = vfmaq_f64(U, Q, R);
        U = vfmaq_f64(U, Q, N);
        vst1q_f64(pU, U);
        pU += 2;

        std::size_t rowend = i1 * n + n; // one after last entry in this row
        // loop over inner blocks in this row
        for (std::size_t index = rowstart + 2; index < rowend - 2; index += 2)
        {
          S = vld1q_f64(pS); // load south
          pS += 2;           // increment immediately after use
          W = C;
          N = vld1q_f64(pN); // load north
          pN += 2;           // increment immediately after use
          C = E;
          E = vld1q_f64(pC); // load east
          pC += 2;           // increment immediately after use

          L = vextq_f64(W, C, 1);
          R = vextq_f64(C, E, 1);
          U = vmovq_n_f64(0.0); // U computes first block including boundary
          U = vfmaq_f64(U, Q, S);
          U = vfmaq_f64(U, Q, L);
          U = vfmaq_f64(U, Q, R);
          U = vfmaq_f64(U, Q, N);
          vst1q_f64(pU, U);
          pU += 2;
        }

        // last block in this row including boundary
        S = vld1q_f64(pS); // load south
        W = C;
        N = vld1q_f64(pN); // load north
        C = E;

        S = vcopyq_laneq_f64(S, 1, Z, 1); // zero out lane 1 = right boundary
        L = vmovq_n_f64(0.0);
        L = vcopyq_laneq_f64(L, 0, W, 1); // left neighbors
        R = vmovq_n_f64(0.0);
        R = vcopyq_laneq_f64(R, 0, C, 1); // left neighbors
        N = vcopyq_laneq_f64(N, 1, Z, 1); // zero out lane 1 = right boundary

        U = vmovq_n_f64(0.0); // U computes first block including boundary
        U = vfmaq_f64(U, Q, S);
        U = vfmaq_f64(U, Q, L);
        U = vfmaq_f64(U, Q, R);
        U = vfmaq_f64(U, Q, N);
        vst1q_f64(pU, U);
      }
}
#endif

// main function runs the experiments and outputs results as csv
int main(int argc, char **argv)
{
  // read parameters
  int n = 1024;
  int iterations = 1000;
  if (argc != 3)
  {
    std::cout << "usage: ./jacobi_vanilla <size> <iterations>"
              << std::endl;
    exit(1);
  }
  sscanf(argv[1], "%d", &n);
  sscanf(argv[2], "%d", &iterations);
  // std::cout << "jacobi_vanilla: n=" << n
  //        << " iterations=" << iterations
  //        << " memory (mbytes)=" << (n*n)*8.0*2.0/1024.0/1024.0
  //        << std::endl;

  // check sizes
  if (K % 2 == 1)
  {
    std::cout << "K must be even" << std::endl;
    exit(1);
  }
  if (iterations % K != 0)
  {
    std::cout << "iterations must be a multiple of K=" << K << std::endl;
    exit(1);
  }

  // get global context shared by aall threads
  auto context = std::make_shared<GlobalContext>(n);
  context->iterations = iterations;

  // allocate aligned arrays
  context->u0 = new (std::align_val_t(64)) double[n * n];
  context->u1 = new (std::align_val_t(64)) double[n * n];

  // fill boundary values and initial values
  auto g = [&](int i0, int i1)
  { return (i0 > 0 && i0 < n - 1 && i1 > 0 && i1 < n - 1)
               ? 0.0
               : ((double)(i0 + i1)) / n; };

  std::cout << "N,";
  std::cout << "vanilla,";
  std::cout << "blocked,";
  std::cout << "wave,";
#ifdef HAVE_NEON
  std::cout << "neon,";
  std::cout << "neon_v2,";
  std::cout << "neon_v3,";
  std::cout << "wave_neon";
#endif
  std::cout << std::endl;
  std::cout << n * n;

  // warmup
  for (int i1 = 0; i1 < n; i1++)
    for (int i0 = 0; i0 < n; i0++)
      context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
  auto start = get_time_stamp();
  jacobi_blocked(context);
  auto stop = get_time_stamp();
  double elapsed = get_duration_seconds(start, stop);
  double updates = context->iterations;
  updates *= (n - 2) * (n - 2);

  // vanilla
  for (int i1 = 0; i1 < n; i1++)
    for (int i0 = 0; i0 < n; i0++)
      context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
  start = get_time_stamp();
  jacobi_vanilla(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start, stop);
  std::cout << "," << updates / elapsed / 1e9;

  // blocked
  for (int i1 = 0; i1 < n; i1++)
    for (int i0 = 0; i0 < n; i0++)
      context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
  start = get_time_stamp();
  jacobi_blocked(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start, stop);
  std::cout << "," << updates / elapsed / 1e9;

  // wave
  for (int i1 = 0; i1 < n; i1++)
    for (int i0 = 0; i0 < n; i0++)
      context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
  start = get_time_stamp();
  jacobi_wave(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start, stop);
  std::cout << "," << updates / elapsed / 1e9;

// vectorized
#ifdef HAVE_VCL
  for (int i1 = 0; i1 < n; i1++)
    for (int i0 = 0; i0 < n; i0++)
      context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
  start = get_time_stamp();
  jacobi_vectorized(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start, stop);
  std::cout << "," << updates / elapsed / 1e9;
#endif

  // vectorized wave
#ifdef HAVE_VCL
  for (int i1 = 0; i1 < n; i1++)
    for (int i0 = 0; i0 < n; i0++)
      context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
  start = get_time_stamp();
  jacobi_vectorized_wave(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start, stop);
  std::cout << "," << updates / elapsed / 1e9;
#endif

// vectorized
#ifdef HAVE_NEON
  for (int i1 = 0; i1 < n; i1++)
    for (int i0 = 0; i0 < n; i0++)
      context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
  start = get_time_stamp();
  jacobi_vectorized_neon(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start, stop);
  std::cout << "," << updates / elapsed / 1e9;
#endif

// vectorized
#ifdef HAVE_NEON
  for (int i1 = 0; i1 < n; i1++)
    for (int i0 = 0; i0 < n; i0++)
      context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
  start = get_time_stamp();
  jacobi_vectorized_neon_v2(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start, stop);
  std::cout << "," << updates / elapsed / 1e9;
#endif

// vectorized
#ifdef HAVE_NEON
  for (int i1 = 0; i1 < n; i1++)
    for (int i0 = 0; i0 < n; i0++)
      context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
  start = get_time_stamp();
  jacobi_vectorized_neon_v3(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start, stop);
  std::cout << "," << updates / elapsed / 1e9;
#endif

#ifdef HAVE_NEON
  for (int i1 = 0; i1 < n; i1++)
    for (int i0 = 0; i0 < n; i0++)
      context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
  start = get_time_stamp();
  jacobi_vectorized_wave_neon(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start, stop);
  std::cout << "," << updates / elapsed / 1e9;
#endif

  std::cout << std::endl;

  // deallocate arrays
  delete[] context->u1;
  delete[] context->u0;

  return 0;
}
