/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <type_traits>

/**
 * this workaround is for compatibility with Cuda architectures
 * before 6.0, which lack atomic addition of double numbers
 **/
#ifdef CUDA_DOUBLE_ATOMIC_ADD_WORKAROUND
// workaround for double on <6 architectures
__device__ inline double atomicAddWorkaround(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_ull, assumed,
                      ::__double_as_longlong(val + __longlong_as_double(assumed)));
    // integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}

template <typename T>
__device__ inline T atomicAddWorkaround(T* address, T val) {
  return atomicAdd(address, val);
}

#define atomicAdd atomicAddWorkaround
#endif

// [atomic] A -= B * C.T
template <typename A, typename B, typename C>
__device__ void locked_sub_product(A& aMat, const B& bMat, const C& cMatT) {
  using T = std::remove_reference_t<decltype(aMat(0, 0))>;
  for (int i = 0; i < bMat.rows(); i++) {
    for (int j = 0; j < cMatT.rows(); j++) {
      T* addr = &aMat(i, j);
      T val = -bMat.row(i).dot(cMatT.row(j));
      atomicAdd(addr, val);
    }
  }
}

// [atomic] A -= B * C
template <typename A, typename B, typename C>
__device__ void locked_sub_AxB(A& aMat, const B& bMat, const C& cMatT) {
  using T = std::remove_reference_t<decltype(aMat(0, 0))>;
  for (int i = 0; i < bMat.rows(); i++) {
    for (int j = 0; j < cMatT.cols(); j++) {
      T* addr = &aMat(i, j);
      T val = -bMat.row(i).dot(cMatT.col(j));
      atomicAdd(addr, val);
    }
  }
}

// [atomic] A -= B.T * C
template <typename A, typename B, typename C>
__device__ void locked_sub_ATxB(A& aMat, const B& bMat, const C& cMatT) {
  using T = std::remove_reference_t<decltype(aMat(0, 0))>;
  for (int i = 0; i < bMat.cols(); i++) {
    for (int j = 0; j < cMatT.cols(); j++) {
      T* addr = &aMat(i, j);
      T val = -bMat.col(i).dot(cMatT.col(j));
      atomicAdd(addr, val);
    }
  }
}
