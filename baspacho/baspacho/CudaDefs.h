/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <cstdio>
#include <vector>

namespace BaSpaCho {

const char* cublasGetErrorEnum(cublasStatus_t error);
const char* cusparseGetErrorEnum(cusparseStatus_t error);
const char* cusolverGetErrorEnum(cusolverStatus_t error);

}  // end namespace BaSpaCho

#define cuCHECK(call)                                                                           \
  do {                                                                                          \
    cudaError_t err = (call);                                                                   \
    if (cudaSuccess != err) {                                                                   \
      fprintf(stderr, "[%s:%d] CUDA Error: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      cudaDeviceReset();                                                                        \
      abort();                                                                                  \
    }                                                                                           \
  } while (0)

#define cublasCHECK(call)                                                            \
  do {                                                                               \
    cublasStatus_t status = (call);                                                  \
    if (CUBLAS_STATUS_SUCCESS != status) {                                           \
      fprintf(stderr, "CUBLAS Error: %s\n", ::BaSpaCho::cublasGetErrorEnum(status)); \
      cudaDeviceReset();                                                             \
      abort();                                                                       \
    }                                                                                \
  } while (0)

#define cusparseCHECK(call)                                                              \
  do {                                                                                   \
    cusparseStatus_t status = (call);                                                    \
    if (CUSPARSE_STATUS_SUCCESS != status) {                                             \
      fprintf(stderr, "CUSPARSE Error: %s\n", ::BaSpaCho::cusparseGetErrorEnum(status)); \
      cudaDeviceReset();                                                                 \
      abort();                                                                           \
    }                                                                                    \
  } while (0)

#define cusolverCHECK(call)                                                              \
  do {                                                                                   \
    cusolverStatus_t status = (call);                                                    \
    if (CUSOLVER_STATUS_SUCCESS != status) {                                             \
      fprintf(stderr, "CUSOLVER Error: %s\n", ::BaSpaCho::cusolverGetErrorEnum(status)); \
      cudaDeviceReset();                                                                 \
      abort();                                                                           \
    }                                                                                    \
  } while (0)

#define CHECK_ALLOCATION(ptr, size)                                           \
  if (ptr == nullptr) {                                                       \
    fprintf(stderr, "CUDA: allocation of block of %ld bytes failed\n", size); \
    cudaDeviceReset();                                                        \
    abort();                                                                  \
  }

// utility class to mirror an std::vector on the gpu
template <typename T>
struct DevMirror {
  DevMirror() {}
  DevMirror(const std::vector<T>& vec) { load(vec); }
  ~DevMirror() { clear(); }
  void clear() {
    if (ptr) {
      cuCHECK(cudaFree(ptr));
      ptr = nullptr;
      allocSize = 0;
    }
  }
  void resizeToAtLeast(size_t size) {
    if (allocSize < size) {
      clear();
    }
    if (!ptr && size > 0) {
      cuCHECK(cudaMalloc((void**)&ptr, size * sizeof(T)));
      CHECK_ALLOCATION(ptr, size * sizeof(T));
      allocSize = size;
    }
  }
  void load(const std::vector<T>& vec) {
    if (allocSize < vec.size()) {
      clear();
    }
    if (!vec.size()) {
      return;  // cudaMalloc of size 0 fails
    }
    if (!ptr) {
      cuCHECK(cudaMalloc((void**)&ptr, vec.size() * sizeof(T)));
      CHECK_ALLOCATION(ptr, vec.size() * sizeof(T));
      allocSize = vec.size();
    }
    cuCHECK(cudaMemcpy(ptr, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  void get(std::vector<T>& vec) const {
    cuCHECK(cudaMemcpy(vec.data(), ptr, vec.size() * sizeof(T), cudaMemcpyDeviceToHost));
  }
  T* ptr = nullptr;
  size_t allocSize = 0;
};

// utility class to mirror an std::vector of pointers, applying an offset
template <typename T>
struct DevPtrMirror {
  DevPtrMirror() {}
  DevPtrMirror(const std::vector<T*>& vec, int64_t offset = 0) { load(vec, offset); }
  void clear() {
    if (ptr) {
      cuCHECK(cudaFree(ptr));
      ptr = nullptr;
      allocSize = 0;
    }
  }
  void load(const std::vector<T*>& vec, int64_t offset = 0) {
    T** vecCopy = (T**)alloca(vec.size() * sizeof(T*));
    for (size_t i = 0; i < vec.size(); i++) {
      vecCopy[i] = vec[i] + offset;
    }

    if (allocSize < vec.size()) {
      clear();
    }
    if (!vec.size()) {
      return;  // cudaMalloc of size 0 fails
    }
    if (!ptr) {
      cuCHECK(cudaMalloc((void**)&ptr, vec.size() * sizeof(T*)));
      CHECK_ALLOCATION(ptr, vec.size() * sizeof(T*));
      allocSize = vec.size();
    }
    cuCHECK(cudaMemcpy(ptr, vecCopy, vec.size() * sizeof(T*), cudaMemcpyHostToDevice));
  }
  ~DevPtrMirror() { clear(); }
  T** ptr = nullptr;
  size_t allocSize = 0;
};
