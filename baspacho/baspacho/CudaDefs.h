#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include <vector>

namespace BaSpaCho {

const char* cublasGetErrorEnum(cublasStatus_t error);
const char* cusparseGetErrorEnum(cusparseStatus_t error);
const char* cusolverGetErrorEnum(cusolverStatus_t error);

}  // end namespace BaSpaCho

#define cuCHECK(call)                                                 \
  do {                                                                \
    cudaError_t err = (call);                                         \
    if (cudaSuccess != err) {                                         \
      fprintf(stderr, "[%s:%d] CUDA Error: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                               \
      cudaDeviceReset();                                              \
      abort();                                                        \
    }                                                                 \
  } while (0)

#define cublasCHECK(call)                              \
  do {                                                 \
    cublasStatus_t status = (call);                    \
    if (CUBLAS_STATUS_SUCCESS != status) {             \
      fprintf(stderr, "CUBLAS Error: %s\n",            \
              ::BaSpaCho::cublasGetErrorEnum(status)); \
      cudaDeviceReset();                               \
      abort();                                         \
    }                                                  \
  } while (0)

#define cusparseCHECK(call)                              \
  do {                                                   \
    cusparseStatus_t status = (call);                    \
    if (CUSPARSE_STATUS_SUCCESS != status) {             \
      fprintf(stderr, "CUSPARSE Error: %s\n",            \
              ::BaSpaCho::cusparseGetErrorEnum(status)); \
      cudaDeviceReset();                                 \
      abort();                                           \
    }                                                    \
  } while (0)

#define cusolverCHECK(call)                              \
  do {                                                   \
    cusolverStatus_t status = (call);                    \
    if (CUSOLVER_STATUS_SUCCESS != status) {             \
      fprintf(stderr, "CUSOLVER Error: %s\n",            \
              ::BaSpaCho::cusolverGetErrorEnum(status)); \
      cudaDeviceReset();                                 \
      abort();                                           \
    }                                                    \
  } while (0)

#define CHECK_ALLOCATION(ptr, size)                                           \
  if (ptr == nullptr) {                                                       \
    fprintf(stderr, "CUDA: allocation of block of %ld bytes failed\n", size); \
    cudaDeviceReset();                                                        \
    abort();                                                                  \
  }

// utility class to mirror an std::vector
template <typename T>
struct DevMirror {
  DevMirror() {}
  DevMirror(const std::vector<T>& vec) { load(vec); }
  ~DevMirror() { clear(); }
  void clear() {
    if (ptr) {
      cuCHECK(cudaFree(ptr));
      ptr = nullptr;
    }
  }
  void load(const std::vector<T>& vec) {
    clear();
    if (!vec.size()) {
      return;  // cudaMalloc of size 0 fails
    }
    cuCHECK(cudaMalloc((void**)&ptr, vec.size() * sizeof(T)));
    CHECK_ALLOCATION(ptr, vec.size() * sizeof(T));
    cuCHECK(cudaMemcpy(ptr, vec.data(), vec.size() * sizeof(T),
                       cudaMemcpyHostToDevice));
  }
  void get(std::vector<T>& vec) const {
    cuCHECK(cudaMemcpy(vec.data(), ptr, vec.size() * sizeof(T),
                       cudaMemcpyDeviceToHost));
  }
  T* ptr = nullptr;
};

// utility class to mirror an std::vector of pointers, applying an offset
template <typename T>
struct DevPtrMirror {
  DevPtrMirror(const std::vector<T*>& vec, int64_t offset = 0) {
    T** vecCopy = (T**)alloca(vec.size() * sizeof(T*));
    for (size_t i = 0; i < vec.size(); i++) {
      vecCopy[i] = vec[i] + offset;
    }
    cuCHECK(cudaMalloc((void**)&ptr, vec.size() * sizeof(T*)));
    CHECK_ALLOCATION(ptr, vec.size() * sizeof(T*));
    cuCHECK(cudaMemcpy(ptr, vecCopy, vec.size() * sizeof(T*),
                       cudaMemcpyHostToDevice));
  }
  ~DevPtrMirror() {
    if (ptr) {
      cuCHECK(cudaFree(ptr));
    }
  }
  T** ptr = nullptr;
};