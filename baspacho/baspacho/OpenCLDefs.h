/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file OpenCLDefs.h
 * @brief OpenCL GPU backend definitions for BaSpaCho
 *
 * This file provides the OpenCL compute backend for portable GPU acceleration.
 * It uses CLBlast for high-performance BLAS operations and custom OpenCL
 * kernels for BaSpaCho-specific sparse operations.
 *
 * ## Usage
 *
 * ```cpp
 * #include "baspacho/baspacho/Solver.h"
 *
 * // Create solver with OpenCL backend
 * Settings settings;
 * settings.backend = BackendOpenCL;
 * auto solver = createSolver<float>(paramSize, structure, settings);
 *
 * // Use OpenCLMirror for GPU memory
 * OpenCLMirror<float> dataGpu(hostData);
 * solver.factor(dataGpu.ptr());
 * dataGpu.get(hostData);  // Copy back to CPU
 * ```
 *
 * ## Notes
 *
 * - OpenCL supports both float and double precision (unlike Metal)
 * - CLBlast provides gemm, trsm, syrk; potrf uses CPU fallback (Eigen/BLAS)
 * - Custom kernels handle sparse elimination and assembly operations
 */

#pragma once

#ifdef BASPACHO_USE_OPENCL

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace BaSpaCho {

// Error checking macro for OpenCL operations
#define clCHECK(err)                                                             \
  do {                                                                           \
    cl_int _err = (err);                                                         \
    if (_err != CL_SUCCESS) {                                                    \
      fprintf(stderr, "[%s:%d] OpenCL Error: %d\n", __FILE__, __LINE__, _err);   \
      abort();                                                                   \
    }                                                                            \
  } while (0)

#define CHECK_OPENCL_ALLOCATION(ptr, size)                                  \
  if (ptr == nullptr) {                                                     \
    fprintf(stderr, "OpenCL: allocation of block of %ld bytes failed\n",    \
            static_cast<long>(size));                                       \
    abort();                                                                 \
  }

// Forward declarations
class OpenCLContextImpl;

/**
 * OpenCL context singleton - manages device, command queue, and program
 */
class OpenCLContext {
 public:
  static OpenCLContext& instance();

  // Get OpenCL handles
  cl_context context() const { return context_; }
  cl_command_queue queue() const { return queue_; }
  cl_device_id device() const { return device_; }
  cl_program program() const { return program_; }

  // Wait for all GPU operations to complete
  void synchronize();

  // Get a kernel by name (aborts if not found)
  cl_kernel getKernel(const char* kernelName);

  // Check if a kernel is available (returns false if not, doesn't abort)
  bool hasKernel(const char* kernelName);

 private:
  OpenCLContext();
  ~OpenCLContext();
  OpenCLContext(const OpenCLContext&) = delete;
  OpenCLContext& operator=(const OpenCLContext&) = delete;

  void buildProgram();

  cl_platform_id platform_;
  cl_device_id device_;
  cl_context context_;
  cl_command_queue queue_;
  cl_program program_;
  std::mutex kernelMutex_;
  std::unordered_map<std::string, cl_kernel> kernelCache_;
};

/**
 * Utility class to mirror an std::vector on the GPU via OpenCL buffer
 * Registers with OpenCLBufferRegistry so GPU buffers can be found from host pointers
 */
template <typename T>
class OpenCLMirror {
 public:
  OpenCLMirror() : buffer_(nullptr), hostPtr_(nullptr), allocSize_(0) {}

  OpenCLMirror(const std::vector<T>& vec) : buffer_(nullptr), hostPtr_(nullptr), allocSize_(0) {
    load(vec);
  }

  ~OpenCLMirror() { clear(); }

  // Non-copyable
  OpenCLMirror(const OpenCLMirror&) = delete;
  OpenCLMirror& operator=(const OpenCLMirror&) = delete;

  // Movable
  OpenCLMirror(OpenCLMirror&& other) noexcept
      : buffer_(other.buffer_), hostPtr_(other.hostPtr_), allocSize_(other.allocSize_) {
    other.buffer_ = nullptr;
    other.hostPtr_ = nullptr;
    other.allocSize_ = 0;
  }

  OpenCLMirror& operator=(OpenCLMirror&& other) noexcept {
    if (this != &other) {
      clear();
      buffer_ = other.buffer_;
      hostPtr_ = other.hostPtr_;
      allocSize_ = other.allocSize_;
      other.buffer_ = nullptr;
      other.hostPtr_ = nullptr;
      other.allocSize_ = 0;
    }
    return *this;
  }

  void clear();
  void resizeToAtLeast(size_t size);
  void load(const std::vector<T>& vec);
  void get(std::vector<T>& vec) const;

  // Get OpenCL buffer for kernel binding
  cl_mem buffer() const { return buffer_; }

  // Get the host pointer that was used to load data (for buffer registry lookup)
  T* hostPtr() const { return hostPtr_; }

  size_t allocSize() const { return allocSize_; }

 private:
  cl_mem buffer_;
  T* hostPtr_;       // Host pointer used for buffer registry
  size_t allocSize_;
};

/**
 * Utility class to mirror an std::vector of pointers for batched operations
 */
template <typename T>
class OpenCLPtrMirror {
 public:
  OpenCLPtrMirror() : buffer_(nullptr), allocSize_(0) {}

  OpenCLPtrMirror(const std::vector<T*>& vec, int64_t offset = 0)
      : buffer_(nullptr), allocSize_(0) {
    load(vec, offset);
  }

  ~OpenCLPtrMirror() { clear(); }

  // Non-copyable
  OpenCLPtrMirror(const OpenCLPtrMirror&) = delete;
  OpenCLPtrMirror& operator=(const OpenCLPtrMirror&) = delete;

  void clear();
  void load(const std::vector<T*>& vec, int64_t offset = 0);

  cl_mem buffer() const { return buffer_; }
  size_t allocSize() const { return allocSize_; }

 private:
  cl_mem buffer_;
  size_t allocSize_;
};

// Explicit template instantiations declared (defined in OpenCLDefs.cpp)
extern template class OpenCLMirror<float>;
extern template class OpenCLMirror<double>;
extern template class OpenCLMirror<int64_t>;
extern template class OpenCLMirror<int32_t>;

extern template class OpenCLPtrMirror<float>;
extern template class OpenCLPtrMirror<double>;

/**
 * Buffer registry for mapping raw pointers back to their cl_mem buffers
 * This is needed because OpenCL operations require cl_mem objects, but the
 * solver API passes raw pointers. When data is loaded via OpenCLMirror,
 * we track the mapping so we can find the GPU buffer from the CPU pointer.
 */
class OpenCLBufferRegistry {
 public:
  static OpenCLBufferRegistry& instance();

  // Register a buffer with its base pointer and size
  void registerBuffer(void* hostPtr, cl_mem buffer, size_t sizeBytes);

  // Unregister a buffer
  void unregisterBuffer(void* hostPtr);

  // Find the buffer containing a given pointer, returns {buffer, byteOffset}
  // Returns {nullptr, 0} if not found
  std::pair<cl_mem, size_t> findBuffer(const void* ptr) const;

 private:
  OpenCLBufferRegistry() = default;
  ~OpenCLBufferRegistry() = default;
  OpenCLBufferRegistry(const OpenCLBufferRegistry&) = delete;
  OpenCLBufferRegistry& operator=(const OpenCLBufferRegistry&) = delete;

  struct BufferInfo {
    cl_mem buffer;     // OpenCL buffer handle
    void* hostPtr;     // Host pointer that was used to load data
    size_t sizeBytes;  // Size in bytes
  };
  mutable std::mutex mutex_;
  std::vector<BufferInfo> buffers_;
};

}  // end namespace BaSpaCho

#endif  // BASPACHO_USE_OPENCL
