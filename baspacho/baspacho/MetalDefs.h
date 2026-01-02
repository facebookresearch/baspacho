/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file MetalDefs.h
 * @brief Metal GPU backend definitions for BaSpaCho
 *
 * This file provides the Metal compute backend for Apple Silicon GPUs.
 *
 * ## Precision Limitation
 *
 * **The Metal backend only supports single-precision (float) operations.**
 *
 * Apple Silicon GPUs (M1, M2, M3, etc.) have limited double-precision support:
 * - No native FP64 compute units in the GPU
 * - Double operations are emulated at ~1/32 the speed of float
 * - Metal Performance Shaders (MPS) only supports float
 *
 * Attempting to use double precision with the Metal backend will result in a
 * runtime error with a clear message. Use BackendFast (CPU) or BackendCuda
 * (NVIDIA GPU) for double precision requirements.
 *
 * ## Usage
 *
 * ```cpp
 * #include "baspacho/baspacho/Solver.h"
 *
 * // Create solver with Metal backend (float only)
 * Settings settings;
 * settings.backend = BackendMetal;
 * auto solver = createSolver<float>(paramSize, structure, settings);
 *
 * // Use MetalMirror for GPU memory
 * MetalMirror<float> dataGpu(hostData);
 * solver.factor(dataGpu.ptr());
 * dataGpu.get(hostData);  // Copy back to CPU
 * ```
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace BaSpaCho {

// Error checking macro for Metal operations
#define mtlCHECK(condition, msg)                                               \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "[%s:%d] Metal Error: %s\n", __FILE__, __LINE__, (msg)); \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_METAL_ALLOCATION(ptr, size)                                  \
  if (ptr == nullptr) {                                                    \
    fprintf(stderr, "Metal: allocation of block of %ld bytes failed\n",    \
            static_cast<long>(size));                                      \
    abort();                                                               \
  }

// Forward declarations for Metal context (implemented in MetalDefs.mm)
// These allow C++ code to use Metal without including Objective-C headers
class MetalContextImpl;

// Metal context singleton - manages device, command queue, and shader library
class MetalContext {
 public:
  static MetalContext& instance();

  // Get raw pointers to Metal objects (cast to appropriate type in .mm files)
  void* device();        // Returns id<MTLDevice>
  void* commandQueue();  // Returns id<MTLCommandQueue>
  void* library();       // Returns id<MTLLibrary>

  // Wait for all GPU operations to complete
  void synchronize();

  // Get a compute pipeline state for a kernel function
  void* getPipelineState(const char* functionName);  // Returns id<MTLComputePipelineState>

 private:
  MetalContext();
  ~MetalContext();
  MetalContext(const MetalContext&) = delete;
  MetalContext& operator=(const MetalContext&) = delete;

  MetalContextImpl* impl;
};

// Utility class to mirror an std::vector on the GPU via Metal buffer
// Uses shared memory mode for CPU/GPU access
template <typename T>
class MetalMirror {
 public:
  MetalMirror() : buffer_(nullptr), ptr_(nullptr), allocSize_(0) {}

  MetalMirror(const std::vector<T>& vec) : buffer_(nullptr), ptr_(nullptr), allocSize_(0) {
    load(vec);
  }

  ~MetalMirror() { clear(); }

  // Non-copyable
  MetalMirror(const MetalMirror&) = delete;
  MetalMirror& operator=(const MetalMirror&) = delete;

  // Movable
  MetalMirror(MetalMirror&& other) noexcept
      : buffer_(other.buffer_), ptr_(other.ptr_), allocSize_(other.allocSize_) {
    other.buffer_ = nullptr;
    other.ptr_ = nullptr;
    other.allocSize_ = 0;
  }

  MetalMirror& operator=(MetalMirror&& other) noexcept {
    if (this != &other) {
      clear();
      buffer_ = other.buffer_;
      ptr_ = other.ptr_;
      allocSize_ = other.allocSize_;
      other.buffer_ = nullptr;
      other.ptr_ = nullptr;
      other.allocSize_ = 0;
    }
    return *this;
  }

  void clear();
  void resizeToAtLeast(size_t size);
  void load(const std::vector<T>& vec);
  void get(std::vector<T>& vec) const;

  // Get raw pointer for kernel access
  T* ptr() const { return ptr_; }

  // Get Metal buffer handle (for binding to compute encoder)
  void* buffer() const { return buffer_; }

  size_t allocSize() const { return allocSize_; }

 private:
  void* buffer_;   // id<MTLBuffer> - stored as void* for C++ compatibility
  T* ptr_;         // CPU-visible pointer from shared buffer
  size_t allocSize_;
};

// Utility class to mirror an std::vector of pointers, applying an offset
// Used for batched operations
template <typename T>
class MetalPtrMirror {
 public:
  MetalPtrMirror() : buffer_(nullptr), ptr_(nullptr), allocSize_(0) {}

  MetalPtrMirror(const std::vector<T*>& vec, int64_t offset = 0)
      : buffer_(nullptr), ptr_(nullptr), allocSize_(0) {
    load(vec, offset);
  }

  ~MetalPtrMirror() { clear(); }

  // Non-copyable
  MetalPtrMirror(const MetalPtrMirror&) = delete;
  MetalPtrMirror& operator=(const MetalPtrMirror&) = delete;

  void clear();
  void load(const std::vector<T*>& vec, int64_t offset = 0);

  T** ptr() const { return ptr_; }
  void* buffer() const { return buffer_; }
  size_t allocSize() const { return allocSize_; }

 private:
  void* buffer_;
  T** ptr_;
  size_t allocSize_;
};

// Explicit template instantiations declared (defined in MetalDefs.mm)
extern template class MetalMirror<float>;
extern template class MetalMirror<double>;
extern template class MetalMirror<int64_t>;
extern template class MetalMirror<int32_t>;

extern template class MetalPtrMirror<float>;
extern template class MetalPtrMirror<double>;

// Buffer registry for mapping raw pointers back to their MTLBuffers
// This is needed because MPS operations require MTLBuffer objects
class MetalBufferRegistry {
 public:
  static MetalBufferRegistry& instance();

  // Register a buffer with its base pointer and size
  void registerBuffer(void* ptr, void* buffer, size_t sizeBytes);

  // Unregister a buffer
  void unregisterBuffer(void* ptr);

  // Find the buffer containing a given pointer, returns {buffer, byteOffset}
  // Returns {nullptr, 0} if not found
  std::pair<void*, size_t> findBuffer(const void* ptr) const;

 private:
  MetalBufferRegistry() = default;
  ~MetalBufferRegistry() = default;
  MetalBufferRegistry(const MetalBufferRegistry&) = delete;
  MetalBufferRegistry& operator=(const MetalBufferRegistry&) = delete;

  struct BufferInfo {
    void* buffer;      // id<MTLBuffer>
    void* basePtr;     // Base pointer from buffer contents
    size_t sizeBytes;  // Size in bytes
  };
  mutable std::mutex mutex_;
  std::vector<BufferInfo> buffers_;
};

}  // end namespace BaSpaCho
