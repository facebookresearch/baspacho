/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "baspacho/baspacho/MetalDefs.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace BaSpaCho {

// Implementation class for MetalContext - holds Objective-C objects
class MetalContextImpl {
 public:
  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;
  id<MTLLibrary> library;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelineCache;
  std::mutex pipelineMutex;

  MetalContextImpl() {
    @autoreleasepool {
      // Get the default Metal device
      device = MTLCreateSystemDefaultDevice();
      mtlCHECK(device != nil, "Failed to create Metal device");

      // Create command queue
      commandQueue = [device newCommandQueue];
      mtlCHECK(commandQueue != nil, "Failed to create Metal command queue");

      // Load the compiled shader library
#ifdef BASPACHO_METAL_LIBRARY_PATH
      NSError* error = nil;
      NSString* libraryPath = @BASPACHO_METAL_LIBRARY_PATH;
      NSURL* libraryURL = [NSURL fileURLWithPath:libraryPath];
      library = [device newLibraryWithURL:libraryURL error:&error];
      if (error != nil) {
        NSLog(@"Failed to load Metal library from %@: %@", libraryPath, error);
        // Try loading default library as fallback
        library = [device newDefaultLibrary];
      }
#else
      library = [device newDefaultLibrary];
#endif
      mtlCHECK(library != nil, "Failed to load Metal shader library");

      NSLog(@"Metal initialized: %@", device.name);
    }
  }

  ~MetalContextImpl() {
    @autoreleasepool {
      pipelineCache.clear();
      library = nil;
      commandQueue = nil;
      device = nil;
    }
  }

  id<MTLComputePipelineState> getPipelineState(const char* functionName) {
    std::string name(functionName);

    std::lock_guard<std::mutex> lock(pipelineMutex);

    auto it = pipelineCache.find(name);
    if (it != pipelineCache.end()) {
      return it->second;
    }

    @autoreleasepool {
      NSError* error = nil;
      NSString* nsName = [NSString stringWithUTF8String:functionName];
      id<MTLFunction> function = [library newFunctionWithName:nsName];
      mtlCHECK(function != nil, "Failed to find Metal function");

      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      mtlCHECK(pipeline != nil && error == nil, "Failed to create compute pipeline");

      pipelineCache[name] = pipeline;
      return pipeline;
    }
  }
};

// MetalContext singleton implementation
MetalContext& MetalContext::instance() {
  static MetalContext ctx;
  return ctx;
}

MetalContext::MetalContext() { impl = new MetalContextImpl(); }

MetalContext::~MetalContext() { delete impl; }

void* MetalContext::device() { return (__bridge void*)impl->device; }

void* MetalContext::commandQueue() { return (__bridge void*)impl->commandQueue; }

void* MetalContext::library() { return (__bridge void*)impl->library; }

void MetalContext::synchronize() {
  @autoreleasepool {
    id<MTLCommandBuffer> cmdBuf = [impl->commandQueue commandBuffer];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
  }
}

void* MetalContext::getPipelineState(const char* functionName) {
  return (__bridge void*)impl->getPipelineState(functionName);
}

// MetalMirror template implementation
template <typename T>
void MetalMirror<T>::clear() {
  @autoreleasepool {
    if (buffer_) {
      // Unregister from buffer registry
      MetalBufferRegistry::instance().unregisterBuffer(ptr_);
      // Release the buffer by setting to nil (ARC handles release)
      id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)buffer_;
      buf = nil;
      buffer_ = nullptr;
      ptr_ = nullptr;
      allocSize_ = 0;
    }
  }
}

template <typename T>
void MetalMirror<T>::resizeToAtLeast(size_t size) {
  if (allocSize_ < size) {
    clear();
  }
  if (!buffer_ && size > 0) {
    @autoreleasepool {
      id<MTLDevice> device = (__bridge id<MTLDevice>)MetalContext::instance().device();
      // Use shared storage mode for CPU/GPU access
      id<MTLBuffer> buf = [device newBufferWithLength:size * sizeof(T)
                                              options:MTLResourceStorageModeShared];
      CHECK_METAL_ALLOCATION(buf, size * sizeof(T));
      buffer_ = (__bridge_retained void*)buf;
      ptr_ = static_cast<T*>([buf contents]);
      allocSize_ = size;
      // Register with buffer registry for MPS operations
      MetalBufferRegistry::instance().registerBuffer(ptr_, buffer_, size * sizeof(T));
    }
  }
}

template <typename T>
void MetalMirror<T>::load(const std::vector<T>& vec) {
  if (vec.empty()) {
    return;
  }
  resizeToAtLeast(vec.size());
  // Copy data to shared buffer (directly accessible from CPU)
  memcpy(ptr_, vec.data(), vec.size() * sizeof(T));
}

template <typename T>
void MetalMirror<T>::get(std::vector<T>& vec) const {
  if (!ptr_ || vec.empty()) {
    return;
  }
  // Copy data from shared buffer (directly accessible from CPU)
  memcpy(vec.data(), ptr_, vec.size() * sizeof(T));
}

// MetalPtrMirror template implementation
template <typename T>
void MetalPtrMirror<T>::clear() {
  @autoreleasepool {
    if (buffer_) {
      id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)buffer_;
      buf = nil;
      buffer_ = nullptr;
      ptr_ = nullptr;
      allocSize_ = 0;
    }
  }
}

template <typename T>
void MetalPtrMirror<T>::load(const std::vector<T*>& vec, int64_t offset) {
  if (vec.empty()) {
    return;
  }

  if (allocSize_ < vec.size()) {
    clear();
  }

  @autoreleasepool {
    if (!buffer_) {
      id<MTLDevice> device = (__bridge id<MTLDevice>)MetalContext::instance().device();
      id<MTLBuffer> buf = [device newBufferWithLength:vec.size() * sizeof(T*)
                                              options:MTLResourceStorageModeShared];
      CHECK_METAL_ALLOCATION(buf, vec.size() * sizeof(T*));
      buffer_ = (__bridge_retained void*)buf;
      ptr_ = static_cast<T**>([buf contents]);
      allocSize_ = vec.size();
    }

    // Copy pointers with offset applied
    for (size_t i = 0; i < vec.size(); i++) {
      ptr_[i] = vec[i] + offset;
    }
  }
}

// Explicit template instantiations
template class MetalMirror<float>;
template class MetalMirror<double>;
template class MetalMirror<int64_t>;
template class MetalMirror<int32_t>;

template class MetalPtrMirror<float>;
template class MetalPtrMirror<double>;

// MetalBufferRegistry implementation
MetalBufferRegistry& MetalBufferRegistry::instance() {
  static MetalBufferRegistry registry;
  return registry;
}

void MetalBufferRegistry::registerBuffer(void* ptr, void* buffer, size_t sizeBytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  // Remove any existing entry for this pointer
  buffers_.erase(std::remove_if(buffers_.begin(), buffers_.end(),
                                [ptr](const BufferInfo& info) { return info.basePtr == ptr; }),
                 buffers_.end());
  // Add new entry
  buffers_.push_back({buffer, ptr, sizeBytes});
}

void MetalBufferRegistry::unregisterBuffer(void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);
  buffers_.erase(std::remove_if(buffers_.begin(), buffers_.end(),
                                [ptr](const BufferInfo& info) { return info.basePtr == ptr; }),
                 buffers_.end());
}

std::pair<void*, size_t> MetalBufferRegistry::findBuffer(const void* ptr) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const char* charPtr = static_cast<const char*>(ptr);

  for (const auto& info : buffers_) {
    const char* baseCharPtr = static_cast<const char*>(info.basePtr);
    if (charPtr >= baseCharPtr && charPtr < baseCharPtr + info.sizeBytes) {
      size_t offset = static_cast<size_t>(charPtr - baseCharPtr);
      return {info.buffer, offset};
    }
  }
  return {nullptr, 0};
}

}  // end namespace BaSpaCho
