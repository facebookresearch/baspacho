/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef BASPACHO_USE_OPENCL

#include "baspacho/baspacho/OpenCLDefs.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

namespace BaSpaCho {

// Read kernel source from file
static std::string readKernelSource(const char* path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "OpenCL: Failed to open kernel file: " << path << std::endl;
    abort();
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

OpenCLContext& OpenCLContext::instance() {
  static OpenCLContext ctx;
  return ctx;
}

OpenCLContext::OpenCLContext() {
  cl_int err;

  // Get platform
  cl_uint numPlatforms;
  clCHECK(clGetPlatformIDs(0, nullptr, &numPlatforms));
  if (numPlatforms == 0) {
    std::cerr << "OpenCL: No platforms found" << std::endl;
    abort();
  }

  std::vector<cl_platform_id> platforms(numPlatforms);
  clCHECK(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr));
  platform_ = platforms[0];  // Use first platform

  // Get device (prefer GPU)
  cl_uint numDevices;
  err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
  if (err != CL_SUCCESS || numDevices == 0) {
    // Fall back to any device
    clCHECK(clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices));
  }

  if (numDevices == 0) {
    std::cerr << "OpenCL: No devices found" << std::endl;
    abort();
  }

  std::vector<cl_device_id> devices(numDevices);
  err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
  if (err != CL_SUCCESS) {
    clCHECK(clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr));
  }
  device_ = devices[0];

  // Print device info
  char deviceName[256];
  clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
  std::cout << "OpenCL: Using device: " << deviceName << std::endl;

  // Create context
  context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
  clCHECK(err);

  // Create command queue
  queue_ = clCreateCommandQueue(context_, device_, 0, &err);
  clCHECK(err);

  // Build program from kernel source
  buildProgram();
}

void OpenCLContext::buildProgram() {
#ifdef BASPACHO_OPENCL_KERNEL_PATH
  std::string source = readKernelSource(BASPACHO_OPENCL_KERNEL_PATH);
#else
  std::cerr << "OpenCL: Kernel path not defined" << std::endl;
  abort();
#endif

  cl_int err;
  const char* sources[] = {source.c_str()};
  size_t lengths[] = {source.size()};

  program_ = clCreateProgramWithSource(context_, 1, sources, lengths, &err);
  clCHECK(err);

  err = clBuildProgram(program_, 1, &device_, "-cl-std=CL1.2", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    // Get build log
    size_t logSize;
    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    std::vector<char> log(logSize);
    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
    std::cerr << "OpenCL: Build failed:\n" << log.data() << std::endl;
    abort();
  }
}

OpenCLContext::~OpenCLContext() {
  // Release all cached kernels
  for (auto& kv : kernelCache_) {
    if (kv.second) {
      clReleaseKernel(kv.second);
    }
  }
  kernelCache_.clear();

  if (program_) clReleaseProgram(program_);
  if (queue_) clReleaseCommandQueue(queue_);
  if (context_) clReleaseContext(context_);
}

void OpenCLContext::synchronize() {
  clCHECK(clFinish(queue_));
}

cl_kernel OpenCLContext::getKernel(const char* kernelName) {
  std::lock_guard<std::mutex> lock(kernelMutex_);

  // Check cache first
  auto it = kernelCache_.find(kernelName);
  if (it != kernelCache_.end()) {
    return it->second;
  }

  // Create kernel and cache it
  cl_int err;
  cl_kernel kernel = clCreateKernel(program_, kernelName, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "OpenCL: Failed to create kernel: " << kernelName << std::endl;
    abort();
  }

  kernelCache_[kernelName] = kernel;
  return kernel;
}

bool OpenCLContext::hasKernel(const char* kernelName) {
  std::lock_guard<std::mutex> lock(kernelMutex_);

  // Check cache first
  auto it = kernelCache_.find(kernelName);
  if (it != kernelCache_.end()) {
    return true;
  }

  // Try to create kernel
  cl_int err;
  cl_kernel kernel = clCreateKernel(program_, kernelName, &err);
  if (err != CL_SUCCESS) {
    return false;
  }

  // Cache it for future use
  kernelCache_[kernelName] = kernel;
  return true;
}

// OpenCLBufferRegistry implementation
OpenCLBufferRegistry& OpenCLBufferRegistry::instance() {
  static OpenCLBufferRegistry registry;
  return registry;
}

void OpenCLBufferRegistry::registerBuffer(void* hostPtr, cl_mem buffer, size_t sizeBytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  // Check if already registered
  for (auto& info : buffers_) {
    if (info.hostPtr == hostPtr) {
      info.buffer = buffer;
      info.sizeBytes = sizeBytes;
      return;
    }
  }
  buffers_.push_back({buffer, hostPtr, sizeBytes});
}

void OpenCLBufferRegistry::unregisterBuffer(void* hostPtr) {
  std::lock_guard<std::mutex> lock(mutex_);
  buffers_.erase(
      std::remove_if(buffers_.begin(), buffers_.end(),
                     [hostPtr](const BufferInfo& info) { return info.hostPtr == hostPtr; }),
      buffers_.end());
}

std::pair<cl_mem, size_t> OpenCLBufferRegistry::findBuffer(const void* ptr) const {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& info : buffers_) {
    const char* base = static_cast<const char*>(info.hostPtr);
    const char* target = static_cast<const char*>(ptr);
    if (target >= base && target < base + info.sizeBytes) {
      size_t byteOffset = target - base;
      return {info.buffer, byteOffset};
    }
  }
  return {nullptr, 0};
}

// OpenCLMirror implementation
template <typename T>
void OpenCLMirror<T>::clear() {
  if (buffer_) {
    // Unregister from buffer registry before releasing
    if (hostPtr_) {
      OpenCLBufferRegistry::instance().unregisterBuffer(hostPtr_);
    }
    clReleaseMemObject(buffer_);
    buffer_ = nullptr;
    hostPtr_ = nullptr;
  }
  allocSize_ = 0;
}

template <typename T>
void OpenCLMirror<T>::resizeToAtLeast(size_t size) {
  if (size <= allocSize_) {
    return;
  }

  clear();

  cl_int err;
  size_t sizeBytes = size * sizeof(T);
  buffer_ = clCreateBuffer(OpenCLContext::instance().context(),
                           CL_MEM_READ_WRITE, sizeBytes, nullptr, &err);
  clCHECK(err);
  CHECK_OPENCL_ALLOCATION(buffer_, sizeBytes);

  allocSize_ = size;
}

template <typename T>
void OpenCLMirror<T>::load(const std::vector<T>& vec) {
  resizeToAtLeast(vec.size());

  if (vec.empty()) return;

  // Store the host pointer and register with buffer registry
  hostPtr_ = const_cast<T*>(vec.data());
  OpenCLBufferRegistry::instance().registerBuffer(hostPtr_, buffer_, vec.size() * sizeof(T));

  clCHECK(clEnqueueWriteBuffer(OpenCLContext::instance().queue(),
                               buffer_, CL_TRUE, 0,
                               vec.size() * sizeof(T), vec.data(),
                               0, nullptr, nullptr));
}

template <typename T>
void OpenCLMirror<T>::get(std::vector<T>& vec) const {
  if (vec.empty() || !buffer_) return;

  clCHECK(clEnqueueReadBuffer(OpenCLContext::instance().queue(),
                              buffer_, CL_TRUE, 0,
                              vec.size() * sizeof(T), vec.data(),
                              0, nullptr, nullptr));
}

// OpenCLPtrMirror implementation
template <typename T>
void OpenCLPtrMirror<T>::clear() {
  if (buffer_) {
    clReleaseMemObject(buffer_);
    buffer_ = nullptr;
  }
  allocSize_ = 0;
}

template <typename T>
void OpenCLPtrMirror<T>::load(const std::vector<T*>& vec, int64_t offset) {
  if (vec.size() > allocSize_) {
    clear();

    cl_int err;
    size_t sizeBytes = vec.size() * sizeof(cl_ulong);  // Use 64-bit for pointers
    buffer_ = clCreateBuffer(OpenCLContext::instance().context(),
                             CL_MEM_READ_WRITE, sizeBytes, nullptr, &err);
    clCHECK(err);
    CHECK_OPENCL_ALLOCATION(buffer_, sizeBytes);
    allocSize_ = vec.size();
  }

  // Convert pointers to offsets (GPU-relative addressing)
  std::vector<cl_ulong> offsets(vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    offsets[i] = reinterpret_cast<cl_ulong>(vec[i]) + offset * sizeof(T);
  }

  clCHECK(clEnqueueWriteBuffer(OpenCLContext::instance().queue(),
                               buffer_, CL_TRUE, 0,
                               offsets.size() * sizeof(cl_ulong), offsets.data(),
                               0, nullptr, nullptr));
}

// Explicit template instantiations
template class OpenCLMirror<float>;
template class OpenCLMirror<double>;
template class OpenCLMirror<int64_t>;
template class OpenCLMirror<int32_t>;

template class OpenCLPtrMirror<float>;
template class OpenCLPtrMirror<double>;

}  // end namespace BaSpaCho

#endif  // BASPACHO_USE_OPENCL
