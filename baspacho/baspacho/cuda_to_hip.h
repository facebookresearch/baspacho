/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// CUDA-to-HIP compatibility header for BaSpaCho
// On ROCm builds (USE_HIP defined), this aliases CUDA API calls to their HIP
// equivalents. On CUDA builds, this is a no-op include of the CUDA headers.

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <hipsparse/hipsparse.h>

// Runtime API
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceReset hipDeviceReset
#define cudaGetErrorString hipGetErrorString
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess

// cuBLAS -> hipBLAS
#define cublasHandle_t hipblasHandle_t
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasStatus_t hipblasStatus_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED
#define CUBLAS_STATUS_LICENSE_ERROR HIPBLAS_STATUS_NOT_SUPPORTED

// cuBLAS enums
#define CUBLAS_SIDE_LEFT HIPBLAS_SIDE_LEFT
#define CUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#define CUBLAS_OP_C HIPBLAS_OP_C
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT

// cuBLAS BLAS routines
#define cublasDgemm hipblasDgemm
#define cublasSgemm hipblasSgemm
#define cublasDgemmBatched hipblasDgemmBatched
#define cublasSgemmBatched hipblasSgemmBatched
#define cublasDtrsm hipblasDtrsm
#define cublasStrsm hipblasStrsm
#define cublasDtrsmBatched hipblasDtrsmBatched
#define cublasStrsmBatched hipblasStrsmBatched
#define cublasDsymm hipblasDsymm
#define cublasSsymm hipblasSsymm

// cuSolver -> hipSolver
#define cusolverDnHandle_t hipsolverDnHandle_t
#define cusolverDnCreate hipsolverDnCreate
#define cusolverDnDestroy hipsolverDnDestroy
#define cusolverDnSetStream hipsolverDnSetStream
#define cusolverStatus_t hipsolverStatus_t
#define CUSOLVER_STATUS_SUCCESS HIPSOLVER_STATUS_SUCCESS
#define CUSOLVER_STATUS_NOT_INITIALIZED HIPSOLVER_STATUS_NOT_INITIALIZED
#define CUSOLVER_STATUS_ALLOC_FAILED HIPSOLVER_STATUS_ALLOC_FAILED
#define CUSOLVER_STATUS_INVALID_VALUE HIPSOLVER_STATUS_INVALID_VALUE
#define CUSOLVER_STATUS_ARCH_MISMATCH HIPSOLVER_STATUS_ARCH_MISMATCH
#define CUSOLVER_STATUS_MAPPING_ERROR HIPSOLVER_STATUS_MAPPING_ERROR
#define CUSOLVER_STATUS_EXECUTION_FAILED HIPSOLVER_STATUS_EXECUTION_FAILED
#define CUSOLVER_STATUS_INTERNAL_ERROR HIPSOLVER_STATUS_INTERNAL_ERROR
#define CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED
#define CUSOLVER_STATUS_NOT_SUPPORTED HIPSOLVER_STATUS_NOT_SUPPORTED
#define CUSOLVER_STATUS_ZERO_PIVOT HIPSOLVER_STATUS_ZERO_PIVOT
// hipSOLVER does not have INVALID_LICENSE; use UNKNOWN as fallback
#define CUSOLVER_STATUS_INVALID_LICENSE HIPSOLVER_STATUS_UNKNOWN

// cuSolver potrf routines
#define cusolverDnDpotrf_bufferSize hipsolverDnDpotrf_bufferSize
#define cusolverDnSpotrf_bufferSize hipsolverDnSpotrf_bufferSize
#define cusolverDnDpotrf hipsolverDnDpotrf
#define cusolverDnSpotrf hipsolverDnSpotrf
#define cusolverDnDpotrfBatched hipsolverDnDpotrfBatched
#define cusolverDnSpotrfBatched hipsolverDnSpotrfBatched

// cuSPARSE -> hipSPARSE (only error enums used)
#define cusparseStatus_t hipsparseStatus_t
#define CUSPARSE_STATUS_SUCCESS HIPSPARSE_STATUS_SUCCESS
#define CUSPARSE_STATUS_NOT_INITIALIZED HIPSPARSE_STATUS_NOT_INITIALIZED
#define CUSPARSE_STATUS_ALLOC_FAILED HIPSPARSE_STATUS_ALLOC_FAILED
#define CUSPARSE_STATUS_INVALID_VALUE HIPSPARSE_STATUS_INVALID_VALUE
#define CUSPARSE_STATUS_ARCH_MISMATCH HIPSPARSE_STATUS_ARCH_MISMATCH
#define CUSPARSE_STATUS_MAPPING_ERROR HIPSPARSE_STATUS_MAPPING_ERROR
#define CUSPARSE_STATUS_EXECUTION_FAILED HIPSPARSE_STATUS_EXECUTION_FAILED
#define CUSPARSE_STATUS_INTERNAL_ERROR HIPSPARSE_STATUS_INTERNAL_ERROR
#define CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
#define CUSPARSE_STATUS_ZERO_PIVOT HIPSPARSE_STATUS_ZERO_PIVOT
#define CUSPARSE_STATUS_NOT_SUPPORTED HIPSPARSE_STATUS_NOT_SUPPORTED
#define CUSPARSE_STATUS_INSUFFICIENT_RESOURCES HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES

#else

// CUDA build: include the standard CUDA headers
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusparse.h>

#endif
