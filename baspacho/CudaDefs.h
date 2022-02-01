#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusparse.h>

namespace BaSpaCho {

const char* cublasGetErrorEnum(cublasStatus_t error);
const char* cusparseGetErrorEnum(cusparseStatus_t error);
const char* cusolverGetErrorEnum(cusolverStatus_t error);

}  // end namespace BaSpaCho

#define cuCHECK(call)                                                     \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (cudaSuccess != err) {                                         \
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
            cudaDeviceReset();                                            \
            abort();                                                      \
        }                                                                 \
    } while (0)

#define cublasCHECK(call)                                    \
    do {                                                     \
        cublasStatus_t status = (call);                      \
        if (CUBLAS_STATUS_SUCCESS != status) {               \
            fprintf(stderr, "CUBLAS Error: %s\n",            \
                    ::BaSpaCho::cublasGetErrorEnum(status)); \
            cudaDeviceReset();                               \
            abort();                                         \
        }                                                    \
    } while (0)

#define cusparseCHECK(call)                                    \
    do {                                                       \
        cusparseStatus_t status = (call);                      \
        if (CUSPARSE_STATUS_SUCCESS != status) {               \
            fprintf(stderr, "CUSPARSE Error: %s\n",            \
                    ::BaSpaCho::cusparseGetErrorEnum(status)); \
            cudaDeviceReset();                                 \
            abort();                                           \
        }                                                      \
    } while (0)

#define cusolverCHECK(call)                                    \
    do {                                                       \
        cusolverStatus_t status = (call);                      \
        if (CUSOLVER_STATUS_SUCCESS != status) {               \
            fprintf(stderr, "CUSOLVER Error: %s\n",            \
                    ::BaSpaCho::cusolverGetErrorEnum(status)); \
            cudaDeviceReset();                                 \
            abort();                                           \
        }                                                      \
    } while (0)
