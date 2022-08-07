/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// BLAS/LAPACK famously go without headers.
extern "C" {

// TODO: detect in build if blas type is long or int
#if 0
#define BLAS_INT long
#else
#define BLAS_INT int
#endif

void dpotrf_(const char* uplo, BLAS_INT* n, double* A, const BLAS_INT* lda, BLAS_INT* info);

void dtrsm_(const char* side, const char* uplo, const char* transa, const char* diag,
            const BLAS_INT* m, const BLAS_INT* n, const double* alpha, const double* A,
            const BLAS_INT* lda, double* B, const BLAS_INT* ldb);

void dgemm_(const char* transa, const char* transb, const BLAS_INT* m, const BLAS_INT* n,
            const BLAS_INT* k, const double* alpha, const double* A, const BLAS_INT* lda,
            const double* B, const BLAS_INT* ldb, const double* beta, double* C,
            const BLAS_INT* ldc);

void dsyrk_(const char* uplo, const char* transa, const BLAS_INT* n, const BLAS_INT* k,
            const double* alpha, const double* A, const BLAS_INT* lda, const double* beta,
            double* C, const BLAS_INT* ldc);

void dsymm_(const char* uplo, const char* transa, const BLAS_INT* m, const BLAS_INT* n,
            const double* alpha, const double* A, const BLAS_INT* lda, const double* B,
            const BLAS_INT* ldb, const double* beta, double* C, const BLAS_INT* ldc);

void spotrf_(const char* uplo, BLAS_INT* n, float* A, const BLAS_INT* lda, BLAS_INT* info);

void strsm_(const char* side, const char* uplo, const char* transa, const char* diag,
            const BLAS_INT* m, const BLAS_INT* n, const float* alpha, const float* A,
            const BLAS_INT* lda, float* B, const BLAS_INT* ldb);

void sgemm_(const char* transa, const char* transb, const BLAS_INT* m, const BLAS_INT* n,
            const BLAS_INT* k, const float* alpha, const float* A, const BLAS_INT* lda,
            const float* B, const BLAS_INT* ldb, const float* beta, float* C, const BLAS_INT* ldc);

void ssyrk_(const char* uplo, const char* transa, const BLAS_INT* n, const BLAS_INT* k,
            const float* alpha, const float* A, const BLAS_INT* lda, const float* beta, float* C,
            const BLAS_INT* ldc);

void ssymm_(const char* uplo, const char* transa, const BLAS_INT* m, const BLAS_INT* n,
            const float* alpha, const float* A, const BLAS_INT* lda, const float* B,
            const BLAS_INT* ldb, const float* beta, float* C, const BLAS_INT* ldc);
}

#define CBLAS_LAYOUT int
#define CBLAS_TRANSPOSE char
#define CBLAS_SIDE char
#define CBLAS_UPLO char
#define CBLAS_DIAG char

#define CblasColMajor 0
// #define CblasRowMajor ... (not supported)

#define CblasLeft 'L'
#define CblasUpper 'U'
#define CblasConjTrans 'C'
#define CblasNoTrans 'N'
#define CblasNonUnit 'N'

#define LAPACK_COL_MAJOR 0
// #define LAPACK_ROW_MAJOR ... (not supported)

namespace BaSpaCho {

inline void cblas_dgemm(const CBLAS_LAYOUT /* Layout */, const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const BLAS_INT M, const BLAS_INT N,
                        const BLAS_INT K, const double alpha, const double* A, const BLAS_INT lda,
                        const double* B, const BLAS_INT ldb, const double beta, double* C,
                        const BLAS_INT ldc) {
  dgemm_(&TransA, &TransB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

inline void cblas_dtrsm(const CBLAS_LAYOUT /* Layout */, const CBLAS_SIDE Side,
                        const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                        const BLAS_INT M, const BLAS_INT N, const double alpha, const double* A,
                        const BLAS_INT lda, double* B, const BLAS_INT ldb) {
  dtrsm_(&Side, &Uplo, &TransA, &Diag, &M, &N, &alpha, A, &lda, B, &ldb);
}

inline void cblas_dsyrk(const CBLAS_LAYOUT /* Layout */, const CBLAS_UPLO Uplo,
                        const CBLAS_TRANSPOSE Trans, const BLAS_INT N, const BLAS_INT K,
                        const double alpha, const double* A, const BLAS_INT lda, const double beta,
                        double* C, const BLAS_INT ldc) {
  dsyrk_(&Uplo, &Trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc);
}

inline void cblas_dsymm(const CBLAS_LAYOUT /* Layout */, const CBLAS_SIDE side,
                        const CBLAS_UPLO uplo, const BLAS_INT m, const BLAS_INT n,
                        const double alpha, const double* a, const BLAS_INT lda, const double* b,
                        const BLAS_INT ldb, const double beta, double* c, const BLAS_INT ldc) {
  dsymm_(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

BLAS_INT LAPACKE_dpotrf(int /* matrix_layout */, char uplo, BLAS_INT n, double* a, BLAS_INT lda) {
  BLAS_INT info;
  dpotrf_(&uplo, &n, a, &lda, &info);
  return info;
}

inline void cblas_sgemm(const CBLAS_LAYOUT /* Layout */, const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const BLAS_INT M, const BLAS_INT N,
                        const BLAS_INT K, const float alpha, const float* A, const BLAS_INT lda,
                        const float* B, const BLAS_INT ldb, const float beta, float* C,
                        const BLAS_INT ldc) {
  sgemm_(&TransA, &TransB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

inline void cblas_strsm(const CBLAS_LAYOUT /* Layout */, const CBLAS_SIDE Side,
                        const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                        const BLAS_INT M, const BLAS_INT N, const float alpha, const float* A,
                        const BLAS_INT lda, float* B, const BLAS_INT ldb) {
  strsm_(&Side, &Uplo, &TransA, &Diag, &M, &N, &alpha, A, &lda, B, &ldb);
}

inline void cblas_ssyrk(const CBLAS_LAYOUT /* Layout */, const CBLAS_UPLO Uplo,
                        const CBLAS_TRANSPOSE Trans, const BLAS_INT N, const BLAS_INT K,
                        const float alpha, const float* A, const BLAS_INT lda, const float beta,
                        float* C, const BLAS_INT ldc) {
  ssyrk_(&Uplo, &Trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc);
}

inline void cblas_ssymm(const CBLAS_LAYOUT /* Layout */, const CBLAS_SIDE side,
                        const CBLAS_UPLO uplo, const BLAS_INT m, const BLAS_INT n,
                        const float alpha, const float* a, const BLAS_INT lda, const float* b,
                        const BLAS_INT ldb, const float beta, float* c, const BLAS_INT ldc) {
  ssymm_(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

BLAS_INT LAPACKE_spotrf(int /* matrix_layout */, char uplo, BLAS_INT n, float* a, BLAS_INT lda) {
  BLAS_INT info;
  spotrf_(&uplo, &n, a, &lda, &info);
  return info;
}

}  // end namespace BaSpaCho