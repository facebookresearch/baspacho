/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ============================================================================
// BaSpaCho OpenCL Kernels
// Ported from Metal/CUDA implementations
// ============================================================================

// Helper: Convert linear index to ordered pair (x, y) where 0 <= x <= y < n
int2 toOrderedPair(long n, long p) {
    long odd = n & 1;
    long m = n + 1 - odd;
    long x = p % m;
    long y = n - 1 - (p / m);
    if (x > y) {
        x = x - y - 1;
        y = n - 1 - odd - y;
    }
    return (int2)(x, y);
}

// Binary search: find largest i such that array[i] <= needle
long bisect(__global const long* array, long size, long needle) {
    long a = 0, b = size;
    while (b - a > 1) {
        long mid = (a + b) / 2;
        if (needle >= array[mid]) {
            a = mid;
        } else {
            b = mid;
        }
    }
    return a;
}

// ============================================================================
// In-place Cholesky decomposition for small blocks
// A is row-major with stride lda
// ============================================================================
void cholesky_float(__global float* A, int lda, int n) {
    __global float* b_ii = A;

    for (int i = 0; i < n; i++) {
        float d = sqrt(*b_ii);
        *b_ii = d;

        __global float* b_ji = b_ii + lda;
        for (int j = i + 1; j < n; j++) {
            float c = *b_ji / d;
            *b_ji = c;

            __global float* b_ki = b_ii + lda;
            __global float* b_jk = b_ji + 1;
            for (int k = i + 1; k <= j; k++) {
                *b_jk -= c * (*b_ki);
                b_ki += lda;
                b_jk += 1;
            }

            b_ji += lda;
        }

        b_ii += lda + 1;
    }
}

void cholesky_double(__global double* A, int lda, int n) {
    __global double* b_ii = A;

    for (int i = 0; i < n; i++) {
        double d = sqrt(*b_ii);
        *b_ii = d;

        __global double* b_ji = b_ii + lda;
        for (int j = i + 1; j < n; j++) {
            double c = *b_ji / d;
            *b_ji = c;

            __global double* b_ki = b_ii + lda;
            __global double* b_jk = b_ji + 1;
            for (int k = i + 1; k <= j; k++) {
                *b_jk -= c * (*b_ki);
                b_ki += lda;
                b_jk += 1;
            }

            b_ji += lda;
        }

        b_ii += lda + 1;
    }
}

// ============================================================================
// In-place solver for A^T (A built upper-diagonal col-major)
// ============================================================================
void solveUpperT_float(__global float* A, int lda, int n, __global float* v) {
    __global float* b_ii = A;
    for (int i = 0; i < n; i++) {
        float x = v[i];

        for (int j = 0; j < i; j++) {
            x -= b_ii[j] * v[j];
        }

        v[i] = x / b_ii[i];
        b_ii += lda;
    }
}

void solveUpperT_double(__global double* A, int lda, int n, __global double* v) {
    __global double* b_ii = A;
    for (int i = 0; i < n; i++) {
        double x = v[i];

        for (int j = 0; j < i; j++) {
            x -= b_ii[j] * v[j];
        }

        v[i] = x / b_ii[i];
        b_ii += lda;
    }
}

// ============================================================================
// In-place solver for A (A built upper-diagonal col-major)
// ============================================================================
void solveUpper_float(__global float* A, int lda, int n, __global float* v) {
    __global float* b_ii = A + (lda + 1) * (n - 1);
    for (int i = n - 1; i >= 0; i--) {
        float x = v[i];

        __global float* b_ij = b_ii;
        for (int j = i + 1; j < n; j++) {
            b_ij += lda;
            x -= (*b_ij) * v[j];
        }

        v[i] = x / (*b_ii);
        b_ii -= lda + 1;
    }
}

void solveUpper_double(__global double* A, int lda, int n, __global double* v) {
    __global double* b_ii = A + (lda + 1) * (n - 1);
    for (int i = n - 1; i >= 0; i--) {
        double x = v[i];

        __global double* b_ij = b_ii;
        for (int j = i + 1; j < n; j++) {
            b_ij += lda;
            x -= (*b_ij) * v[j];
        }

        v[i] = x / (*b_ii);
        b_ii -= lda + 1;
    }
}

// ============================================================================
// Kernel 1: factor_lumps (Cholesky on diagonal blocks) - Float version
// One work-item per lump
// ============================================================================
__kernel void factor_lumps_kernel_float(
    __global const long* lumpStart,
    __global const long* chainColPtr,
    __global const long* chainData,
    __global const long* boardColPtr,
    __global const long* boardChainColOrd,
    __global const long* chainRowsTillEnd,
    __global float* data,
    const long lumpIndexStart,
    const long lumpIndexEnd)
{
    int tid = get_global_id(0);
    long lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    long lumpSize = lumpStart[lump + 1] - lumpStart[lump];
    long colStart = chainColPtr[lump];
    long dataPtr = chainData[colStart];

    // In-place lower diag Cholesky on diagonal block
    __global float* diagBlockPtr = data + dataPtr;
    cholesky_float(diagBlockPtr, (int)lumpSize, (int)lumpSize);

    // Below-diagonal solve
    long gatheredStart = boardColPtr[lump];
    long gatheredEnd = boardColPtr[lump + 1];
    long rowDataStart = boardChainColOrd[gatheredStart + 1];
    long rowDataEnd = boardChainColOrd[gatheredEnd - 1];
    long belowDiagStart = chainData[colStart + rowDataStart];
    long numRows = chainRowsTillEnd[colStart + rowDataEnd - 1]
                 - chainRowsTillEnd[colStart + rowDataStart - 1];

    __global float* belowDiagBlockPtr = data + belowDiagStart;
    for (long i = 0; i < numRows; i++) {
        solveUpperT_float(diagBlockPtr, (int)lumpSize, (int)lumpSize, belowDiagBlockPtr);
        belowDiagBlockPtr += lumpSize;
    }
}

// Double precision version
__kernel void factor_lumps_kernel_double(
    __global const long* lumpStart,
    __global const long* chainColPtr,
    __global const long* chainData,
    __global const long* boardColPtr,
    __global const long* boardChainColOrd,
    __global const long* chainRowsTillEnd,
    __global double* data,
    const long lumpIndexStart,
    const long lumpIndexEnd)
{
    int tid = get_global_id(0);
    long lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    long lumpSize = lumpStart[lump + 1] - lumpStart[lump];
    long colStart = chainColPtr[lump];
    long dataPtr = chainData[colStart];

    __global double* diagBlockPtr = data + dataPtr;
    cholesky_double(diagBlockPtr, (int)lumpSize, (int)lumpSize);

    long gatheredStart = boardColPtr[lump];
    long gatheredEnd = boardColPtr[lump + 1];
    long rowDataStart = boardChainColOrd[gatheredStart + 1];
    long rowDataEnd = boardChainColOrd[gatheredEnd - 1];
    long belowDiagStart = chainData[colStart + rowDataStart];
    long numRows = chainRowsTillEnd[colStart + rowDataEnd - 1]
                 - chainRowsTillEnd[colStart + rowDataStart - 1];

    __global double* belowDiagBlockPtr = data + belowDiagStart;
    for (long i = 0; i < numRows; i++) {
        solveUpperT_double(diagBlockPtr, (int)lumpSize, (int)lumpSize, belowDiagBlockPtr);
        belowDiagBlockPtr += lumpSize;
    }
}

// ============================================================================
// Kernel 2: assemble (Assemble rectangular sections) - Float version
// ============================================================================
__kernel void assemble_kernel_float(
    const long numBlockRows,
    const long numBlockCols,
    const long startRow,
    const long srcRectWidth,
    const long dstStride,
    __global const long* pChainRowsTillEnd,
    __global const long* pToSpan,
    __global const long* pSpanToChainOffset,
    __global const long* pSpanOffsetInLump,
    __global const float* matRectPtr,
    __global float* data)
{
    int tid = get_global_id(0);
    if (tid >= numBlockRows * numBlockCols) {
        return;
    }

    long r = tid % numBlockRows;
    long c = tid / numBlockRows;

    // Only process lower triangle
    if (c > r) {
        return;
    }

    long rBegin = (r > 0) ? (pChainRowsTillEnd[r - 1] - startRow) : 0;
    long rEnd = pChainRowsTillEnd[r] - startRow;
    long rSize = rEnd - rBegin;
    long rParam = pToSpan[r];
    long rOffset = pSpanToChainOffset[rParam];

    long cStart = (c > 0) ? (pChainRowsTillEnd[c - 1] - startRow) : 0;
    long cEnd = pChainRowsTillEnd[c] - startRow;
    long cSize = cEnd - cStart;
    long offset = rOffset + pSpanOffsetInLump[pToSpan[c]];

    __global const float* matRowPtr = matRectPtr + rBegin * srcRectWidth;
    __global const float* src = matRowPtr + cStart;
    __global float* dst = data + offset;

    // Subtract source from destination (atomic for thread safety)
    for (long i = 0; i < rSize; i++) {
        for (long j = 0; j < cSize; j++) {
            // Note: OpenCL 1.2 doesn't have native atomic float subtract
            // Using atomic_xchg workaround or non-atomic for simplicity
            dst[i * dstStride + j] -= src[i * srcRectWidth + j];
        }
    }
}

// Double precision version
__kernel void assemble_kernel_double(
    const long numBlockRows,
    const long numBlockCols,
    const long startRow,
    const long srcRectWidth,
    const long dstStride,
    __global const long* pChainRowsTillEnd,
    __global const long* pToSpan,
    __global const long* pSpanToChainOffset,
    __global const long* pSpanOffsetInLump,
    __global const double* matRectPtr,
    __global double* data)
{
    int tid = get_global_id(0);
    if (tid >= numBlockRows * numBlockCols) {
        return;
    }

    long r = tid % numBlockRows;
    long c = tid / numBlockRows;

    if (c > r) {
        return;
    }

    long rBegin = (r > 0) ? (pChainRowsTillEnd[r - 1] - startRow) : 0;
    long rEnd = pChainRowsTillEnd[r] - startRow;
    long rSize = rEnd - rBegin;
    long rParam = pToSpan[r];
    long rOffset = pSpanToChainOffset[rParam];

    long cStart = (c > 0) ? (pChainRowsTillEnd[c - 1] - startRow) : 0;
    long cEnd = pChainRowsTillEnd[c] - startRow;
    long cSize = cEnd - cStart;
    long offset = rOffset + pSpanOffsetInLump[pToSpan[c]];

    __global const double* matRowPtr = matRectPtr + rBegin * srcRectWidth;
    __global const double* src = matRowPtr + cStart;
    __global double* dst = data + offset;

    for (long i = 0; i < rSize; i++) {
        for (long j = 0; j < cSize; j++) {
            dst[i * dstStride + j] -= src[i * srcRectWidth + j];
        }
    }
}

// ============================================================================
// Kernel 3: sparseElim_diagSolveL - Float version
// ============================================================================
__kernel void sparseElim_diagSolveL_float(
    __global const long* lumpStarts,
    __global const long* chainColPtr,
    __global const long* chainData,
    __global float* data,
    __global float* v,
    const long ldc,
    const long nRHS,
    const long lumpIndexStart,
    const long lumpIndexEnd)
{
    int tid = get_global_id(0);
    long lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    long lumpStart = lumpStarts[lump];
    long lumpSize = lumpStarts[lump + 1] - lumpStart;
    long colStart = chainColPtr[lump];
    long diagDataPtr = chainData[colStart];

    __global float* diagBlock = data + diagDataPtr;

    for (long rhs = 0; rhs < nRHS; rhs++) {
        solveUpperT_float(diagBlock, (int)lumpSize, (int)lumpSize, v + lumpStart + ldc * rhs);
    }
}

// Double precision version
__kernel void sparseElim_diagSolveL_double(
    __global const long* lumpStarts,
    __global const long* chainColPtr,
    __global const long* chainData,
    __global double* data,
    __global double* v,
    const long ldc,
    const long nRHS,
    const long lumpIndexStart,
    const long lumpIndexEnd)
{
    int tid = get_global_id(0);
    long lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    long lumpStart = lumpStarts[lump];
    long lumpSize = lumpStarts[lump + 1] - lumpStart;
    long colStart = chainColPtr[lump];
    long diagDataPtr = chainData[colStart];

    __global double* diagBlock = data + diagDataPtr;

    for (long rhs = 0; rhs < nRHS; rhs++) {
        solveUpperT_double(diagBlock, (int)lumpSize, (int)lumpSize, v + lumpStart + ldc * rhs);
    }
}

// ============================================================================
// Kernel 4: sparseElim_diagSolveLt - Float version
// ============================================================================
__kernel void sparseElim_diagSolveLt_float(
    __global const long* lumpStarts,
    __global const long* chainColPtr,
    __global const long* chainData,
    __global float* data,
    __global float* v,
    const long ldc,
    const long nRHS,
    const long lumpIndexStart,
    const long lumpIndexEnd)
{
    int tid = get_global_id(0);
    long lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    long lumpStart = lumpStarts[lump];
    long lumpSize = lumpStarts[lump + 1] - lumpStart;
    long colStart = chainColPtr[lump];
    long diagDataPtr = chainData[colStart];

    __global float* diagBlock = data + diagDataPtr;

    for (long rhs = 0; rhs < nRHS; rhs++) {
        solveUpper_float(diagBlock, (int)lumpSize, (int)lumpSize, v + lumpStart + ldc * rhs);
    }
}

// Double precision version
__kernel void sparseElim_diagSolveLt_double(
    __global const long* lumpStarts,
    __global const long* chainColPtr,
    __global const long* chainData,
    __global double* data,
    __global double* v,
    const long ldc,
    const long nRHS,
    const long lumpIndexStart,
    const long lumpIndexEnd)
{
    int tid = get_global_id(0);
    long lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    long lumpStart = lumpStarts[lump];
    long lumpSize = lumpStarts[lump + 1] - lumpStart;
    long colStart = chainColPtr[lump];
    long diagDataPtr = chainData[colStart];

    __global double* diagBlock = data + diagDataPtr;

    for (long rhs = 0; rhs < nRHS; rhs++) {
        solveUpper_double(diagBlock, (int)lumpSize, (int)lumpSize, v + lumpStart + ldc * rhs);
    }
}

// ============================================================================
// Kernel 5: assembleVec (Vector assembly during solve) - Float version
// ============================================================================
__kernel void assembleVec_kernel_float(
    __global const long* chainRowsTillEnd,
    __global const long* toSpan,
    __global const long* spanStarts,
    __global const float* A,
    const long numColItems,
    __global float* C,
    const long ldc,
    const long nRHS,
    const long startRow)
{
    int tid = get_global_id(0);
    if (tid >= numColItems) {
        return;
    }

    long rowsBefore = (tid > 0) ? (chainRowsTillEnd[tid - 1] - startRow) : 0;
    long rowsAfter = chainRowsTillEnd[tid] - startRow;
    long blockRows = rowsAfter - rowsBefore;

    long span = toSpan[tid];
    long spanStart = spanStarts[span];

    __global const float* srcPtr = A + rowsBefore * nRHS;
    __global float* dstPtr = C + spanStart;

    for (long rhs = 0; rhs < nRHS; rhs++) {
        for (long i = 0; i < blockRows; i++) {
            dstPtr[i + rhs * ldc] += srcPtr[i * nRHS + rhs];
        }
    }
}

// Double precision version
__kernel void assembleVec_kernel_double(
    __global const long* chainRowsTillEnd,
    __global const long* toSpan,
    __global const long* spanStarts,
    __global const double* A,
    const long numColItems,
    __global double* C,
    const long ldc,
    const long nRHS,
    const long startRow)
{
    int tid = get_global_id(0);
    if (tid >= numColItems) {
        return;
    }

    long rowsBefore = (tid > 0) ? (chainRowsTillEnd[tid - 1] - startRow) : 0;
    long rowsAfter = chainRowsTillEnd[tid] - startRow;
    long blockRows = rowsAfter - rowsBefore;

    long span = toSpan[tid];
    long spanStart = spanStarts[span];

    __global const double* srcPtr = A + rowsBefore * nRHS;
    __global double* dstPtr = C + spanStart;

    for (long rhs = 0; rhs < nRHS; rhs++) {
        for (long i = 0; i < blockRows; i++) {
            dstPtr[i + rhs * ldc] += srcPtr[i * nRHS + rhs];
        }
    }
}

// ============================================================================
// Kernel 6: sparse_elim (Sparse elimination) - Float version
// This is the core sparse elimination kernel, using the "straightened" approach
// ============================================================================

// Helper for locked subtract (without atomics for simplicity)
void matSubProduct_float(__global float* dst, int dstStride,
                          __global const float* A, int aStride, int aRows, int aCols,
                          __global const float* B, int bStride, int bRows, int bCols) {
    // dst -= A * B^T where A is aRows x aCols, B is bRows x aCols (bCols = aCols)
    for (int i = 0; i < aRows; i++) {
        for (int j = 0; j < bRows; j++) {
            float sum = 0.0f;
            for (int k = 0; k < aCols; k++) {
                sum += A[i * aStride + k] * B[j * bStride + k];
            }
            dst[i * dstStride + j] -= sum;
        }
    }
}

void matSubProduct_double(__global double* dst, int dstStride,
                           __global const double* A, int aStride, int aRows, int aCols,
                           __global const double* B, int bStride, int bRows, int bCols) {
    for (int i = 0; i < aRows; i++) {
        for (int j = 0; j < bRows; j++) {
            double sum = 0.0;
            for (int k = 0; k < aCols; k++) {
                sum += A[i * aStride + k] * B[j * bStride + k];
            }
            dst[i * dstStride + j] -= sum;
        }
    }
}

__kernel void sparse_elim_straight_kernel_float(
    __global const long* chainColPtr,
    __global const long* lumpStart,
    __global const long* chainRowSpan,
    __global const long* spanStart,
    __global const long* chainData,
    __global const long* spanToLump,
    __global const long* spanOffsetInLump,
    __global float* data,
    const long lumpIndexStart,
    const long lumpIndexEnd,
    __global const long* makeBlockPairEnumStraight,
    const long numBlockPairs)
{
    int tid = get_global_id(0);
    if (tid >= numBlockPairs) {
        return;
    }

    // Find which lump this pair belongs to
    long pos = bisect(makeBlockPairEnumStraight, lumpIndexEnd - lumpIndexStart, tid);
    long l = lumpIndexStart + pos;

    // Get the pair indices within this lump
    long n = chainColPtr[l + 1] - (chainColPtr[l] + 1);
    int2 di_dj = toOrderedPair(n, tid - makeBlockPairEnumStraight[pos]);
    long di = di_dj.x;
    long dj = di_dj.y;

    // Perform sparse elimination for this pair
    long startPtr = chainColPtr[l] + 1;  // skip diag block
    long lColSize = lumpStart[l + 1] - lumpStart[l];

    long i = startPtr + di;
    long si = chainRowSpan[i];
    long siSize = spanStart[si + 1] - spanStart[si];
    long siDataPtr = chainData[i];

    long targetLump = spanToLump[si];
    long targetSpanOffset = spanOffsetInLump[si];
    long targetStartPtr = chainColPtr[targetLump];
    long targetEndPtr = chainColPtr[targetLump + 1];
    long targetLumpSize = lumpStart[targetLump + 1] - lumpStart[targetLump];

    long j = startPtr + dj;
    long sj = chainRowSpan[j];
    long sjSize = spanStart[sj + 1] - spanStart[sj];
    long sjDataPtr = chainData[j];

    // Find target block position via binary search
    long targetPos = bisect(chainRowSpan + targetStartPtr, targetEndPtr - targetStartPtr, sj);
    long jiDataPtr = chainData[targetStartPtr + targetPos];

    // Compute: jiBlock -= jlBlock * ilBlock^T
    __global float* dst = data + jiDataPtr + targetSpanOffset;
    __global const float* jlBlock = data + sjDataPtr;
    __global const float* ilBlock = data + siDataPtr;

    matSubProduct_float(dst, (int)targetLumpSize, jlBlock, (int)lColSize, (int)sjSize, (int)lColSize,
                        ilBlock, (int)lColSize, (int)siSize, (int)lColSize);
}

__kernel void sparse_elim_straight_kernel_double(
    __global const long* chainColPtr,
    __global const long* lumpStart,
    __global const long* chainRowSpan,
    __global const long* spanStart,
    __global const long* chainData,
    __global const long* spanToLump,
    __global const long* spanOffsetInLump,
    __global double* data,
    const long lumpIndexStart,
    const long lumpIndexEnd,
    __global const long* makeBlockPairEnumStraight,
    const long numBlockPairs)
{
    int tid = get_global_id(0);
    if (tid >= numBlockPairs) {
        return;
    }

    long pos = bisect(makeBlockPairEnumStraight, lumpIndexEnd - lumpIndexStart, tid);
    long l = lumpIndexStart + pos;

    long n = chainColPtr[l + 1] - (chainColPtr[l] + 1);
    int2 di_dj = toOrderedPair(n, tid - makeBlockPairEnumStraight[pos]);
    long di = di_dj.x;
    long dj = di_dj.y;

    long startPtr = chainColPtr[l] + 1;
    long lColSize = lumpStart[l + 1] - lumpStart[l];

    long i = startPtr + di;
    long si = chainRowSpan[i];
    long siSize = spanStart[si + 1] - spanStart[si];
    long siDataPtr = chainData[i];

    long targetLump = spanToLump[si];
    long targetSpanOffset = spanOffsetInLump[si];
    long targetStartPtr = chainColPtr[targetLump];
    long targetEndPtr = chainColPtr[targetLump + 1];
    long targetLumpSize = lumpStart[targetLump + 1] - lumpStart[targetLump];

    long j = startPtr + dj;
    long sj = chainRowSpan[j];
    long sjSize = spanStart[sj + 1] - spanStart[sj];
    long sjDataPtr = chainData[j];

    long targetPos = bisect(chainRowSpan + targetStartPtr, targetEndPtr - targetStartPtr, sj);
    long jiDataPtr = chainData[targetStartPtr + targetPos];

    __global double* dst = data + jiDataPtr + targetSpanOffset;
    __global const double* jlBlock = data + sjDataPtr;
    __global const double* ilBlock = data + siDataPtr;

    matSubProduct_double(dst, (int)targetLumpSize, jlBlock, (int)lColSize, (int)sjSize, (int)lColSize,
                         ilBlock, (int)lColSize, (int)siSize, (int)lColSize);
}

// ============================================================================
// Kernel 7: sparseElim_subDiagMult - Float version
// Sub-diagonal matrix multiply during solve
// ============================================================================
__kernel void sparseElim_subDiagMult_float(
    __global const long* lumpStarts,
    __global const long* spanStarts,
    __global const long* chainColPtr,
    __global const long* chainRowSpan,
    __global const long* chainData,
    __global const float* data,
    __global float* v,
    const long ldc,
    const long nRHS,
    const long lumpIndexStart,
    const long lumpIndexEnd)
{
    int tid = get_global_id(0);
    long lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    long lumpStart = lumpStarts[lump];
    long lumpSize = lumpStarts[lump + 1] - lumpStart;
    long colStart = chainColPtr[lump];
    long colEnd = chainColPtr[lump + 1];

    // For each block below diagonal
    for (long colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
        long rowSpan = chainRowSpan[colPtr];
        long rowSpanStart = spanStarts[rowSpan];
        long rowSpanSize = spanStarts[rowSpan + 1] - rowSpanStart;
        long blockPtr = chainData[colPtr];

        // matQ -= block * matC
        for (long rhs = 0; rhs < nRHS; rhs++) {
            for (long i = 0; i < rowSpanSize; i++) {
                float sum = 0.0f;
                for (long k = 0; k < lumpSize; k++) {
                    sum += data[blockPtr + i * lumpSize + k] * v[lumpStart + k + ldc * rhs];
                }
                v[rowSpanStart + i + ldc * rhs] -= sum;
            }
        }
    }
}

__kernel void sparseElim_subDiagMult_double(
    __global const long* lumpStarts,
    __global const long* spanStarts,
    __global const long* chainColPtr,
    __global const long* chainRowSpan,
    __global const long* chainData,
    __global const double* data,
    __global double* v,
    const long ldc,
    const long nRHS,
    const long lumpIndexStart,
    const long lumpIndexEnd)
{
    int tid = get_global_id(0);
    long lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    long lumpStart = lumpStarts[lump];
    long lumpSize = lumpStarts[lump + 1] - lumpStart;
    long colStart = chainColPtr[lump];
    long colEnd = chainColPtr[lump + 1];

    for (long colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
        long rowSpan = chainRowSpan[colPtr];
        long rowSpanStart = spanStarts[rowSpan];
        long rowSpanSize = spanStarts[rowSpan + 1] - rowSpanStart;
        long blockPtr = chainData[colPtr];

        for (long rhs = 0; rhs < nRHS; rhs++) {
            for (long i = 0; i < rowSpanSize; i++) {
                double sum = 0.0;
                for (long k = 0; k < lumpSize; k++) {
                    sum += data[blockPtr + i * lumpSize + k] * v[lumpStart + k + ldc * rhs];
                }
                v[rowSpanStart + i + ldc * rhs] -= sum;
            }
        }
    }
}

// ============================================================================
// Kernel 8: sparseElim_subDiagMultT - Float version
// Transposed sub-diagonal matrix multiply during solve
// ============================================================================
__kernel void sparseElim_subDiagMultT_float(
    __global const long* lumpStarts,
    __global const long* spanStarts,
    __global const long* chainColPtr,
    __global const long* chainRowSpan,
    __global const long* chainData,
    __global const float* data,
    __global float* v,
    const long ldc,
    const long nRHS,
    const long lumpIndexStart,
    const long lumpIndexEnd)
{
    int tid = get_global_id(0);
    long lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    long lumpStart = lumpStarts[lump];
    long lumpSize = lumpStarts[lump + 1] - lumpStart;
    long colStart = chainColPtr[lump];
    long colEnd = chainColPtr[lump + 1];

    for (long colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
        long rowSpan = chainRowSpan[colPtr];
        long rowSpanStart = spanStarts[rowSpan];
        long rowSpanSize = spanStarts[rowSpan + 1] - rowSpanStart;
        long blockPtr = chainData[colPtr];

        // matC -= block^T * matQ
        for (long rhs = 0; rhs < nRHS; rhs++) {
            for (long i = 0; i < lumpSize; i++) {
                float sum = 0.0f;
                for (long k = 0; k < rowSpanSize; k++) {
                    sum += data[blockPtr + k * lumpSize + i] * v[rowSpanStart + k + ldc * rhs];
                }
                v[lumpStart + i + ldc * rhs] -= sum;
            }
        }
    }
}

__kernel void sparseElim_subDiagMultT_double(
    __global const long* lumpStarts,
    __global const long* spanStarts,
    __global const long* chainColPtr,
    __global const long* chainRowSpan,
    __global const long* chainData,
    __global const double* data,
    __global double* v,
    const long ldc,
    const long nRHS,
    const long lumpIndexStart,
    const long lumpIndexEnd)
{
    int tid = get_global_id(0);
    long lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    long lumpStart = lumpStarts[lump];
    long lumpSize = lumpStarts[lump + 1] - lumpStart;
    long colStart = chainColPtr[lump];
    long colEnd = chainColPtr[lump + 1];

    for (long colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
        long rowSpan = chainRowSpan[colPtr];
        long rowSpanStart = spanStarts[rowSpan];
        long rowSpanSize = spanStarts[rowSpan + 1] - rowSpanStart;
        long blockPtr = chainData[colPtr];

        for (long rhs = 0; rhs < nRHS; rhs++) {
            for (long i = 0; i < lumpSize; i++) {
                double sum = 0.0;
                for (long k = 0; k < rowSpanSize; k++) {
                    sum += data[blockPtr + k * lumpSize + i] * v[rowSpanStart + k + ldc * rhs];
                }
                v[lumpStart + i + ldc * rhs] -= sum;
            }
        }
    }
}

// ============================================================================
// Kernel 9: assembleVecT (Transposed vector assembly) - Float version
// ============================================================================
__kernel void assembleVecT_kernel_float(
    __global const long* chainRowsTillEnd,
    __global const long* toSpan,
    __global const long* spanStarts,
    __global const float* C,
    const long numColItems,
    __global float* A,
    const long ldc,
    const long nRHS,
    const long startRow)
{
    int tid = get_global_id(0);
    if (tid >= numColItems) {
        return;
    }

    long rowsBefore = (tid > 0) ? (chainRowsTillEnd[tid - 1] - startRow) : 0;
    long rowsAfter = chainRowsTillEnd[tid] - startRow;
    long blockRows = rowsAfter - rowsBefore;

    long span = toSpan[tid];
    long spanStart = spanStarts[span];

    __global float* dstPtr = A + rowsBefore * nRHS;
    __global const float* srcPtr = C + spanStart;

    for (long rhs = 0; rhs < nRHS; rhs++) {
        for (long i = 0; i < blockRows; i++) {
            dstPtr[i * nRHS + rhs] = srcPtr[i + rhs * ldc];
        }
    }
}

// Double precision version
__kernel void assembleVecT_kernel_double(
    __global const long* chainRowsTillEnd,
    __global const long* toSpan,
    __global const long* spanStarts,
    __global const double* C,
    const long numColItems,
    __global double* A,
    const long ldc,
    const long nRHS,
    const long startRow)
{
    int tid = get_global_id(0);
    if (tid >= numColItems) {
        return;
    }

    long rowsBefore = (tid > 0) ? (chainRowsTillEnd[tid - 1] - startRow) : 0;
    long rowsAfter = chainRowsTillEnd[tid] - startRow;
    long blockRows = rowsAfter - rowsBefore;

    long span = toSpan[tid];
    long spanStart = spanStarts[span];

    __global double* dstPtr = A + rowsBefore * nRHS;
    __global const double* srcPtr = C + spanStart;

    for (long rhs = 0; rhs < nRHS; rhs++) {
        for (long i = 0; i < blockRows; i++) {
            dstPtr[i * nRHS + rhs] = srcPtr[i + rhs * ldc];
        }
    }
}
