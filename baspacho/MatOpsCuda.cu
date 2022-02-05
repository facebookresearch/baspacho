#pragma nv_diag_suppress 20236
#pragma nv_diag_suppress 20012

#include <chrono>
#include <iostream>

#include "baspacho/CudaAtomic.cuh"
#include "baspacho/CudaDefs.h"
#include "baspacho/DebugMacros.h"
#include "baspacho/MatOpsCpuBase.h"
#include "baspacho/MathUtils.h"
#include "baspacho/Utils.h"

namespace BaSpaCho {

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

using OuterStride = Eigen::OuterStride<>;
template <typename T>
using OuterStridedMatM = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0,
    OuterStride>;
template <typename T>
using OuterStridedCMajMatM = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
    OuterStride>;
template <typename T>
using OuterStridedCMajMatK = Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
    OuterStride>;

struct CudaSymElimCtx : SymElimCtx {
    CudaSymElimCtx() {}
    virtual ~CudaSymElimCtx() override {}

    int64_t numColumns;
    int64_t numBlockPairs;
    DevMirror<int64_t> makeBlockPairEnumStraight;
};

struct CudaSymbolicCtx : CpuBaseSymbolicCtx {
    CudaSymbolicCtx(const CoalescedBlockMatrixSkel& skel)
        : CpuBaseSymbolicCtx(skel) {
        // TODO: support custom stream in the future
        cublasCHECK(cublasCreate(&cublasH));
        // cublasCHECK(cublasSetStream(cublasH, stream));
        cusolverCHECK(cusolverDnCreate(&cusolverDnH));
        // cusolverCHECK(cusolverDnSetStream(cusolverDnH, stream));

        devChainRowsTillEnd.load(skel.chainRowsTillEnd);
        devChainRowSpan.load(skel.chainRowSpan);
        devSpanOffsetInLump.load(skel.spanOffsetInLump);
        devLumpStart.load(skel.lumpStart);
        devChainColPtr.load(skel.chainColPtr);
        devChainData.load(skel.chainData);
        devBoardColPtr.load(skel.boardColPtr);
        devBoardChainColOrd.load(skel.boardChainColOrd);
        devSpanStart.load(skel.spanStart);
        devSpanToLump.load(skel.spanToLump);
    }

    virtual ~CudaSymbolicCtx() override {
        if (cublasH) {
            cublasCHECK(cublasDestroy(cublasH));
        }
        if (cusolverDnH) {
            cusolverCHECK(cusolverDnDestroy(cusolverDnH));
        }
    }

    virtual SymElimCtxPtr prepareElimination(int64_t lumpsBegin,
                                             int64_t lumpsEnd) override {
        CudaSymElimCtx* elim = new CudaSymElimCtx;

        vector<int64_t> makeStraight(lumpsEnd - lumpsBegin + 1);

        // for each lump, compute number of pairs contributing to elim
        for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
            int64_t startPtr = skel.chainColPtr[l] + 1;  // skip diag block
            int64_t endPtr = skel.chainColPtr[l + 1];
            int64_t n = endPtr - startPtr;
            makeStraight[l - lumpsBegin] = n * (n + 1) / 2;
        }
        cumSumVec(makeStraight);

        elim->numColumns = lumpsEnd - lumpsBegin;
        elim->numBlockPairs = makeStraight[makeStraight.size() - 1];
        elim->makeBlockPairEnumStraight.load(makeStraight);

        return SymElimCtxPtr(elim);
    }

    virtual NumericCtxBase* createNumericCtxForType(
        std::type_index tIdx, int64_t tempBufSize) override;

    virtual SolveCtxBase* createSolveCtxForType(std::type_index tIdx,
                                                int nRHS) override;

    cublasHandle_t cublasH = nullptr;
    cusolverDnHandle_t cusolverDnH = nullptr;

    DevMirror<int64_t> devChainRowsTillEnd;
    DevMirror<int64_t> devChainRowSpan;
    DevMirror<int64_t> devSpanOffsetInLump;
    DevMirror<int64_t> devLumpStart;
    DevMirror<int64_t> devChainColPtr;
    DevMirror<int64_t> devChainData;
    DevMirror<int64_t> devBoardColPtr;
    DevMirror<int64_t> devBoardChainColOrd;
    DevMirror<int64_t> devSpanStart;
    DevMirror<int64_t> devSpanToLump;
};

// cuda ops implemented using CUBLAS and custom kernels
struct CudaOps : Ops {
    virtual SymbolicCtxPtr createSymbolicCtx(
        const CoalescedBlockMatrixSkel& skel) override {
        // cout << "create sym..." << endl;
        return SymbolicCtxPtr(new CudaSymbolicCtx(skel));
    }
};

template <typename T>
__global__ static inline void factor_lumps_kernel(
    const int64_t* lumpStart, const int64_t* chainColPtr,
    const int64_t* chainData, const int64_t* boardColPtr,
    const int64_t* boardChainColOrd, const int64_t* chainRowsTillEnd, T* data,
    int64_t lumpIndexStart, int64_t lumpIndexEnd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t lump = lumpIndexStart + i;
    if (lump >= lumpIndexEnd) {
        return;
    }
    int64_t lumpSize = lumpStart[lump + 1] - lumpStart[lump];
    int64_t colStart = chainColPtr[lump];
    int64_t dataPtr = chainData[colStart];

    // in-place lower diag cholesky dec on diagonal block
    T* diagBlockPtr = data + dataPtr;
    cholesky(diagBlockPtr, lumpSize);

    int64_t gatheredStart = boardColPtr[lump];
    int64_t gatheredEnd = boardColPtr[lump + 1];
    int64_t rowDataStart = boardChainColOrd[gatheredStart + 1];
    int64_t rowDataEnd = boardChainColOrd[gatheredEnd - 1];
    int64_t belowDiagStart = chainData[colStart + rowDataStart];
    int64_t numRows = chainRowsTillEnd[colStart + rowDataEnd - 1] -
                      chainRowsTillEnd[colStart + rowDataStart - 1];

    T* belowDiagBlockPtr = data + belowDiagStart;
    for (int i = 0; i < numRows; i++) {
        solveUpperT(diagBlockPtr, lumpSize, belowDiagBlockPtr);
        belowDiagBlockPtr += lumpSize;
    }
}

template <typename T>
__device__ static inline void do_sparse_elim(
    const int64_t* chainColPtr, const int64_t* lumpStart,
    const int64_t* chainRowSpan, const int64_t* spanStart,
    const int64_t* chainData, const int64_t* spanToLump,
    const int64_t* spanOffsetInLump, T* data, int64_t l, int64_t di,
    int64_t dj) {
    int64_t startPtr = chainColPtr[l] + 1;  // skip diag block
    int64_t lColSize = lumpStart[l + 1] - lumpStart[l];

    int64_t i = startPtr + di;
    int64_t si = chainRowSpan[i];
    int64_t siSize = spanStart[si + 1] - spanStart[si];
    int64_t siDataPtr = chainData[i];
    Eigen::Map<MatRMaj<T>> ilBlock(data + siDataPtr, siSize, lColSize);

    int64_t targetLump = spanToLump[si];
    int64_t targetSpanOffsetInLump = spanOffsetInLump[si];
    int64_t targetStartPtr = chainColPtr[targetLump];  // skip diag block
    int64_t targetEndPtr = chainColPtr[targetLump + 1];
    int64_t targetLumpSize = lumpStart[targetLump + 1] - lumpStart[targetLump];

    int64_t j = startPtr + dj;
    int64_t sj = chainRowSpan[j];
    int64_t sjSize = spanStart[sj + 1] - spanStart[sj];
    int64_t sjDataPtr = chainData[j];

    Eigen::Map<MatRMaj<T>> jlBlock(data + sjDataPtr, sjSize, lColSize);

    uint64_t pos = bisect(chainRowSpan + targetStartPtr,
                          targetEndPtr - targetStartPtr, sj);
    int64_t jiDataPtr = chainData[targetStartPtr + pos];
    OuterStridedMatM<T> jiBlock(data + jiDataPtr + targetSpanOffsetInLump,
                                sjSize, siSize, OuterStride(targetLumpSize));
    // jiBlock -= jlBlock * ilBlock.transpose();
    locked_sub_product(jiBlock, jlBlock, ilBlock);
}

// "naive" elimination kernel, in the sense there is one kernel instance
// per column, and will internally iterate over pairs of blocks (two
// nested loops). Not meant for performance, but as a simpler testing
// version of the below "straigthened" kernel.
template <typename T>
__global__ static inline void sparse_elim_2loops_kernel(
    const int64_t* chainColPtr, const int64_t* lumpStart,
    const int64_t* chainRowSpan, const int64_t* spanStart,
    const int64_t* chainData, const int64_t* spanToLump,
    const int64_t* spanOffsetInLump, T* data, int64_t lumpIndexStart,
    int64_t lumpIndexEnd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t l = lumpIndexStart + i;
    if (l >= lumpIndexEnd) {
        return;
    }
    int64_t startPtr = chainColPtr[l] + 1;  // skip diag block
    int64_t endPtr = chainColPtr[l + 1];
    for (int64_t i = startPtr; i < endPtr; i++) {
        for (int64_t j = i; j < endPtr; j++) {
            do_sparse_elim(chainColPtr, lumpStart, chainRowSpan, spanStart,
                           chainData, spanToLump, spanOffsetInLump, data, l,
                           i - startPtr, j - startPtr);
        }
    }
}

// makeBlockPairEnumStraight contains the cumulated sum of nb*(nb+1)/2
// over all columns, nb being the number of blocks in the columns.
// `i` is the index in the index in the list of all pairs of block in
// the same column. We bisect and get as position the column (relative to
// lumpIndexStart), and as offset to the found value the index in range
// 0..nb*(nb+1)/2-1 in the list of *pairs* of blocks in the column. Such
// index if converted to the ordered pair di/dj
template <typename T>
__global__ static inline void sparse_elim_straight_kernel(
    const int64_t* chainColPtr, const int64_t* lumpStart,
    const int64_t* chainRowSpan, const int64_t* spanStart,
    const int64_t* chainData, const int64_t* spanToLump,
    const int64_t* spanOffsetInLump, T* data, int64_t lumpIndexStart,
    int64_t lumpIndexEnd, const int64_t* makeBlockPairEnumStraight,
    int64_t numBlockPairs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBlockPairs) {
        return;
    }
    int64_t pos =
        bisect(makeBlockPairEnumStraight, lumpIndexEnd - lumpIndexStart, i);
    int64_t l = lumpIndexStart + pos;
    int64_t n = chainColPtr[l + 1] - (chainColPtr[l] + 1);
    auto [di, dj] = toOrderedPair(n, i - makeBlockPairEnumStraight[pos]);
    do_sparse_elim(chainColPtr, lumpStart, chainRowSpan, spanStart, chainData,
                   spanToLump, spanOffsetInLump, data, l, di, dj);
}

template <typename T>
__device__ static inline void stridedMatSubDev(T* dst, int64_t dstStride,
                                               const T* src, int64_t srcStride,
                                               int64_t rSize, int64_t cSize) {
    for (uint j = 0; j < rSize; j++) {
        for (uint i = 0; i < cSize; i++) {
            dst[i] -= src[i];
        }
        dst += dstStride;
        src += srcStride;
    }
}

template <typename T>
__global__ void assemble_kernel(
    int64_t numBlockRows, int64_t numBlockCols, int64_t rectRowBegin,
    int64_t srcRectWidth, int64_t dstStride, const int64_t* pChainRowsTillEnd,
    const int64_t* pToSpan, const int64_t* pSpanToChainOffset,
    const int64_t* pSpanOffsetInLump, const T* matRectPtr, T* data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > numBlockRows * numBlockCols) {
        return;
    }
    int64_t r = i % numBlockRows;
    int64_t c = i / numBlockRows;
    if (c > r) {
        return;
    }

    int64_t rBegin = pChainRowsTillEnd[r - 1] - rectRowBegin;
    int64_t rSize = pChainRowsTillEnd[r] - rBegin - rectRowBegin;
    int64_t rParam = pToSpan[r];
    int64_t rOffset = pSpanToChainOffset[rParam];
    const T* matRowPtr = matRectPtr + rBegin * srcRectWidth;

    int64_t cStart = pChainRowsTillEnd[c - 1] - rectRowBegin;
    int64_t cSize = pChainRowsTillEnd[c] - cStart - rectRowBegin;
    int64_t offset = rOffset + pSpanOffsetInLump[pToSpan[c]];

    T* dst = data + offset;
    const T* src = matRowPtr + cStart;
    stridedMatSubDev(dst, dstStride, src, srcRectWidth, rSize, cSize);
}

template <typename T>
struct CudaNumericCtx : NumericCtx<T> {
    CudaNumericCtx(const CudaSymbolicCtx& sym, int64_t bufSize,
                   int64_t numSpans)
        : spanToChainOffset(numSpans), sym(sym) {
        cuCHECK(cudaMalloc((void**)&devTempBuffer, bufSize * sizeof(T)));
        cuCHECK(cudaMalloc((void**)&devSpanToChainOffset,
                           spanToChainOffset.size() * sizeof(int64_t)));
    }

    virtual ~CudaNumericCtx() override {
        if (devTempBuffer) {
            cuCHECK(cudaFree(devTempBuffer));
        }
        if (devSpanToChainOffset) {
            cuCHECK(cudaFree(devSpanToChainOffset));
        }
    }

    virtual void doElimination(const SymElimCtx& elimData, T* data,
                               int64_t lumpsBegin, int64_t lumpsEnd) override {
        const CudaSymElimCtx* pElim =
            dynamic_cast<const CudaSymElimCtx*>(&elimData);
        BASPACHO_CHECK_NOTNULL(pElim);
        const CudaSymElimCtx& elim = *pElim;

        OpInstance timer(elim.elimStat);

        int wgs = 32;
        int numGroups = (lumpsEnd - lumpsBegin + wgs - 1) / wgs;
        factor_lumps_kernel<T><<<numGroups, wgs>>>(
            sym.devLumpStart.ptr, sym.devChainColPtr.ptr, sym.devChainData.ptr,
            sym.devBoardColPtr.ptr, sym.devBoardChainColOrd.ptr,
            sym.devChainRowsTillEnd.ptr, data, lumpsBegin, lumpsEnd);

        cuCHECK(cudaDeviceSynchronize());
        /*cout << "elim 1st part: " << tdelta(hrc::now() - timer.start).count()
             << "s" << endl;*/

#if 0
        // double inner loop
        sparse_elim_2loops_kernel<T><<<numGroups, wgs>>>(
            sym.devChainColPtr.ptr, sym.devLumpStart.ptr,
            sym.devChainRowSpan.ptr, sym.devSpanStart.ptr, sym.devChainData.ptr,
            sym.devSpanToLump.ptr, sym.devSpanOffsetInLump.ptr, data,
            lumpsBegin, lumpsEnd);
#else
        int wgs2 = 32;
        int numGroups2 = (elim.numBlockPairs + wgs2 - 1) / wgs2;
        sparse_elim_straight_kernel<T><<<numGroups2, wgs2>>>(
            sym.devChainColPtr.ptr, sym.devLumpStart.ptr,
            sym.devChainRowSpan.ptr, sym.devSpanStart.ptr, sym.devChainData.ptr,
            sym.devSpanToLump.ptr, sym.devSpanOffsetInLump.ptr, data,
            lumpsBegin, lumpsEnd, elim.makeBlockPairEnumStraight.ptr,
            elim.numBlockPairs);
#endif

        cuCHECK(cudaDeviceSynchronize());
    }

    virtual void potrf(int64_t n, T* A) override;

    virtual void trsm(int64_t n, int64_t k, const T* A, T* B) override;

    virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const T* data,
                              int64_t offset) override;

    virtual void prepareAssemble(int64_t targetLump) override {
        const CoalescedBlockMatrixSkel& skel = sym.skel;

        // FIXME: compute on CPU and copy, not ideal
        for (int64_t i = skel.chainColPtr[targetLump],
                     iEnd = skel.chainColPtr[targetLump + 1];
             i < iEnd; i++) {
            spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
        }
        cuCHECK(cudaMemcpy(devSpanToChainOffset, spanToChainOffset.data(),
                           spanToChainOffset.size() * sizeof(int64_t),
                           cudaMemcpyHostToDevice));
    }

    virtual void assemble(T* data, int64_t rectRowBegin,
                          int64_t dstStride,  //
                          int64_t srcColDataOffset, int64_t srcRectWidth,
                          int64_t numBlockRows, int64_t numBlockCols) override {
        OpInstance timer(sym.asmblStat);
        const int64_t* pChainRowsTillEnd =
            sym.devChainRowsTillEnd.ptr + srcColDataOffset;
        const int64_t* pToSpan = sym.devChainRowSpan.ptr + srcColDataOffset;
        const int64_t* pSpanToChainOffset = devSpanToChainOffset;
        const int64_t* pSpanOffsetInLump = sym.devSpanOffsetInLump.ptr;

        const T* matRectPtr = devTempBuffer;

        int wgs = 32;
        int numGroups = (numBlockRows * numBlockCols + wgs - 1) / wgs;
        assemble_kernel<T><<<numGroups, wgs>>>(
            numBlockRows, numBlockCols, rectRowBegin, srcRectWidth, dstStride,
            pChainRowsTillEnd, pToSpan, pSpanToChainOffset, pSpanOffsetInLump,
            matRectPtr, data);
    }

    T* devTempBuffer;
    int64_t* devSpanToChainOffset;
    std::vector<int64_t> spanToChainOffset;

    const CudaSymbolicCtx& sym;
};

template <>
void CudaNumericCtx<double>::potrf(int64_t n, double* A) {
    OpInstance timer(sym.potrfStat);
    sym.potrfBiggestN = std::max(sym.potrfBiggestN, n);

    int workspaceSize;
    cusolverCHECK(cusolverDnDpotrf_bufferSize(
        sym.cusolverDnH, CUBLAS_FILL_MODE_UPPER, n, A, n, &workspaceSize));

    double* workspace;
    int* devInfo;
    cuCHECK(cudaMalloc((void**)&workspace, workspaceSize * sizeof(double)));
    cuCHECK(cudaMalloc((void**)&devInfo, 1 * sizeof(int)));

    cusolverCHECK(cusolverDnDpotrf(sym.cusolverDnH, CUBLAS_FILL_MODE_UPPER, n,
                                   A, n, workspace, workspaceSize, devInfo));

    int info;
    cuCHECK(
        cudaMemcpy(&info, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost));
    cuCHECK(cudaFree(devInfo));
    cuCHECK(cudaFree(workspace));
}

template <>
void CudaNumericCtx<float>::potrf(int64_t n, float* A) {
    OpInstance timer(sym.potrfStat);
    sym.potrfBiggestN = std::max(sym.potrfBiggestN, n);

    int workspaceSize;
    cusolverCHECK(cusolverDnSpotrf_bufferSize(
        sym.cusolverDnH, CUBLAS_FILL_MODE_UPPER, n, A, n, &workspaceSize));

    float* workspace;
    int* devInfo;
    cuCHECK(cudaMalloc((void**)&workspace, workspaceSize * sizeof(float)));
    cuCHECK(cudaMalloc((void**)&devInfo, 1 * sizeof(int)));

    cusolverCHECK(cusolverDnSpotrf(sym.cusolverDnH, CUBLAS_FILL_MODE_UPPER, n,
                                   A, n, workspace, workspaceSize, devInfo));

    int info;
    cuCHECK(
        cudaMemcpy(&info, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost));
    cuCHECK(cudaFree(devInfo));
    cuCHECK(cudaFree(workspace));
}

template <>
void CudaNumericCtx<double>::trsm(int64_t n, int64_t k, const double* A,
                                  double* B) {
    OpInstance timer(sym.trsmStat);

    double alpha(1.0);
    cublasCHECK(cublasDtrsm(sym.cublasH, CUBLAS_SIDE_LEFT,
                            CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                            CUBLAS_DIAG_NON_UNIT, n, k, &alpha, A, n, B, n));
}

template <>
void CudaNumericCtx<float>::trsm(int64_t n, int64_t k, const float* A,
                                 float* B) {
    OpInstance timer(sym.trsmStat);

    float alpha(1.0);
    cublasCHECK(cublasStrsm(sym.cublasH, CUBLAS_SIDE_LEFT,
                            CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                            CUBLAS_DIAG_NON_UNIT, n, k, &alpha, A, n, B, n));
}

template <>
void CudaNumericCtx<double>::saveSyrkGemm(int64_t m, int64_t n, int64_t k,
                                          const double* data, int64_t offset) {
    OpInstance timer(sym.sygeStat);

    double alpha(1.0), beta(0.0);
    cublasCHECK(cublasDgemm(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, m, n, k,
                            &alpha, data + offset, k, data + offset, k, &beta,
                            devTempBuffer, m));

    sym.gemmCalls++;
}

template <>
void CudaNumericCtx<float>::saveSyrkGemm(int64_t m, int64_t n, int64_t k,
                                         const float* data, int64_t offset) {
    OpInstance timer(sym.sygeStat);

    float alpha(1.0), beta(0.0);
    cublasCHECK(cublasSgemm(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, m, n, k,
                            &alpha, data + offset, k, data + offset, k, &beta,
                            devTempBuffer, m));

    sym.gemmCalls++;
}

template <typename T>
__device__ static inline void stridedTransSub(T* dst, int64_t dstStride,
                                              const T* src, int64_t srcStride,
                                              int64_t rSize, int64_t cSize) {
    for (uint j = 0; j < rSize; j++) {
        T* pDst = dst + j;
        for (uint i = 0; i < cSize; i++) {
            *pDst -= src[i];
            pDst += dstStride;
        }
        src += srcStride;
    }
}

template <typename T>
__device__ static inline void stridedTransSet(T* dst, int64_t dstStride,
                                              const T* src, int64_t srcStride,
                                              int64_t rSize, int64_t cSize) {
    for (uint j = 0; j < rSize; j++) {
        T* pDst = dst + j;
        for (uint i = 0; i < cSize; i++) {
            *pDst = src[i];
            pDst += dstStride;
        }
        src += srcStride;
    }
}

template <typename T>
__global__ void assembleVec_kernel(const int64_t* chainRowsTillEnd,
                                   const int64_t* toSpan,
                                   const int64_t* spanStarts, const T* A,
                                   int64_t numColItems, T* C, int64_t ldc,
                                   int64_t nRHS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > numColItems) {
        return;
    }
    int64_t rowOffset = chainRowsTillEnd[i - 1] - chainRowsTillEnd[-1];
    int64_t span = toSpan[i];
    int64_t spanStart = spanStarts[span];
    int64_t spanSize = spanStarts[span + 1] - spanStart;

    stridedTransSub(C + spanStart, ldc, A + rowOffset * nRHS, nRHS, spanSize,
                    nRHS);
}

template <typename T>
__global__ void assembleVecT_kernel(const int64_t* chainRowsTillEnd,
                                    const int64_t* toSpan,
                                    const int64_t* spanStarts, const T* C,
                                    int64_t ldc, int64_t nRHS, T* A,
                                    int64_t numColItems) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > numColItems) {
        return;
    }
    int64_t rowOffset = chainRowsTillEnd[i - 1] - chainRowsTillEnd[-1];
    int64_t span = toSpan[i];
    int64_t spanStart = spanStarts[span];
    int64_t spanSize = spanStarts[span + 1] - spanStart;

    stridedTransSet(A + rowOffset * nRHS, nRHS, C + spanStart, ldc, nRHS,
                    spanSize);
}

template <typename T>
struct CudaSolveCtx : SolveCtx<T> {
    CudaSolveCtx(const CudaSymbolicCtx& sym, int64_t nRHS)
        : sym(sym), nRHS(nRHS) {
        cuCHECK(cudaMalloc((void**)&devSolveBuf,
                           sym.skel.order() * nRHS * sizeof(T)));
    }
    virtual ~CudaSolveCtx() override {
        if (devSolveBuf) {
            cuCHECK(cudaFree(devSolveBuf));
        }
    }

    virtual void solveL(const T* data, int64_t offM, int64_t n, T* C,
                        int64_t offC, int64_t ldc) override {
        T alpha(1.0);
        cublasCHECK(cublasDtrsm(sym.cublasH, CUBLAS_SIDE_LEFT,
                                CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                                CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha,
                                data + offM, n, C + offC, ldc));
    }

    virtual void gemv(const T* data, int64_t offM, int64_t nRows, int64_t nCols,
                      const T* A, int64_t offA, int64_t lda) override {
        T alpha(1.0), beta(0.0);
        cublasCHECK(cublasDgemm(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, nRHS,
                                nRows, nCols, &alpha, A + offA, lda,
                                data + offM, nCols, &beta, devSolveBuf, nRHS));
    }

    virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, T* C,
                             int64_t ldc) override {
        int wgs = 32;
        int numGroups = (numColItems + wgs - 1) / wgs;
        assembleVec_kernel<T><<<numGroups, wgs>>>(
            sym.devChainRowsTillEnd.ptr + chainColPtr,
            sym.devChainRowSpan.ptr + chainColPtr, sym.devSpanStart.ptr,
            devSolveBuf, numColItems, C, ldc, nRHS);
    }

    virtual void solveLt(const T* data, int64_t offM, int64_t n, T* C,
                         int64_t offC, int64_t ldc) override {
        T alpha(1.0);
        cublasCHECK(cublasDtrsm(sym.cublasH, CUBLAS_SIDE_LEFT,
                                CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha,
                                data + offM, n, C + offC, ldc));
    }

    virtual void gemvT(const T* data, int64_t offM, int64_t nRows,
                       int64_t nCols, T* A, int64_t offA,
                       int64_t lda) override {
        T alpha(-1.0), beta(1.0);
        cublasCHECK(cublasDgemm(sym.cublasH, CUBLAS_OP_N, CUBLAS_OP_C, nCols,
                                nRHS, nRows, &alpha, data + offM, nCols,
                                devSolveBuf, nRHS, &beta, A + offA, lda));
    }

    virtual void assembleVecT(const T* C, int64_t ldc, int64_t chainColPtr,
                              int64_t numColItems) override {
        int wgs = 32;
        int numGroups = (numColItems + wgs - 1) / wgs;
        assembleVecT_kernel<T><<<numGroups, wgs>>>(
            sym.devChainRowsTillEnd.ptr + chainColPtr,
            sym.devChainRowSpan.ptr + chainColPtr, sym.devSpanStart.ptr, C, ldc,
            nRHS, devSolveBuf, numColItems);
    }

    const CudaSymbolicCtx& sym;
    int64_t nRHS;
    T* devSolveBuf;
};

template <>
void CudaSolveCtx<double>::solveL(const double* data, int64_t offM, int64_t n,
                                  double* C, int64_t offC, int64_t ldc) {
    double alpha(1.0);
    cublasCHECK(cublasDtrsm(
        sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
        CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, data + offM, n, C + offC, ldc));
}

template <>
void CudaSolveCtx<float>::solveL(const float* data, int64_t offM, int64_t n,
                                 float* C, int64_t offC, int64_t ldc) {
    float alpha(1.0);
    cublasCHECK(cublasStrsm(
        sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
        CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, data + offM, n, C + offC, ldc));
}

template <>
void CudaSolveCtx<double>::gemv(const double* data, int64_t offM, int64_t nRows,
                                int64_t nCols, const double* A, int64_t offA,
                                int64_t lda) {
    double alpha(1.0), beta(0.0);
    cublasCHECK(cublasDgemm(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, nRHS, nRows,
                            nCols, &alpha, A + offA, lda, data + offM, nCols,
                            &beta, devSolveBuf, nRHS));
}

template <>
void CudaSolveCtx<float>::gemv(const float* data, int64_t offM, int64_t nRows,
                               int64_t nCols, const float* A, int64_t offA,
                               int64_t lda) {
    float alpha(1.0), beta(0.0);
    cublasCHECK(cublasSgemm(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, nRHS, nRows,
                            nCols, &alpha, A + offA, lda, data + offM, nCols,
                            &beta, devSolveBuf, nRHS));
}

template <>
void CudaSolveCtx<double>::solveLt(const double* data, int64_t offM, int64_t n,
                                   double* C, int64_t offC, int64_t ldc) {
    double alpha(1.0);
    cublasCHECK(cublasDtrsm(
        sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, data + offM, n, C + offC, ldc));
}

template <>
void CudaSolveCtx<float>::solveLt(const float* data, int64_t offM, int64_t n,
                                  float* C, int64_t offC, int64_t ldc) {
    float alpha(1.0);
    cublasCHECK(cublasStrsm(
        sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, data + offM, n, C + offC, ldc));
}

template <>
void CudaSolveCtx<double>::gemvT(const double* data, int64_t offM,
                                 int64_t nRows, int64_t nCols, double* A,
                                 int64_t offA, int64_t lda) {
    double alpha(-1.0), beta(1.0);
    cublasCHECK(cublasDgemm(sym.cublasH, CUBLAS_OP_N, CUBLAS_OP_C, nCols, nRHS,
                            nRows, &alpha, data + offM, nCols, devSolveBuf,
                            nRHS, &beta, A + offA, lda));
}

template <>
void CudaSolveCtx<float>::gemvT(const float* data, int64_t offM, int64_t nRows,
                                int64_t nCols, float* A, int64_t offA,
                                int64_t lda) {
    float alpha(-1.0), beta(1.0);
    cublasCHECK(cublasSgemm(sym.cublasH, CUBLAS_OP_N, CUBLAS_OP_C, nCols, nRHS,
                            nRows, &alpha, data + offM, nCols, devSolveBuf,
                            nRHS, &beta, A + offA, lda));
}

NumericCtxBase* CudaSymbolicCtx::createNumericCtxForType(std::type_index tIdx,
                                                         int64_t tempBufSize) {
    if (tIdx == std::type_index(typeid(double))) {
        return new CudaNumericCtx<double>(*this, tempBufSize,
                                          skel.spanStart.size() - 1);
    } else if (tIdx == std::type_index(typeid(float))) {
        return new CudaNumericCtx<float>(*this, tempBufSize,
                                         skel.spanStart.size() - 1);
    } else {
        return nullptr;
    }
}

SolveCtxBase* CudaSymbolicCtx::createSolveCtxForType(std::type_index tIdx,
                                                     int nRHS) {
    if (tIdx == std::type_index(typeid(double))) {
        return new CudaSolveCtx<double>(*this, nRHS);
    } else if (tIdx == std::type_index(typeid(float))) {
        return new CudaSolveCtx<float>(*this, nRHS);
    } else {
        return nullptr;
    }
}

OpsPtr cudaOps() { return OpsPtr(new CudaOps); }

}  // end namespace BaSpaCho