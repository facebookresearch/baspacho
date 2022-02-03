#pragma nv_diag_suppress 20236
#pragma nv_diag_suppress 20012

#include <chrono>
#include <iostream>

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
        // cout << "sym-init Hs" << endl;
        cublasCHECK(cublasCreate(&cublasH));
        // cublasCHECK(cublasSetStream(cublasH, stream));
        cusolverCHECK(cusolverDnCreate(&cusolverDnH));
        // cusolverCHECK(cusolverDnSetStream(cusolverDnH, stream));

        // cout << "sym-init..." << endl;
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
        std::type_index tIdx, int64_t tempBufSize,
        int maxBatchSize = 1) override;

    virtual SolveCtxBase* createSolveCtxForType(std::type_index tIdx) override;

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

#if 0
// magic IEEE double number locking mechanism
struct MagicLock {
    // this is a magic value (a NaN with "content" 0xb10cd, "blocked")
    // which his functionally equivalent to a NaN but cannot be ever
    // obtained as result of a computation, and can therefore be used as
    // a "magic" value to block matrix of double numbers
    // For future reference a magic value for floats could be 0x7f80b1cdu
    static constexpr uint64_t magicValue = 0x7ff00000000b10cdul;

    static inline __attribute__((always_inline)) double lock(double* address) {
        uint64_t retv;
        do {
            retv = __atomic_exchange_n((uint64_t*)address, magicValue,
                                       __ATOMIC_ACQUIRE);
        } while (retv == magicValue);
        return *(double*)(&retv);
    }

    static inline __attribute__((always_inline)) void unlock(double* address,
                                                             double val) {
        __atomic_store_n((uint64_t*)address, *(uint64_t*)(&val),
                         __ATOMIC_RELEASE);
    }
};

template <typename A, typename B, typename C>
__device__ void locked_sub_product(A& aMat, const B& bMat, const C& cMatT) {}
#endif

__device__ static inline float aAdd(float* address, float val) {
    return atomicAdd(address, val);
}

__device__ static inline double aAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(
            address_as_ull, assumed,
            ::__double_as_longlong(val + __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN
        // != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
}

template <typename A, typename B, typename C>
__device__ void locked_sub_product(A& aMat, const B& bMat, const C& cMatT) {
    for (int i = 0; i < bMat.rows(); i++) {
        for (int j = 0; j < cMatT.rows(); j++) {
            double* addr = &aMat(i, j);
            double val = -bMat.row(i).dot(cMatT.row(j));
            aAdd(addr, val);
        }
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
    int64_t endPtr = chainColPtr[l + 1];
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
    // BASPACHO_CHECK_EQ(chainRowSpan[targetStartPtr + pos], sj);
    int64_t jiDataPtr = chainData[targetStartPtr + pos];
    OuterStridedMatM<T> jiBlock(data + jiDataPtr + targetSpanOffsetInLump,
                                sjSize, siSize, OuterStride(targetLumpSize));
    // jiBlock -= jlBlock * ilBlock.transpose();
    locked_sub_product(jiBlock, jlBlock, ilBlock);
}

template <typename T>
__global__ static inline void sparse_elim_kernel(
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
    // makeBlockPairEnumStraight contains the cumulated sum of nb*(nb+1)/2
    // over all columns, nb being the number of blocks in the columns.
    // `i` is the index in the index in the list of all pairs of block in
    // the same column. We bisect and get as position the column (relative to
    // lumpIndexStart), and as offset to the found value the index
    // 0..nb*(nb+1)/2-1 in the list of pairs of blocks in the column. Such
    // index if converted to ordered the pair di/dj
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
        // cout << "num-init..." << endl;
        cuCHECK(cudaMalloc((void**)&devTempBuffer, bufSize * sizeof(T)));
        cuCHECK(cudaMalloc((void**)&devSpanToChainOffset,
                           spanToChainOffset.size() * sizeof(int64_t)));
        // cout << "num-init done!" << endl;
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
        // const CoalescedBlockMatrixSkel& skel = sym.skel;

        OpInstance timer(elim.elimStat);

        int wgs = 32;
        int numGroups = (lumpsEnd - lumpsBegin + wgs - 1) / wgs;
        factor_lumps_kernel<double><<<numGroups, wgs>>>(
            sym.devLumpStart.ptr, sym.devChainColPtr.ptr, sym.devChainData.ptr,
            sym.devBoardColPtr.ptr, sym.devBoardChainColOrd.ptr,
            sym.devChainRowsTillEnd.ptr, data, lumpsBegin, lumpsEnd);

#if 0
        sparse_elim_kernel<double><<<numGroups, wgs>>>(
            sym.devChainColPtr.ptr, sym.devLumpStart.ptr,
            sym.devChainRowSpan.ptr, sym.devSpanStart.ptr, sym.devChainData.ptr,
            sym.devSpanToLump.ptr, sym.devSpanOffsetInLump.ptr, data,
            lumpsBegin, lumpsEnd);
#else
        /*int64_t numColumns;
        int64_t numBlockPairs;
        DevMirror<int64_t> makeBlockPairEnumStraight;*/
        int wgs2 = 32;
        int numGroups2 = (elim.numBlockPairs + wgs2 - 1) / wgs2;
        sparse_elim_straight_kernel<double><<<numGroups2, wgs2>>>(
            sym.devChainColPtr.ptr, sym.devLumpStart.ptr,
            sym.devChainRowSpan.ptr, sym.devSpanStart.ptr, sym.devChainData.ptr,
            sym.devSpanToLump.ptr, sym.devSpanOffsetInLump.ptr, data,
            lumpsBegin, lumpsEnd, elim.makeBlockPairEnumStraight.ptr,
            elim.numBlockPairs);
#endif
        /*for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
            factorLump(skel, data, l);
        }*/

#if 0  // not yet
        int64_t numElimRows = elim.rowPtr.size() - 1;
        int64_t numSpans = skel.spanStart.size() - 1;
        std::vector<T> tempBuffer(elim.maxBufferSize);
        std::vector<int64_t> spanToChainOffset(numSpans);
        for (int64_t sRel = 0UL; sRel < numElimRows; sRel++) {
            eliminateRowChain(elim, skel, data, sRel, spanToChainOffset,
                              tempBuffer);
        }
#endif
    }

    virtual void potrf(int64_t n, T* A) override {
        // cout << "potrf n=" << n << endl;
        OpInstance timer(sym.potrfStat);
        sym.potrfBiggestN = std::max(sym.potrfBiggestN, n);

        int workspaceSize;
        cusolverCHECK(cusolverDnDpotrf_bufferSize(
            sym.cusolverDnH, CUBLAS_FILL_MODE_UPPER, n, A, n, &workspaceSize));

        T* workspace;
        int* devInfo;
        cuCHECK(cudaMalloc((void**)&workspace, workspaceSize * sizeof(T)));
        cuCHECK(cudaMalloc((void**)&devInfo, 1 * sizeof(int)));

        cusolverCHECK(cusolverDnDpotrf(sym.cusolverDnH, CUBLAS_FILL_MODE_UPPER,
                                       n, A, n, workspace, workspaceSize,
                                       devInfo));

        int info;
        cuCHECK(cudaMemcpy(&info, devInfo, 1 * sizeof(int),
                           cudaMemcpyDeviceToHost));
        cuCHECK(cudaFree(devInfo));
        cuCHECK(cudaFree(workspace));
    }

    virtual void trsm(int64_t n, int64_t k, const T* A, T* B) override {
        // cout << "trsm n=" << n << ", k=" << k << endl;
        OpInstance timer(sym.trsmStat);

        T alpha(1.0);
        cublasCHECK(cublasDtrsm(
            sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
            CUBLAS_DIAG_NON_UNIT, n, k, &alpha, A, n, B, n));
    }

    virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const T* data,
                              int64_t offset) override {
        // cout << "gemm m=" << m << ", n=" << n << ", k=" << k << endl;
        OpInstance timer(sym.sygeStat);

        T alpha(1.0), beta(0.0);
        cublasCHECK(cublasDgemm(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, m, n, k,
                                &alpha, data + offset, k, data + offset, k,
                                &beta, devTempBuffer, m));

        sym.gemmCalls++;
    }

    virtual void saveSyrkGemmBatched(int64_t* ms, int64_t* ns, int64_t* ks,
                                     const T* data, int64_t* offsets,
                                     int batchSize) {
        BASPACHO_CHECK(!"Batching not supported");
    }

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
                          int64_t numBlockRows, int64_t numBlockCols,
                          int numBatch = -1) override {
        // cout << "assembl rows=" << numBlockRows << ", cols=" << numBlockCols
        //     << endl;
        // BASPACHO_CHECK(!"Not implemented yet!");

        BASPACHO_CHECK_EQ(numBatch, -1);  // batching not supported
        OpInstance timer(sym.asmblStat);
        const int64_t* pChainRowsTillEnd =
            sym.devChainRowsTillEnd.ptr + srcColDataOffset;
        const int64_t* pToSpan = sym.devChainRowSpan.ptr + srcColDataOffset;
        const int64_t* pSpanToChainOffset = devSpanToChainOffset;
        const int64_t* pSpanOffsetInLump = sym.devSpanOffsetInLump.ptr;

        const T* matRectPtr = devTempBuffer;

        int wgs = 32;
        int numGroups = (numBlockRows * numBlockCols + wgs - 1) / wgs;
        assemble_kernel<double><<<numGroups, wgs>>>(
            numBlockRows, numBlockCols, rectRowBegin, srcRectWidth, dstStride,
            pChainRowsTillEnd, pToSpan, pSpanToChainOffset, pSpanOffsetInLump,
            matRectPtr, data);
    }

    T* devTempBuffer;
    int64_t* devSpanToChainOffset;
    std::vector<int64_t> spanToChainOffset;

    const CudaSymbolicCtx& sym;
};

template <typename T>
struct CudaSolveCtx : SolveCtx<T> {
    CudaSolveCtx(const CudaSymbolicCtx& sym) : sym(sym) {}
    virtual ~CudaSolveCtx() override {}

    virtual void solveL(const T* data, int64_t offM, int64_t n, T* C,
                        int64_t offC, int64_t ldc, int64_t nRHS) override {
        // BASPACHO_CHECK(!"Not implemented yet!");
#if 0
        Eigen::Map<const MatRMaj<T>> matA(data + offM, n, n);
        OuterStridedCMajMatM<T> matC(C + offC, n, nRHS, OuterStride(ldc));
        matA.template triangularView<Eigen::Lower>().solveInPlace(matC);
#endif
    }

    virtual void gemv(const T* data, int64_t offM, int64_t nRows, int64_t nCols,
                      const T* A, int64_t offA, int64_t lda, T* C,
                      int64_t nRHS) override {
        // BASPACHO_CHECK(!"Not implemented yet!");
#if 0
        Eigen::Map<const MatRMaj<T>> matM(data + offM, nRows, nCols);
        OuterStridedCMajMatK<T> matA(A + offA, nCols, nRHS, OuterStride(lda));
        Eigen::Map<MatRMaj<T>> matC(C, nRows, nRHS);
        matC.noalias() = matM * matA;
#endif
    }

    virtual void assembleVec(const T* A, int64_t chainColPtr,
                             int64_t numColItems, T* C, int64_t ldc,
                             int64_t nRHS) override {
        // BASPACHO_CHECK(!"Not implemented yet!");
#if 0
        const CoalescedBlockMatrixSkel& skel = sym.skel;
        const int64_t* chainRowsTillEnd =
            skel.chainRowsTillEnd.data() + chainColPtr;
        const int64_t* toSpan = skel.chainRowSpan.data() + chainColPtr;
        int64_t startRow = chainRowsTillEnd[-1];
        for (int64_t i = 0; i < numColItems; i++) {
            int64_t rowOffset = chainRowsTillEnd[i - 1] - startRow;
            int64_t span = toSpan[i];
            int64_t spanStart = skel.spanStart[span];
            int64_t spanSize = skel.spanStart[span + 1] - spanStart;

            Eigen::Map<const MatRMaj<T>> matA(A + rowOffset * nRHS, spanSize,
                                              nRHS);
            OuterStridedCMajMatM<T> matC(C + spanStart, spanSize, nRHS,
                                         OuterStride(ldc));
            matC -= matA;
        }
#endif
    }

    virtual void solveLt(const T* data, int64_t offM, int64_t n, T* C,
                         int64_t offC, int64_t ldc, int64_t nRHS) override {
        // BASPACHO_CHECK(!"Not implemented yet!");
#if 0
        Eigen::Map<const MatRMaj<T>> matA(data + offM, n, n);
        OuterStridedCMajMatM<T> matC(C + offC, n, nRHS, OuterStride(ldc));
        matA.template triangularView<Eigen::Lower>().adjoint().solveInPlace(
            matC);
#endif
    }

    virtual void gemvT(const T* data, int64_t offM, int64_t nRows,
                       int64_t nCols, const T* C, int64_t nRHS, T* A,
                       int64_t offA, int64_t lda) override {
        // BASPACHO_CHECK(!"Not implemented yet!");
#if 0
        Eigen::Map<const MatRMaj<T>> matM(data + offM, nRows, nCols);
        OuterStridedCMajMatM<T> matA(A + offA, nCols, nRHS, OuterStride(lda));
        Eigen::Map<const MatRMaj<T>> matC(C, nRows, nRHS);
        matA.noalias() -= matM.transpose() * matC;
#endif
    }

    virtual void assembleVecT(const T* C, int64_t ldc, int64_t nRHS, T* A,
                              int64_t chainColPtr,
                              int64_t numColItems) override {
        // BASPACHO_CHECK(!"Not implemented yet!");
#if 0
        const CoalescedBlockMatrixSkel& skel = sym.skel;
        const int64_t* chainRowsTillEnd =
            skel.chainRowsTillEnd.data() + chainColPtr;
        const int64_t* toSpan = skel.chainRowSpan.data() + chainColPtr;
        int64_t startRow = chainRowsTillEnd[-1];
        for (int64_t i = 0; i < numColItems; i++) {
            int64_t rowOffset = chainRowsTillEnd[i - 1] - startRow;
            int64_t span = toSpan[i];
            int64_t spanStart = skel.spanStart[span];
            int64_t spanSize = skel.spanStart[span + 1] - spanStart;

            Eigen::Map<MatRMaj<T>> matA(A + rowOffset * nRHS, spanSize, nRHS);
            OuterStridedCMajMatK<T> matC(C + spanStart, spanSize, nRHS,
                                         OuterStride(ldc));
            matA = matC;
        }
#endif
    }

    const CudaSymbolicCtx& sym;
};

NumericCtxBase* CudaSymbolicCtx::createNumericCtxForType(std::type_index tIdx,
                                                         int64_t tempBufSize,
                                                         int maxBatchSize) {
    if (tIdx == std::type_index(typeid(double))) {
        return new CudaNumericCtx<double>(*this, tempBufSize,
                                          skel.spanStart.size() - 1);
        /*} else if (tIdx == std::type_index(typeid(float))) {
            return new CudaNumericCtx<float>(*this, tempBufSize,
                                             skel.spanStart.size() - 1);*/
    } else {
        return nullptr;
    }
}

SolveCtxBase* CudaSymbolicCtx::createSolveCtxForType(std::type_index tIdx) {
    if (tIdx == std::type_index(typeid(double))) {
        return new CudaSolveCtx<double>(*this);
        /*} else if (tIdx == std::type_index(typeid(float))) {
            return new CudaSolveCtx<float>(*this);*/
    } else {
        return nullptr;
    }
}

OpsPtr cudaOps() { return OpsPtr(new CudaOps); }

}  // end namespace BaSpaCho