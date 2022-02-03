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

// returns all pairs (x, y) with 0 <= x <= y < n, while p varies in 0 <= p <
// n*(n+1)/2
__device__ inline std::pair<int64_t, int64_t> toOrderedPair(int64_t n,
                                                            int64_t p) {
    // the trick below converts p that varies in the range 0,1,...,n*(n+1)/2
    // to a pair x<=y, where y varies in the range 0,1,...,(n-1) and,
    // for each y, x varies in the range 0,1,...,y
    // furthermore, x increases sequentially in all pairs that are generated,
    // and this will optimize the memory accesses
    int64_t odd = n & 1;
    int64_t m = n + 1 - odd;
    int64_t x = p % m;            // here: x = 0,1,...,n-1
    int64_t y = n - 1 - (p / m);  // here: y = n-1,n-2,...,floor(n/2)
    if (x > y) {  // flip the triangle formed by points with x>y, and move it
                  // close to the origin
        x = x - y - 1;
        y = n - 1 - odd - y;
    }
    return std::make_pair(x, y);
}

struct CudaSymElimCtx : SymElimCtx {
    CudaSymElimCtx() {}
    virtual ~CudaSymElimCtx() override {}

#if 0  // TODO
    // per-row pointers to chains in a rectagle:
    // * span-rows from lumpToSpan[lumpsEnd],
    // * board cols in interval lumpsBegin:lumpsEnd
    int64_t spanRowBegin;
    int64_t maxBufferSize;
    std::vector<int64_t> rowPtr;       // row data pointer
    std::vector<int64_t> colLump;      // col-lump
    std::vector<int64_t> chainColOrd;  // order in col chain elements
#endif
};

template <typename T>
struct DevMirror {
    DevMirror() {}
    ~DevMirror() {
        if (ptr) {
            cuCHECK(cudaFree(ptr));
        }
    }
    void load(const vector<T>& vec) {
        if (ptr) {
            cuCHECK(cudaFree(ptr));
        }
        cuCHECK(cudaMalloc((void**)&ptr, vec.size() * sizeof(T)));
        cuCHECK(cudaMemcpy(ptr, vec.data(), vec.size() * sizeof(T),
                           cudaMemcpyHostToDevice));
    }
    T* ptr = nullptr;
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

        // TODO

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