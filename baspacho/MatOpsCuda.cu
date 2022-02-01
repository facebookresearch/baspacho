#pragma nv_diag_suppress 20236
#pragma nv_diag_suppress 20012

#include <chrono>
#include <iostream>

#include "baspacho/CudaDefs.h"
#include "baspacho/DebugMacros.h"
#include "baspacho/MatOpsCpuBase.h"
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

struct CudaSymbolicCtx : CpuBaseSymbolicCtx {
    CudaSymbolicCtx(const CoalescedBlockMatrixSkel& skel)
        : CpuBaseSymbolicCtx(skel) {
        cublasCHECK(cublasCreate(&cublasH));
        // cublasCHECK(cublasSetStream(cublasH, stream));
        cusolverCHECK(cusolverDnCreate(&cusolverDnH));
        // cusolverCHECK(cusolverDnSetStream(cusolverDnH, stream));

        cuCHECK(cudaMalloc((void**)&devChainRowsTillEnd,
                           skel.chainRowsTillEnd.size() * sizeof(int64_t)));
        cuCHECK(cudaMemcpy(devChainRowsTillEnd, skel.chainRowsTillEnd.data(),
                           skel.chainRowsTillEnd.size() * sizeof(int64_t),
                           cudaMemcpyHostToDevice));
        cuCHECK(cudaMalloc((void**)&devChainRowSpan,
                           skel.chainRowSpan.size() * sizeof(int64_t)));
        cuCHECK(cudaMemcpy(devChainRowSpan, skel.chainRowSpan.data(),
                           skel.chainRowSpan.size() * sizeof(int64_t),
                           cudaMemcpyHostToDevice));
        cuCHECK(cudaMalloc((void**)&devSpanOffsetInLump,
                           skel.spanOffsetInLump.size() * sizeof(int64_t)));
        cuCHECK(cudaMemcpy(devSpanOffsetInLump, skel.spanOffsetInLump.data(),
                           skel.spanOffsetInLump.size() * sizeof(int64_t),
                           cudaMemcpyHostToDevice));
    }

    virtual ~CudaSymbolicCtx() override {
        if (cublasH) {
            cublasCHECK(cublasDestroy(cublasH));
        }
        if (cusolverDnH) {
            cusolverCHECK(cusolverDnDestroy(cusolverDnH));
        }
        if (devChainRowsTillEnd) {
            cuCHECK(cudaFree(devChainRowsTillEnd));
        }
        if (devChainRowSpan) {
            cuCHECK(cudaFree(devChainRowSpan));
        }
        if (devSpanOffsetInLump) {
            cuCHECK(cudaFree(devSpanOffsetInLump));
        }
    }

    virtual SymElimCtxPtr prepareElimination(int64_t lumpsBegin,
                                             int64_t lumpsEnd) override {
        BASPACHO_CHECK(!"Not implemented yet!");
        return SymElimCtxPtr();
    }

    virtual NumericCtxBase* createNumericCtxForType(
        std::type_index tIdx, int64_t tempBufSize,
        int maxBatchSize = 1) override;

    virtual SolveCtxBase* createSolveCtxForType(std::type_index tIdx) override;

    cublasHandle_t cublasH = nullptr;
    cusolverDnHandle_t cusolverDnH = nullptr;

    int64_t* devChainRowsTillEnd = nullptr;
    int64_t* devChainRowSpan = nullptr;
    int64_t* devSpanOffsetInLump = nullptr;
};

// cuda ops implemented using CUBLAS and custom kernels
struct CudaOps : Ops {
    virtual SymbolicCtxPtr createSymbolicCtx(
        const CoalescedBlockMatrixSkel& skel) override {
        return SymbolicCtxPtr(new CudaSymbolicCtx(skel));
    }
};

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
    if (i > numBlockRows + numBlockCols) {
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

    virtual void printStats() const override {
        std::cout << "matOp stats:"
                  << "\nelim: " << elimStat.toString()
                  << "\nBiggest dense block: " << potrfBiggestN
                  << "\npotrf: " << potrfStat.toString()
                  << "\ntrsm: " << trsmStat.toString()  //
                  << "\nsyrk/gemm(" << syrkCalls << "+" << gemmCalls
                  << "): " << sygeStat.toString()
                  << "\nasmbl: " << asmblStat.toString() << std::endl;
    }

    virtual void doElimination(const SymElimCtx& elimData, T* data,
                               int64_t lumpsBegin, int64_t lumpsEnd) override {
        BASPACHO_CHECK(!"Not implemented yet!");
#if 0
        OpInstance timer(elimStat);
        const CpuBaseSymElimCtx* pElim =
            dynamic_cast<const CpuBaseSymElimCtx*>(&elimData);
        BASPACHO_CHECK_NOTNULL(pElim);
        const CpuBaseSymElimCtx& elim = *pElim;
        const CoalescedBlockMatrixSkel& skel = sym.skel;

        for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
            factorLump(skel, data, l);
        }

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
        OpInstance timer(potrfStat);

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
        OpInstance timer(trsmStat);

        T alpha(1.0);
        cublasCHECK(cublasDtrsm(
            sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
            CUBLAS_DIAG_NON_UNIT, n, k, &alpha, A, n, B, n));
    }

    virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const T* data,
                              int64_t offset) override {
        T alpha(1.0), beta(0.0);
        cublasCHECK(cublasDgemm(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, m, n, k,
                                &alpha, data + offset, k, data + offset, k,
                                &beta, devTempBuffer, m));
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
        // BASPACHO_CHECK(!"Not implemented yet!");

        BASPACHO_CHECK_EQ(numBatch, -1);  // batching not supported
        OpInstance timer(asmblStat);
        const int64_t* pChainRowsTillEnd =
            sym.devChainRowsTillEnd + srcColDataOffset;
        const int64_t* pToSpan = sym.devChainRowSpan + srcColDataOffset;
        const int64_t* pSpanToChainOffset = devSpanToChainOffset;
        const int64_t* pSpanOffsetInLump = sym.devSpanOffsetInLump;

        const T* matRectPtr = devTempBuffer;

        int wgs = 32;
        int numGroups = (numBlockRows * numBlockCols + wgs - 1) / wgs;
        assemble_kernel<double><<<numGroups, wgs>>>(
            numBlockRows, numBlockCols, rectRowBegin, srcRectWidth, dstStride,
            pChainRowsTillEnd, pToSpan, pSpanToChainOffset, pSpanOffsetInLump,
            matRectPtr, data);
    }

    OpStat elimStat;
    OpStat potrfStat;
    int64_t potrfBiggestN = 0;
    OpStat trsmStat;
    OpStat sygeStat;
    int64_t gemmCalls = 0;
    int64_t syrkCalls = 0;
    OpStat asmblStat;

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