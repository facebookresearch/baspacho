
#include <chrono>
#include <iostream>

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

struct SimpleSymbolicCtx : CpuBaseSymbolicCtx {
    SimpleSymbolicCtx(const CoalescedBlockMatrixSkel& skel)
        : CpuBaseSymbolicCtx(skel) {}

    virtual NumericCtxBase* createNumericCtxForType(
        std::type_index tIdx, int64_t tempBufSize,
        int maxBatchSize = 1) override;

    virtual SolveCtxBase* createSolveCtxForType(std::type_index tIdx) override;
};

// simple ops implemented using Eigen (therefore single thread)
struct SimpleOps : Ops {
    virtual SymbolicCtxPtr createSymbolicCtx(
        const CoalescedBlockMatrixSkel& skel) override {
        return SymbolicCtxPtr(new SimpleSymbolicCtx(skel));
    }
};

template <typename T>
struct SimpleNumericCtx : CpuBaseNumericCtx<T> {
    SimpleNumericCtx(const SimpleSymbolicCtx& sym, int64_t bufSize,
                     int64_t numSpans)
        : CpuBaseNumericCtx<T>(bufSize, numSpans), sym(sym) {}

    virtual void doElimination(const SymElimCtx& elimData, T* data,
                               int64_t lumpsBegin, int64_t lumpsEnd) override {
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
    }

    virtual void potrf(int64_t n, T* A) override {
        OpInstance timer(potrfStat);

        Eigen::Map<MatRMaj<T>> matA(A, n, n);
        Eigen::LLT<Eigen::Ref<MatRMaj<T>>> llt(matA);

        potrfBiggestN = std::max(potrfBiggestN, n);
    }

    virtual void trsm(int64_t n, int64_t k, const T* A, T* B) override {
        OpInstance timer(trsmStat);

        using MatCMajD =
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

        // col-major's upper = (row-major's lower).transpose()
        Eigen::Map<const MatCMajD> matA(A, n, n);
        Eigen::Map<MatRMaj<T>> matB(B, k, n);
        matA.template triangularView<Eigen::Upper>()
            .template solveInPlace<Eigen::OnTheRight>(matB);
    }

    virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const T* data,
                              int64_t offset) override {
        OpInstance timer(sygeStat);
        BASPACHO_CHECK_LE(m * n, tempBuffer.size());

        const T* AB = data + offset;
        T* C = tempBuffer.data();
        Eigen::Map<const MatRMaj<T>> matA(AB, m, k);
        Eigen::Map<const MatRMaj<T>> matB(AB, n, k);
        Eigen::Map<MatRMaj<T>> matC(C, n, m);
        matC.noalias() = matB * matA.transpose();
    }

    virtual void saveSyrkGemmBatched(int64_t* ms, int64_t* ns, int64_t* ks,
                                     const T* data, int64_t* offsets,
                                     int batchSize) {
        BASPACHO_CHECK(!"Batching not supported");
    }

    virtual void prepareAssemble(int64_t targetLump) override {
        const CoalescedBlockMatrixSkel& skel = sym.skel;

        for (int64_t i = skel.chainColPtr[targetLump],
                     iEnd = skel.chainColPtr[targetLump + 1];
             i < iEnd; i++) {
            spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
        }
    }

    virtual void assemble(T* data, int64_t rectRowBegin,
                          int64_t dstStride,  //
                          int64_t srcColDataOffset, int64_t srcRectWidth,
                          int64_t numBlockRows, int64_t numBlockCols,
                          int numBatch = -1) override {
        BASPACHO_CHECK_EQ(numBatch, -1);  // batching not supported
        OpInstance timer(asmblStat);
        const CoalescedBlockMatrixSkel& skel = sym.skel;
        const int64_t* chainRowsTillEnd =
            skel.chainRowsTillEnd.data() + srcColDataOffset;
        const int64_t* pToSpan = skel.chainRowSpan.data() + srcColDataOffset;
        const int64_t* pSpanToChainOffset = spanToChainOffset.data();
        const int64_t* pSpanOffsetInLump = skel.spanOffsetInLump.data();

        const T* matRectPtr = tempBuffer.data();

        for (int64_t r = 0; r < numBlockRows; r++) {
            int64_t rBegin = chainRowsTillEnd[r - 1] - rectRowBegin;
            int64_t rSize = chainRowsTillEnd[r] - rBegin - rectRowBegin;
            int64_t rParam = pToSpan[r];
            int64_t rOffset = pSpanToChainOffset[rParam];
            const T* matRowPtr = matRectPtr + rBegin * srcRectWidth;

            int64_t cEnd = std::min(numBlockCols, r + 1);
            for (int64_t c = 0; c < cEnd; c++) {
                int64_t cStart = chainRowsTillEnd[c - 1] - rectRowBegin;
                int64_t cSize = chainRowsTillEnd[c] - cStart - rectRowBegin;
                int64_t offset = rOffset + pSpanOffsetInLump[pToSpan[c]];

                T* dst = data + offset;
                const T* src = matRowPtr + cStart;
                stridedMatSub(dst, dstStride, src, srcRectWidth, rSize, cSize);
            }
        }
    }

    using CpuBaseNumericCtx<T>::factorLump;
    using CpuBaseNumericCtx<T>::eliminateRowChain;
    using CpuBaseNumericCtx<T>::stridedMatSub;

    using CpuBaseNumericCtx<T>::tempBuffer;
    using CpuBaseNumericCtx<T>::spanToChainOffset;

    using CpuBaseNumericCtx<T>::elimStat;
    using CpuBaseNumericCtx<T>::potrfStat;
    using CpuBaseNumericCtx<T>::potrfBiggestN;
    using CpuBaseNumericCtx<T>::trsmStat;
    using CpuBaseNumericCtx<T>::sygeStat;
    using CpuBaseNumericCtx<T>::gemmCalls;
    using CpuBaseNumericCtx<T>::syrkCalls;
    using CpuBaseNumericCtx<T>::asmblStat;

    const SimpleSymbolicCtx& sym;
};

template <typename T>
struct SimpleSolveCtx : SolveCtx<T> {
    SimpleSolveCtx(const SimpleSymbolicCtx& sym) : sym(sym) {}
    virtual ~SimpleSolveCtx() override {}

    virtual void solveL(const T* data, int64_t offM, int64_t n, T* C,
                        int64_t offC, int64_t ldc, int64_t nRHS) override {
        Eigen::Map<const MatRMaj<T>> matA(data + offM, n, n);
        OuterStridedCMajMatM<T> matC(C + offC, n, nRHS, OuterStride(ldc));
        matA.template triangularView<Eigen::Lower>().solveInPlace(matC);
    }

    virtual void gemv(const T* data, int64_t offM, int64_t nRows, int64_t nCols,
                      const T* A, int64_t offA, int64_t lda, T* C,
                      int64_t nRHS) override {
        Eigen::Map<const MatRMaj<T>> matM(data + offM, nRows, nCols);
        OuterStridedCMajMatK<T> matA(A + offA, nCols, nRHS, OuterStride(lda));
        Eigen::Map<MatRMaj<T>> matC(C, nRows, nRHS);
        matC.noalias() = matM * matA;
    }

    virtual void assembleVec(const T* A, int64_t chainColPtr,
                             int64_t numColItems, T* C, int64_t ldc,
                             int64_t nRHS) override {
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
    }

    virtual void solveLt(const T* data, int64_t offM, int64_t n, T* C,
                         int64_t offC, int64_t ldc, int64_t nRHS) override {
        Eigen::Map<const MatRMaj<T>> matA(data + offM, n, n);
        OuterStridedCMajMatM<T> matC(C + offC, n, nRHS, OuterStride(ldc));
        matA.template triangularView<Eigen::Lower>().adjoint().solveInPlace(
            matC);
    }

    virtual void gemvT(const T* data, int64_t offM, int64_t nRows,
                       int64_t nCols, const T* C, int64_t nRHS, T* A,
                       int64_t offA, int64_t lda) override {
        Eigen::Map<const MatRMaj<T>> matM(data + offM, nRows, nCols);
        OuterStridedCMajMatM<T> matA(A + offA, nCols, nRHS, OuterStride(lda));
        Eigen::Map<const MatRMaj<T>> matC(C, nRows, nRHS);
        matA.noalias() -= matM.transpose() * matC;
    }

    virtual void assembleVecT(const T* C, int64_t ldc, int64_t nRHS, T* A,
                              int64_t chainColPtr,
                              int64_t numColItems) override {
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
    }

    const SimpleSymbolicCtx& sym;
};

NumericCtxBase* SimpleSymbolicCtx::createNumericCtxForType(std::type_index tIdx,
                                                           int64_t tempBufSize,
                                                           int maxBatchSize) {
    if (tIdx == std::type_index(typeid(double))) {
        return new SimpleNumericCtx<double>(*this, tempBufSize,
                                            skel.spanStart.size() - 1);
    } else if (tIdx == std::type_index(typeid(float))) {
        return new SimpleNumericCtx<float>(*this, tempBufSize,
                                           skel.spanStart.size() - 1);
    } else {
        return nullptr;
    }
}

SolveCtxBase* SimpleSymbolicCtx::createSolveCtxForType(std::type_index tIdx) {
    if (tIdx == std::type_index(typeid(double))) {
        return new SimpleSolveCtx<double>(*this);
    } else if (tIdx == std::type_index(typeid(float))) {
        return new SimpleSolveCtx<float>(*this);
    } else {
        return nullptr;
    }
}

OpsPtr simpleOps() { return OpsPtr(new SimpleOps); }

}  // end namespace BaSpaCho