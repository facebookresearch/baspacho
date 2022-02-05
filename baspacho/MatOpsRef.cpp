
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

    virtual SolveCtxBase* createSolveCtxForType(std::type_index tIdx,
                                                int nRHS) override;
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
        const CpuBaseSymElimCtx* pElim =
            dynamic_cast<const CpuBaseSymElimCtx*>(&elimData);
        BASPACHO_CHECK_NOTNULL(pElim);
        const CpuBaseSymElimCtx& elim = *pElim;
        OpInstance timer(elim.elimStat);
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

    // This is an alternative of doElimination, which is however written in such
    // a way to mock the sparse elimination as done on the GPU. This is in order
    // to test the logic here, and then create a kernel doing something similar
    void doEliminationMockSparse(const SymElimCtx& elimData, T* data,
                                 int64_t lumpsBegin, int64_t lumpsEnd) {
        const CpuBaseSymElimCtx* pElim =
            dynamic_cast<const CpuBaseSymElimCtx*>(&elimData);
        BASPACHO_CHECK_NOTNULL(pElim);
        const CpuBaseSymElimCtx& elim = *pElim;
        OpInstance timer(elim.elimStat);
        const CoalescedBlockMatrixSkel& skel = sym.skel;

        for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
            factorLump(skel, data, l);
        }

        for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
            int64_t startPtr = skel.chainColPtr[l] + 1;  // skip diag block
            int64_t endPtr = skel.chainColPtr[l + 1];
            int64_t lColSize = skel.lumpStart[l + 1] - skel.lumpStart[l];

            for (int64_t i = startPtr; i < endPtr; i++) {
                int64_t si = skel.chainRowSpan[i];
                int64_t siSize = skel.spanStart[si + 1] - skel.spanStart[si];
                int64_t siDataPtr = skel.chainData[i];
                Eigen::Map<MatRMaj<T>> ilBlock(data + siDataPtr, siSize,
                                               lColSize);

                int64_t targetLump = skel.spanToLump[si];
                int64_t targetSpanOffsetInLump = skel.spanOffsetInLump[si];
                int64_t targetStartPtr =
                    skel.chainColPtr[targetLump];  // skip diag block
                int64_t targetEndPtr = skel.chainColPtr[targetLump + 1];
                int64_t targetLumpSize =
                    skel.lumpStart[targetLump + 1] - skel.lumpStart[targetLump];

                for (int64_t j = i; j < endPtr; j++) {
                    int64_t sj = skel.chainRowSpan[j];
                    int64_t sjSize =
                        skel.spanStart[sj + 1] - skel.spanStart[sj];
                    int64_t sjDataPtr = skel.chainData[j];

                    Eigen::Map<MatRMaj<T>> jlBlock(data + sjDataPtr, sjSize,
                                                   lColSize);

                    uint64_t pos =
                        bisect(skel.chainRowSpan.data() + targetStartPtr,
                               targetEndPtr - targetStartPtr, sj);
                    BASPACHO_CHECK_EQ(skel.chainRowSpan[targetStartPtr + pos],
                                      sj);
                    int64_t jiDataPtr = skel.chainData[targetStartPtr + pos];
                    OuterStridedMatM<T> jiBlock(
                        data + jiDataPtr + targetSpanOffsetInLump, sjSize,
                        siSize, OuterStride(targetLumpSize));
                    jiBlock -= jlBlock * ilBlock.transpose();
                }
            }
        }
    }

    virtual void potrf(int64_t n, T* A) override {
        OpInstance timer(sym.potrfStat);
        sym.potrfBiggestN = std::max(sym.potrfBiggestN, n);

        Eigen::Map<MatRMaj<T>> matA(A, n, n);
        Eigen::LLT<Eigen::Ref<MatRMaj<T>>> llt(matA);
    }

    virtual void trsm(int64_t n, int64_t k, const T* A, T* B) override {
        OpInstance timer(sym.trsmStat);

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
        OpInstance timer(sym.sygeStat);
        BASPACHO_CHECK_LE(m * n, (int64_t)tempBuffer.size());

        const T* AB = data + offset;
        T* C = tempBuffer.data();
        Eigen::Map<const MatRMaj<T>> matA(AB, m, k);
        Eigen::Map<const MatRMaj<T>> matB(AB, n, k);
        Eigen::Map<MatRMaj<T>> matC(C, n, m);
        matC.noalias() = matB * matA.transpose();

        sym.gemmCalls++;
    }

    virtual void saveSyrkGemmBatched(int64_t* ms, int64_t* ns, int64_t* ks,
                                     const T* data, int64_t* offsets,
                                     int batchSize) {
        UNUSED(ms, ns, ks, data, offsets, batchSize);
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
        OpInstance timer(sym.asmblStat);
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

    const SimpleSymbolicCtx& sym;
};

template <typename T>
struct SimpleSolveCtx : SolveCtx<T> {
    SimpleSolveCtx(const SimpleSymbolicCtx& sym, int nRHS)
        : sym(sym), nRHS(nRHS), tmpBuf(sym.skel.order() * nRHS) {}
    virtual ~SimpleSolveCtx() override {}

    virtual void solveL(const T* data, int64_t offM, int64_t n, T* C,
                        int64_t offC,
                        int64_t ldc /* , int64_t nRHS */) override {
        Eigen::Map<const MatRMaj<T>> matA(data + offM, n, n);
        OuterStridedCMajMatM<T> matC(C + offC, n, nRHS, OuterStride(ldc));
        matA.template triangularView<Eigen::Lower>().solveInPlace(matC);
    }

    virtual void gemv(const T* data, int64_t offM, int64_t nRows, int64_t nCols,
                      const T* A, int64_t offA, int64_t lda) override {
        Eigen::Map<const MatRMaj<T>> matM(data + offM, nRows, nCols);
        OuterStridedCMajMatK<T> matA(A + offA, nCols, nRHS, OuterStride(lda));
        Eigen::Map<MatRMaj<T>> matC(tmpBuf.data(), nRows, nRHS);
        matC.noalias() = matM * matA;
    }

    virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, T* C,
                             int64_t ldc) override {
        const T* A = tmpBuf.data();
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
                         int64_t offC, int64_t ldc) override {
        Eigen::Map<const MatRMaj<T>> matA(data + offM, n, n);
        OuterStridedCMajMatM<T> matC(C + offC, n, nRHS, OuterStride(ldc));
        matA.template triangularView<Eigen::Lower>().adjoint().solveInPlace(
            matC);
    }

    virtual void gemvT(const T* data, int64_t offM, int64_t nRows,
                       int64_t nCols, T* A, int64_t offA,
                       int64_t lda) override {
        Eigen::Map<const MatRMaj<T>> matM(data + offM, nRows, nCols);
        OuterStridedCMajMatM<T> matA(A + offA, nCols, nRHS, OuterStride(lda));
        Eigen::Map<const MatRMaj<T>> matC(tmpBuf.data(), nRows, nRHS);
        matA.noalias() -= matM.transpose() * matC;
    }

    virtual void assembleVecT(const T* C, int64_t ldc, int64_t chainColPtr,
                              int64_t numColItems) override {
        T* A = tmpBuf.data();
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
    int64_t nRHS;
    vector<T> tmpBuf;
};

NumericCtxBase* SimpleSymbolicCtx::createNumericCtxForType(
    std::type_index tIdx, int64_t tempBufSize, int /* maxBatchSize */) {
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

SolveCtxBase* SimpleSymbolicCtx::createSolveCtxForType(std::type_index tIdx,
                                                       int nRHS) {
    if (tIdx == std::type_index(typeid(double))) {
        return new SimpleSolveCtx<double>(*this, nRHS);
    } else if (tIdx == std::type_index(typeid(float))) {
        return new SimpleSolveCtx<float>(*this, nRHS);
    } else {
        return nullptr;
    }
}

OpsPtr simpleOps() { return OpsPtr(new SimpleOps); }

}  // end namespace BaSpaCho