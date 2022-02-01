
#include <chrono>
#include <iostream>

#include "DebugMacros.h"
#include "MatOpsCpuBase.h"
#include "Utils.h"

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

    virtual NumericCtxPtr<double> createDoubleContext(
        uint64_t tempBufSize, int maxBatchSize = 1) override;

    virtual SolveCtxPtr<double> createDoubleSolveContext() override;
};

// simple ops implemented using Eigen (therefore single thread)
struct SimpleOps : Ops {
    virtual SymbolicCtxPtr initSymbolicInfo(
        const CoalescedBlockMatrixSkel& skel) override {
        return SymbolicCtxPtr(new SimpleSymbolicCtx(skel));
    }
};

template <typename T>
struct SimpleNumericCtx : CpuBaseNumericCtx<T> {
    SimpleNumericCtx(const SimpleSymbolicCtx& sym, uint64_t bufSize,
                     uint64_t numSpans)
        : CpuBaseNumericCtx<T>(bufSize, numSpans), sym(sym) {}

    virtual void doElimination(const SymElimCtx& elimData, T* data,
                               uint64_t lumpsBegin,
                               uint64_t lumpsEnd) override {
        OpInstance timer(elimStat);
        const CpuBaseSymElimCtx* pElim =
            dynamic_cast<const CpuBaseSymElimCtx*>(&elimData);
        BASPACHO_CHECK_NOTNULL(pElim);
        const CpuBaseSymElimCtx& elim = *pElim;
        const CoalescedBlockMatrixSkel& skel = sym.skel;

        for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
            factorLump(skel, data, l);
        }

        uint64_t numElimRows = elim.rowPtr.size() - 1;
        uint64_t numSpans = skel.spanStart.size() - 1;
        std::vector<T> tempBuffer(elim.maxBufferSize);
        std::vector<uint64_t> spanToChainOffset(numSpans);
        for (uint64_t sRel = 0UL; sRel < numElimRows; sRel++) {
            eliminateRowChain(elim, skel, data, sRel, spanToChainOffset,
                              tempBuffer);
        }
    }

    /*virtual void doEliminationQ(const OpaqueData& info, T* data,
                                uint64_t lumpsBegin, uint64_t lumpsEnd,
                                const OpaqueData& elimData) {
        OpInstance timer(elimStat);
        const SimpleSymbolicInfo* pInfo =
            dynamic_cast<const SimpleSymbolicInfo*>(&info);
        const SparseEliminationInfo* pElim =
            dynamic_cast<const SparseEliminationInfo*>(&elimData);
        BASPACHO_CHECK_NOTNULL(pInfo);
        BASPACHO_CHECK_NOTNULL(pElim);
        const CoalescedBlockMatrixSkel& skel = pInfo->skel;
        const SparseEliminationInfo& elim = *pElim;

        for (uint64_t a = lumpsBegin; a < lumpsEnd; a++) {
            factorLump(skel, data, a);
        }

        std::vector<T> tempBuffer;                // ctx
        std::vector<uint64_t> spanToChainOffset;  // ctx

        uint64_t spanRowBegin = skel.lumpToSpan[lumpsEnd];
        for (uint64_t sRel = 0; sRel < elim.rowPtr.size() - 1; sRel++) {
            uint64_t s = sRel + spanRowBegin;
            uint64_t targetLump = skel.spanToLump[s];
            uint64_t targetLumpSize =
                skel.lumpStart[targetLump + 1] - skel.lumpStart[targetLump];
            uint64_t spanOffsetInLump =
                skel.spanStart[s] - skel.lumpStart[targetLump];
            prepareContextForTargetLump(skel, targetLump, spanToChainOffset);

            // iterate over chains present in this row
            for (uint64_t i = elim.rowPtr[sRel], iEnd = elim.rowPtr[sRel + 1];
                 i < iEnd; i++) {
                uint64_t lump = elim.colLump[i];
                uint64_t chainColOrd = elim.chainColOrd[i];
                BASPACHO_CHECK_GE(chainColOrd,
                                  1);  // there must be a diagonal block

                uint64_t ptrStart = skel.chainColPtr[lump] + chainColOrd;
                uint64_t ptrEnd = skel.chainColPtr[lump + 1];
                BASPACHO_CHECK_EQ(skel.chainRowSpan[ptrStart], s);

                uint64_t nRowsAbove = skel.chainRowsTillEnd[ptrStart - 1];
                uint64_t nRowsChain =
                    skel.chainRowsTillEnd[ptrStart] - nRowsAbove;
                uint64_t nRowsOnward = skel.chainRowsTillEnd[ptrEnd - 1];
                uint64_t dataOffset = skel.chainData[ptrStart];
                BASPACHO_CHECK_EQ(nRowsChain,
                                  skel.spanStart[s + 1] - skel.spanStart[s]);
                uint64_t lumpSize =
                    skel.lumpStart[lump + 1] - skel.lumpStart[lump];

                Eigen::Map<MatRMaj<T>> chainSubMat(data + dataOffset,
                                                   nRowsChain, lumpSize);
                Eigen::Map<MatRMaj<T>> chainOnwardSubMat(data + dataOffset,
                                                         nRowsOnward, lumpSize);

                tempBuffer.resize(nRowsOnward * nRowsChain);
                Eigen::Map<MatRMaj<T>> prod(tempBuffer.data(), nRowsOnward,
                                            nRowsChain);
                prod = chainOnwardSubMat * chainSubMat.transpose();

                // assemble blocks, iterating on chain and below chains
                for (uint64_t ptr = ptrStart; ptr < ptrEnd; ptr++) {
                    uint64_t s2 = skel.chainRowSpan[ptr];
                    uint64_t relRow =
                        skel.chainRowsTillEnd[ptr - 1] - nRowsAbove;
                    uint64_t s2_size =
                        skel.chainRowsTillEnd[ptr] - nRowsAbove - relRow;
                    BASPACHO_CHECK_EQ(
                        s2_size, skel.spanStart[s2 + 1] - skel.spanStart[s2]);

                    T* targetData =
                        data + spanOffsetInLump + spanToChainOffset[s2];
                    BASPACHO_CHECK(spanToChainOffset[s2] != kInvalid);

                    OuterStridedMatM targetBlock(targetData, s2_size,
                                                 nRowsChain,
                                                 OuterStride(targetLumpSize));
                    targetBlock -= prod.block(relRow, 0, s2_size, nRowsChain);
                }
            }
        }
    }*/

    virtual void potrf(uint64_t n, T* A) override {
        OpInstance timer(potrfStat);

        Eigen::Map<MatRMaj<T>> matA(A, n, n);
        Eigen::LLT<Eigen::Ref<MatRMaj<T>>> llt(matA);

        potrfBiggestN = std::max(potrfBiggestN, n);
    }

    virtual void trsm(uint64_t n, uint64_t k, const T* A, T* B) override {
        OpInstance timer(trsmStat);

        using MatCMajD =
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

        // col-major's upper = (row-major's lower).transpose()
        Eigen::Map<const MatCMajD> matA(A, n, n);
        Eigen::Map<MatRMaj<T>> matB(B, k, n);
        matA.template triangularView<Eigen::Upper>()
            .template solveInPlace<Eigen::OnTheRight>(matB);
    }

    virtual void saveSyrkGemm(uint64_t m, uint64_t n, uint64_t k, const T* data,
                              uint64_t offset) override {
        OpInstance timer(sygeStat);
        BASPACHO_CHECK_LE(m * n, tempBuffer.size());

        const T* AB = data + offset;
        T* C = tempBuffer.data();
        Eigen::Map<const MatRMaj<T>> matA(AB, m, k);
        Eigen::Map<const MatRMaj<T>> matB(AB, n, k);
        Eigen::Map<MatRMaj<T>> matC(C, n, m);
        matC.noalias() = matB * matA.transpose();
    }

    virtual void saveSyrkGemmBatched(uint64_t* ms, uint64_t* ns, uint64_t* ks,
                                     const T* data, uint64_t* offsets,
                                     int batchSize) {
        BASPACHO_CHECK(!"Batching not supported");
    }

    virtual void prepareAssemble(uint64_t targetLump) override {
        const CoalescedBlockMatrixSkel& skel = sym.skel;

        for (uint64_t i = skel.chainColPtr[targetLump],
                      iEnd = skel.chainColPtr[targetLump + 1];
             i < iEnd; i++) {
            spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
        }
    }

    virtual void assemble(T* data, uint64_t rectRowBegin,
                          uint64_t dstStride,  //
                          uint64_t srcColDataOffset, uint64_t srcRectWidth,
                          uint64_t numBlockRows, uint64_t numBlockCols,
                          int numBatch = -1) override {
        BASPACHO_CHECK_EQ(numBatch, -1);  // batching not supported
        OpInstance timer(asmblStat);
        const CoalescedBlockMatrixSkel& skel = sym.skel;
        const uint64_t* chainRowsTillEnd =
            skel.chainRowsTillEnd.data() + srcColDataOffset;
        const uint64_t* pToSpan = skel.chainRowSpan.data() + srcColDataOffset;
        const uint64_t* pSpanToChainOffset = spanToChainOffset.data();
        const uint64_t* pSpanOffsetInLump = skel.spanOffsetInLump.data();

        const T* matRectPtr = tempBuffer.data();

        for (uint64_t r = 0; r < numBlockRows; r++) {
            uint64_t rBegin = chainRowsTillEnd[r - 1] - rectRowBegin;
            uint64_t rSize = chainRowsTillEnd[r] - rBegin - rectRowBegin;
            uint64_t rParam = pToSpan[r];
            uint64_t rOffset = pSpanToChainOffset[rParam];
            const T* matRowPtr = matRectPtr + rBegin * srcRectWidth;

            uint64_t cEnd = std::min(numBlockCols, r + 1);
            for (uint64_t c = 0; c < cEnd; c++) {
                uint64_t cStart = chainRowsTillEnd[c - 1] - rectRowBegin;
                uint64_t cSize = chainRowsTillEnd[c] - cStart - rectRowBegin;
                uint64_t offset = rOffset + pSpanOffsetInLump[pToSpan[c]];

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

    virtual void solveL(const T* data, uint64_t offM, uint64_t n, T* C,
                        uint64_t offC, uint64_t ldc, uint64_t nRHS) override {
        Eigen::Map<const MatRMaj<T>> matA(data + offM, n, n);
        OuterStridedCMajMatM<T> matC(C + offC, n, nRHS, OuterStride(ldc));
        matA.template triangularView<Eigen::Lower>().solveInPlace(matC);
    }

    virtual void gemv(const T* data, uint64_t offM, uint64_t nRows,
                      uint64_t nCols, const T* A, uint64_t offA, uint64_t lda,
                      T* C, uint64_t nRHS) override {
        Eigen::Map<const MatRMaj<T>> matM(data + offM, nRows, nCols);
        OuterStridedCMajMatK<T> matA(A + offA, nCols, nRHS, OuterStride(lda));
        Eigen::Map<MatRMaj<T>> matC(C, nRows, nRHS);
        matC.noalias() = matM * matA;
    }

    virtual void assembleVec(const T* A, uint64_t chainColPtr,
                             uint64_t numColItems, T* C, uint64_t ldc,
                             uint64_t nRHS) override {
        const CoalescedBlockMatrixSkel& skel = sym.skel;
        const uint64_t* chainRowsTillEnd =
            skel.chainRowsTillEnd.data() + chainColPtr;
        const uint64_t* toSpan = skel.chainRowSpan.data() + chainColPtr;
        uint64_t startRow = chainRowsTillEnd[-1];
        for (uint64_t i = 0; i < numColItems; i++) {
            uint64_t rowOffset = chainRowsTillEnd[i - 1] - startRow;
            uint64_t span = toSpan[i];
            uint64_t spanStart = skel.spanStart[span];
            uint64_t spanSize = skel.spanStart[span + 1] - spanStart;

            Eigen::Map<const MatRMaj<T>> matA(A + rowOffset * nRHS, spanSize,
                                              nRHS);
            OuterStridedCMajMatM<T> matC(C + spanStart, spanSize, nRHS,
                                         OuterStride(ldc));
            matC -= matA;
        }
    }

    virtual void solveLt(const T* data, uint64_t offM, uint64_t n, T* C,
                         uint64_t offC, uint64_t ldc, uint64_t nRHS) override {
        Eigen::Map<const MatRMaj<T>> matA(data + offM, n, n);
        OuterStridedCMajMatM<T> matC(C + offC, n, nRHS, OuterStride(ldc));
        matA.template triangularView<Eigen::Lower>().adjoint().solveInPlace(
            matC);
    }

    virtual void gemvT(const T* data, uint64_t offM, uint64_t nRows,
                       uint64_t nCols, const T* C, uint64_t nRHS, T* A,
                       uint64_t offA, uint64_t lda) override {
        Eigen::Map<const MatRMaj<T>> matM(data + offM, nRows, nCols);
        OuterStridedCMajMatM<T> matA(A + offA, nCols, nRHS, OuterStride(lda));
        Eigen::Map<const MatRMaj<T>> matC(C, nRows, nRHS);
        matA.noalias() -= matM.transpose() * matC;
    }

    virtual void assembleVecT(const T* C, uint64_t ldc, uint64_t nRHS, T* A,
                              uint64_t chainColPtr,
                              uint64_t numColItems) override {
        const CoalescedBlockMatrixSkel& skel = sym.skel;
        const uint64_t* chainRowsTillEnd =
            skel.chainRowsTillEnd.data() + chainColPtr;
        const uint64_t* toSpan = skel.chainRowSpan.data() + chainColPtr;
        uint64_t startRow = chainRowsTillEnd[-1];
        for (uint64_t i = 0; i < numColItems; i++) {
            uint64_t rowOffset = chainRowsTillEnd[i - 1] - startRow;
            uint64_t span = toSpan[i];
            uint64_t spanStart = skel.spanStart[span];
            uint64_t spanSize = skel.spanStart[span + 1] - spanStart;

            Eigen::Map<MatRMaj<T>> matA(A + rowOffset * nRHS, spanSize, nRHS);
            OuterStridedCMajMatK<T> matC(C + spanStart, spanSize, nRHS,
                                         OuterStride(ldc));
            matA = matC;
        }
    }

    const SimpleSymbolicCtx& sym;
};

NumericCtxPtr<double> SimpleSymbolicCtx::createDoubleContext(
    uint64_t tempBufSize, int maxBatchSize) {
    return NumericCtxPtr<double>(new SimpleNumericCtx<double>(
        *this, tempBufSize, skel.spanStart.size() - 1));
}

SolveCtxPtr<double> SimpleSymbolicCtx::createDoubleSolveContext() {
    return SolveCtxPtr<double>(new SimpleSolveCtx<double>(*this));
}

OpsPtr simpleOps() { return OpsPtr(new SimpleOps); }

OpsPtr blasOps() { return OpsPtr(new SimpleOps); }
