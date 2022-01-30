
#include <glog/logging.h>

#include <chrono>
#include <iostream>

#include "MatOpsCpuBase.h"
#include "Utils.h"

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

using OuterStride = Eigen::OuterStride<>;
using OuterStridedMatM = Eigen::Map<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0,
    OuterStride>;
using OuterStridedCMajMatM = Eigen::Map<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
    OuterStride>;
using OuterStridedCMajMatK =
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::ColMajor>,
               0, OuterStride>;

// simple ops implemented using Eigen (therefore single thread)
struct SimpleOps : CpuBaseOps {
    // will just contain a reference to the skel
    struct SimpleSymbolicInfo : OpaqueData {
        SimpleSymbolicInfo(const CoalescedBlockMatrixSkel& skel) : skel(skel) {}
        virtual ~SimpleSymbolicInfo() {}
        const CoalescedBlockMatrixSkel& skel;
    };

    virtual OpaqueDataPtr initSymbolicInfo(
        const CoalescedBlockMatrixSkel& skel) override {
        return OpaqueDataPtr(new SimpleSymbolicInfo(skel));
    }

    virtual void doElimination(const OpaqueData& info, double* data,
                               uint64_t lumpsBegin, uint64_t lumpsEnd,
                               const OpaqueData& elimData) override {
        OpInstance timer(elimStat);
        const SimpleSymbolicInfo* pInfo =
            dynamic_cast<const SimpleSymbolicInfo*>(&info);
        const SparseEliminationInfo* pElim =
            dynamic_cast<const SparseEliminationInfo*>(&elimData);
        CHECK_NOTNULL(pInfo);
        CHECK_NOTNULL(pElim);
        const CoalescedBlockMatrixSkel& skel = pInfo->skel;
        const SparseEliminationInfo& elim = *pElim;

        for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
            factorLump(skel, data, l);
        }

        uint64_t numElimRows = elim.rowPtr.size() - 1;
        uint64_t numSpans = skel.spanStart.size() - 1;
        std::vector<double> tempBuffer(elim.maxBufferSize);
        std::vector<uint64_t> spanToChainOffset(numSpans);
        for (uint64_t sRel = 0UL; sRel < numElimRows; sRel++) {
            eliminateRowChain(elim, skel, data, sRel, spanToChainOffset,
                              tempBuffer);
        }
    }

    virtual void doEliminationQ(const OpaqueData& info, double* data,
                                uint64_t lumpsBegin, uint64_t lumpsEnd,
                                const OpaqueData& elimData) {
        OpInstance timer(elimStat);
        const SimpleSymbolicInfo* pInfo =
            dynamic_cast<const SimpleSymbolicInfo*>(&info);
        const SparseEliminationInfo* pElim =
            dynamic_cast<const SparseEliminationInfo*>(&elimData);
        CHECK_NOTNULL(pInfo);
        CHECK_NOTNULL(pElim);
        const CoalescedBlockMatrixSkel& skel = pInfo->skel;
        const SparseEliminationInfo& elim = *pElim;

        for (uint64_t a = lumpsBegin; a < lumpsEnd; a++) {
            factorLump(skel, data, a);
        }

        std::vector<double> tempBuffer;           // ctx
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
                CHECK_GE(chainColOrd, 1);  // there must be a diagonal block

                uint64_t ptrStart = skel.chainColPtr[lump] + chainColOrd;
                uint64_t ptrEnd = skel.chainColPtr[lump + 1];
                CHECK_EQ(skel.chainRowSpan[ptrStart], s);

                uint64_t nRowsAbove = skel.chainRowsTillEnd[ptrStart - 1];
                uint64_t nRowsChain =
                    skel.chainRowsTillEnd[ptrStart] - nRowsAbove;
                uint64_t nRowsOnward = skel.chainRowsTillEnd[ptrEnd - 1];
                uint64_t dataOffset = skel.chainData[ptrStart];
                CHECK_EQ(nRowsChain, skel.spanStart[s + 1] - skel.spanStart[s]);
                uint64_t lumpSize =
                    skel.lumpStart[lump + 1] - skel.lumpStart[lump];

                Eigen::Map<MatRMaj<double>> chainSubMat(data + dataOffset,
                                                        nRowsChain, lumpSize);
                Eigen::Map<MatRMaj<double>> chainOnwardSubMat(
                    data + dataOffset, nRowsOnward, lumpSize);

                tempBuffer.resize(nRowsOnward * nRowsChain);
                Eigen::Map<MatRMaj<double>> prod(tempBuffer.data(), nRowsOnward,
                                                 nRowsChain);
                prod = chainOnwardSubMat * chainSubMat.transpose();

                // assemble blocks, iterating on chain and below chains
                for (uint64_t ptr = ptrStart; ptr < ptrEnd; ptr++) {
                    uint64_t s2 = skel.chainRowSpan[ptr];
                    uint64_t relRow =
                        skel.chainRowsTillEnd[ptr - 1] - nRowsAbove;
                    uint64_t s2_size =
                        skel.chainRowsTillEnd[ptr] - nRowsAbove - relRow;
                    CHECK_EQ(s2_size,
                             skel.spanStart[s2 + 1] - skel.spanStart[s2]);

                    double* targetData =
                        data + spanOffsetInLump + spanToChainOffset[s2];
                    CHECK(spanToChainOffset[s2] != kInvalid);

                    OuterStridedMatM targetBlock(targetData, s2_size,
                                                 nRowsChain,
                                                 OuterStride(targetLumpSize));
                    targetBlock -= prod.block(relRow, 0, s2_size, nRowsChain);
                }
            }
        }
    }

    virtual void potrf(uint64_t n, double* A) override {
        OpInstance timer(potrfStat);

        Eigen::Map<MatRMaj<double>> matA(A, n, n);
        Eigen::LLT<Eigen::Ref<MatRMaj<double>>> llt(matA);

        potrfBiggestN = std::max(potrfBiggestN, n);
    }

    virtual void trsm(uint64_t n, uint64_t k, const double* A,
                      double* B) override {
        OpInstance timer(trsmStat);

        using MatCMajD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::ColMajor>;

        // col-major's upper = (row-major's lower).transpose()
        Eigen::Map<const MatCMajD> matA(A, n, n);
        Eigen::Map<MatRMaj<double>> matB(B, k, n);
        matA.triangularView<Eigen::Upper>().solveInPlace<Eigen::OnTheRight>(
            matB);
    }

    struct AssembleContext : OpaqueData {
        std::vector<uint64_t> spanToChainOffset;
        std::vector<double> tempBuffer;
    };

    virtual void saveSyrkGemm(OpaqueData& assCtx, uint64_t m, uint64_t n,
                              uint64_t k, const double* data,
                              uint64_t offset) override {
        OpInstance timer(sygeStat);
        AssembleContext* pAx = dynamic_cast<AssembleContext*>(&assCtx);
        CHECK_NOTNULL(pAx);
        AssembleContext& ax = *pAx;
        CHECK_LE(m * n, ax.tempBuffer.size());

        const double* AB = data + offset;
        double* C = ax.tempBuffer.data();
        Eigen::Map<const MatRMaj<double>> matA(AB, m, k);
        Eigen::Map<const MatRMaj<double>> matB(AB, n, k);
        Eigen::Map<MatRMaj<double>> matC(C, n, m);
        matC.noalias() = matB * matA.transpose();
    }

    virtual void saveSyrkGemmBatched(OpaqueData& assCtx, uint64_t* ms,
                                     uint64_t* ns, uint64_t* ks,
                                     const double* data, uint64_t* offsets,
                                     int batchSize) {
        LOG(FATAL) << "Batching not supported";
    }

    virtual OpaqueDataPtr createAssembleContext(const OpaqueData& info,
                                                uint64_t tempBufSize,
                                                int maxBatchSize = 1) override {
        const SimpleSymbolicInfo* pInfo =
            dynamic_cast<const SimpleSymbolicInfo*>(&info);
        CHECK_NOTNULL(pInfo);
        const CoalescedBlockMatrixSkel& skel = pInfo->skel;
        AssembleContext* ax = new AssembleContext;
        ax->spanToChainOffset.resize(skel.spanStart.size() - 1);
        ax->tempBuffer.resize(tempBufSize);
        return OpaqueDataPtr(ax);
    }

    virtual void prepareAssembleContext(const OpaqueData& info,
                                        OpaqueData& assCtx,
                                        uint64_t targetLump) override {
        const SimpleSymbolicInfo* pInfo =
            dynamic_cast<const SimpleSymbolicInfo*>(&info);
        AssembleContext* pAx = dynamic_cast<AssembleContext*>(&assCtx);
        CHECK_NOTNULL(pInfo);
        CHECK_NOTNULL(pAx);
        const CoalescedBlockMatrixSkel& skel = pInfo->skel;
        AssembleContext& ax = *pAx;

        for (uint64_t i = skel.chainColPtr[targetLump],
                      iEnd = skel.chainColPtr[targetLump + 1];
             i < iEnd; i++) {
            ax.spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
        }
    }

    virtual void assemble(const OpaqueData& info, const OpaqueData& assCtx,
                          double* data, uint64_t rectRowBegin,
                          uint64_t dstStride,  //
                          uint64_t srcColDataOffset, uint64_t srcRectWidth,
                          uint64_t numBlockRows, uint64_t numBlockCols,
                          int numBatch = -1) override {
        CHECK_EQ(numBatch, -1) << "Batching not supported";
        OpInstance timer(asmblStat);
        const SimpleSymbolicInfo* pInfo =
            dynamic_cast<const SimpleSymbolicInfo*>(&info);
        const AssembleContext* pAx =
            dynamic_cast<const AssembleContext*>(&assCtx);
        CHECK_NOTNULL(pInfo);
        CHECK_NOTNULL(pAx);
        const CoalescedBlockMatrixSkel& skel = pInfo->skel;
        const AssembleContext& ax = *pAx;
        const uint64_t* chainRowsTillEnd =
            skel.chainRowsTillEnd.data() + srcColDataOffset;
        const uint64_t* toSpan = skel.chainRowSpan.data() + srcColDataOffset;
        const uint64_t* spanToChainOffset = ax.spanToChainOffset.data();
        const uint64_t* spanOffsetInLump = skel.spanOffsetInLump.data();

        const double* matRectPtr = ax.tempBuffer.data();

        for (uint64_t r = 0; r < numBlockRows; r++) {
            uint64_t rBegin = chainRowsTillEnd[r - 1] - rectRowBegin;
            uint64_t rSize = chainRowsTillEnd[r] - rBegin - rectRowBegin;
            uint64_t rParam = toSpan[r];
            uint64_t rOffset = spanToChainOffset[rParam];
            const double* matRowPtr = matRectPtr + rBegin * srcRectWidth;

            uint64_t cEnd = std::min(numBlockCols, r + 1);
            for (uint64_t c = 0; c < cEnd; c++) {
                uint64_t cStart = chainRowsTillEnd[c - 1] - rectRowBegin;
                uint64_t cSize = chainRowsTillEnd[c] - cStart - rectRowBegin;
                uint64_t offset = rOffset + spanOffsetInLump[toSpan[c]];

                double* dst = data + offset;
                const double* src = matRowPtr + cStart;
                stridedMatSub(dst, dstStride, src, srcRectWidth, rSize, cSize);
            }
        }
    }

    virtual void solveL(const double* data, uint64_t offM, uint64_t n,
                        double* C, uint64_t offC, uint64_t ldc,
                        uint64_t nRHS) override {
        Eigen::Map<const MatRMaj<double>> matA(data + offM, n, n);
        OuterStridedCMajMatM matC(C + offC, n, nRHS, OuterStride(ldc));
        matA.triangularView<Eigen::Lower>().solveInPlace(matC);
    }

    virtual void gemv(const double* data, uint64_t offM, uint64_t nRows,
                      uint64_t nCols, const double* A, uint64_t offA,
                      uint64_t lda, double* C, uint64_t nRHS) override {
        Eigen::Map<const MatRMaj<double>> matM(data + offM, nRows, nCols);
        OuterStridedCMajMatK matA(A + offA, nCols, nRHS, OuterStride(lda));
        Eigen::Map<MatRMaj<double>> matC(C, nRows, nRHS);
        matC.noalias() = matM * matA;
    }

    virtual void assembleVec(const OpaqueData& info, const double* A,
                             uint64_t chainColPtr, uint64_t numColItems,
                             double* C, uint64_t ldc, uint64_t nRHS) override {
        const SimpleSymbolicInfo* pInfo =
            dynamic_cast<const SimpleSymbolicInfo*>(&info);
        CHECK_NOTNULL(pInfo);
        const CoalescedBlockMatrixSkel& skel = pInfo->skel;
        const uint64_t* chainRowsTillEnd =
            skel.chainRowsTillEnd.data() + chainColPtr;
        const uint64_t* toSpan = skel.chainRowSpan.data() + chainColPtr;
        uint64_t startRow = chainRowsTillEnd[-1];
        for (uint64_t i = 0; i < numColItems; i++) {
            uint64_t rowOffset = chainRowsTillEnd[i - 1] - startRow;
            uint64_t span = toSpan[i];
            uint64_t spanStart = skel.spanStart[span];
            uint64_t spanSize = skel.spanStart[span + 1] - spanStart;

            Eigen::Map<const MatRMaj<double>> matA(A + rowOffset * nRHS,
                                                   spanSize, nRHS);
            OuterStridedCMajMatM matC(C + spanStart, spanSize, nRHS,
                                      OuterStride(ldc));
            matC -= matA;
        }
    }

    virtual void solveLt(const double* data, uint64_t offM, uint64_t n,
                         double* C, uint64_t offC, uint64_t ldc,
                         uint64_t nRHS) override {
        Eigen::Map<const MatRMaj<double>> matA(data + offM, n, n);
        OuterStridedCMajMatM matC(C + offC, n, nRHS, OuterStride(ldc));
        matA.triangularView<Eigen::Lower>().adjoint().solveInPlace(matC);
    }

    virtual void gemvT(const double* data, uint64_t offM, uint64_t nRows,
                       uint64_t nCols, const double* C, uint64_t nRHS,
                       double* A, uint64_t offA, uint64_t lda) override {
        Eigen::Map<const MatRMaj<double>> matM(data + offM, nRows, nCols);
        OuterStridedCMajMatM matA(A + offA, nCols, nRHS, OuterStride(lda));
        Eigen::Map<const MatRMaj<double>> matC(C, nRows, nRHS);
        matA.noalias() -= matM.transpose() * matC;
    }

    virtual void assembleVecT(const OpaqueData& info, const double* C,
                              uint64_t ldc, uint64_t nRHS, double* A,
                              uint64_t chainColPtr,
                              uint64_t numColItems) override {
        const SimpleSymbolicInfo* pInfo =
            dynamic_cast<const SimpleSymbolicInfo*>(&info);
        CHECK_NOTNULL(pInfo);
        const CoalescedBlockMatrixSkel& skel = pInfo->skel;
        const uint64_t* chainRowsTillEnd =
            skel.chainRowsTillEnd.data() + chainColPtr;
        const uint64_t* toSpan = skel.chainRowSpan.data() + chainColPtr;
        uint64_t startRow = chainRowsTillEnd[-1];
        for (uint64_t i = 0; i < numColItems; i++) {
            uint64_t rowOffset = chainRowsTillEnd[i - 1] - startRow;
            uint64_t span = toSpan[i];
            uint64_t spanStart = skel.spanStart[span];
            uint64_t spanSize = skel.spanStart[span + 1] - spanStart;

            Eigen::Map<MatRMaj<double>> matA(A + rowOffset * nRHS, spanSize,
                                             nRHS);
            OuterStridedCMajMatK matC(C + spanStart, spanSize, nRHS,
                                      OuterStride(ldc));
            matA = matC;
        }
    }
};

OpsPtr simpleOps() { return OpsPtr(new SimpleOps); }
