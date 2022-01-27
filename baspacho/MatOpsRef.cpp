
#include <glog/logging.h>

#include <chrono>

#include "MatOps.h"
#include "Utils.h"

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

using OuterStride = Eigen::OuterStride<>;
using OuterStridedMatM = Eigen::Map<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0,
    OuterStride>;

// simple ops implemented using Eigen (therefore single thread)
struct SimpleOps : Ops {
    // will just contain a reference to the skel
    struct OpaqueDataMatrixSkel : OpaqueData {
        OpaqueDataMatrixSkel(const BlockMatrixSkel& skel) : skel(skel) {}
        virtual ~OpaqueDataMatrixSkel() {}
        const BlockMatrixSkel& skel;
    };

    virtual OpaqueDataPtr prepareMatrixSkel(
        const BlockMatrixSkel& skel) override {
        return OpaqueDataPtr(new OpaqueDataMatrixSkel(skel));
    }

    virtual void printStats() const override {
        LOG(INFO) << "matOp stats:"
                  << "\nelim: " << elimStat.toString()
                  << "\nBiggest dense block: " << potrfBiggestN
                  << "\npotrf: " << potrfStat.toString()
                  << "\ntrsm: " << trsmStat.toString()  //
                  << "\nsyrk/gemm: " << sygeStat.toString()
                  << "\nasmbl: " << asmblStat.toString();
    }

    static void factorLump(const BlockMatrixSkel& skel, double* data,
                           uint64_t lump) {
        uint64_t lumpStart = skel.lumpStart[lump];
        uint64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
        uint64_t colStart = skel.chainColPtr[lump];
        uint64_t dataPtr = skel.chainData[colStart];

        // compute lower diag cholesky dec on diagonal block
        Eigen::Map<MatRMaj<double>> diagBlock(data + dataPtr, lumpSize,
                                              lumpSize);
        { Eigen::LLT<Eigen::Ref<MatRMaj<double>>> llt(diagBlock); }
        uint64_t gatheredStart = skel.boardColPtr[lump];
        uint64_t gatheredEnd = skel.boardColPtr[lump + 1];
        uint64_t rowDataStart = skel.boardChainColOrd[gatheredStart + 1];
        uint64_t rowDataEnd = skel.boardChainColOrd[gatheredEnd - 1];
        uint64_t belowDiagStart = skel.chainData[colStart + rowDataStart];
        uint64_t numRows = skel.chainRowsTillEnd[colStart + rowDataEnd - 1] -
                           skel.chainRowsTillEnd[colStart + rowDataStart - 1];

        Eigen::Map<MatRMaj<double>> belowDiagBlock(data + belowDiagStart,
                                                   numRows, lumpSize);
        diagBlock.triangularView<Eigen::Lower>()
            .transpose()
            .solveInPlace<Eigen::OnTheRight>(belowDiagBlock);
    }

    static void prepareContextForTargetLump(
        const BlockMatrixSkel& skel, uint64_t targetLump,
        vector<uint64_t>& spanToChainOffset) {
        spanToChainOffset.assign(skel.spanStart.size() - 1, kInvalid);
        for (uint64_t i = skel.chainColPtr[targetLump],
                      iEnd = skel.chainColPtr[targetLump + 1];
             i < iEnd; i++) {
            spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
        }
    }

    static inline void stridedMatSub(double* dst, uint64_t dstStride,
                                     const double* src, uint64_t srcStride,
                                     uint64_t rSize, uint64_t cSize) {
        for (uint j = 0; j < rSize; j++) {
            for (uint i = 0; i < cSize; i++) {
                dst[i] -= src[i];
            }
            dst += dstStride;
            src += srcStride;
        }
    }

    struct OpaqueDataElimData : OpaqueData {
        OpaqueDataElimData() {}
        virtual ~OpaqueDataElimData() {}

        // per-row pointers to chains in a rectagle:
        // * span-rows from lumpToSpan[lumpsEnd],
        // * board cols in interval lumpsBegin:lumpsEnd
        vector<uint64_t> rowPtr;       // row data pointer
        vector<uint64_t> colLump;      // col-lump
        vector<uint64_t> chainColOrd;  // order in col chain elements
    };

    // TODO: unit test
    virtual OpaqueDataPtr prepareElimination(const BlockMatrixSkel& skel,
                                             uint64_t lumpsBegin,
                                             uint64_t lumpsEnd) override {
        OpaqueDataElimData* elim = new OpaqueDataElimData;

        uint64_t spanRowBegin = skel.lumpToSpan[lumpsEnd];
        uint64_t numSpanRows = skel.spanStart.size() - 1 - spanRowBegin;
        elim->rowPtr.assign(numSpanRows + 1, 0);
        for (uint64_t l = lumpsBegin; l < lumpsEnd; l++) {
            for (uint64_t i = skel.chainColPtr[l],
                          iEnd = skel.chainColPtr[l + 1];
                 i < iEnd; i++) {
                uint64_t s = skel.chainRowSpan[i];
                if (s < spanRowBegin) {
                    continue;
                }
                uint64_t sRel = s - spanRowBegin;
                elim->rowPtr[sRel]++;
            }
        }
        uint64_t totNumChains = cumSumVec(elim->rowPtr);
        elim->colLump.resize(totNumChains);
        elim->chainColOrd.resize(totNumChains);
        for (uint64_t l = lumpsBegin; l < lumpsEnd; l++) {
            for (uint64_t iBegin = skel.chainColPtr[l],
                          iEnd = skel.chainColPtr[l + 1], i = iBegin;
                 i < iEnd; i++) {
                uint64_t s = skel.chainRowSpan[i];
                if (s < spanRowBegin) {
                    continue;
                }
                uint64_t sRel = s - spanRowBegin;
                elim->colLump[elim->rowPtr[sRel]] = l;
                elim->chainColOrd[elim->rowPtr[sRel]] = i - iBegin;
                elim->rowPtr[sRel]++;
            }
        }
        rewindVec(elim->rowPtr);
        return OpaqueDataPtr(elim);
    }

    virtual void doElimination(const OpaqueData& ref, double* data,
                               uint64_t lumpsBegin, uint64_t lumpsEnd,
                               const OpaqueData& elimData) override {
        OpInstance timer(elimStat);
        const OpaqueDataMatrixSkel* pSkel =
            dynamic_cast<const OpaqueDataMatrixSkel*>(&ref);
        const OpaqueDataElimData* pElim =
            dynamic_cast<const OpaqueDataElimData*>(&elimData);
        CHECK_NOTNULL(pSkel);
        CHECK_NOTNULL(pElim);
        const BlockMatrixSkel& skel = pSkel->skel;
        const OpaqueDataElimData& elim = *pElim;

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
        std::vector<uint64_t> paramToChainOffset;
        uint64_t stride;
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

    virtual OpaqueDataPtr createAssembleContext(const OpaqueData& ref,
                                                uint64_t tempBufSize,
                                                int maxBatchSize = 1) override {
        const OpaqueDataMatrixSkel* pSkel =
            dynamic_cast<const OpaqueDataMatrixSkel*>(&ref);
        CHECK_NOTNULL(pSkel);
        const BlockMatrixSkel& skel = pSkel->skel;
        AssembleContext* ax = new AssembleContext;
        ax->paramToChainOffset.resize(skel.spanStart.size() - 1);
        ax->tempBuffer.resize(tempBufSize);
        return OpaqueDataPtr(ax);
    }

    virtual void prepareAssembleContext(const OpaqueData& ref,
                                        OpaqueData& assCtx,
                                        uint64_t targetLump) override {
        const OpaqueDataMatrixSkel* pSkel =
            dynamic_cast<const OpaqueDataMatrixSkel*>(&ref);
        AssembleContext* pAx = dynamic_cast<AssembleContext*>(&assCtx);
        CHECK_NOTNULL(pSkel);
        CHECK_NOTNULL(pAx);
        const BlockMatrixSkel& skel = pSkel->skel;
        AssembleContext& ax = *pAx;

        for (uint64_t i = skel.chainColPtr[targetLump],
                      iEnd = skel.chainColPtr[targetLump + 1];
             i < iEnd; i++) {
            ax.paramToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
        }
    }

    virtual void assemble(const OpaqueData& ref, const OpaqueData& assCtx,
                          double* data, uint64_t rectRowBegin,
                          uint64_t dstStride,  //
                          uint64_t srcColDataOffset, uint64_t srcRectWidth,
                          uint64_t numBlockRows, uint64_t numBlockCols,
                          int numBatch = -1) override {
        CHECK_EQ(numBatch, -1) << "Batching not supported";
        OpInstance timer(asmblStat);
        const OpaqueDataMatrixSkel* pSkel =
            dynamic_cast<const OpaqueDataMatrixSkel*>(&ref);
        const AssembleContext* pAx =
            dynamic_cast<const AssembleContext*>(&assCtx);
        CHECK_NOTNULL(pSkel);
        CHECK_NOTNULL(pAx);
        const BlockMatrixSkel& skel = pSkel->skel;
        const AssembleContext& ax = *pAx;
        const uint64_t* chainRowsTillEnd =
            skel.chainRowsTillEnd.data() + srcColDataOffset;
        const uint64_t* toSpan = skel.chainRowSpan.data() + srcColDataOffset;
        const uint64_t* paramToChainOffset = ax.paramToChainOffset.data();
        const uint64_t* spanOffsetInLump = skel.spanOffsetInLump.data();

        const double* matRectPtr = ax.tempBuffer.data();

        for (uint64_t r = 0; r < numBlockRows; r++) {
            uint64_t rBegin = chainRowsTillEnd[r - 1] - rectRowBegin;
            uint64_t rSize = chainRowsTillEnd[r] - rBegin - rectRowBegin;
            uint64_t rParam = toSpan[r];
            uint64_t rOffset = paramToChainOffset[rParam];
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

    OpStat elimStat;
    OpStat potrfStat;
    uint64_t potrfBiggestN = 0;
    OpStat trsmStat;
    OpStat sygeStat;
    OpStat asmblStat;
};

OpsPtr simpleOps() { return OpsPtr(new SimpleOps); }
