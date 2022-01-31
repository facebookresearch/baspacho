#pragma once

#include <vector>

#include "MatOps.h"
#include "Utils.h"

struct CpuBaseSymElimCtx : SymElimCtx {
    CpuBaseSymElimCtx() {}
    virtual ~CpuBaseSymElimCtx() override {}

    // per-row pointers to chains in a rectagle:
    // * span-rows from lumpToSpan[lumpsEnd],
    // * board cols in interval lumpsBegin:lumpsEnd
    uint64_t spanRowBegin;
    uint64_t maxBufferSize;
    std::vector<uint64_t> rowPtr;       // row data pointer
    std::vector<uint64_t> colLump;      // col-lump
    std::vector<uint64_t> chainColOrd;  // order in col chain elements
};

// common code for ref and blas implementations
struct CpuBaseSymbolicCtx : SymbolicCtx {
    CpuBaseSymbolicCtx(const CoalescedBlockMatrixSkel& skel) : skel(skel) {}

    static uint64_t computeMaxBufSize(const CpuBaseSymElimCtx& elim,
                                      const CoalescedBlockMatrixSkel& skel,
                                      uint64_t sRel) {
        uint64_t maxBufferSize = 0;

        // iterate over chains present in this row
        for (uint64_t i = elim.rowPtr[sRel], iEnd = elim.rowPtr[sRel + 1];
             i < iEnd; i++) {
            uint64_t lump = elim.colLump[i];
            uint64_t chainColOrd = elim.chainColOrd[i];
            BASPACHO_CHECK_GE(chainColOrd,
                              1);  // there must be a diagonal block

            uint64_t ptrStart = skel.chainColPtr[lump] + chainColOrd;
            uint64_t ptrEnd = skel.chainColPtr[lump + 1];

            uint64_t nRowsAbove = skel.chainRowsTillEnd[ptrStart - 1];
            uint64_t nRowsChain = skel.chainRowsTillEnd[ptrStart] - nRowsAbove;
            uint64_t nRowsOnward = skel.chainRowsTillEnd[ptrEnd - 1];

            maxBufferSize = std::max(maxBufferSize, nRowsOnward * nRowsChain);
        }

        return maxBufferSize;
    }

    virtual SymElimCtxPtr prepareElimination(uint64_t lumpsBegin,
                                             uint64_t lumpsEnd) override {
        CpuBaseSymElimCtx* elim = new CpuBaseSymElimCtx;

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
        elim->spanRowBegin = spanRowBegin;

        elim->maxBufferSize = 0;
        for (uint64_t r = 0; r < elim->rowPtr.size() - 1; r++) {
            elim->maxBufferSize = std::max(elim->maxBufferSize,
                                           computeMaxBufSize(*elim, skel, r));
        }
        return SymElimCtxPtr(elim);
    }

    const CoalescedBlockMatrixSkel& skel;
};

template <typename T>
struct CpuBaseNumericCtx : NumericCtx<T> {
    CpuBaseNumericCtx(uint64_t bufSize, uint64_t numSpans)
        : tempBuffer(bufSize), spanToChainOffset(numSpans) {}

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

    // helper for elimination
    static inline void factorLump(const CoalescedBlockMatrixSkel& skel, T* data,
                                  uint64_t lump) {
        uint64_t lumpStart = skel.lumpStart[lump];
        uint64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
        uint64_t colStart = skel.chainColPtr[lump];
        uint64_t dataPtr = skel.chainData[colStart];

        // in-place lower diag cholesky dec on diagonal block
        Eigen::Map<MatRMaj<T>> diagBlock(data + dataPtr, lumpSize, lumpSize);
        { Eigen::LLT<Eigen::Ref<MatRMaj<T>>> llt(diagBlock); }

        uint64_t gatheredStart = skel.boardColPtr[lump];
        uint64_t gatheredEnd = skel.boardColPtr[lump + 1];
        uint64_t rowDataStart = skel.boardChainColOrd[gatheredStart + 1];
        uint64_t rowDataEnd = skel.boardChainColOrd[gatheredEnd - 1];
        uint64_t belowDiagStart = skel.chainData[colStart + rowDataStart];
        uint64_t numRows = skel.chainRowsTillEnd[colStart + rowDataEnd - 1] -
                           skel.chainRowsTillEnd[colStart + rowDataStart - 1];

        Eigen::Map<MatRMaj<T>> belowDiagBlock(data + belowDiagStart, numRows,
                                              lumpSize);
        diagBlock.template triangularView<Eigen::Lower>()
            .transpose()
            .template solveInPlace<Eigen::OnTheRight>(belowDiagBlock);
    }

    static inline void stridedMatSub(T* dst, uint64_t dstStride, const T* src,
                                     uint64_t srcStride, uint64_t rSize,
                                     uint64_t cSize) {
        for (uint j = 0; j < rSize; j++) {
            for (uint i = 0; i < cSize; i++) {
                dst[i] -= src[i];
            }
            dst += dstStride;
            src += srcStride;
        }
    }

    static void prepareContextForTargetLump(
        const CoalescedBlockMatrixSkel& skel, uint64_t targetLump,
        std::vector<uint64_t>& spanToChainOffset) {
        for (uint64_t i = skel.chainColPtr[targetLump],
                      iEnd = skel.chainColPtr[targetLump + 1];
             i < iEnd; i++) {
            spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
        }
    }

    static void eliminateRowChain(const CpuBaseSymElimCtx& elim,
                                  const CoalescedBlockMatrixSkel& skel, T* data,
                                  uint64_t sRel,
                                  std::vector<uint64_t>& spanToChainOffset,
                                  std::vector<T>& tempBuffer) {
        uint64_t s = sRel + elim.spanRowBegin;
        if (elim.rowPtr[sRel] == elim.rowPtr[sRel + 1]) {
            return;
        }
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
            uint64_t nRowsChain = skel.chainRowsTillEnd[ptrStart] - nRowsAbove;
            uint64_t nRowsOnward = skel.chainRowsTillEnd[ptrEnd - 1];
            uint64_t dataOffset = skel.chainData[ptrStart];
            BASPACHO_CHECK_EQ(nRowsChain,
                              skel.spanStart[s + 1] - skel.spanStart[s]);
            uint64_t lumpSize = skel.lumpStart[lump + 1] - skel.lumpStart[lump];

            Eigen::Map<MatRMaj<T>> chainSubMat(data + dataOffset, nRowsChain,
                                               lumpSize);
            Eigen::Map<MatRMaj<T>> chainOnwardSubMat(data + dataOffset,
                                                     nRowsOnward, lumpSize);

            BASPACHO_CHECK_GE(tempBuffer.size(), nRowsOnward * nRowsChain);
            Eigen::Map<MatRMaj<T>> prod(tempBuffer.data(), nRowsOnward,
                                        nRowsChain);
            prod.noalias() = chainOnwardSubMat * chainSubMat.transpose();

            // assemble blocks, iterating on chain and below chains
            for (uint64_t ptr = ptrStart; ptr < ptrEnd; ptr++) {
                uint64_t s2 = skel.chainRowSpan[ptr];
                uint64_t relRow = skel.chainRowsTillEnd[ptr - 1] - nRowsAbove;
                uint64_t s2_size =
                    skel.chainRowsTillEnd[ptr] - nRowsAbove - relRow;

                // incomment below if check is needed
                // BASPACHO_CHECK(spanToChainOffset[s2] != kInvalid);
                T* targetData = data + spanOffsetInLump + spanToChainOffset[s2];

                stridedMatSub(targetData, targetLumpSize,
                              tempBuffer.data() + nRowsChain * relRow,
                              nRowsChain, s2_size, nRowsChain);
            }
        }
    }

    static void eliminateVerySparseRowChain(
        const CpuBaseSymElimCtx& elim, const CoalescedBlockMatrixSkel& skel,
        T* data, uint64_t sRel) {
        uint64_t s = sRel + elim.spanRowBegin;
        if (elim.rowPtr[sRel] == elim.rowPtr[sRel + 1]) {
            return;
        }
        uint64_t targetLump = skel.spanToLump[s];
        uint64_t targetLumpSize =
            skel.lumpStart[targetLump + 1] - skel.lumpStart[targetLump];
        uint64_t spanOffsetInLump =
            skel.spanStart[s] - skel.lumpStart[targetLump];
        uint64_t bisectStart = skel.chainColPtr[targetLump];
        uint64_t bisectEnd = skel.chainColPtr[targetLump + 1];

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
            uint64_t nRowsChain = skel.chainRowsTillEnd[ptrStart] - nRowsAbove;
            uint64_t nRowsOnward = skel.chainRowsTillEnd[ptrEnd - 1];
            uint64_t dataOffset = skel.chainData[ptrStart];
            BASPACHO_CHECK_EQ(nRowsChain,
                              skel.spanStart[s + 1] - skel.spanStart[s]);
            uint64_t lumpSize = skel.lumpStart[lump + 1] - skel.lumpStart[lump];

            Eigen::Map<MatRMaj<T>> chainSubMat(data + dataOffset, nRowsChain,
                                               lumpSize);
            Eigen::Map<MatRMaj<T>> chainOnwardSubMat(data + dataOffset,
                                                     nRowsOnward, lumpSize);

            T* tempBuffer = (T*)alloca(sizeof(T) * nRowsOnward * nRowsChain);
            Eigen::Map<MatRMaj<T>> prod(tempBuffer, nRowsOnward, nRowsChain);
            prod.noalias() = chainOnwardSubMat * chainSubMat.transpose();

            // assemble blocks, iterating on chain and below chains
            for (uint64_t ptr = ptrStart; ptr < ptrEnd; ptr++) {
                uint64_t s2 = skel.chainRowSpan[ptr];
                uint64_t relRow = skel.chainRowsTillEnd[ptr - 1] - nRowsAbove;
                uint64_t s2_size =
                    skel.chainRowsTillEnd[ptr] - nRowsAbove - relRow;

                uint64_t pos = bisect(skel.chainRowSpan.data() + bisectStart,
                                      bisectEnd - bisectStart, s2);
                uint64_t chainOffset = skel.chainData[bisectStart + pos];
                T* targetData = data + spanOffsetInLump + chainOffset;

                stridedMatSub(targetData, targetLumpSize,
                              tempBuffer + nRowsChain * relRow, nRowsChain,
                              s2_size, nRowsChain);
            }
        }
    }

    // temporary data
    std::vector<T> tempBuffer;
    std::vector<uint64_t> spanToChainOffset;

    OpStat elimStat;
    OpStat potrfStat;
    uint64_t potrfBiggestN = 0;
    OpStat trsmStat;
    OpStat sygeStat;
    uint64_t gemmCalls = 0;
    uint64_t syrkCalls = 0;
    OpStat asmblStat;
};