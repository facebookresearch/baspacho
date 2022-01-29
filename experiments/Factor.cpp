#include "Factor.h"

#include <glog/logging.h>

#include <Eigen/Eigenvalues>

void factor(const CoalescedBlockMatrixSkel& skel, std::vector<double>& data) {
    for (size_t l = 0; l < skel.lumpToSpan.size() - 1; l++) {
        factorLump(skel, data, l);

        uint64_t ptrBegin = skel.boardColPtr[l];
        uint64_t numBoards = skel.boardColPtr[l + 1] - ptrBegin;

        CHECK_EQ(skel.boardRowLump[ptrBegin], l);

        for (uint64_t i = 1; i < numBoards - 1; i++) {
            eliminateBoard(skel, data, l, i);
        }
    }
}

void factorLump(const CoalescedBlockMatrixSkel& skel, std::vector<double>& data,
                uint64_t lump) {
    LOG(INFO) << "a: " << lump;
    uint64_t lumpStart = skel.lumpStart[lump];
    uint64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
    uint64_t colStart = skel.chainColPtr[lump];
    uint64_t dataPtr = skel.chainData[colStart];

    // compute lower diag cholesky dec on diagonal block
    LOG(INFO) << "d: " << data.data() << ", ptr: " << dataPtr
              << ", asz: " << lumpSize;
    Eigen::Map<MatRMaj<double>> diagBlock(data.data() + dataPtr, lumpSize,
                                          lumpSize);
    { Eigen::LLT<Eigen::Ref<MatRMaj<double>>> llt(diagBlock); }
    uint64_t gatheredStart = skel.boardColPtr[lump];
    uint64_t gatheredEnd = skel.boardColPtr[lump + 1];
    uint64_t rowDataStart = skel.boardChainColOrd[gatheredStart + 1];
    uint64_t rowDataEnd = skel.boardChainColOrd[gatheredEnd - 1];
    uint64_t belowDiagStart = skel.chainData[colStart + rowDataStart];
    uint64_t numRows = skel.chainRowsTillEnd[colStart + rowDataEnd - 1] -
                       skel.chainRowsTillEnd[colStart + rowDataStart - 1];

    LOG(INFO) << "d: " << data.data() << ", ptr: " << belowDiagStart
              << ", nrows: " << numRows;
    Eigen::Map<MatRMaj<double>> belowDiagBlock(data.data() + belowDiagStart,
                                               numRows, lumpSize);
    diagBlock.triangularView<Eigen::Lower>()
        .transpose()
        .solveInPlace<Eigen::OnTheRight>(belowDiagBlock);
}

uint64_t bisect(const uint64_t* array, uint64_t size, uint64_t needle) {
    uint64_t a = 0, b = size;
    while (b - a > 1) {
        uint64_t m = (a + b) / 2;
        if (needle >= array[m]) {
            a = m;
        } else {
            b = m;
        }
    }
    return a;
}

// returns (offset, stride)
std::pair<uint64_t, uint64_t> findBlock(const CoalescedBlockMatrixSkel& skel,
                                        uint64_t cParam, uint64_t rParam) {
    uint64_t lump = skel.spanToLump[cParam];
    uint64_t lumpSize = skel.lumpStart[lump + 1] - skel.lumpStart[lump];
    uint64_t offsetInAggreg = skel.spanStart[cParam] - skel.lumpStart[lump];
    uint64_t start = skel.chainColPtr[lump];
    uint64_t end = skel.chainColPtr[lump + 1];
    // bisect to find rParam in chainRowSpan[start:end]
    uint64_t pos =
        bisect(skel.chainRowSpan.data() + start, end - start, rParam);
    CHECK_EQ(skel.chainRowSpan[start + pos], rParam);
    return std::make_pair(skel.chainData[start + pos] + offsetInAggreg,
                          lumpSize);
}

using OuterStride = Eigen::OuterStride<>;
using OuterStridedMatM = Eigen::Map<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0,
    OuterStride>;

void eliminateBoard(const CoalescedBlockMatrixSkel& skel,
                    std::vector<double>& data, uint64_t lump,
                    uint64_t rowItem) {
    uint64_t lumpStart = skel.lumpStart[lump];
    uint64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
    uint64_t colStart = skel.chainColPtr[lump];

    uint64_t gatheredStart = skel.boardColPtr[lump];
    uint64_t gatheredEnd = skel.boardColPtr[lump + 1];
    uint64_t rowDataStart = skel.boardChainColOrd[gatheredStart + rowItem];
    uint64_t rowDataEnd0 = skel.boardChainColOrd[gatheredStart + rowItem + 1];
    uint64_t rowDataEnd1 = skel.boardChainColOrd[gatheredEnd - 1];
    uint64_t belowDiagStart = skel.chainData[colStart + rowDataStart];
    uint64_t rowStart = skel.chainRowsTillEnd[colStart + rowDataStart - 1];
    uint64_t numRowsSub =
        skel.chainRowsTillEnd[colStart + rowDataEnd0 - 1] - rowStart;
    uint64_t numRowsFull =
        skel.chainRowsTillEnd[colStart + rowDataEnd1 - 1] - rowStart;

    // LOG(INFO) << "lump: " << lump << ", item: " << rowItem;
    // LOG(INFO) << "sizes: " << numRowsSub << " " << numRowsFull << " " <<
    // lumpSize;
    Eigen::Map<MatRMaj<double>> belowDiagBlockSub(data.data() + belowDiagStart,
                                                  numRowsSub, lumpSize);
    Eigen::Map<MatRMaj<double>> belowDiagBlockFull(data.data() + belowDiagStart,
                                                   numRowsFull, lumpSize);

    // LOG(INFO) << "sub:\n" << belowDiagBlockSub;
    // LOG(INFO) << "full:\n" << belowDiagBlockFull;
    MatRMaj<double> prod = belowDiagBlockFull * belowDiagBlockSub.transpose();
    // LOG(INFO) << "prod:\n" << prod;

    for (uint64_t c = rowDataStart; c < rowDataEnd0; c++) {
        uint64_t cStart = skel.chainRowsTillEnd[colStart + c - 1] - rowStart;
        uint64_t cSize =
            skel.chainRowsTillEnd[colStart + c] - cStart - rowStart;
        uint64_t cParam = skel.chainRowSpan[colStart + c];
        for (uint64_t r = rowDataStart; r < rowDataEnd1; r++) {
            uint64_t rBegin =
                skel.chainRowsTillEnd[colStart + r - 1] - rowStart;
            uint64_t rSize =
                skel.chainRowsTillEnd[colStart + r] - rBegin - rowStart;
            uint64_t rParam = skel.chainRowSpan[colStart + r];
            // LOG(INFO) << rBegin << ":" << rSize << "@" << rParam << " x " <<
            // cStart << ":" << cSize << "@" << cParam;

            auto [offset, stride] = findBlock(skel, cParam, rParam);
            OuterStridedMatM target(data.data() + offset, rSize, cSize,
                                    OuterStride(stride));
            auto orig = prod.block(rBegin, cStart, rSize, cSize);
            // LOG(INFO) << "orig:\n" << orig;
            // LOG(INFO) << "target:\n" << target;
            target -= orig;
        }
    }
}
