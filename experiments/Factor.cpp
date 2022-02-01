#include "Factor.h"

#include <Eigen/Eigenvalues>

#include "../baspacho/DebugMacros.h"

void factor(const CoalescedBlockMatrixSkel& skel, std::vector<double>& data) {
    for (size_t l = 0; l < skel.lumpToSpan.size() - 1; l++) {
        factorLump(skel, data, l);

        int64_t ptrBegin = skel.boardColPtr[l];
        int64_t numBoards = skel.boardColPtr[l + 1] - ptrBegin;

        BASPACHO_CHECK_EQ(skel.boardRowLump[ptrBegin], l);

        for (int64_t i = 1; i < numBoards - 1; i++) {
            eliminateBoard(skel, data, l, i);
        }
    }
}

void factorLump(const CoalescedBlockMatrixSkel& skel, std::vector<double>& data,
                int64_t lump) {
    int64_t lumpStart = skel.lumpStart[lump];
    int64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
    int64_t colStart = skel.chainColPtr[lump];
    int64_t dataPtr = skel.chainData[colStart];

    // compute lower diag cholesky dec on diagonal block
    Eigen::Map<MatRMaj<double>> diagBlock(data.data() + dataPtr, lumpSize,
                                          lumpSize);
    { Eigen::LLT<Eigen::Ref<MatRMaj<double>>> llt(diagBlock); }
    int64_t gatheredStart = skel.boardColPtr[lump];
    int64_t gatheredEnd = skel.boardColPtr[lump + 1];
    int64_t rowDataStart = skel.boardChainColOrd[gatheredStart + 1];
    int64_t rowDataEnd = skel.boardChainColOrd[gatheredEnd - 1];
    int64_t belowDiagStart = skel.chainData[colStart + rowDataStart];
    int64_t numRows = skel.chainRowsTillEnd[colStart + rowDataEnd - 1] -
                      skel.chainRowsTillEnd[colStart + rowDataStart - 1];

    Eigen::Map<MatRMaj<double>> belowDiagBlock(data.data() + belowDiagStart,
                                               numRows, lumpSize);
    diagBlock.triangularView<Eigen::Lower>()
        .transpose()
        .solveInPlace<Eigen::OnTheRight>(belowDiagBlock);
}

// returns (offset, stride)
std::pair<int64_t, int64_t> findBlock(const CoalescedBlockMatrixSkel& skel,
                                      int64_t cParam, int64_t rParam) {
    int64_t lump = skel.spanToLump[cParam];
    int64_t lumpSize = skel.lumpStart[lump + 1] - skel.lumpStart[lump];
    int64_t offsetInAggreg = skel.spanStart[cParam] - skel.lumpStart[lump];
    int64_t start = skel.chainColPtr[lump];
    int64_t end = skel.chainColPtr[lump + 1];
    // bisect to find rParam in chainRowSpan[start:end]
    int64_t pos = bisect(skel.chainRowSpan.data() + start, end - start, rParam);
    BASPACHO_CHECK_EQ(skel.chainRowSpan[start + pos], rParam);
    return std::make_pair(skel.chainData[start + pos] + offsetInAggreg,
                          lumpSize);
}

using OuterStride = Eigen::OuterStride<>;
using OuterStridedMatM = Eigen::Map<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0,
    OuterStride>;

void eliminateBoard(const CoalescedBlockMatrixSkel& skel,
                    std::vector<double>& data, int64_t lump, int64_t rowItem) {
    int64_t lumpStart = skel.lumpStart[lump];
    int64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
    int64_t colStart = skel.chainColPtr[lump];

    int64_t gatheredStart = skel.boardColPtr[lump];
    int64_t gatheredEnd = skel.boardColPtr[lump + 1];
    int64_t rowDataStart = skel.boardChainColOrd[gatheredStart + rowItem];
    int64_t rowDataEnd0 = skel.boardChainColOrd[gatheredStart + rowItem + 1];
    int64_t rowDataEnd1 = skel.boardChainColOrd[gatheredEnd - 1];
    int64_t belowDiagStart = skel.chainData[colStart + rowDataStart];
    int64_t rowStart = skel.chainRowsTillEnd[colStart + rowDataStart - 1];
    int64_t numRowsSub =
        skel.chainRowsTillEnd[colStart + rowDataEnd0 - 1] - rowStart;
    int64_t numRowsFull =
        skel.chainRowsTillEnd[colStart + rowDataEnd1 - 1] - rowStart;

    // std::cout  << "lump: " << lump << ", item: " << rowItem << std::endl;
    // LOG(INFO) << "sizes: " << numRowsSub << " " << numRowsFull << " " <<
    // lumpSize;
    Eigen::Map<MatRMaj<double>> belowDiagBlockSub(data.data() + belowDiagStart,
                                                  numRowsSub, lumpSize);
    Eigen::Map<MatRMaj<double>> belowDiagBlockFull(data.data() + belowDiagStart,
                                                   numRowsFull, lumpSize);

    // std::cout  << "sub:\n" << belowDiagBlockSub << std::endl;
    // std::cout  << "full:\n" << belowDiagBlockFull << std::endl;
    MatRMaj<double> prod = belowDiagBlockFull * belowDiagBlockSub.transpose();
    // std::cout  << "prod:\n" << prod << std::endl;

    for (int64_t c = rowDataStart; c < rowDataEnd0; c++) {
        int64_t cStart = skel.chainRowsTillEnd[colStart + c - 1] - rowStart;
        int64_t cSize = skel.chainRowsTillEnd[colStart + c] - cStart - rowStart;
        int64_t cParam = skel.chainRowSpan[colStart + c];
        for (int64_t r = rowDataStart; r < rowDataEnd1; r++) {
            int64_t rBegin = skel.chainRowsTillEnd[colStart + r - 1] - rowStart;
            int64_t rSize =
                skel.chainRowsTillEnd[colStart + r] - rBegin - rowStart;
            int64_t rParam = skel.chainRowSpan[colStart + r];
            // LOG(INFO) << rBegin << ":" << rSize << "@" << rParam << " x " <<
            // cStart << ":" << cSize << "@" << cParam;

            auto [offset, stride] = findBlock(skel, cParam, rParam);
            OuterStridedMatM target(data.data() + offset, rSize, cSize,
                                    OuterStride(stride));
            auto orig = prod.block(rBegin, cStart, rSize, cSize);
            // std::cout  << "orig:\n" << orig << std::endl;
            // std::cout  << "target:\n" << target << std::endl;
            target -= orig;
        }
    }
}
