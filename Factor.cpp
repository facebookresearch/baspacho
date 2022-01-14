#include "Factor.h"

#include <glog/logging.h>

#include <Eigen/Eigenvalues>

void factor(const BlockMatrixSkel& skel, std::vector<double>& data) {
    for (size_t a = 0; a < skel.blockColDataPtr.size() - 1; a++) {
        factorAggreg(skel, data, a);

        uint64_t rowAggregStart = skel.blockColGatheredDataPtr[a];
        uint64_t rowNumAggregs =
            skel.blockColGatheredDataPtr[a + 1] - rowAggregStart;

        CHECK_EQ(skel.blockRowAggreg[rowAggregStart], a);

        for (uint64_t i = 1; i < rowNumAggregs - 1; i++) {
            eliminateAggregItem(skel, data, a, i);
        }
    }
}

void factorAggreg(const BlockMatrixSkel& skel, std::vector<double>& data,
                  uint64_t aggreg) {
    uint64_t aggregStart = skel.aggregStart[aggreg];
    uint64_t aggregSize = skel.aggregStart[aggreg + 1] - aggregStart;
    uint64_t colStart = skel.blockColDataPtr[aggreg];
    uint64_t dataPtr = skel.blockData[colStart];

    // compute lower diag cholesky dec on diagonal block
    Eigen::Map<MatRMaj<double>> diagBlock(data.data() + dataPtr, aggregSize,
                                          aggregSize);
    { Eigen::LLT<Eigen::Ref<MatRMaj<double>>> llt(diagBlock); }
    uint64_t gatheredStart = skel.blockColGatheredDataPtr[aggreg];
    uint64_t gatheredEnd = skel.blockColGatheredDataPtr[aggreg + 1];
    uint64_t rowDataStart = skel.blockRowAggregParamPtr[gatheredStart + 1];
    uint64_t rowDataEnd = skel.blockRowAggregParamPtr[gatheredEnd - 1];
    uint64_t belowDiagStart = skel.blockData[colStart + rowDataStart];
    uint64_t numRows = skel.endBlockNumRowsAbove[colStart + rowDataEnd - 1] -
                       skel.endBlockNumRowsAbove[colStart + rowDataStart - 1];

    Eigen::Map<MatRMaj<double>> belowDiagBlock(data.data() + belowDiagStart,
                                               numRows, aggregSize);
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
std::pair<uint64_t, uint64_t> findBlock(const BlockMatrixSkel& skel,
                                        uint64_t cParam, uint64_t rParam) {
    uint64_t aggreg = skel.paramToAggreg[cParam];
    uint64_t aggregSize =
        skel.aggregStart[aggreg + 1] - skel.aggregStart[aggreg];
    uint64_t offsetInAggreg =
        skel.paramStart[cParam] - skel.aggregStart[aggreg];
    uint64_t start = skel.blockColDataPtr[aggreg];
    uint64_t end = skel.blockColDataPtr[aggreg + 1];
    // bisect to find rParam in blockRowParam[start:end]
    uint64_t pos =
        bisect(skel.blockRowParam.data() + start, end - start, rParam);
    CHECK_EQ(skel.blockRowParam[start + pos], rParam);
    return std::make_pair(skel.blockData[start + pos] + offsetInAggreg,
                          aggregSize);
}

using OuterStride = Eigen::OuterStride<>;
using OuterStridedMatM = Eigen::Map<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0,
    OuterStride>;

void eliminateAggregItem(const BlockMatrixSkel& skel, std::vector<double>& data,
                         uint64_t aggreg, uint64_t rowItem) {
    uint64_t aggregStart = skel.aggregStart[aggreg];
    uint64_t aggregSize = skel.aggregStart[aggreg + 1] - aggregStart;
    uint64_t colStart = skel.blockColDataPtr[aggreg];

    uint64_t gatheredStart = skel.blockColGatheredDataPtr[aggreg];
    uint64_t gatheredEnd = skel.blockColGatheredDataPtr[aggreg + 1];
    uint64_t rowDataStart =
        skel.blockRowAggregParamPtr[gatheredStart + rowItem];
    uint64_t rowDataEnd0 =
        skel.blockRowAggregParamPtr[gatheredStart + rowItem + 1];
    uint64_t rowDataEnd1 = skel.blockRowAggregParamPtr[gatheredEnd - 1];
    uint64_t belowDiagStart = skel.blockData[colStart + rowDataStart];
    uint64_t rowStart = skel.endBlockNumRowsAbove[colStart + rowDataStart - 1];
    uint64_t numRowsSub =
        skel.endBlockNumRowsAbove[colStart + rowDataEnd0 - 1] - rowStart;
    uint64_t numRowsFull =
        skel.endBlockNumRowsAbove[colStart + rowDataEnd1 - 1] - rowStart;

    // LOG(INFO) << "aggreg: " << aggreg << ", item: " << rowItem;
    // LOG(INFO) << "sizes: " << numRowsSub << " " << numRowsFull << " " <<
    // aggregSize;
    Eigen::Map<MatRMaj<double>> belowDiagBlockSub(data.data() + belowDiagStart,
                                                  numRowsSub, aggregSize);
    Eigen::Map<MatRMaj<double>> belowDiagBlockFull(data.data() + belowDiagStart,
                                                   numRowsFull, aggregSize);

    // LOG(INFO) << "sub:\n" << belowDiagBlockSub;
    // LOG(INFO) << "full:\n" << belowDiagBlockFull;
    MatRMaj<double> prod = belowDiagBlockFull * belowDiagBlockSub.transpose();
    // LOG(INFO) << "prod:\n" << prod;

    for (uint64_t c = rowDataStart; c < rowDataEnd0; c++) {
        uint64_t cStart =
            skel.endBlockNumRowsAbove[colStart + c - 1] - rowStart;
        uint64_t cSize =
            skel.endBlockNumRowsAbove[colStart + c] - cStart - rowStart;
        uint64_t cParam = skel.blockRowParam[colStart + c];
        for (uint64_t r = rowDataStart; r < rowDataEnd1; r++) {
            uint64_t rStart =
                skel.endBlockNumRowsAbove[colStart + r - 1] - rowStart;
            uint64_t rSize =
                skel.endBlockNumRowsAbove[colStart + r] - rStart - rowStart;
            uint64_t rParam = skel.blockRowParam[colStart + r];
            // LOG(INFO) << rStart << ":" << rSize << "@" << rParam << " x " <<
            // cStart << ":" << cSize << "@" << cParam;

            auto [offset, stride] = findBlock(skel, cParam, rParam);
            OuterStridedMatM target(data.data() + offset, rSize, cSize,
                                    OuterStride(stride));
            auto orig = prod.block(rStart, cStart, rSize, cSize);
            // LOG(INFO) << "orig:\n" << orig;
            // LOG(INFO) << "target:\n" << target;
            target -= orig;
        }
    }
}
