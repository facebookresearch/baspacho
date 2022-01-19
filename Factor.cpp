#include "Factor.h"

#include <glog/logging.h>

#include <Eigen/Eigenvalues>

void factor(const BlockMatrixSkel& skel, std::vector<double>& data) {
    for (size_t a = 0; a < skel.sliceColPtr.size() - 1; a++) {
        factorAggreg(skel, data, a);

        uint64_t rowAggregStart = skel.slabColPtr[a];
        uint64_t rowNumAggregs = skel.slabColPtr[a + 1] - rowAggregStart;

        CHECK_EQ(skel.slabRowRange[rowAggregStart], a);

        for (uint64_t i = 1; i < rowNumAggregs - 1; i++) {
            eliminateAggregItem(skel, data, a, i);
        }
    }
}

void factorAggreg(const BlockMatrixSkel& skel, std::vector<double>& data,
                  uint64_t aggreg) {
    LOG(INFO) << "a: " << aggreg;
    uint64_t rangeStart = skel.rangeStart[aggreg];
    uint64_t aggregSize = skel.rangeStart[aggreg + 1] - rangeStart;
    uint64_t colStart = skel.sliceColPtr[aggreg];
    uint64_t dataPtr = skel.sliceData[colStart];

    // compute lower diag cholesky dec on diagonal block
    LOG(INFO) << "d: " << data.data() << ", ptr: " << dataPtr
              << ", asz: " << aggregSize;
    Eigen::Map<MatRMaj<double>> diagBlock(data.data() + dataPtr, aggregSize,
                                          aggregSize);
    { Eigen::LLT<Eigen::Ref<MatRMaj<double>>> llt(diagBlock); }
    uint64_t gatheredStart = skel.slabColPtr[aggreg];
    uint64_t gatheredEnd = skel.slabColPtr[aggreg + 1];
    uint64_t rowDataStart = skel.slabSliceColOrd[gatheredStart + 1];
    uint64_t rowDataEnd = skel.slabSliceColOrd[gatheredEnd - 1];
    uint64_t belowDiagStart = skel.sliceData[colStart + rowDataStart];
    uint64_t numRows = skel.sliceRowsTillEnd[colStart + rowDataEnd - 1] -
                       skel.sliceRowsTillEnd[colStart + rowDataStart - 1];

    LOG(INFO) << "d: " << data.data() << ", ptr: " << belowDiagStart
              << ", nrows: " << numRows;
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
    uint64_t aggreg = skel.spanToRange[cParam];
    uint64_t aggregSize = skel.rangeStart[aggreg + 1] - skel.rangeStart[aggreg];
    uint64_t offsetInAggreg = skel.spanStart[cParam] - skel.rangeStart[aggreg];
    uint64_t start = skel.sliceColPtr[aggreg];
    uint64_t end = skel.sliceColPtr[aggreg + 1];
    // bisect to find rParam in sliceRowSpan[start:end]
    uint64_t pos =
        bisect(skel.sliceRowSpan.data() + start, end - start, rParam);
    CHECK_EQ(skel.sliceRowSpan[start + pos], rParam);
    return std::make_pair(skel.sliceData[start + pos] + offsetInAggreg,
                          aggregSize);
}

using OuterStride = Eigen::OuterStride<>;
using OuterStridedMatM = Eigen::Map<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0,
    OuterStride>;

void eliminateAggregItem(const BlockMatrixSkel& skel, std::vector<double>& data,
                         uint64_t aggreg, uint64_t rowItem) {
    uint64_t rangeStart = skel.rangeStart[aggreg];
    uint64_t aggregSize = skel.rangeStart[aggreg + 1] - rangeStart;
    uint64_t colStart = skel.sliceColPtr[aggreg];

    uint64_t gatheredStart = skel.slabColPtr[aggreg];
    uint64_t gatheredEnd = skel.slabColPtr[aggreg + 1];
    uint64_t rowDataStart = skel.slabSliceColOrd[gatheredStart + rowItem];
    uint64_t rowDataEnd0 = skel.slabSliceColOrd[gatheredStart + rowItem + 1];
    uint64_t rowDataEnd1 = skel.slabSliceColOrd[gatheredEnd - 1];
    uint64_t belowDiagStart = skel.sliceData[colStart + rowDataStart];
    uint64_t rowStart = skel.sliceRowsTillEnd[colStart + rowDataStart - 1];
    uint64_t numRowsSub =
        skel.sliceRowsTillEnd[colStart + rowDataEnd0 - 1] - rowStart;
    uint64_t numRowsFull =
        skel.sliceRowsTillEnd[colStart + rowDataEnd1 - 1] - rowStart;

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
        uint64_t cStart = skel.sliceRowsTillEnd[colStart + c - 1] - rowStart;
        uint64_t cSize =
            skel.sliceRowsTillEnd[colStart + c] - cStart - rowStart;
        uint64_t cParam = skel.sliceRowSpan[colStart + c];
        for (uint64_t r = rowDataStart; r < rowDataEnd1; r++) {
            uint64_t rStart =
                skel.sliceRowsTillEnd[colStart + r - 1] - rowStart;
            uint64_t rSize =
                skel.sliceRowsTillEnd[colStart + r] - rStart - rowStart;
            uint64_t rParam = skel.sliceRowSpan[colStart + r];
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
