#include "BlockMatrix.h"

#include <glog/logging.h>

#include <iostream>

#include "Utils.h"

using namespace std;

BlockMatrixSkel::BlockMatrixSkel(const vector<uint64_t>& spanStart,
                                 const vector<uint64_t>& rangeToSpan,
                                 const vector<uint64_t>& colPtr,
                                 const vector<uint64_t>& rowInd)
    : spanStart(spanStart), rangeToSpan(rangeToSpan) {
    CHECK_GE(spanStart.size(), rangeToSpan.size());
    CHECK_GE(rangeToSpan.size(), 1);
    CHECK_EQ(spanStart.size() - 1, rangeToSpan[rangeToSpan.size() - 1]);
    CHECK_EQ(colPtr.size(), rangeToSpan.size());
    CHECK(isStrictlyIncreasing(spanStart, 0, spanStart.size()));
    CHECK(isStrictlyIncreasing(rangeToSpan, 0, rangeToSpan.size()));

    uint64_t totSize = spanStart[spanStart.size() - 1];
    uint64_t numSpans = spanStart.size() - 1;
    uint64_t numRanges = rangeToSpan.size() - 1;

    spanToRange.resize(numSpans);
    rangeStart.resize(numRanges + 1);
    for (size_t a = 0; a < numRanges; a++) {
        uint64_t aStart = rangeToSpan[a];
        uint64_t aEnd = rangeToSpan[a + 1];
        rangeStart[a] = spanStart[aStart];
        for (size_t i = aStart; i < aEnd; i++) {
            spanToRange[i] = a;
        }
    }
    rangeStart[numRanges] = totSize;

    sliceColPtr.resize(numRanges + 1);
    sliceRowSpan.clear();
    sliceData.clear();

    slabColPtr.resize(numRanges + 1);
    slabRowRange.clear();
    slabSliceColOrd.clear();
    uint64_t dataPtr = 0;
    uint64_t gatheredDataPtr = 0;
    for (size_t a = 0; a < numRanges; a++) {
        uint64_t colStart = colPtr[a];
        uint64_t colEnd = colPtr[a + 1];
        CHECK(isStrictlyIncreasing(rowInd, colStart, colEnd));
        uint64_t aParamStart = rangeToSpan[a];
        uint64_t aParamEnd = rangeToSpan[a + 1];
        uint64_t aParamSize = aParamEnd - aParamStart;
        uint64_t aDataSize = rangeStart[a + 1] - rangeStart[a];

        // check the initial section is the set of params from `a`, and
        // therefore the full diagonal block is contained in the matrix
        CHECK_GE(colEnd - colStart, aParamSize)
            << "Column must contain full diagonal block";
        CHECK_EQ(rowInd[colStart], aParamStart)
            << "Column data must start at diagonal block";
        CHECK_EQ(rowInd[colStart + aParamSize - 1], aParamEnd - 1)
            << "Column must contain full diagonal block";

        sliceColPtr[a] = sliceRowSpan.size();
        slabColPtr[a] = slabRowRange.size();
        uint64_t currentRowAggreg = kInvalid;
        uint64_t numRowsSkipped = 0;
        for (size_t i = colStart; i < colEnd; i++) {
            uint64_t p = rowInd[i];
            sliceRowSpan.push_back(p);
            sliceData.push_back(dataPtr);
            dataPtr += aDataSize * (spanStart[p + 1] - spanStart[p]);
            numRowsSkipped += spanStart[p + 1] - spanStart[p];
            sliceRowsTillEnd.push_back(numRowsSkipped);

            uint64_t rowAggreg = spanToRange[p];
            if (rowAggreg != currentRowAggreg) {
                currentRowAggreg = rowAggreg;
                slabRowRange.push_back(rowAggreg);
                slabSliceColOrd.push_back(i - colStart);
            }
        }
        slabRowRange.push_back(kInvalid);
        slabSliceColOrd.push_back(colEnd - colStart);
    }
    sliceColPtr[numRanges] = sliceRowSpan.size();
    slabColPtr[numRanges] = slabRowRange.size();
    sliceData.push_back(dataPtr);

    /*
    std::vector<uint64_t> slabRowPtr;
    std::vector<uint64_t> slabColRange;
    std::vector<uint64_t> slabSliceColOrd;
    */

    slabRowPtr.assign(numRanges + 1, 0);
    for (size_t a = 0; a < numRanges; a++) {
        for (uint64_t i = slabColPtr[a]; i < slabColPtr[a + 1] - 1; i++) {
            uint64_t rowAggreg = slabRowRange[i];
            slabRowPtr[rowAggreg]++;
        }
    }
    uint64_t numSlabs = cumSum(slabRowPtr);
    slabColRange.resize(numSlabs);
    slabColOrd.resize(numSlabs);
    for (size_t a = 0; a < numRanges; a++) {
        for (uint64_t i = slabColPtr[a]; i < slabColPtr[a + 1] - 1; i++) {
            uint64_t rowAggreg = slabRowRange[i];
            slabColRange[slabRowPtr[rowAggreg]] = a;
            slabColOrd[slabRowPtr[rowAggreg]] = i - slabColPtr[a];
            slabRowPtr[rowAggreg]++;
        }
    }
    rewind(slabRowPtr);
}

Eigen::MatrixXd BlockMatrixSkel::densify(const std::vector<double>& data) {
    uint64_t totData = sliceData[sliceData.size() - 1];
    CHECK_EQ(totData, data.size());

    uint64_t totSize = spanStart[spanStart.size() - 1];
    Eigen::MatrixXd retv(totSize, totSize);
    retv.setZero();

    for (size_t a = 0; a < sliceColPtr.size() - 1; a++) {
        uint64_t aStart = rangeStart[a];
        uint64_t aSize = rangeStart[a + 1] - aStart;
        uint64_t colStart = sliceColPtr[a];
        uint64_t colEnd = sliceColPtr[a + 1];
        for (uint64_t i = colStart; i < colEnd; i++) {
            uint64_t p = sliceRowSpan[i];
            uint64_t pStart = spanStart[p];
            uint64_t pSize = spanStart[p + 1] - pStart;
            uint64_t dataPtr = sliceData[i];

            retv.block(pStart, aStart, pSize, aSize) =
                Eigen::Map<const MatRMaj<double>>(data.data() + dataPtr, pSize,
                                                  aSize);
        }
    }

    return retv;
}

void BlockMatrixSkel::damp(std::vector<double>& data, double alpha,
                           double beta) {
    uint64_t totData = sliceData[sliceData.size() - 1];
    CHECK_EQ(totData, data.size());

    uint64_t totSize = spanStart[spanStart.size() - 1];

    for (size_t a = 0; a < sliceColPtr.size() - 1; a++) {
        uint64_t aStart = rangeStart[a];
        uint64_t aSize = rangeStart[a + 1] - aStart;
        uint64_t colStart = sliceColPtr[a];
        uint64_t dataPtr = sliceData[colStart];

        Eigen::Map<MatRMaj<double>> block(data.data() + dataPtr, aSize, aSize);
        block.diagonal() *= (1 + alpha);
        block.diagonal().array() += beta;
    }
}
