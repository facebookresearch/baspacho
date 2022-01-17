#include "BlockMatrix.h"

#include <glog/logging.h>

#include <iostream>

#include "Utils.h"

using namespace std;

BlockMatrixSkel::BlockMatrixSkel(const vector<uint64_t>& paramStart,
                                 const vector<uint64_t>& aggregParamStart,
                                 const vector<uint64_t>& colPtr,
                                 const vector<uint64_t>& rowInd)
    : paramStart(paramStart), aggregParamStart(aggregParamStart) {
    CHECK_GE(paramStart.size(), aggregParamStart.size());
    CHECK_GE(aggregParamStart.size(), 1);
    CHECK_EQ(paramStart.size() - 1,
             aggregParamStart[aggregParamStart.size() - 1]);
    CHECK_EQ(colPtr.size(), aggregParamStart.size());
    CHECK(isStrictlyIncreasing(paramStart, 0, paramStart.size()));
    CHECK(isStrictlyIncreasing(aggregParamStart, 0, aggregParamStart.size()));

    uint64_t totSize = paramStart[paramStart.size() - 1];
    uint64_t numParams = paramStart.size() - 1;
    uint64_t numAggregs = aggregParamStart.size() - 1;

    paramToAggreg.resize(numParams);
    aggregStart.resize(numAggregs + 1);
    for (size_t a = 0; a < numAggregs; a++) {
        uint64_t aStart = aggregParamStart[a];
        uint64_t aEnd = aggregParamStart[a + 1];
        aggregStart[a] = paramStart[aStart];
        for (size_t i = aStart; i < aEnd; i++) {
            paramToAggreg[i] = a;
        }
    }
    aggregStart[numAggregs] = totSize;

    blockColDataPtr.resize(numAggregs + 1);
    blockRowParam.clear();
    blockData.clear();

    blockColGatheredDataPtr.resize(numAggregs + 1);
    blockRowAggreg.clear();
    blockRowAggregParamPtr.clear();
    uint64_t dataPtr = 0;
    uint64_t gatheredDataPtr = 0;
    for (size_t a = 0; a < numAggregs; a++) {
        uint64_t colStart = colPtr[a];
        uint64_t colEnd = colPtr[a + 1];
        CHECK(isStrictlyIncreasing(rowInd, colStart, colEnd));
        uint64_t aParamStart = aggregParamStart[a];
        uint64_t aParamEnd = aggregParamStart[a + 1];
        uint64_t aParamSize = aParamEnd - aParamStart;
        uint64_t aDataSize = aggregStart[a + 1] - aggregStart[a];

        // check the initial section is the set of params from `a`, and
        // therefore the full diagonal block is contained in the matrix
        CHECK_GE(colEnd - colStart, aParamSize)
            << "Column must contain full diagonal block";
        CHECK_EQ(rowInd[colStart], aParamStart)
            << "Column data must start at diagonal block";
        CHECK_EQ(rowInd[colStart + aParamSize - 1], aParamEnd - 1)
            << "Column must contain full diagonal block";

        blockColDataPtr[a] = blockRowParam.size();
        blockColGatheredDataPtr[a] = blockRowAggreg.size();
        uint64_t currentRowAggreg = kInvalid;
        uint64_t numRowsSkipped = 0;
        for (size_t i = colStart; i < colEnd; i++) {
            uint64_t p = rowInd[i];
            blockRowParam.push_back(p);
            blockData.push_back(dataPtr);
            dataPtr += aDataSize * (paramStart[p + 1] - paramStart[p]);
            numRowsSkipped += paramStart[p + 1] - paramStart[p];
            endBlockNumRowsAbove.push_back(numRowsSkipped);

            uint64_t rowAggreg = paramToAggreg[p];
            if (rowAggreg != currentRowAggreg) {
                currentRowAggreg = rowAggreg;
                blockRowAggreg.push_back(rowAggreg);
                blockRowAggregParamPtr.push_back(i - colStart);
            }
        }
        blockRowAggreg.push_back(kInvalid);
        blockRowAggregParamPtr.push_back(colEnd - colStart);
    }
    blockColDataPtr[numAggregs] = blockRowParam.size();
    blockColGatheredDataPtr[numAggregs] = blockRowAggreg.size();
    blockData.push_back(dataPtr);

    /*
    std::vector<uint64_t> slabRowPtr;
    std::vector<uint64_t> slabAggregInd;
    std::vector<uint64_t> slabSliceColInd;
    */

    slabRowPtr.assign(numAggregs + 1, 0);
    for (size_t a = 0; a < numAggregs; a++) {
        for (uint64_t i = blockColGatheredDataPtr[a];
             i < blockColGatheredDataPtr[a + 1] - 1; i++) {
            uint64_t rowAggreg = blockRowAggreg[i];
            slabRowPtr[rowAggreg]++;
        }
    }
    uint64_t numSlabs = cumSum(slabRowPtr);
    slabAggregInd.resize(numSlabs);
    slabColInd.resize(numSlabs);
    for (size_t a = 0; a < numAggregs; a++) {
        for (uint64_t i = blockColGatheredDataPtr[a];
             i < blockColGatheredDataPtr[a + 1] - 1; i++) {
            uint64_t rowAggreg = blockRowAggreg[i];
            slabAggregInd[slabRowPtr[rowAggreg]] = a;
            slabColInd[slabRowPtr[rowAggreg]] = i - blockColGatheredDataPtr[a];
            slabRowPtr[rowAggreg]++;
        }
    }
    rewind(slabRowPtr);
}

Eigen::MatrixXd BlockMatrixSkel::densify(const std::vector<double>& data) {
    uint64_t totData = blockData[blockData.size() - 1];
    CHECK_EQ(totData, data.size());

    uint64_t totSize = paramStart[paramStart.size() - 1];
    Eigen::MatrixXd retv(totSize, totSize);
    retv.setZero();

    for (size_t a = 0; a < blockColDataPtr.size() - 1; a++) {
        uint64_t aStart = aggregStart[a];
        uint64_t aSize = aggregStart[a + 1] - aStart;
        uint64_t colStart = blockColDataPtr[a];
        uint64_t colEnd = blockColDataPtr[a + 1];
        for (uint64_t i = colStart; i < colEnd; i++) {
            uint64_t p = blockRowParam[i];
            uint64_t pStart = paramStart[p];
            uint64_t pSize = paramStart[p + 1] - pStart;
            uint64_t dataPtr = blockData[i];

            retv.block(pStart, aStart, pSize, aSize) =
                Eigen::Map<const MatRMaj<double>>(data.data() + dataPtr, pSize,
                                                  aSize);
        }
    }

    return retv;
}

void BlockMatrixSkel::damp(std::vector<double>& data, double alpha,
                           double beta) {
    uint64_t totData = blockData[blockData.size() - 1];
    CHECK_EQ(totData, data.size());

    uint64_t totSize = paramStart[paramStart.size() - 1];

    for (size_t a = 0; a < blockColDataPtr.size() - 1; a++) {
        uint64_t aStart = aggregStart[a];
        uint64_t aSize = aggregStart[a + 1] - aStart;
        uint64_t colStart = blockColDataPtr[a];
        uint64_t dataPtr = blockData[colStart];

        Eigen::Map<MatRMaj<double>> block(data.data() + dataPtr, aSize, aSize);
        block.diagonal() *= (1 + alpha);
        block.diagonal().array() += beta;
    }
}
