
#include <glog/logging.h>
#include <iostream>
#include "BlockMatrix.h"
#include "Utils.h"

using namespace std;


BlockMatrixSkel initBlockMatrixSkel(const vector<uint64_t>& paramStart,
                                    const vector<uint64_t>& aggregParamStart,
                                    const vector<vector<uint64_t>>& columnParams) {

    CHECK_GE(paramStart.size(), aggregParamStart.size());
    CHECK_GE(aggregParamStart.size(), 1);
    CHECK_EQ(paramStart.size() - 1, aggregParamStart[aggregParamStart.size()-1]);
    CHECK_EQ(columnParams.size(), aggregParamStart.size() - 1);
    CHECK(isStrictlyIncreasing(paramStart, 0, paramStart.size()));
    CHECK(isStrictlyIncreasing(aggregParamStart, 0, aggregParamStart.size()));

    uint64_t totSize = paramStart[paramStart.size() - 1];
    uint64_t numParams = paramStart.size() - 1;
    uint64_t numAggregs = aggregParamStart.size() - 1;
    
    BlockMatrixSkel retv;
    retv.paramStart = paramStart;
    retv.aggregParamStart = aggregParamStart;

    retv.paramToAggreg.resize(numParams);
    retv.aggregStart.resize(numAggregs + 1);
    for(size_t a = 0; a < numAggregs; a++) {
        uint64_t aggregStart = aggregParamStart[a];
        uint64_t aggregEnd = aggregParamStart[a+1];
        retv.aggregStart[a] = paramStart[aggregStart];
        for(size_t i = aggregStart; i < aggregEnd; i++) {
            retv.paramToAggreg[i] = a;
        }
    }
    retv.aggregStart[numAggregs] = totSize;

    retv.blockColDataPtr.resize(numAggregs + 1);
    retv.blockRowParam.clear();
    retv.blockData.clear();
    
    retv.blockColGatheredDataPtr.resize(numAggregs + 1);
    retv.blockRowAggreg.clear();
    retv.blockRowAggregParamPtr.clear();
    uint64_t dataPtr = 0;
    uint64_t gatheredDataPtr = 0;
    for(size_t a = 0; a < numAggregs; a++) {
        CHECK(isStrictlyIncreasing(columnParams[a], 0, columnParams[a].size()));
        uint64_t aParamStart = aggregParamStart[a];
        uint64_t aParamEnd = aggregParamStart[a+1];
        uint64_t aParamSize = aParamEnd - aParamStart;
        uint64_t aDataSize = retv.aggregStart[a+1] - retv.aggregStart[a];

        // check the initial section is the set of params from `a`, and therefore
        // the full diagonal block is contained in the matrix
        CHECK_GE(columnParams[a].size(), aParamSize) << "Column must contain full diagonal block";
        CHECK_EQ(columnParams[a][0], aParamStart) << "Column data must start at diagonal block";
        CHECK_EQ(columnParams[a][aParamSize-1], aParamEnd-1) << "Column must contain full diagonal block";

        retv.blockColDataPtr[a] = retv.blockRowParam.size();
        retv.blockColGatheredDataPtr[a] = retv.blockRowAggreg.size();
        uint64_t currentRowAggreg = kInvalid;
        uint64_t numRowsSkipped = 0;
        for(size_t i = 0; i < columnParams[a].size(); i++) {
            uint64_t p = columnParams[a][i];
            retv.blockRowParam.push_back(p);
            retv.blockData.push_back(dataPtr);
            dataPtr += aDataSize * (paramStart[p+1] - paramStart[p]);
            numRowsSkipped += paramStart[p+1] - paramStart[p];
            retv.endBlockNumRowsAbove.push_back(numRowsSkipped);

            uint64_t rowAggreg = retv.paramToAggreg[p];
            if(rowAggreg != currentRowAggreg) {
                currentRowAggreg = rowAggreg;
                retv.blockRowAggreg.push_back(rowAggreg);
                retv.blockRowAggregParamPtr.push_back(i);
            }
        }
        retv.blockRowAggreg.push_back(kInvalid);
        retv.blockRowAggregParamPtr.push_back(columnParams[a].size());
   }
   retv.blockColDataPtr[numAggregs] = retv.blockRowParam.size();
   retv.blockColGatheredDataPtr[numAggregs] = retv.blockRowAggreg.size();
   retv.blockData.push_back(dataPtr);
    
   return retv;
}

Eigen::MatrixXd densify(const BlockMatrixSkel& skel, const std::vector<double>& data) {
    uint64_t totData = skel.blockData[skel.blockData.size() - 1];
    CHECK_EQ(totData, data.size());

    uint64_t totSize = skel.paramStart[skel.paramStart.size() - 1];
    Eigen::MatrixXd retv(totSize, totSize);
    retv.setZero();

    for(size_t a = 0; a < skel.blockColDataPtr.size() - 1; a++) {
        uint64_t aggregStart = skel.aggregStart[a];
        uint64_t aggregSize = skel.aggregStart[a+1] - aggregStart;
        uint64_t colStart = skel.blockColDataPtr[a];
        uint64_t colEnd = skel.blockColDataPtr[a+1];
        for(uint64_t i = colStart; i < colEnd; i++) {
            uint64_t p = skel.blockRowParam[i];
            uint64_t paramStart = skel.paramStart[p];
            uint64_t paramSize = skel.paramStart[p+1] - paramStart;
            uint64_t dataPtr = skel.blockData[i];

            retv.block(paramStart, aggregStart, paramSize, aggregSize) =
                Eigen::Map<const MatRMaj<double>>(data.data() + dataPtr, paramSize, aggregSize);
        }
    }

    return retv;
}

void damp(const BlockMatrixSkel& skel, std::vector<double>& data, double alpha, double beta) {
    uint64_t totData = skel.blockData[skel.blockData.size() - 1];
    CHECK_EQ(totData, data.size());

    uint64_t totSize = skel.paramStart[skel.paramStart.size() - 1];

    for(size_t a = 0; a < skel.blockColDataPtr.size() - 1; a++) {
        uint64_t aggregStart = skel.aggregStart[a];
        uint64_t aggregSize = skel.aggregStart[a+1] - aggregStart;
        uint64_t colStart = skel.blockColDataPtr[a];
        uint64_t dataPtr = skel.blockData[colStart];

        Eigen::Map<MatRMaj<double>> block(data.data() + dataPtr, aggregSize, aggregSize);
        block.diagonal() *= (1 + alpha);
        block.diagonal().array() += beta;
    }
}
