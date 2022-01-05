
#include <glog/logging.h>
#include "BlockMatrix.h"
#include "Utils.h"

using namespace std;

#if 0
std::vector<uint64_t> paramStart; // num_params + 1
std::vector<uint64_t> paramToAggreg; // num_params
std::vector<uint64_t> aggregStart; // num_aggregs + 1
std::vector<uint64_t> aggregParamStart; // num_aggregs + 1

// A matrix block is identified by a pair of param x aggreg
std::vector<uint64_t> blockColDataPtr; // num_aggregs + 1
std::vector<uint64_t> blockRowParam; // num_blocks
std::vector<uint64_t> blockData; // num_blocks

// We also need to know about the "gathered" blocks, where we have
// grouped the consecutive row params into aggregates.
// This is because we will process the colum of blocks taking not
// one row at a time, but an aggregate of rows.
std::vector<uint64_t> blockColGatheredDataPtr; // num_aggregs + 1
std::vector<uint64_t> blockRowAggreg; // num_gathered_blocks
std::vector<uint64_t> blockRowAggregParamPtr; // num_gathered_blocks
#endif

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
        for(size_t i = 0; i < columnParams[a].size(); i++) {
            uint64_t p = columnParams[a][i];
            retv.blockRowParam.push_back(p);
            retv.blockData.push_back(dataPtr);
            dataPtr += aDataSize * (paramStart[p+1] - paramStart[p]);

            uint64_t rowAggreg = retv.paramToAggreg[p];
            if(rowAggreg != currentRowAggreg) {
                currentRowAggreg = rowAggreg;
                retv.blockRowAggreg.push_back(rowAggreg);
                retv.blockRowAggregParamPtr.push_back(i);
            }
        }
   }
   retv.blockColDataPtr[numAggregs] = retv.blockRowParam.size();
   retv.blockColGatheredDataPtr[numAggregs] = retv.blockRowAggreg.size();
   retv.blockData.push_back(dataPtr);
    
   return retv;
}
