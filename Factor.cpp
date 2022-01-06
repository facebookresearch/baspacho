
#include <glog/logging.h>
#include <Eigen/Eigenvalues>
#include "Factor.h"

void factor(const BlockMatrixSkel& skel,
            std::vector<double>& data) {

    for(size_t a = 0; a < skel.blockColDataPtr.size() - 1; a++) {

	    factorAggreg(skel, data, a);
	    
        uint64_t rowAggregStart = skel.blockColGatheredDataPtr[a];
        uint64_t rowNumAggregs = skel.blockColGatheredDataPtr[a+1] - rowAggregStart;

        CHECK_EQ(skel.blockRowAggreg[rowAggregStart], a);

        for(uint64_t i = 1; i < rowNumAggregs; i++) {
	        eliminateAggregItem(skel, data, a, i);
        }
    }
}

void factorAggreg(const BlockMatrixSkel& skel,
                  std::vector<double>& data,
                  uint64_t aggreg) {

	uint64_t aggregStart = skel.aggregStart[aggreg];
	uint64_t aggregSize = skel.aggregStart[aggreg + 1] - aggregStart;
	uint64_t colStart = skel.blockColDataPtr[aggreg];
	uint64_t dataPtr = skel.blockData[colStart];

	// compute lower diag cholesky dec on diagonal block
	Eigen::Map<MatRMaj<double>> diagBlock(data.data() + dataPtr, aggregSize, aggregSize);
	{
        Eigen::LLT<Eigen::Ref<MatRMaj<double>>> llt(diagBlock);
	}
	uint64_t gatheredStart = skel.blockColGatheredDataPtr[aggreg];
	uint64_t gatheredEnd = skel.blockColGatheredDataPtr[aggreg+1];
	uint64_t rowDataStart = skel.blockRowAggregParamPtr[gatheredStart + 1];
	uint64_t rowDataEnd = skel.blockRowAggregParamPtr[gatheredEnd-1];
	uint64_t belowDiagStart = skel.blockData[colStart + rowDataStart];
	uint64_t numRows = skel.endBlockNumRowsAbove[colStart + rowDataEnd - 1]
		- skel.endBlockNumRowsAbove[colStart + rowDataStart - 1];

	Eigen::Map<MatRMaj<double>> belowDiagBlock(data.data() + belowDiagStart, numRows, aggregSize);
	diagBlock.triangularView<Eigen::Lower>().transpose().solveInPlace<Eigen::OnTheRight>(belowDiagBlock);
}

void eliminateAggregItem(const BlockMatrixSkel& skel,
                         std::vector<double>& data,
                         uint64_t aggreg,
                         uint64_t rowItem) {
}
