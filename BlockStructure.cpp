#include "BlockStructure.h"

#include <glog/logging.h>

#include "Utils.h"

BlockStructure::BlockStructure(const std::vector<uint64_t> &paramSize,
                               const std::vector<std::set<uint64_t>> &colBlocks)
    : paramSize(paramSize), colBlocks(colBlocks) {
    CHECK_EQ(paramSize.size(), colBlocks.size());
}

// naive ATM, but shoud work. in the future, adapt simplicial cholesky algo.
void BlockStructure::addBlocksForEliminationOfRange(uint64_t start,
                                                    uint64_t end) {
    for (int i = start; i < end; i++) {
        std::set<uint64_t> &cBlocks = colBlocks[i];
        auto it = cBlocks.begin();
        CHECK(it != cBlocks.end());
        CHECK_EQ(i, *it) << "Expecting diagonal block!";
        while (++it != cBlocks.end()) {
            auto it2 = it;
            std::set<uint64_t> &cAltBlocks = colBlocks[*it];
            while (++it2 != cBlocks.end()) {
                cAltBlocks.insert(*it2);
            }
        }
    }
}

GroupedBlockStructure::GroupedBlockStructure(
    const BlockStructure &blockStructure,
    const std::vector<uint64_t> &aggregParamStart)
    : aggregParamStart(aggregParamStart) {
    CHECK(isStrictlyIncreasing(aggregParamStart, 0, aggregParamStart.size()));
    CHECK_GE(aggregParamStart.size(), 1);
    CHECK_EQ(blockStructure.paramSize.size(),
             aggregParamStart[aggregParamStart.size() - 1]);

    size_t totSize = 0;
    paramStart.push_back(0);
    for (uint64_t s : blockStructure.paramSize) {
        totSize += s;
        paramStart.push_back(totSize);
    }

    columnParams.resize(aggregParamStart.size() - 1);
    for (int a = 0; a < aggregParamStart.size() - 1; a++) {
        uint64_t pStart = aggregParamStart[a];
        uint64_t pEnd = aggregParamStart[a + 1];
        std::set<uint64_t> allRows;
        for (uint64_t p = pStart; p < pEnd; p++) {
            allRows.insert(blockStructure.colBlocks[p].begin(),
                           blockStructure.colBlocks[p].end());
        }
        columnParams[a].assign(allRows.begin(), allRows.end());
    }
}
