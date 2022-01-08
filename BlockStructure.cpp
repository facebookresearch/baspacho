#include "BlockStructure.h"

#include <glog/logging.h>
#include <suitesparse/amd.h>

#include <algorithm>

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

uint64_t BlockStructure::numBlocksInCols(uint64_t start, uint64_t end) {
    uint64_t tot = 0;
    for (int i = start; i < end; i++) {
        tot += colBlocks[i].size();
    }
    return tot;
}

void BlockStructure::applyAmdFrom(uint64_t start) {
    LOG(INFO) << "prep for AMD...";
    std::vector<int64_t> colPtr, rowInd;
    colPtr.push_back(0);
    for (int i = start; i < colBlocks.size(); i++) {
        std::set<uint64_t> &cBlocks = colBlocks[i];
        auto it = cBlocks.begin();
        CHECK(it != cBlocks.end());
        CHECK_EQ(i, *it) << "Expecting diagonal block!";
        while (it != cBlocks.end()) {
            rowInd.push_back(*it - start);
            ++it;
        }
        colPtr.push_back(rowInd.size());
    }
    uint64_t n = colPtr.size() - 1;
    std::vector<int64_t> P(n);
    double Control[AMD_CONTROL], Info[AMD_INFO];

    LOG(INFO) << "run AMD...";
    amd_l_defaults(Control);
    amd_l_control(Control);

    int result =
        amd_l_order(n, colPtr.data(), rowInd.data(), P.data(), Control, Info);
    LOG(INFO) << "result: " << result;

    /*std::sort(P.begin(), P.end());
    for (uint64_t i = 0; i < n; i++) {
        CHECK_EQ(P[i], i) << "Unexpected P[" << i << "] = " << P[i];
    }*/

    std::vector<int64_t> invP(n);
    for (uint64_t i = 0; i < n; i++) {
        invP[P[i]] = i;
    }
    for (uint64_t i = start; i < colBlocks.size(); i++) {
        colBlocks[i].clear();
    }
    for (uint64_t c = 0; c < colPtr.size() - 1; c++) {
        uint64_t i0 = colPtr[c];
        uint64_t iz = colPtr[c + 1];
        for (uint64_t i = i0; i < iz; i++) {
            uint64_t r = rowInd[i];
            uint64_t newMappedC = invP[c];
            uint64_t newMappedR = invP[r];
            uint64_t newC = std::min(newMappedC, newMappedR);
            uint64_t newR = std::max(newMappedC, newMappedR);
            colBlocks[newC + start].insert(newR + start);
        }
    }
    for (uint64_t i = start; i < colBlocks.size(); i++) {
        std::set<uint64_t> &cBlocks = colBlocks[i];
        auto it = cBlocks.begin();
        CHECK(it != cBlocks.end());
        CHECK_EQ(i, *it) << "Expecting diagonal block: " << i << " " << start
                         << " " << colBlocks.size();
    }

    LOG(INFO) << "rebuilt!";
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
