#pragma once

#include "baspacho/SparseStructure.h"

namespace BaSpaCho {

struct EliminationTree {
    EliminationTree(const std::vector<int64_t>& paramSize,
                    const SparseStructure& ss);

    void buildTree();

    void computeMerges();

    void computeMerges2();

    void computeAggregateStruct();

    std::vector<int64_t> paramSize;
    const SparseStructure& ss;  // input

    // generated data, buildTree
    std::vector<int64_t> parent;
    std::vector<int64_t> nodeSize;
    std::vector<int64_t> nodeRows;
    std::vector<int64_t> nodeFill;
    std::vector<int64_t> firstChild;
    std::vector<int64_t> nextSibling;

    // generated data, computeMerges
    std::vector<int64_t> mergeWith;
    int64_t numMerges;

    // generated computing aggregate struct
    std::vector<int64_t> permutation;
    std::vector<int64_t> permInverse;
    std::vector<int64_t> spanStart;
    std::vector<int64_t> lumpStart;
    std::vector<int64_t> lumpToSpan;
    std::vector<int64_t> colStart;
    std::vector<int64_t> rowParam;
};

}  // end namespace BaSpaCho