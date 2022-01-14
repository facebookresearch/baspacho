#pragma once

#include "SparseStructure.h"

struct EliminationTree {
    EliminationTree(const std::vector<uint64_t>& paramSize,
                    const SparseStructure& ss);

    void buildTree();

    void computeMerges();

    void computeAggregateStruct();

    std::vector<uint64_t> paramSize;
    const SparseStructure& ss;  // input

    // generated data, buildTree
    std::vector<int64_t> parent;
    std::vector<uint64_t> nodeSize;
    std::vector<uint64_t> nodeRows;
    std::vector<uint64_t> nodeFill;
    std::vector<int64_t> firstChild;
    std::vector<int64_t> nextSibling;

    // generated data, computeMerges
    std::vector<int64_t> mergeWith;
    uint64_t numMerges;

    // generated computing aggregate struct
    std::vector<uint64_t> colStart;
    std::vector<uint64_t> rowParam;

    std::vector<uint64_t> permutation;
    std::vector<uint64_t> permInverse;
    std::vector<uint64_t> aggregStart;
};