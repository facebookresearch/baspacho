#pragma once

#include "SparseStructure.h"

struct EliminationTree {
    EliminationTree(const std::vector<uint64_t>& paramSize,
                    const SparseStructure& ss);

    void buildTree();

    void computeMerges();

    void computeAggregateStruct();

    std::vector<uint64_t> paramSize;
    const SparseStructure& ss;

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
};