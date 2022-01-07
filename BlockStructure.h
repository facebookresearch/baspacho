#pragma once

#include <cstdint>
#include <set>
#include <vector>

// TODO: this can certainly be improved, and made to use no containers with
// dynamic allocation
struct BlockStructure {
    std::vector<uint64_t> paramSize;
    std::vector<std::set<uint64_t>> colBlocks;

    BlockStructure(const std::vector<uint64_t> &paramSize,
                   const std::vector<std::set<uint64_t>> &colBlocks);

    void addBlocksForEliminationOfRange(uint64_t start, uint64_t end);
};

struct GroupedBlockStructure {
    std::vector<uint64_t> paramStart;
    std::vector<uint64_t> aggregParamStart;
    std::vector<std::vector<uint64_t>> columnParams;

    GroupedBlockStructure(const BlockStructure &blockStructure,
                          const std::vector<uint64_t> &aggregParamStart);
};
