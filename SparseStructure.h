#pragma once

#include <cstdint>
#include <set>
#include <vector>

// assumed square matrix
struct SparseStructure {
    std::vector<uint64_t> ptrs;
    std::vector<uint64_t> inds;

    SparseStructure() {}
    SparseStructure(std::vector<uint64_t>&& ptrs_,
                    std::vector<uint64_t>&& inds_)
        : ptrs(std::move(ptrs_)), inds(std::move(inds_)) {}
    SparseStructure(const std::vector<uint64_t>& ptrs_,
                    const std::vector<uint64_t>& inds_)
        : ptrs(ptrs_), inds(inds_) {}

    uint64_t order() const { return ptrs.size() - 1; }

    // makes sure row/col indices are sorted
    void sortIndices();

    // transpose
    SparseStructure transpose(bool sortIndices = true) const;

    // clear upper/lower half (csc)
    SparseStructure clear(bool clearLower = true) const;

    // assumes only a half (any) is present,
    // result is lower/upper half csc (= upper/lower half csr)
    SparseStructure symmetricPermutation(const std::vector<uint64_t>& mapPerm,
                                         bool lowerHalf = true,
                                         bool sortIndices = true) const;

    // assumes CSR lower-diagonal matrix
    SparseStructure addIndependentEliminationFill(uint64_t start, uint64_t end,
                                                  bool sortIdx = true) const;

    SparseStructure addFullEliminationFill() const;

    std::vector<uint64_t> fillReducingPermutation() const;

    SparseStructure extractRightBottom(uint64_t start);
};