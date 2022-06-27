#pragma once

#include <cstdint>
#include <set>
#include <vector>

namespace BaSpaCho {

// assumed square matrix
struct SparseStructure {
  std::vector<int64_t> ptrs;
  std::vector<int64_t> inds;

  SparseStructure() {}
  SparseStructure(std::vector<int64_t>&& ptrs_, std::vector<int64_t>&& inds_)
      : ptrs(std::move(ptrs_)), inds(std::move(inds_)) {}
  SparseStructure(const std::vector<int64_t>& ptrs_,
                  const std::vector<int64_t>& inds_)
      : ptrs(ptrs_), inds(inds_) {}

  int64_t order() const { return ptrs.size() - 1; }

  // makes sure row/col indices are sorted
  void sortIndices();

  // transpose
  SparseStructure transpose() const;

  // clear upper/lower half (csc)
  SparseStructure clear(bool clearLower = true) const;

  // assumes only a half (any) is present,
  // result is lower/upper half csc (= upper/lower half csr)
  // `mapPerm[i]` is the new index: `i`-th row will move to `mapPerm[i]`
  SparseStructure symmetricPermutation(const std::vector<int64_t>& mapPerm,
                                       bool lowerHalf = true,
                                       bool sortIndices = true) const;

  // assumes CSR lower-diagonal matrix
  SparseStructure addIndependentEliminationFill(int64_t start, int64_t end,
                                                bool sortIdx = true) const;

  SparseStructure addFullEliminationFill() const;

  // return `perm`: `perm[i]` is old index that should move in `i`-th position
  std::vector<int64_t> fillReducingPermutation() const;

  SparseStructure extractRightBottom(int64_t start);
};

}  // end namespace BaSpaCho