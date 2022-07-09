#pragma once

#include "baspacho/baspacho/SparseStructure.h"

namespace BaSpaCho {

struct EliminationTree {
  EliminationTree(const std::vector<int64_t> &paramSize,
                  const SparseStructure &ss);

  void buildTree();

  void computeMerges(bool computeSparseElimRanges,
                     const std::vector<int64_t> &noCrossPoints = {},
                     bool findOnlyElims = false);

  void computeAggregateStruct(bool fillOnlyForElims = false);

  // utility helper
  std::vector<int64_t> computeSpanStart();

  std::vector<int64_t> paramSize;
  const SparseStructure &ss;  // input

  // generated data, buildTree
  std::vector<int64_t> parent;
  std::vector<int64_t> nodeSize;
  std::vector<int64_t> nodeRows;

  // generated data, computeMerges
  std::vector<int64_t> sparseElimRanges;
  std::vector<int64_t> mergeWith;
  int64_t numMerges;

  // generated computing aggregate struct
  std::vector<int64_t> permInverse;
  std::vector<int64_t> lumpStart;
  std::vector<int64_t> lumpToSpan;
  std::vector<int64_t> colStart;
  std::vector<int64_t> rowParam;
};

}  // end namespace BaSpaCho