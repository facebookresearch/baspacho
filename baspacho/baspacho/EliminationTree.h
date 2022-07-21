#pragma once

#include "baspacho/baspacho/SparseStructure.h"

namespace BaSpaCho {

struct ElimTreeProc;

// (internal) class encapsulating the computation of an elimination-tree
// and the heuristics applied on top of the tree structure. In patricular
// it's responsible of
// 1. automated computation of `sparse elimination ranges`, which is triggered
//   when there are large sets of leaf nodes that can be simulatenously eliminated
//   in parallel. Such sets of leafs are progressively removed from the tree under
//   analysis and related ranges are added to `sparseElimRanges`.
// 2. on the remaining part of the tree, merge child nodes with parent, according
//   to some heuristics depending on the zero-fill of the resulting nodes and the
//   expected runtimes of the resulting dense operations
struct EliminationTree {
  EliminationTree(const std::vector<int64_t>& paramSize, const SparseStructure& ss);

  // build the tree for processing
  void buildTree();

  // executed the processing
  void processTree(bool computeSparseElimRanges, const std::vector<int64_t>& noCrossPoints = {},
                   bool findOnlyElims = false);

  // computes the aggregate struct (where nodes have been merged)
  void computeAggregateStruct(bool fillOnlyForElims = false);

  // internal
  void computeNodeHeights(ElimTreeProc& proc, const std::vector<int64_t>& noCrossPoints);

  // utility helper
  std::vector<int64_t> computeSpanStart();

  std::vector<int64_t> paramSize;
  const SparseStructure& ss;  // input

  // generated data, buildTree
  std::vector<int64_t> parent;
  std::vector<int64_t> nodeSize;
  std::vector<int64_t> nodeRows;

  // generated data, processTree
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