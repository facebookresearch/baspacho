#pragma once

#include <Eigen/Geometry>
#include "baspacho/baspacho/SparseStructure.h"

namespace BaSpaCho {

struct ComputationModel;

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
  EliminationTree(const std::vector<int64_t>& paramSize, const SparseStructure& ss,
                  const ComputationModel* compMod = nullptr);

  // build the tree for processing
  void buildTree();

  // executed the processing
  void processTree(bool detectSparseElimRanges, const std::vector<int64_t>& noCrossPoints = {},
                   bool findOnlyElims = false);

  // computes the aggregate struct (where nodes have been merged)
  void computeAggregateStruct(bool fillOnlyForElims = false);

  // utility helper to get span starts from interally stored (sorted) param sizes
  std::vector<int64_t> computeSpanStart();

  // internal (called by processTree)
  void computeNodeHeights(const std::vector<int64_t>& noCrossPoints);

  // internal (called by processTree)
  void computeSparseElimRanges(const std::vector<int64_t>& noCrossPoints);

  // internal (called by processTree)
  void computeMerges();

  // internal (called by processTree)
  void collapseMergePointers();

  std::vector<int64_t> paramSize;
  const SparseStructure& ss;  // input
  const ComputationModel& compMod;

  // generated data, buildTree
  std::vector<int64_t> parent;
  std::vector<int64_t> nodeSize;
  std::vector<int64_t> nodeRows;
  std::vector<int64_t> nodeRowBlocks;
  std::vector<std::vector<int64_t>> perColNodes;
  struct NodeStats {
    int64_t colIdx;
    int64_t rBlocks;
    int64_t rows;
    int64_t rBlocksDown;
    int64_t rowsDown;
  };
  std::vector<std::vector<NodeStats>> perRowNodeStats;
  std::vector<NodeStats> tmpRowStats;

  // modelled costs of syge/asmbl ops for a node, as linear
  // function of node size (resp. size as number of col blocks)
  std::vector<Eigen::Vector2d> sygeCosts;
  std::vector<Eigen::Vector2d> asmblCosts;

  // generated data, processTree
  std::vector<int64_t> sparseElimRanges;
  std::vector<std::tuple<int64_t, int64_t, int64_t>> unmergedHeightNode;
  std::vector<bool> forbidMerge;
  std::vector<int64_t> numMergedNodes;
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