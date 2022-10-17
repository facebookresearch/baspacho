/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "baspacho/baspacho/EliminationTree.h"
#include <algorithm>
#include <queue>
#include <tuple>
#include "baspacho/baspacho/ComputationModel.h"
#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/Utils.h"

namespace BaSpaCho {

using namespace std;

EliminationTree::EliminationTree(const vector<int64_t>& paramSize, const SparseStructure& ss,
                                 const ComputationModel* compMod)
    : paramSize(paramSize),
      ss(ss),
      compMod(compMod ? *compMod : ComputationModel::model_OpenBlas_i7_1185g7) {
  BASPACHO_CHECK_EQ(paramSize.size(), ss.ptrs.size() - 1);
}

void EliminationTree::buildTree() {
  int64_t ord = ss.order();
  parent.assign(ord, -1);

  nodeSize = paramSize;
  nodeRows.assign(ord, 0);
  nodeRowBlocks.assign(ord, 0);
  vector<int64_t> tags(ord);

  // skeleton of the algo to iterate on fillup's nodes is from Eigen's
  // `SimplicialCholesky_impl.h` (by Gael Guennebaud),
  // in turn from LDL by Timothy A. Davis.
  perColNodes.resize(ord);
  for (int64_t k = 0; k < ord; ++k) {
    /* L(k,:) pattern: all nodes reachable in etree from nz in A(0:k-1,k) */
    parent[k] = -1; /* parent of k is not yet known */
    tags[k] = k;    /* mark node k as visited */
    /* L(k,k) is nonzero */

    int64_t start = ss.ptrs[k];
    int64_t end = ss.ptrs[k + 1];
    for (int64_t q = start; q < end; q++) {
      int64_t i = ss.inds[q];
      if (i >= k) {
        continue;
      }
      /* follow path from i to root of etree, stop at flagged node */
      for (; tags[i] != k; i = parent[i]) {
        /* find parent of i if not yet determined */
        if (parent[i] == -1) {
          parent[i] = k;
        }

        tags[i] = k;
        /* L(k,i) is nonzero */
        nodeRows[i] += paramSize[k];
        nodeRowBlocks[i]++;
        perColNodes[i].push_back(k);
      }
    }
  }

  sygeCosts.resize(ord);
  asmblCosts.resize(ord);
  perRowNodeStats.resize(ord);
  for (int64_t col = 0; col < (int64_t)perColNodes.size(); col++) {
    auto& c = perColNodes[col];
    c.push_back(col);
    sort(c.begin(), c.end());

    int64_t skippedRows = 0;
    int64_t skippedBlocks = 0;
    Eigen::Vector2d sygeC{0, 0}, asmblC{0, 0};
    for (int64_t i = c.size() - 1; i >= 0; i--) {
      int64_t row = c[i];

      sygeC += compMod.sygeLinEst(skippedRows + paramSize[row], paramSize[row]);
      asmblC += compMod.asmblLinEst(skippedBlocks + 1);

      perRowNodeStats[row].emplace_back(col, 1, paramSize[row], skippedBlocks, skippedRows);

      skippedRows += paramSize[row];
      skippedBlocks++;
    }
    sygeCosts[col] = sygeC;
    asmblCosts[col] = asmblC;
  }
}

// Compute node heights:
// depending on the structure of the matrix, if we have many small leaf
// nodes (height = 0) we can eliminate them with a sparse elimination
// step that skips the node-merge adding fill in. It's then possible
// to eliminate progressively nodes with height=1, 2, etc till when
// this strategy is no longer effective and we're better off with
// merged nodes and blas operations.
// In order to do so we sort nodes according to height.
// TODO: consider allowing some initial merge of very small nodes?
void EliminationTree::computeNodeHeights(const vector<int64_t>& noCrossPoints) {
  int64_t ord = ss.order();

  unmergedHeightNode.resize(ord);
  forbidMerge.assign(ord, false);

  vector<int64_t> height(ord, 0);
  for (size_t rangeIndex = 0; rangeIndex < noCrossPoints.size() + 1; rangeIndex++) {
    int64_t rangeStart = rangeIndex == 0 ? 0 : noCrossPoints[rangeIndex - 1];
    int64_t rangeEnd = rangeIndex < noCrossPoints.size() ? noCrossPoints[rangeIndex] : ord;

    for (int64_t k = rangeStart; k < rangeEnd; k++) {
      unmergedHeightNode[k] = make_tuple(height[k], nodeSize[k], k);

      int64_t par = parent[k];
      if (par == -1) {
        continue;
      }
      if (par >= rangeEnd) {
        forbidMerge[k] = true;
      }
      height[par] = max(height[par], height[k] + 1);
    }
    sort(unmergedHeightNode.begin() + rangeStart, unmergedHeightNode.begin() + rangeEnd);
  }
}

static constexpr int64_t maxSparseElimNodeSize = 12;
static constexpr int64_t minNumSparseElimNodes = 50;

void EliminationTree::computeSparseElimRanges(const vector<int64_t>& noCrossPoints) {
  int64_t ord = ss.order();
  sparseElimRanges.push_back(0);

  for (size_t rangeIndex = 0; rangeIndex < noCrossPoints.size() + 1; rangeIndex++) {
    int64_t rangeStart = rangeIndex == 0 ? 0 : noCrossPoints[rangeIndex - 1];
    int64_t rangeEnd = rangeIndex < noCrossPoints.size() ? noCrossPoints[rangeIndex] : ord;

    int64_t k0 = rangeStart;
    while (k0 < rangeEnd) {
      int64_t k1 = k0;
      int64_t mergeHeight = get<0>(unmergedHeightNode[k0]);

      int64_t numEasyMerge = 0;
      while (k1 < rangeEnd && get<0>(unmergedHeightNode[k1]) == mergeHeight &&
             get<1>(unmergedHeightNode[k1]) <= maxSparseElimNodeSize) {
        int64_t p = parent[k1];
        if (p >= 0) {
          double fillAfterMerge = ((double)nodeRows[k1]) / (nodeRows[p] + nodeSize[p]);
          if (fillAfterMerge > 0.8) {
            numEasyMerge++;
          }
        }
        k1++;
      }

      // skip and stop searching if 1. too small, or 2. most nodes are easily merged
      if ((k1 - k0) < minNumSparseElimNodes || (k1 - k0) < numEasyMerge * 3) {
        break;
      }

      for (int64_t k = k0; k < k1; k++) {
        forbidMerge[get<2>(unmergedHeightNode[k])] = true;
      }
      sparseElimRanges.push_back(k1);
      k0 = k1;
    }
    if (k0 < rangeEnd) {
      break;
    }
  }
  if (sparseElimRanges.size() == 1) {
    sparseElimRanges.pop_back();
  }
}

void EliminationTree::computeMerges() {
  int64_t ord = ss.order();
  numMergedNodes.assign(ord, 1);
  mergeWith.assign(ord, -1);
  numMerges = 0;

  // score used to select which child/parent pair we will try to merge first
  auto pickUpScore = [&](int64_t k, int64_t p) -> double {
    return ((double)nodeRows[k]) / (nodeRows[p] + nodeSize[p]);
  };

  priority_queue<tuple<double, int64_t, int64_t>> mergeCandidates;
  for (int64_t k = ord - 1; k >= 0; k--) {
    if (forbidMerge[k]) {
      continue;
    }
    int64_t p = parent[k];
    if (p == -1) {
      continue;
    }
    mergeCandidates.emplace(pickUpScore(k, p), k, p);
  }

  while (!mergeCandidates.empty()) {
    auto wasFillAfterMergeKP = mergeCandidates.top();
    int64_t k = std::get<1>(wasFillAfterMergeKP);
    int64_t p = std::get<2>(wasFillAfterMergeKP);
    mergeCandidates.pop();

    auto oldP = p;
    BASPACHO_CHECK_LT(p, (int64_t)mergeWith.size());
    while (mergeWith[p] != -1) {
      p = mergeWith[p];
      BASPACHO_CHECK_LT(p, (int64_t)mergeWith.size());
    }

    // parent was merged? value changed, re-prioritize
    if (oldP != p) {
      mergeCandidates.emplace(pickUpScore(k, p), k, p);
      continue;
    }

    double sk = nodeSize[k], rk = nodeRows[k], sp = nodeSize[p], rp = nodeRows[p], sm = sp + sk;

    // To decide if we're merging the nodes, we compare the estimated runtimes of the (factor)
    // operations related to the individual unmerged nodes with the runtime of the operations
    // in the merged node. This isn't 100% accurate because some children elimination operations
    // might also merge and this is not accounted, but the current estimate should be rather
    // accurate.
    double tk = compMod.potrfEst(sk) + compMod.trsmEst(sk, rk) +  //
                sygeCosts[k][0] + sygeCosts[k][1] * sk +          //
                asmblCosts[k][0] + asmblCosts[k][1] * numMergedNodes[k];
    double tp = compMod.potrfEst(sp) + compMod.trsmEst(sp, rp) +  //
                sygeCosts[p][0] + sygeCosts[p][1] * sp +          //
                asmblCosts[p][0] + asmblCosts[p][1] * numMergedNodes[p];
    double tm = compMod.potrfEst(sm) + compMod.trsmEst(sm, rp) +  //
                sygeCosts[p][0] + sygeCosts[p][1] * sm +          //
                asmblCosts[p][0] + asmblCosts[p][1] * (numMergedNodes[k] + numMergedNodes[p]);
    bool willMerge = tm < tk + tp;

    if (willMerge) {
      int64_t prevNodeSize = nodeSize[p];
      int64_t prevNumMergedNodes = numMergedNodes[p];
      mergeWith[k] = p;
      nodeSize[p] += nodeSize[k];
      numMergedNodes[p] += numMergedNodes[k];
      numMerges++;

      auto& kRD = perRowNodeStats[k];
      auto& pRD = perRowNodeStats[p];
      for (size_t ik = 0, ip = 0; ik < kRD.size() || ip < pRD.size(); /* */) {
        if ((ip >= pRD.size()) || (ik < kRD.size() && kRD[ik].colIdx < pRD[ip].colIdx)) {
          if (kRD[ik].colIdx != k) {
            tmpRowStats.push_back(kRD[ik]);
          }
          ik++;
        } else if ((ik >= kRD.size()) || (ip < pRD.size() && kRD[ik].colIdx > pRD[ip].colIdx)) {
          if (pRD[ip].colIdx != p) {
            tmpRowStats.push_back(pRD[ip]);
          }
          ip++;
        } else {
          int64_t c = pRD[ip].colIdx;
          auto& sygeC = sygeCosts[c];
          auto& asmblC = asmblCosts[c];
          sygeC -= compMod.sygeLinEst(kRD[ik].rowsDown + kRD[ik].rows, kRD[ik].rows);
          asmblC -= compMod.asmblLinEst(kRD[ik].rBlocksDown + kRD[ik].rBlocks);
          sygeC -= compMod.sygeLinEst(pRD[ip].rowsDown + pRD[ip].rows, pRD[ip].rows);
          asmblC -= compMod.asmblLinEst(pRD[ip].rBlocksDown + pRD[ip].rBlocks);
          sygeC += compMod.sygeLinEst(pRD[ip].rowsDown + (kRD[ik].rows + pRD[ip].rows),
                                      (kRD[ik].rows + pRD[ip].rows));
          asmblC += compMod.asmblLinEst(pRD[ip].rBlocksDown + (kRD[ik].rBlocks + pRD[ip].rBlocks));

          tmpRowStats.emplace_back(c, kRD[ik].rBlocks + pRD[ip].rBlocks,
                                   kRD[ik].rows + pRD[ip].rows, pRD[ip].rBlocksDown,
                                   pRD[ip].rowsDown);
          ik++;
          ip++;
        }
      }
      auto& sygeC = sygeCosts[p];
      auto& asmblC = asmblCosts[p];
      sygeC -= compMod.sygeLinEst(nodeRows[p] + prevNodeSize, prevNodeSize);
      asmblC -= compMod.asmblLinEst(nodeRowBlocks[p] + prevNumMergedNodes);
      sygeC += compMod.sygeLinEst(nodeRows[p] + nodeSize[p], nodeSize[p]);
      asmblC += compMod.asmblLinEst(nodeRowBlocks[p] + numMergedNodes[p]);
      tmpRowStats.emplace_back(p, numMergedNodes[p], nodeSize[p], nodeRowBlocks[p], nodeRows[p]);
      swap(perRowNodeStats[p], tmpRowStats);
      tmpRowStats.clear();
    }  // willmerge
  }
}

// collapse pointer to parent, make parent become root ancestor
void EliminationTree::collapseMergePointers() {
  int64_t ord = ss.order();

  for (int64_t k = ord - 1; k >= 0; k--) {
    int64_t p = mergeWith[k];
    if (p == -1) {
      continue;
    }
    int64_t a = mergeWith[p];
    if (a != -1) {
      mergeWith[k] = a;
    }
  }
}

void EliminationTree::processTree(bool detectSparseElimRanges, const vector<int64_t>& noCrossPoints,
                                  bool findOnlyElims) {
  int64_t ord = ss.order();

  computeNodeHeights(noCrossPoints);

  // Compute the sparse elimination ranges (after permutation is applied),
  // and set a flag to forbid merge of nodes which will be sparse-eliminated
  if (detectSparseElimRanges) {
    computeSparseElimRanges(noCrossPoints);
  }

  if (findOnlyElims) {
    mergeWith.assign(ord, -1);
    numMergedNodes.assign(ord, 1);
    numMerges = 0;
  } else {
    computeMerges();
    collapseMergePointers();
  }

  // We create `lumpStart` and `lumpToSpan` arrays for the
  // permuted aggregated parameters.
  // We also set an `unpermutedRootSpanToLump` to register
  // the lumpIndex of an unpermuted root node.
  int64_t numLumps = ord - numMerges, lumpIndex = 0;
  lumpStart.resize(numLumps + 1);   // permuted
  lumpToSpan.resize(numLumps + 1);  // permuted
  vector<int64_t> unpermutedRootSpanToLump(ord, -1);

  for (int64_t i = 0; i < ord; i++) {
    int64_t k = std::get<2>(unmergedHeightNode[i]);
    if (mergeWith[k] != -1) {
      continue;
    }
    unpermutedRootSpanToLump[k] = lumpIndex;
    lumpStart[lumpIndex] = nodeSize[k];
    lumpToSpan[lumpIndex] = numMergedNodes[k];
    lumpIndex++;
  }
  BASPACHO_CHECK_EQ(lumpIndex, numLumps);

  cumSumVec(lumpStart);
  cumSumVec(lumpToSpan);

  // we compute the (inverse) permutation: for each span we have a pointer
  // to the beginning of the lump, it is advanced to get the index the span
  // is mapped to (the lumpToSpan pointers are rewinded later)
  permInverse.resize(ord);
  for (int64_t i = 0; i < ord; i++) {
    int64_t p = mergeWith[i];
    int64_t lumpIndex1 = unpermutedRootSpanToLump[p == -1 ? i : p];
    permInverse[i] = lumpToSpan[lumpIndex1]++;  // advance
  }
  rewindVec(lumpToSpan);  // restore after advancing
}

void EliminationTree::computeAggregateStruct(bool fillOnlyForElims) {
  int64_t ord = ss.order();
  int64_t numLumps = ord - numMerges;

  SparseStructure tperm =  // lower-half csc
      ss.symmetricPermutation(permInverse, /* lowerHalf = */ false,
                              /* sortIndices = */ false);

  if (fillOnlyForElims) {
    for (int64_t e = 0; e < (int64_t)sparseElimRanges.size() - 1; e++) {
      tperm = tperm.addIndependentEliminationFill(sparseElimRanges[e], sparseElimRanges[e + 1]);
    }
  } else {
    tperm = tperm.addFullEliminationFill();
  }
  tperm = tperm.transpose();

  vector<int64_t> tags(ord, -1);  // check if row el was added already
  colStart.push_back(0);
  for (int64_t a = 0; a < numLumps; a++) {
    int64_t aStart = lumpToSpan[a];
    int64_t aEnd = lumpToSpan[a + 1];
    int64_t pStart = tperm.ptrs[aStart];
    int64_t pEnd = tperm.ptrs[aEnd];
    for (int64_t i = pStart; i < pEnd; i++) {
      int64_t p = tperm.inds[i];
      if (tags[p] < (int64_t)a) {
        rowParam.push_back(p);  // L(p,a) is set
        tags[p] = a;
      }
    }
    sort(rowParam.begin() + colStart[colStart.size() - 1], rowParam.end());
    colStart.push_back(rowParam.size());
  }
}

vector<int64_t> EliminationTree::computeSpanStart() {
  vector<int64_t> spanStart(paramSize.size() + 1);
  leftPermute(spanStart.begin(), permInverse, paramSize);
  cumSumVec(spanStart);

  return spanStart;
}

}  // end namespace BaSpaCho
