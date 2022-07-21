
#include "baspacho/baspacho/EliminationTree.h"
#include <algorithm>
#include <queue>
#include <tuple>
#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/Utils.h"

namespace BaSpaCho {

using namespace std;

EliminationTree::EliminationTree(const vector<int64_t>& paramSize, const SparseStructure& ss)
    : paramSize(paramSize), ss(ss) {
  BASPACHO_CHECK_EQ(paramSize.size(), ss.ptrs.size() - 1);
}

void EliminationTree::buildTree() {
  int64_t ord = ss.order();
  parent.assign(ord, -1);

  nodeSize = paramSize;
  nodeRows.assign(ord, 0);
  vector<int64_t> tags(ord);

  // skeleton of the algo to iterate on fillup's nodes is from Eigen's
  // `SimplicialCholesky_impl.h` (by Gael Guennebaud),
  // in turn from LDL by Timothy A. Davis.
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
      }
    }
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
  int64_t mergeHeight = 0;
  sparseElimRanges.push_back(0);

  for (size_t rangeIndex = 0; rangeIndex < noCrossPoints.size() + 1; rangeIndex++) {
    int64_t rangeStart = rangeIndex == 0 ? 0 : noCrossPoints[rangeIndex - 1];
    int64_t rangeEnd = rangeIndex < noCrossPoints.size() ? noCrossPoints[rangeIndex] : ord;

    int64_t k0 = rangeStart;
    while (k0 < rangeEnd) {
      int64_t k1 = k0;

      while (k1 < rangeEnd && get<0>(unmergedHeightNode[k1]) == mergeHeight &&
             get<1>(unmergedHeightNode[k1]) <= maxSparseElimNodeSize) {
        k1++;
      }
      if (k1 - k0 < minNumSparseElimNodes) {  // skip, too small
        break;
      }
      mergeHeight++;
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

static constexpr double flopsColOverhead = 2e7;

void EliminationTree::processTree(bool detectSparseElimRanges, const vector<int64_t>& noCrossPoints,
                                  bool findOnlyElims) {
  int64_t ord = ss.order();

  computeNodeHeights(noCrossPoints);

  // Compute the sparse elimination ranges (after permutation is applied),
  // and set a flag to forbid merge of nodes which will be sparse-eliminated
  if (detectSparseElimRanges) {
    computeSparseElimRanges(noCrossPoints);
  }

  priority_queue<tuple<double, int64_t, int64_t>> mergeCandidates;
  if (!findOnlyElims) {
    for (int64_t k = ord - 1; k >= 0; k--) {
      if (forbidMerge[k]) {
        continue;
      }
      int64_t p = parent[k];
      if (p == -1) {
        continue;
      }

      double fillAfterMerge = ((double)nodeRows[k]) / (nodeRows[p] + nodeSize[p]);
      mergeCandidates.emplace(fillAfterMerge, k, p);
    }
  }

  mergeWith.assign(ord, -1);
  vector<int64_t> numMergedNodes(ord, 1);
  numMerges = 0;
  while (!mergeCandidates.empty()) {
    auto [wasFillAfterMerge, k, p] = mergeCandidates.top();
    mergeCandidates.pop();

    auto oldP = p;
    BASPACHO_CHECK_LT(p, (int64_t)mergeWith.size());
    while (mergeWith[p] != -1) {
      p = mergeWith[p];
      BASPACHO_CHECK_LT(p, (int64_t)mergeWith.size());
    }

    // parent was merged? value changed, re-prioritize
    if (oldP != p) {
      double fillAfterMerge = ((double)nodeRows[k]) / (nodeRows[p] + nodeSize[p]);
      mergeCandidates.emplace(fillAfterMerge, k, p);
      continue;
    }

    double sk = nodeSize[k], rk = nodeRows[k], sp = nodeSize[p], rp = nodeRows[p], sm = sp + sk;

    // To decide if we're merging the nodes, we compute the flops of the
    // independent eliminations, and compare it with the flops of the
    // elimination of the merged node. If the flops of the merged node
    // is smaller to the sum of the two nodes (plus an overhead constant)
    // we will merge the nodes.
    double elimFlopsK = sk * sk * sk + sk * sk * rk + sk * rk * rk;
    double elimFlopsP = sp * sp * sp + sp * sp * rp + sp * rp * rp;
    double elimFlopsMerg = sm * sm * sm + sm * sm * rp + sm * rp * rp;

    bool willMerge = elimFlopsMerg < elimFlopsK + elimFlopsP + flopsColOverhead;

    if (willMerge) {
      mergeWith[k] = p;
      nodeSize[p] += nodeSize[k];
      numMergedNodes[p] += numMergedNodes[k];
      numMerges++;
    }
  }

  // collapse pointer to parent, make parent become root ancestor
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

  // We create `lumpStart` and `lumpToSpan` arrays for the
  // permuted aggregated parameters.
  // We also set an `unpermutedRootSpanToLump` to register
  // the lumpIndex of an unpermuted root node.
  int64_t numLumps = ord - numMerges, lumpIndex = 0;
  lumpStart.resize(numLumps + 1);   // permuted
  lumpToSpan.resize(numLumps + 1);  // permuted
  vector<int64_t> unpermutedRootSpanToLump(ord, -1);

  for (int64_t i = 0; i < ord; i++) {
    auto [height, unmergedSize, k] = unmergedHeightNode[i];
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
    int64_t lumpIndex = unpermutedRootSpanToLump[p == -1 ? i : p];
    permInverse[i] = lumpToSpan[lumpIndex]++;  // advance
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