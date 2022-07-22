
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
// t ~= a + b*n + c*n^2 + d*n^3
double potrfModel(double n) {
  double c[] = {3.975137677492635046e-07, 0.0 /*-7.384080107915980689e-08*/,
                5.034549574025795499e-09, 6.502293016861755153e-12};
  return c[0] + n * (c[1] + n * (c[2] + n * c[3]));
}

double trsmModel(double n, double k) {
  double c[] = {1.146889197744097553e-06,          6.405404911372597549e-07,
                0.0 /*-1.310823260065062174e-09*/, 0.0 /*-2.734271880834972249e-10*/,
                1.383456786664319804e-09,          2.752945944702002428e-11};
  return c[0] + n * (c[1] + n * c[2]) + k * (c[3] + n * (c[4] + n * c[5]));
}

double sygeModel(double m, double n, double k) {
  double c[] = {7.807840624901230139e-07, 0.0 /*-6.422313720003776029e-09*/,
                5.625680355500814005e-10, 0.0 /*-2.622329776837938154e-08*/,
                1.591634874952530293e-09, 2.316558505904654318e-11};
  return c[0] + (m + n) * c[1] + (m * n) * c[2] +  //
         k * (c[3] + (m + n) * c[4] + (m * n) * c[5]);
}

double asmblModel(double br, double bc) {
  double c[] = {3.865027765792498186e-07, 3.542278666359479735e-08, 1.928966025421573037e-07,
                3.78354960153969549e-09};
  return c[0] + br * c[1] + bc * c[2] + br * bc * c[3];
}

Eigen::Vector2d sygeModelV(double m, double n) {
  double c[] = {7.807840624901230139e-07, 0.0 /*-6.422313720003776029e-09*/,
                5.625680355500814005e-10, 0.0 /*-2.622329776837938154e-08*/,
                1.591634874952530293e-09, 2.316558505904654318e-11};
  return Eigen::Vector2d{c[0] + (m + n) * c[1] + (m * n) * c[2],  //
                         c[3] + (m + n) * c[4] + (m * n) * c[5]};
}

Eigen::Vector2d asmblModelV(double br) {
  double c[] = {3.865027765792498186e-07, 3.542278666359479735e-08, 1.928966025421573037e-07,
                3.78354960153969549e-09};
  return Eigen::Vector2d{c[0] + br * c[1], c[2] + br * c[3]};
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
  nodeColEls.resize(ord);
  nodeColDataEls.resize(ord);
  for (int64_t k = 0; k < ord; ++k) {
    /* L(k,:) pattern: all nodes reachable in etree from nz in A(0:k-1,k) */
    parent[k] = -1; /* parent of k is not yet known */
    tags[k] = k;    /* mark node k as visited */
    /* L(k,k) is nonzero */

    int64_t start = ss.ptrs[k];
    int64_t end = ss.ptrs[k + 1];
    auto& col = nodeColEls[k];
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
        col.push_back(i);
        nodeColDataEls[i].push_back(k);
      }
    }
    sort(col.begin(), col.end());
  }

  sygeCosts.resize(ord);
  asmblCosts.resize(ord);
  nodeRowDataEls.resize(ord);
  // cout << "A" << endl;
  for (int64_t col = 0; col < nodeColDataEls.size(); col++) {
    auto& c = nodeColDataEls[col];
    c.push_back(col);
    sort(c.begin(), c.end());

    int64_t skippedRows = 0;
    int64_t skippedBlocks = 0;
    Eigen::Vector2d sygeC{0, 0}, asmblC{0, 0};
    for (int64_t i = c.size() - 1; i >= 0; i--) {
      int64_t row = c[i];

      sygeC += sygeModelV(skippedRows + paramSize[row], paramSize[row]);
      asmblC += asmblModelV(skippedBlocks + 1);

      nodeRowDataEls[row].push_back({.colIdx = col,
                                     .rBlocks = 1,
                                     .rows = paramSize[row],
                                     .rBlocksDown = skippedBlocks,
                                     .rowsDown = skippedRows});

      skippedRows += paramSize[row];
      skippedBlocks++;
    }
    sygeCosts[col] = sygeC;
    asmblCosts[col] = asmblC;
  }
  // cout << "B" << endl;
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
        double fillAfterMerge = ((double)nodeRows[k1]) / (nodeRows[p] + nodeSize[p]);
        if (fillAfterMerge > 0.8) {
          numEasyMerge++;
        }
        k1++;
      }

      // skip and stop searching if 1. too small, or 2. most nodes are easily merged
      if ((k1 - k0) < minNumSparseElimNodes || (k1 - k0) < numEasyMerge * 3) {
        break;
      }
      // cout << k0 << "..." << k1 << " (" << numEasyMerge << "/" << (k1 - k0) << ")" << endl;

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

void EliminationTree::computeMerges() {
  int64_t ord = ss.order();
  numMergedNodes.assign(ord, 1);
  mergeWith.assign(ord, -1);
  numMerges = 0;

  auto pickUpScore0 = [&](int64_t k, int64_t p) -> double {
    return ((double)nodeRows[k]) / (nodeRows[p] + nodeSize[p]);
  };
  auto pickUpScore0b = [&](int64_t k, int64_t p) -> double {
    return ((double)nodeRows[k]) / (nodeRows[p] + nodeSize[p]) *
           (1.0 + log2((nodeRows[p] + nodeSize[p]) * nodeRows[k]) * 0.3);
  };
  auto pickUpScore = pickUpScore0;

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
      mergeCandidates.emplace(pickUpScore(k, p), k, p);
      continue;
    }

    double sk = nodeSize[k], rk = nodeRows[k], sp = nodeSize[p], rp = nodeRows[p], sm = sp + sk;

    // To decide if we're merging the nodes, we compare the estimated runtimes of the (factor)
    // operations related to the individual unmerged nodes with the runtime of the operations
    // in the merged node. This isn't 100% accurate because some children elimination operations
    // might also merge and this is not accounted, but the current estimate should be rather
    // accurate.
#if 0
    double elimFlopsK = sk * sk * sk + sk * sk * rk + sk * rk * rk;
    double elimFlopsP = sp * sp * sp + sp * sp * rp + sp * rp * rp;
    double elimFlopsMerg = sm * sm * sm + sm * sm * rp + sm * rp * rp;

    bool willMerge = elimFlopsMerg < elimFlopsK + elimFlopsP + flopsColOverhead;
#elif 0
    double tk =
        potrfModel(sk) + trsmModel(sk, rk) + sygeModel(rk, rk, sk) + asmblModel(rk / 2, sk / 2);
    double tp =
        potrfModel(sp) + trsmModel(sp, rp) + sygeModel(rp, rp, sp) + asmblModel(rp / 2, sp / 2);
    double tm =
        potrfModel(sm) + trsmModel(sm, rp) + sygeModel(rp, rp, sm) + asmblModel(rp / 2, sm / 2);

    bool willMerge = tm < tk + tp;
#elif 0
    double tk = potrfModel(sk) + trsmModel(sk, rk) + sygeModel(rk, rk, sk) +
                asmblModel(nodeRowBlocks[k], numMergedNodes[k]);
    double tp = potrfModel(sp) + trsmModel(sp, rp) + sygeModel(rp, rp, sp) +
                asmblModel(nodeRowBlocks[p], numMergedNodes[p]);
    double tm = potrfModel(sm) + trsmModel(sm, rp) + sygeModel(rp, rp, sm) +
                asmblModel(nodeRowBlocks[p], numMergedNodes[k] + numMergedNodes[p]);

    bool willMerge = tm < tk + tp;
#elif 1
    double tk = potrfModel(sk) + trsmModel(sk, rk) + sygeCosts[k][0] + sygeCosts[k][1] * sk +
                asmblCosts[k][0] + asmblCosts[k][1] * numMergedNodes[k];
    double tp = potrfModel(sp) + trsmModel(sp, rp) + sygeCosts[p][0] + sygeCosts[p][1] * sp +
                asmblCosts[p][0] + asmblCosts[p][1] * numMergedNodes[p];
    double tm = potrfModel(sm) + trsmModel(sm, rp) + sygeCosts[p][0] + sygeCosts[p][1] * sm +
                asmblCosts[p][0] + asmblCosts[p][1] * (numMergedNodes[k] + numMergedNodes[p]);
    bool willMerge = tm < tk + tp;
#endif

    if (willMerge) {
      int64_t prevNodeSize = nodeSize[p];
      int64_t prevNumMergedNodes = numMergedNodes[p];
      mergeWith[k] = p;
      nodeSize[p] += nodeSize[k];
      numMergedNodes[p] += numMergedNodes[k];
      numMerges++;

      auto& kRD = nodeRowDataEls[k];
      auto& pRD = nodeRowDataEls[p];
      std::vector<OpCost> newRdEls;

#if 1
      for (size_t ik = 0, ip = 0; ik < kRD.size() || ip < pRD.size(); /* */) {
        if ((ip >= pRD.size()) || (ik < kRD.size() && kRD[ik].colIdx < pRD[ip].colIdx)) {
          if (kRD[ik].colIdx != k) {
            newRdEls.push_back(kRD[ik]);
          }
          ik++;
        } else if ((ik >= kRD.size()) || (ip < pRD.size() && kRD[ik].colIdx > pRD[ip].colIdx)) {
          if (pRD[ip].colIdx != p) {
            newRdEls.push_back(pRD[ip]);
          }
          ip++;
        } else {
          int64_t c = pRD[ip].colIdx;
          auto& sygeC = sygeCosts[c];
          auto& asmblC = asmblCosts[c];
          sygeC -= sygeModelV(kRD[ik].rowsDown + kRD[ik].rows, kRD[ik].rows);
          asmblC -= asmblModelV(kRD[ik].rBlocksDown + kRD[ik].rBlocks);
          sygeC -= sygeModelV(pRD[ip].rowsDown + pRD[ip].rows, pRD[ip].rows);
          asmblC -= asmblModelV(pRD[ip].rBlocksDown + pRD[ip].rBlocks);
          sygeC += sygeModelV(pRD[ip].rowsDown + (kRD[ik].rows + pRD[ip].rows),
                              (kRD[ik].rows + pRD[ip].rows));
          asmblC += asmblModelV(pRD[ip].rBlocksDown + (kRD[ik].rBlocks + pRD[ip].rBlocks));

          newRdEls.push_back({.colIdx = c,
                              .rBlocks = kRD[ik].rBlocks + pRD[ip].rBlocks,
                              .rows = kRD[ik].rows + pRD[ip].rows,
                              .rBlocksDown = pRD[ip].rBlocksDown,
                              .rowsDown = pRD[ip].rowsDown});
          ik++;
          ip++;
        }
      }
      auto& sygeC = sygeCosts[p];
      auto& asmblC = asmblCosts[p];
      sygeC -= sygeModelV(nodeRows[p] + prevNodeSize, prevNodeSize);
      asmblC -= asmblModelV(nodeRowBlocks[p] + prevNumMergedNodes);
      sygeC += sygeModelV(nodeRows[p] + nodeSize[p], nodeSize[p]);
      asmblC += asmblModelV(nodeRowBlocks[p] + numMergedNodes[p]);
      newRdEls.push_back({.colIdx = p,
                          .rBlocks = numMergedNodes[p],
                          .rows = nodeSize[p],
                          .rBlocksDown = nodeRowBlocks[p],
                          .rowsDown = nodeRows[p]});
      swap(nodeRowDataEls[p], newRdEls);
#endif
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