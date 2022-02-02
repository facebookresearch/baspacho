
#include "baspacho/EliminationTree.h"

#include <algorithm>

#include "baspacho/DebugMacros.h"
#include "baspacho/Utils.h"

namespace BaSpaCho {

using namespace std;

EliminationTree::EliminationTree(const std::vector<int64_t>& paramSize,
                                 const SparseStructure& ss)
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

    // compute node-fill (just entries below diagonal)
    nodeFill.resize(ord);
    for (int64_t k = 0; k < ord; ++k) {
        nodeFill[k] =
            nodeRows[k] * nodeSize[k] + nodeRows[k] * (nodeRows[k] + 1) / 2;
    }

    // compute next child
    firstChild.assign(ord, -1);
    nextSibling.assign(ord, -1);
    for (int64_t k = 0; k < ord; ++k) {
        int64_t p = parent[k];
        if (p != -1) {
            nextSibling[k] = firstChild[p];
            firstChild[p] = k;
        }
    }
}

// TODO: expand on merge settings
static constexpr double kPropRows = 0.7;

void EliminationTree::computeMerges() {
    int64_t ord = ss.order();
    vector<int64_t> height(ord, 0);
    vector<tuple<int64_t, int64_t, int64_t>> unmergedHeightNode;
    unmergedHeightNode.reserve(ord);
    for (int64_t k = 0; k < ord; k++) {
        unmergedHeightNode.emplace_back(height[k], nodeSize[k], k);

        int64_t par = parent[k];
        if (par == -1) {
            continue;
        }
        height[par] = max(height[par], height[k] + 1);
    }

    mergeWith.assign(ord, -1);
    numMerges = 0;
    for (int64_t k = ord - 1; k >= 0; k--) {
        int64_t p = parent[k];
        if (p == -1) {
            continue;
        }
        while (mergeWith[p] != -1) {
            p = mergeWith[p];
        }

        // determine if k and p should be merged
        bool willMerge = nodeRows[k] > (nodeRows[p] + nodeSize[p]) * kPropRows;

        auto [height, _1, _2] = unmergedHeightNode[k];
        if (height <= 1) {
            willMerge = false;
        }

        if (!willMerge) {
            continue;
        }

        // do merge
        mergeWith[k] = p;  // so it becomes direct merge
        nodeSize[p] += nodeSize[k];
        numMerges++;
    }
}

void EliminationTree::computeAggregateStruct() {
    // compute childern list
    int64_t ord = ss.order();
    vector<int64_t> firstMergeChild(ord, -1);
    vector<int64_t> nextMergeSibling(ord, -1);
    for (int64_t k = 0; k < ord; k++) {
        int64_t p = mergeWith[k];
        if (p != -1) {
            nextMergeSibling[k] = firstMergeChild[p];
            firstMergeChild[p] = k;
        }
    }

    // sort according to height (and secondly size)
    vector<int64_t> height(ord, 0);
    vector<tuple<int64_t, int64_t, int64_t>> heightNode;
    heightNode.reserve(ord - numMerges);
    for (int64_t k = 0; k < ord; k++) {
        if (mergeWith[k] != -1) {
            continue;
        }
        heightNode.emplace_back(height[k], nodeSize[k], k);

        int64_t par = parent[k];
        if (par == -1) {
            continue;
        }
        int64_t mPar = mergeWith[par];
        par = mPar != -1 ? mPar : par;
        height[par] = max(height[par], height[k] + 1);
    }
    sort(heightNode.begin(), heightNode.end());

    // straightening permutation, make merged nodes consecutive
    int64_t numLumps = ord - numMerges;
    vector<int64_t> spanToLump(ord);
    permutation.resize(ord);
    lumpStart.resize(numLumps + 1);
    lumpToSpan.assign(numLumps + 1, 0);
    int64_t pIdx = ord;
    int64_t agIdx = numLumps;
    for (int64_t idx = heightNode.size() - 1; idx >= 0; idx--) {
        auto [_1, _2, k] = heightNode[idx];

        BASPACHO_CHECK_GT(agIdx, 0);
        lumpStart[--agIdx] = nodeSize[k];

        BASPACHO_CHECK_GT(pIdx, 0);
        permutation[--pIdx] = k;
        spanToLump[k] = agIdx;
        lumpToSpan[agIdx]++;
        for (int64_t q = firstMergeChild[k]; q != -1; q = nextMergeSibling[q]) {
            BASPACHO_CHECK_GT(pIdx, 0);
            permutation[--pIdx] = q;
            spanToLump[q] = agIdx;
            lumpToSpan[agIdx]++;
        }
    }
    BASPACHO_CHECK_EQ(pIdx, 0);
    BASPACHO_CHECK_EQ(agIdx, 0);

    // cum-sum lumpStart
    cumSumVec(lumpToSpan);
    int64_t tot = cumSumVec(lumpStart);
    permInverse = inversePermutation(permutation);

    SparseStructure tperm =  // lower-half csc
        ss.symmetricPermutation(permInverse, /* lowerHalf = */ false,
                                /* sortIndices = */ false)
            .addFullEliminationFill()
            .transpose();

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
        std::sort(rowParam.begin() + colStart[colStart.size() - 1],
                  rowParam.end());
        colStart.push_back(rowParam.size());
    }

    // set spanStart to cumSumVec of paramSize
    spanStart.reserve(paramSize.size() + 1);
    spanStart = paramSize;
    spanStart.push_back(0);
    cumSumVec(spanStart);
}

}  // end namespace BaSpaCho