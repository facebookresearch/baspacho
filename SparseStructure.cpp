
#include "SparseStructure.h"

#include <glog/logging.h>
#include <suitesparse/amd.h>

#include <algorithm>

#include "Utils.h"

using namespace std;

void SparseStructure::sortIndices() {
    uint64_t ord = order();
    for (uint64_t i = 0; i < ord; i++) {
        uint64_t start = ptrs[i];
        uint64_t end = ptrs[i + 1];
        std::sort(inds.begin() + start, inds.begin() + end);
    }
}

// assumed square matrix
SparseStructure SparseStructure::transpose(bool sortIndices) const {
    uint64_t ord = order();
    SparseStructure retv;
    retv.ptrs.assign(ord + 1, 0);

    for (uint64_t i = 0; i < ord; i++) {
        uint64_t start = ptrs[i];
        uint64_t end = ptrs[i + 1];
        for (uint64_t k = start; k < end; k++) {
            uint64_t j = inds[k];
            CHECK_LT(j, ord);
            retv.ptrs[j]++;
        }
    }

    uint64_t tot = cumSum(retv.ptrs);
    retv.inds.resize(tot);

    for (uint64_t i = 0; i < ord; i++) {
        uint64_t start = ptrs[i];
        uint64_t end = ptrs[i + 1];
        for (uint64_t k = start; k < end; k++) {
            uint64_t j = inds[k];
            CHECK_LT(j, ord);
            CHECK_LT(retv.ptrs[j], retv.inds.size());
            retv.inds[retv.ptrs[j]++] = i;
        }
    }

    rewind(retv.ptrs);

    if (sortIndices) {
        retv.sortIndices();
    }

    return retv;
}

// assumed square matrix
SparseStructure SparseStructure::clear(bool lowerHalf) const {
    uint64_t ord = order();
    SparseStructure retv;
    retv.ptrs.assign(ord + 1, 0);

    for (uint64_t i = 0; i < ord; i++) {
        uint64_t start = ptrs[i];
        uint64_t end = ptrs[i + 1];
        for (uint64_t k = start; k < end; k++) {
            uint64_t j = inds[k];
            CHECK_LT(j, ord);
            if (i != j && (j > i) == lowerHalf) {
                continue;
            }
            retv.ptrs[i]++;
        }
    }

    uint64_t tot = cumSum(retv.ptrs);
    retv.inds.resize(tot);

    for (uint64_t i = 0; i < ord; i++) {
        uint64_t start = ptrs[i];
        uint64_t end = ptrs[i + 1];
        for (uint64_t k = start; k < end; k++) {
            uint64_t j = inds[k];
            CHECK_LT(j, ord);
            if (i != j && (j > i) == lowerHalf) {
                continue;
            }
            CHECK_LT(retv.ptrs[i], retv.inds.size());
            retv.inds[retv.ptrs[i]++] = j;
        }
    }

    rewind(retv.ptrs);

    return retv;
}

SparseStructure SparseStructure::symmetricPermutation(
    const std::vector<uint64_t>& mapPerm, bool lowerHalf,
    bool sortIndices) const {
    uint64_t ord = order();
    CHECK_EQ(ord, mapPerm.size());

    SparseStructure retv;
    retv.ptrs.assign(ord + 1, 0);

    for (uint64_t i = 0; i < ord; i++) {
        uint64_t start = ptrs[i];
        uint64_t end = ptrs[i + 1];
        uint64_t newI = mapPerm[i];
        CHECK_LT(newI, ord);
        for (uint64_t k = start; k < end; k++) {
            uint64_t j = inds[k];
            CHECK_LT(j, ord);
            uint64_t newJ = mapPerm[j];
            CHECK_LT(newJ, ord);
            uint64_t col = lowerHalf ? min(newI, newJ) : max(newI, newJ);
            retv.ptrs[col]++;
        }
    }

    uint64_t tot = cumSum(retv.ptrs);
    retv.inds.resize(tot);

    for (uint64_t i = 0; i < ord; i++) {
        uint64_t start = ptrs[i];
        uint64_t end = ptrs[i + 1];
        uint64_t newI = mapPerm[i];
        CHECK_LT(newI, ord);
        for (uint64_t k = start; k < end; k++) {
            uint64_t j = inds[k];
            CHECK_LT(j, ord);
            uint64_t newJ = mapPerm[j];
            CHECK_LT(newJ, ord);
            uint64_t col = lowerHalf ? min(newI, newJ) : max(newI, newJ);
            uint64_t row = lowerHalf ? max(newI, newJ) : min(newI, newJ);
            CHECK_LT(retv.ptrs[col], retv.inds.size());
            retv.inds[retv.ptrs[col]++] = row;
        }
    }

    rewind(retv.ptrs);

    if (sortIndices) {
        retv.sortIndices();
    }

    return retv;
}

SparseStructure SparseStructure::addIndependentEliminationFill(
    uint64_t elimStart, uint64_t elimEnd) const {
    uint64_t ord = order();

    SparseStructure retv;
    retv.ptrs.assign(ord + 1, 1);  // write sizes here initially
    for (uint64_t k = 0; k < elimEnd; ++k) {
        uint64_t start = ptrs[k];
        uint64_t end = ptrs[k + 1];
        for (uint64_t q = start; q < end; q++) {
            uint64_t i = inds[q];
            if (i >= k) {
                continue;
            }
            // verify elimStart - elimEnd can be eliminated independently
            if (k >= elimStart) {
                CHECK((i < elimStart) || i >= elimEnd);
            }
            retv.ptrs[k]++;
        }
    }

    // nothing to do, no entries are added
    if (elimEnd == ord) {
        return *this;
    }

    uint64_t dataStart = ptrs[elimEnd];
    uint64_t dataSize = ptrs[ord] - dataStart;
    uint64_t rangeSize = elimEnd - elimStart;
    vector<int64_t> tags(ord, -1);                 // mark added row entries
    vector<int64_t> colListRowIdx(dataSize, -1);   // col-list row index
    vector<int64_t> colListPrevPtr(dataSize, -1);  // col-list pointer to prev
    vector<int64_t> colDataRefs(rangeSize, -1);    // per-col list
    for (uint64_t k = elimEnd; k < ord; ++k) {
        uint64_t start = ptrs[k];
        uint64_t end = ptrs[k + 1];
        tags[k] = k;
        for (uint64_t q = start; q < end; q++) {
            uint64_t qPtr = q - dataStart;
            uint64_t i = inds[q];
            if (i >= k) {
                continue;
            }
            if (tags[i] != k) {
                retv.ptrs[k]++; /* L(k,i) is nonzero */
                tags[i] = k;
            }

            // for i in elim range, walk rows in same column
            if (i >= elimStart && i < elimEnd) {
                int64_t cRef = colDataRefs[i - elimStart];
                for (int64_t ref = cRef; ref != -1; ref = colListPrevPtr[ref]) {
                    uint64_t r = colListRowIdx[ref];
                    if (tags[r] != k) {
                        retv.ptrs[k]++; /* L(k,r) is nonzero */
                        tags[r] = k;
                    }
                }
                // append row value `k` to the col-list
                colListRowIdx[qPtr] = k;
                colListPrevPtr[qPtr] = cRef;
                colDataRefs[i - elimStart] = qPtr;
            }
        }
    }

    // cumulate-sum ptrs: sizes -> pointers
    uint64_t tot = cumSum(retv.ptrs);
    retv.inds.reserve(tot);
    retv.inds.assign(inds.begin(), inds.begin() + dataStart);
    retv.inds.resize(tot);

    tags.assign(ord, -1);               // mark added row entries
    colDataRefs.assign(rangeSize, -1);  // per-col list
    for (uint64_t k = elimEnd; k < ord; ++k) {
        uint64_t start = ptrs[k];
        uint64_t end = ptrs[k + 1];
        tags[k] = k;
        retv.inds[retv.ptrs[k]++] = k;

        for (uint64_t q = start; q < end; q++) {
            uint64_t qPtr = q - dataStart;
            uint64_t i = inds[q];
            if (i >= k) {
                continue;
            }
            if (tags[i] != k) {
                retv.inds[retv.ptrs[k]++] = i; /* L(k,i) is nonzero */
                tags[i] = k;
            }

            // for i in elim range, walk rows in same column
            if (i >= elimStart && i < elimEnd) {
                int64_t cRef = colDataRefs[i - elimStart];
                for (int64_t ref = cRef; ref != -1; ref = colListPrevPtr[ref]) {
                    uint64_t r = colListRowIdx[ref];
                    if (tags[r] != k) {
                        retv.inds[retv.ptrs[k]++] = r; /* L(k,r) is nonzero */
                        tags[r] = k;
                    }
                }
                // move list start pointer
                colDataRefs[i - elimStart] = qPtr;
            }
        }
    }

    // move back pointers advanced while writing entries (from elimEnd)
    rewind(retv.ptrs, elimEnd, dataStart);

    retv.sortIndices();

    return retv;
}

SparseStructure SparseStructure::addFullEliminationFill() const {
    uint64_t ord = order();
    vector<int64_t> tags(ord), parent(ord, -1);

    // skeleton of the algo to iterate on fillup's nodes is from Eigen's
    // `SimplicialCholesky_impl.h` (by Gael Guennebaud),
    // in turn from LDL by Timothy A. Davis.
    SparseStructure retv;
    retv.ptrs.assign(ord + 1, 1);  // write sizes here initially
    for (uint64_t k = 0; k < ord; ++k) {
        /* L(k,:) pattern: all nodes reachable in etree from nz in A(0:k-1,k) */
        parent[k] = -1; /* parent of k is not yet known */
        tags[k] = k;    /* mark node k as visited, L(k,k) is nonzero */

        uint64_t start = ptrs[k];
        uint64_t end = ptrs[k + 1];
        for (uint64_t q = start; q < end; q++) {
            uint64_t i = inds[q];
            if (i >= k) {
                continue;
            }
            /* follow path from i to root of etree, stop at flagged node */
            for (; tags[i] != k; i = parent[i]) {
                /* find parent of i if not yet determined */
                if (parent[i] == -1) {
                    parent[i] = k;
                }

                retv.ptrs[k]++; /* L(k,i) is nonzero */
                tags[i] = k;
            }
        }
    }

    // cumulate-sum ptrs: sizes -> pointers
    uint64_t tot = cumSum(retv.ptrs);
    retv.inds.resize(tot);

    // walk again, saving entries in rows
    for (uint64_t k = 0; k < ord; ++k) {
        /* L(k,:) pattern: all nodes reachable in etree from nz in A(0:k-1,k) */
        parent[k] = -1;                /* parent of k is not yet known */
        tags[k] = k;                   /* mark node k as visited */
        retv.inds[retv.ptrs[k]++] = k; /* L(k,k) is nonzero */

        uint64_t start = ptrs[k];
        uint64_t end = ptrs[k + 1];
        for (uint64_t q = start; q < end; q++) {
            uint64_t i = inds[q];
            if (i < k) {
                /* follow path from i to root of etree, stop at flagged node
                 */
                for (; tags[i] != k; i = parent[i]) {
                    /* find parent of i if not yet determined */
                    if (parent[i] == -1) {
                        parent[i] = k;
                    }
                    retv.inds[retv.ptrs[k]++] = i; /* L(k,i) is nonzero */
                    tags[i] = k;                   /* mark i as visited */
                }
            }
        }
    }

    rewind(retv.ptrs);

    retv.sortIndices();

    return retv;
}

std::vector<uint64_t> SparseStructure::fillReducingPermutation() const {
    std::vector<int64_t> colPtr(ptrs.begin(), ptrs.end()),
        rowInd(inds.begin(), inds.end());
    std::vector<int64_t> P(colPtr.size() - 1);
    double Control[AMD_CONTROL], Info[AMD_INFO];

    LOG(INFO) << "run AMD...";
    amd_l_defaults(Control);
    amd_l_control(Control);

    int result = amd_l_order(P.size(), colPtr.data(), rowInd.data(), P.data(),
                             Control, Info);
    LOG(INFO) << "result: " << result;

    return std::vector<uint64_t>(P.begin(), P.end());
}