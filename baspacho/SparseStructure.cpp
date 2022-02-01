
#include "SparseStructure.h"

#if BASPACHO_USE_SUITESPARSE_AMD
#include <amd.h>
#else
#include <Eigen/OrderingMethods>
#endif

#include <algorithm>

#include "DebugMacros.h"
#include "Utils.h"

using namespace std;

void SparseStructure::sortIndices() {
    int64_t ord = order();
    for (int64_t i = 0; i < ord; i++) {
        int64_t start = ptrs[i];
        int64_t end = ptrs[i + 1];
        std::sort(inds.begin() + start, inds.begin() + end);
    }
}

// assumed square matrix
SparseStructure SparseStructure::transpose(bool sortIndices) const {
    int64_t ord = order();
    SparseStructure retv;
    retv.ptrs.assign(ord + 1, 0);

    for (int64_t i = 0; i < ord; i++) {
        int64_t start = ptrs[i];
        int64_t end = ptrs[i + 1];
        for (int64_t k = start; k < end; k++) {
            int64_t j = inds[k];
            BASPACHO_CHECK_LT(j, ord);
            retv.ptrs[j]++;
        }
    }

    int64_t tot = cumSumVec(retv.ptrs);
    retv.inds.resize(tot);

    for (int64_t i = 0; i < ord; i++) {
        int64_t start = ptrs[i];
        int64_t end = ptrs[i + 1];
        for (int64_t k = start; k < end; k++) {
            int64_t j = inds[k];
            BASPACHO_CHECK_LT(j, ord);
            BASPACHO_CHECK_LT(retv.ptrs[j], retv.inds.size());
            retv.inds[retv.ptrs[j]++] = i;
        }
    }

    rewindVec(retv.ptrs);

    if (sortIndices) {
        retv.sortIndices();
    }

    return retv;
}

// assumed square matrix
SparseStructure SparseStructure::clear(bool lowerHalf) const {
    int64_t ord = order();
    SparseStructure retv;
    retv.ptrs.assign(ord + 1, 0);

    for (int64_t i = 0; i < ord; i++) {
        int64_t start = ptrs[i];
        int64_t end = ptrs[i + 1];
        for (int64_t k = start; k < end; k++) {
            int64_t j = inds[k];
            BASPACHO_CHECK_LT(j, ord);
            if (i != j && (j > i) == lowerHalf) {
                continue;
            }
            retv.ptrs[i]++;
        }
    }

    int64_t tot = cumSumVec(retv.ptrs);
    retv.inds.resize(tot);

    for (int64_t i = 0; i < ord; i++) {
        int64_t start = ptrs[i];
        int64_t end = ptrs[i + 1];
        for (int64_t k = start; k < end; k++) {
            int64_t j = inds[k];
            BASPACHO_CHECK_LT(j, ord);
            if (i != j && (j > i) == lowerHalf) {
                continue;
            }
            BASPACHO_CHECK_LT(retv.ptrs[i], retv.inds.size());
            retv.inds[retv.ptrs[i]++] = j;
        }
    }

    rewindVec(retv.ptrs);

    return retv;
}

SparseStructure SparseStructure::symmetricPermutation(
    const std::vector<int64_t>& mapPerm, bool lowerHalf,
    bool sortIndices) const {
    int64_t ord = order();
    BASPACHO_CHECK_EQ(ord, mapPerm.size());

    SparseStructure retv;
    retv.ptrs.assign(ord + 1, 0);

    for (int64_t i = 0; i < ord; i++) {
        int64_t start = ptrs[i];
        int64_t end = ptrs[i + 1];
        int64_t newI = mapPerm[i];
        BASPACHO_CHECK_LT(newI, ord);
        for (int64_t k = start; k < end; k++) {
            int64_t j = inds[k];
            BASPACHO_CHECK_LT(j, ord);
            int64_t newJ = mapPerm[j];
            BASPACHO_CHECK_LT(newJ, ord);
            int64_t col = lowerHalf ? min(newI, newJ) : max(newI, newJ);
            retv.ptrs[col]++;
        }
    }

    int64_t tot = cumSumVec(retv.ptrs);
    retv.inds.resize(tot);

    for (int64_t i = 0; i < ord; i++) {
        int64_t start = ptrs[i];
        int64_t end = ptrs[i + 1];
        int64_t newI = mapPerm[i];
        BASPACHO_CHECK_LT(newI, ord);
        for (int64_t k = start; k < end; k++) {
            int64_t j = inds[k];
            BASPACHO_CHECK_LT(j, ord);
            int64_t newJ = mapPerm[j];
            BASPACHO_CHECK_LT(newJ, ord);
            int64_t col = lowerHalf ? min(newI, newJ) : max(newI, newJ);
            int64_t row = lowerHalf ? max(newI, newJ) : min(newI, newJ);
            BASPACHO_CHECK_LT(retv.ptrs[col], retv.inds.size());
            retv.inds[retv.ptrs[col]++] = row;
        }
    }

    rewindVec(retv.ptrs);

    if (sortIndices) {
        retv.sortIndices();
    }

    return retv;
}

SparseStructure SparseStructure::addIndependentEliminationFill(
    int64_t elimStart, int64_t elimEnd, bool sortIdx) const {
    int64_t ord = order();

    // nothing to do, no entries are added
    if (elimEnd == ord) {
        return *this;
    }

    SparseStructure tThis = transpose(false);
    for (int64_t i = elimStart; i < elimEnd; i++) {  // sort subset
        int64_t start = tThis.ptrs[i];
        int64_t end = tThis.ptrs[i + 1];
        std::sort(tThis.inds.begin() + start, tThis.inds.begin() + end);
    }

    SparseStructure retv;
    retv.ptrs.reserve(ptrs.size());
    retv.ptrs.assign(ptrs.begin(), ptrs.begin() + elimEnd + 1);
    retv.inds.assign(inds.begin(), inds.begin() + ptrs[elimEnd]);

    vector<int64_t> tags(ord, -1);  // mark added row entries
    int64_t numMatches = 0;
    for (int64_t k = elimEnd; k < ord; ++k) {
        int64_t start = ptrs[k];
        int64_t end = ptrs[k + 1];
        tags[k] = k;
        retv.inds.push_back(k);
        for (int64_t q = start; q < end; q++) {
            int64_t i = inds[q];
            if (i >= k) {
                continue;
            }
            if (tags[i] != k) {
                retv.inds.push_back(i); /* L(k,i) is nonzero */
                tags[i] = k;
            }

            // for i in elim lump, walk rows in same column
            if (i >= elimStart && i < elimEnd) {
                int64_t tStart = tThis.ptrs[i];
                int64_t tEnd = tThis.ptrs[i + 1];
                for (int64_t t = tStart; t < tEnd; t++) {
                    int64_t q = tThis.inds[t];
                    if (q >= k) {
                        break;  // tThis rows are sorted
                    }
                    if (tags[q] < k) {
                        tags[q] = k;
                        retv.inds.push_back(q); /* L(k,q) is nonzero */
                    }
                }
            }
        }
        retv.ptrs.push_back(retv.inds.size());
    }

    if (sortIdx) {
        retv.sortIndices();
    }

    return retv;
}

SparseStructure SparseStructure::addFullEliminationFill() const {
    int64_t ord = order();
    vector<int64_t> tags(ord), parent(ord, -1);

    // skeleton of the algo to iterate on fillup's nodes is from Eigen's
    // `SimplicialCholesky_impl.h` (by Gael Guennebaud),
    // in turn from LDL by Timothy A. Davis.
    SparseStructure retv;
    retv.ptrs.assign(ord + 1, 1);  // write sizes here initially
    for (int64_t k = 0; k < ord; ++k) {
        /* L(k,:) pattern: all nodes reachable in etree from nz in A(0:k-1,k) */
        parent[k] = -1; /* parent of k is not yet known */
        tags[k] = k;    /* mark node k as visited, L(k,k) is nonzero */

        int64_t start = ptrs[k];
        int64_t end = ptrs[k + 1];
        for (int64_t q = start; q < end; q++) {
            int64_t i = inds[q];
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
    int64_t tot = cumSumVec(retv.ptrs);
    retv.inds.resize(tot);

    // walk again, saving entries in rows
    for (int64_t k = 0; k < ord; ++k) {
        /* L(k,:) pattern: all nodes reachable in etree from nz in A(0:k-1,k) */
        parent[k] = -1;                /* parent of k is not yet known */
        tags[k] = k;                   /* mark node k as visited */
        retv.inds[retv.ptrs[k]++] = k; /* L(k,k) is nonzero */

        int64_t start = ptrs[k];
        int64_t end = ptrs[k + 1];
        for (int64_t q = start; q < end; q++) {
            int64_t i = inds[q];
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

    rewindVec(retv.ptrs);

    retv.sortIndices();

    return retv;
}

#if BASPACHO_USE_SUITESPARSE_AMD

std::vector<int64_t> SparseStructure::fillReducingPermutation() const {
    std::vector<int64_t> colPtr(ptrs.begin(), ptrs.end()),
        rowInd(inds.begin(), inds.end());
    std::vector<int64_t> P(colPtr.size() - 1);
    double Control[AMD_CONTROL], Info[AMD_INFO];

    amd_l_defaults(Control);
    // amd_l_control(Control); // print verbose messages

    int result = amd_l_order(P.size(), colPtr.data(), rowInd.data(), P.data(),
                             Control, Info);
    BASPACHO_CHECK_EQ(result, AMD_OK);

    return std::vector<int64_t>(P.begin(), P.end());
}

#else

std::vector<int64_t> SparseStructure::fillReducingPermutation() const {
    using INT = int;
    std::vector<INT> colPtr(ptrs.begin(), ptrs.end()),
        rowInd(inds.begin(), inds.end());

    using namespace Eigen;

    // NOTE: feeding directly Map<...> references to AMDOrdering doesn't work,
    // this is probably a bug/limitation of Eigen
    vector<int8_t> fakeData(rowInd.size());
    Map<SparseMatrix<int8_t, ColMajor, INT> > matMap(
        colPtr.size() - 1, colPtr.size() - 1, rowInd.size(), colPtr.data(),
        rowInd.data(), (int8_t*)fakeData.data());
    SparseMatrix<int8_t, ColMajor, INT> mat = matMap;
    PermutationMatrix<Dynamic, Dynamic, INT> perm;
    AMDOrdering<INT>()(mat, perm);

    return std::vector<int64_t>(perm.indices().data(),
                                perm.indices().data() + perm.size());
}

#endif

SparseStructure SparseStructure::extractRightBottom(int64_t startRow) {
    int64_t ord = order();
    BASPACHO_CHECK_LT(startRow, ord);
    BASPACHO_CHECK_GE(startRow, 0);
    int64_t newOrd = ord - startRow;

    SparseStructure retv;
    retv.ptrs.assign(newOrd + 1, 0);

    for (int64_t i = startRow; i < ord; i++) {
        int64_t start = ptrs[i];
        int64_t end = ptrs[i + 1];
        for (int64_t k = start; k < end; k++) {
            int64_t j = inds[k];
            BASPACHO_CHECK_LT(j, ord);
            if (j >= startRow) {
                retv.ptrs[i - startRow]++;
            }
        }
    }

    int64_t tot = cumSumVec(retv.ptrs);
    retv.inds.resize(tot);

    for (int64_t i = 0; i < ord; i++) {
        int64_t start = ptrs[i];
        int64_t end = ptrs[i + 1];
        for (int64_t k = start; k < end; k++) {
            int64_t j = inds[k];
            BASPACHO_CHECK_LT(j, ord);
            if (j >= startRow) {
                BASPACHO_CHECK_LT(retv.ptrs[i - startRow], retv.inds.size());
                retv.inds[retv.ptrs[i - startRow]++] = j - startRow;
            }
        }
    }

    rewindVec(retv.ptrs);
    return retv;
}
