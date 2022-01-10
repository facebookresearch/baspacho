
#include "SparseStructure.h"

#include <glog/logging.h>

#include <algorithm>

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
    retv.inds.resize(inds.size());

    for (uint64_t i = 0; i < ord; i++) {
        uint64_t start = ptrs[i];
        uint64_t end = ptrs[i + 1];
        for (uint64_t k = start; k < end; k++) {
            uint64_t j = inds[k];
            CHECK_LT(j, ord);
            retv.ptrs[j]++;
        }
    }

    uint64_t tot = 0;
    for (uint64_t i = 0; i < ord; i++) {
        uint64_t oldTot = tot;
        tot += retv.ptrs[i];
        retv.ptrs[i] = oldTot;
    }
    retv.ptrs[ord] = tot;

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

    for (uint64_t i = ord; i >= 1; i--) {
        retv.ptrs[i] = retv.ptrs[i - 1];
    }
    retv.ptrs[0] = 0;

    if (sortIndices) {
        retv.sortIndices();
    }

    return retv;
}

SparseStructure SparseStructure::symmetricPermutation(
    const std::vector<uint64_t>& mapPerm, bool lowerHalf,
    bool sortIndices) const {
    uint64_t ord = order();
    CHECK_EQ(ord, mapPerm.size());

    SparseStructure retv;
    retv.ptrs.assign(ord + 1, 0);
    retv.inds.resize(inds.size());

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

    uint64_t tot = 0;
    for (uint64_t i = 0; i < ord; i++) {
        uint64_t oldTot = tot;
        tot += retv.ptrs[i];
        retv.ptrs[i] = oldTot;
    }
    retv.ptrs[ord] = tot;

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

    for (uint64_t i = ord; i >= 1; i--) {
        retv.ptrs[i] = retv.ptrs[i - 1];
    }
    retv.ptrs[0] = 0;

    if (sortIndices) {
        retv.sortIndices();
    }

    return retv;
}