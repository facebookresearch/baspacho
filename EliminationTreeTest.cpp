#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include "EliminationTree.h"
#include "TestingUtils.h"
#include "Utils.h"

using namespace std;
using namespace testing;

TEST(EliminationTree, Build) {
    vector<uint64_t> ptrs{0,   9,   15,  21,  27,  33,  39,  48,  57,
                          61,  70,  76,  82,  88,  94,  100, 106, 110,
                          119, 128, 137, 143, 152, 156, 160};
    vector<uint64_t> inds{
        /* column  0: */ 0, 5,  6,  12, 13, 17, 18, 19, 21,
        /* column  1: */ 1, 8,  9,  13, 14, 17,
        /* column  2: */ 2, 6,  11, 20, 21, 22,
        /* column  3: */ 3, 7,  10, 15, 18, 19,
        /* column  4: */ 4, 7,  9,  14, 15, 16,
        /* column  5: */ 0, 5,  6,  12, 13, 17,
        /* column  6: */ 0, 2,  5,  6,  11, 12, 19, 21, 23,
        /* column  7: */ 3, 4,  7,  9,  14, 15, 16, 17, 18,
        /* column  8: */ 1, 8,  9,  14,
        /* column  9: */ 1, 4,  7,  8,  9,  13, 14, 17, 18,
        /* column 10: */ 3, 10, 18, 19, 20, 21,
        /* column 11: */ 2, 6,  11, 12, 21, 23,
        /* column 12: */ 0, 5,  6,  11, 12, 23,
        /* column 13: */ 0, 1,  5,  9,  13, 17,
        /* column 14: */ 1, 4,  7,  8,  9,  14,
        /* column 15: */ 3, 4,  7,  15, 16, 18,
        /* column 16: */ 4, 7,  15, 16,
        /* column 17: */ 0, 1,  5,  7,  9,  13, 17, 18, 19,
        /* column 18: */ 0, 3,  7,  9,  10, 15, 17, 18, 19,
        /* column 19: */ 0, 3,  6,  10, 17, 18, 19, 20, 21,
        /* column 20: */ 2, 10, 19, 20, 21, 22,
        /* column 21: */ 0, 2,  6,  10, 11, 19, 20, 21, 22,
        /* column 22: */ 2, 20, 21, 22,
        /* column 23: */ 6, 11, 12, 23};

    SparseStructure ssOrig =
        SparseStructure(ptrs, inds).clear();  // lower half csr

#if 0
    vector<uint64_t> permutation = ssOrig.fillReducingPermutation();
    vector<uint64_t> invPerm = inversePermutation(permutation);

    SparseStructure ss =
        ssOrig.symmetricPermutation(invPerm, false).addFullEliminationFill();
    LOG(INFO) << "perm:\n" << printPattern(ss, false);
    LOG(INFO) << "entries: " << ss.inds.size();
#endif

#if 1
    vector<uint64_t> permutation = ssOrig.fillReducingPermutation();
    vector<uint64_t> invPerm = inversePermutation(permutation);

    SparseStructure ss =
        ssOrig.symmetricPermutation(invPerm, false).addFullEliminationFill();
    /*vector<uint64_t> permutation = randomPermutation(ptrs.size() - 1, 40);
    SparseStructure ss =
        ssOrig.clear().symmetricPermutation(permutation, false);*/

    LOG(INFO) << "perm:\n" << printPattern(ss, false);
    LOG(INFO) << "zz.ptrs: " << printInts(ss.ptrs);
    LOG(INFO) << "zz.inds: " << printInts(ss.inds);

    vector<uint64_t> paramSize(ptrs.size() - 1, 1);
    EliminationTree et(paramSize, ss);

    et.buildTree();
    LOG(INFO) << "parents: " << printInts(et.parent);
    LOG(INFO) << "1st ch: " << printInts(et.firstChild);
    LOG(INFO) << "nextsb: " << printInts(et.nextSibling);

    et.computeMerges();
    LOG(INFO) << "mergeWith: " << printInts(et.mergeWith);

    et.computeAggregateStruct();
#endif
}