#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "baspacho/DebugMacros.h"
#include "baspacho/SparseStructure.h"
#include "baspacho/Utils.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing;
using namespace std;
using namespace ::testing;

TEST(SparseStructure, Transpose) {
    /*
      X _ _ X _
      _ _ X _ X
      X X _ _ X
      _ X X _ _
      _ _ X _ X
    */
    SparseStructure ss({0, 2, 4, 7, 9, 11}, {0, 3, 2, 4, 0, 1, 4, 1, 2, 2, 4});
    SparseStructure t = ss.transpose();

    ASSERT_THAT(t.ptrs, ElementsAre(0, 2, 4, 7, 8, 11));
    ASSERT_THAT(t.inds, ElementsAre(0, 2, 2, 3, 1, 3, 4, 0, 1, 2, 4));
}

TEST(SparseStructure, SymPermutation) {
    /*
      X
      _ X
      X X _
      _ X _ X
      _ _ X _ X
      X X _ _ X X
    */
    SparseStructure ss({0, 1, 2, 4, 6, 8, 12},
                       {0, 1, 0, 1, 1, 3, 2, 4, 0, 1, 4, 5});

    /*
      X
      _ X
      X _ _
      X _ _ X
      _ _ X X X
      _ X X X _ X
    */
    SparseStructure pCsr =
        ss.symmetricPermutation({4, 5, 2, 1, 0, 3}, /* lowerHalf = */ false);

    ASSERT_THAT(pCsr.ptrs, ElementsAre(0, 1, 2, 3, 5, 8, 12));
    ASSERT_THAT(pCsr.inds, ElementsAre(0, 1, 0, 0, 3, 2, 3, 4, 1, 2, 3, 5));

    SparseStructure pCsc =
        ss.symmetricPermutation({4, 5, 2, 1, 0, 3}, /* lowerHalf = */ true);

    ASSERT_THAT(pCsc.ptrs, ElementsAre(0, 3, 5, 7, 10, 11, 12));
    ASSERT_THAT(pCsc.inds, ElementsAre(0, 2, 3, 1, 5, 4, 5, 3, 4, 5, 4, 5));
}

TEST(SparseStructure, IndependentEliminationFill) {
    vector<int64_t> sizes{10, 20, 30, 40};
    vector<double> fills{0.15, 0.23, 0.3};
    int seed = 37;
    for (auto size : sizes) {
        for (auto fill : fills) {
            auto colsOrig = randomCols(size, fill, seed++);
            auto ssOrig = columnsToCscStruct(colsOrig).transpose();

            for (int start = 0; start < size * 2 / 3; start += 3) {
                for (int end = start + 3; end < size; end += 3) {
                    vector<set<int64_t>> cols =
                        makeIndependentElimSet(colsOrig, start, end);

                    auto ss = columnsToCscStruct(cols).transpose();

                    // naive (gt)
                    naiveAddEliminationEntries(cols, start, end);
                    auto ssElim = columnsToCscStruct(cols).transpose();

                    // algo
                    auto ssElim2 = ss.addIndependentEliminationFill(start, end);

                    ASSERT_THAT(ssElim.ptrs, ContainerEq(ssElim2.ptrs));
                    ASSERT_THAT(ssElim.inds, ContainerEq(ssElim2.inds));
                }
            }
        }
    }
}

TEST(SparseStructure, FullEliminationFill) {
    vector<int64_t> sizes{10, 20, 30, 40};
    vector<double> fills{0.15, 0.23, 0.3};
    int seed = 37;
    for (auto size : sizes) {
        for (auto fill : fills) {
            auto cols = randomCols(size, fill, seed++);
            auto ss = columnsToCscStruct(cols).transpose();

            // naive (gt)
            naiveAddEliminationEntries(cols, 0, size);
            auto ssElim = columnsToCscStruct(cols).transpose();

            // algo
            auto ssElim2 = ss.addFullEliminationFill();

            ASSERT_THAT(ssElim.ptrs, ContainerEq(ssElim2.ptrs));
            ASSERT_THAT(ssElim.inds, ContainerEq(ssElim2.inds));
        }
    }
}

TEST(SparseStructure, FillReducingPermutation) {
    vector<int64_t> ptrs{0,   9,   15,  21,  27,  33,  39,  48,  57,
                         61,  70,  76,  82,  88,  94,  100, 106, 110,
                         119, 128, 137, 143, 152, 156, 160};
    vector<int64_t> inds{
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

    vector<int64_t> permutation = ssOrig.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);

    SparseStructure ss =
        ssOrig.symmetricPermutation(invPerm, false).addFullEliminationFill();
    std::cout << "perm:\n" << printPattern(ss, false) << std::endl;
    std::cout << "entries: " << ss.inds.size() << std::endl;
    ASSERT_LE(ss.inds.size(), 130);  // should be 120
}