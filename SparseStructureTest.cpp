#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "SparseStructure.h"
#include "TestingUtils.h"

using namespace std;
using namespace testing;

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
    vector<uint64_t> sizes{10, 20, 30, 40};
    vector<double> fills{0.15, 0.23, 0.3};
    int seed = 37;
    for (auto size : sizes) {
        for (auto fill : fills) {
            auto colsOrig = randomCols(size, fill, seed++);
            auto ssOrig = columnsToCscStruct(colsOrig).transpose();

            for (int start = 0; start < size * 2 / 3; start += 3) {
                for (int end = start + 3; end < size; end += 3) {
                    vector<set<uint64_t>> cols =
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
    vector<uint64_t> sizes{10, 20, 30, 40};
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
