#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>
#include <numeric>
#include <sstream>

#include "SparseStructure.h"

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