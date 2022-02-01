
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include "../../testing/TestingUtils.h"
#include "../CoalescedBlockMatrix.h"
#include "../DebugMacros.h"
#include "../EliminationTree.h"
#include "../Utils.h"

using namespace std;
using namespace testing;

TEST(EliminationTree, Build) {
    for (int h = 0; h < 200; h++) {
        auto colsOrig = randomCols(70, 0.1, h + 37);
        auto ssOrig = columnsToCscStruct(colsOrig).transpose();

        vector<int64_t> permutation = ssOrig.fillReducingPermutation();
        vector<int64_t> invPerm = inversePermutation(permutation);

        SparseStructure ss = ssOrig.symmetricPermutation(invPerm, false);

        std::cout << "perm:\n" << printPattern(ss, false) << std::endl;

        vector<int64_t> paramSize(ssOrig.ptrs.size() - 1, 1);
        EliminationTree et(paramSize, ss);

        et.buildTree();

        et.computeMerges();

        et.computeAggregateStruct();

        CoalescedBlockMatrixSkel skel(et.spanStart, et.lumpToSpan, et.colStart,
                                      et.rowParam);
        int64_t totData = skel.chainData[skel.chainData.size() - 1];
        vector<double> data(totData, 1);
        Eigen::MatrixXd mat = skel.densify(data);
        std::cout << "densified:\n" << mat << std::endl;

        // original must be contained via idMap
        vector<int64_t> idMap = composePermutations(et.permInverse, invPerm);
        for (int64_t i = 0; i < ssOrig.ptrs.size() - 1; i++) {
            int64_t start = ssOrig.ptrs[i];
            int64_t end = ssOrig.ptrs[i + 1];
            int64_t newI = idMap[i];
            for (int64_t q = start; q < end; q++) {
                int64_t j = ssOrig.inds[q];
                int64_t newJ = idMap[j];
                int64_t r = std::max(newI, newJ);
                int64_t c = std::min(newI, newJ);
                ASSERT_GT(mat(r, c), 0.5);
            }
        }

        // permuted and elim-filled must be contained
        SparseStructure checkSs =
            ss.symmetricPermutation(et.permInverse, false, true)
                .addFullEliminationFill();
        std::cout << "check:\n" << printPattern(checkSs, false) << std::endl;
        for (int64_t i = 0; i < checkSs.ptrs.size() - 1; i++) {
            int64_t start = checkSs.ptrs[i];
            int64_t end = checkSs.ptrs[i + 1];
            for (int64_t q = start; q < end; q++) {
                int64_t j = checkSs.inds[q];
                ASSERT_GT(mat(i, j), 0.5);
            }
        }
    }
}