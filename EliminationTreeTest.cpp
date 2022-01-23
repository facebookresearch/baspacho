#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include "BlockMatrix.h"
#include "EliminationTree.h"
#include "TestingUtils.h"
#include "Utils.h"

using namespace std;
using namespace testing;

TEST(EliminationTree, Build) {
    for (int h = 0; h < 200; h++) {
        auto colsOrig = randomCols(70, 0.1, h + 37);
        auto ssOrig = columnsToCscStruct(colsOrig).transpose();

        vector<uint64_t> permutation = ssOrig.fillReducingPermutation();
        vector<uint64_t> invPerm = inversePermutation(permutation);

        SparseStructure ss = ssOrig.symmetricPermutation(invPerm, false);

        LOG(INFO) << "perm:\n" << printPattern(ss, false);

        vector<uint64_t> paramSize(ssOrig.ptrs.size() - 1, 1);
        EliminationTree et(paramSize, ss);

        et.buildTree();

        et.computeMerges();

        et.computeAggregateStruct();

        BlockMatrixSkel skel(et.spanStart, et.lumpToSpan, et.colStart,
                             et.rowParam);
        uint64_t totData = skel.sliceData[skel.sliceData.size() - 1];
        vector<double> data(totData, 1);
        Eigen::MatrixXd mat = skel.densify(data);
        LOG(INFO) << "densified:\n" << mat;

        // original must be contained via idMap
        vector<uint64_t> idMap = composePermutations(et.permInverse, invPerm);
        for (uint64_t i = 0; i < ssOrig.ptrs.size() - 1; i++) {
            uint64_t start = ssOrig.ptrs[i];
            uint64_t end = ssOrig.ptrs[i + 1];
            uint64_t newI = idMap[i];
            for (uint64_t q = start; q < end; q++) {
                uint64_t j = ssOrig.inds[q];
                uint64_t newJ = idMap[j];
                uint64_t r = std::max(newI, newJ);
                uint64_t c = std::min(newI, newJ);
                ASSERT_GT(mat(r, c), 0.5);
            }
        }

        // permuted and elim-filled must be contained
        SparseStructure checkSs =
            ss.symmetricPermutation(et.permInverse, false, true)
                .addFullEliminationFill();
        LOG(INFO) << "check:\n" << printPattern(checkSs, false);
        for (uint64_t i = 0; i < checkSs.ptrs.size() - 1; i++) {
            uint64_t start = checkSs.ptrs[i];
            uint64_t end = checkSs.ptrs[i + 1];
            for (uint64_t q = start; q < end; q++) {
                uint64_t j = checkSs.inds[q];
                ASSERT_GT(mat(i, j), 0.5);
            }
        }
    }
}