#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include "baspacho/CoalescedBlockMatrix.h"
#include "baspacho/EliminationTree.h"
#include "baspacho/Solver.h"
#include "baspacho/SparseStructure.h"
#include "baspacho/Utils.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing;
using namespace std;
using namespace ::testing;

void testSolveL(OpsPtr&& ops, int nRHS = 1) {
    vector<set<int64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
    SparseStructure ss =
        columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
    vector<int64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
    vector<int64_t> lumpToSpan{0, 2, 4, 6};
    SparseStructure groupedSs =
        columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
    CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, groupedSs.ptrs,
                                  groupedSs.inds);

    vector<double> data(skel.dataSize());
    iota(data.begin(), data.end(), 13);
    skel.damp(data, 5, 50);

    vector<double> rhsData = randomData(15 * nRHS, -1.0, 1.0, 37);
    vector<double> rhsVerif(15 * nRHS);
    Eigen::MatrixXd verifyMat = skel.densify(data);
    Eigen::Map<Eigen::MatrixXd>(rhsVerif.data(), 15, nRHS) =
        verifyMat.triangularView<Eigen::Lower>().solve(
            Eigen::Map<Eigen::MatrixXd>(rhsData.data(), 15, nRHS));

    Solver solver(std::move(skel), {}, {}, std::move(ops));
    solver.solveL(data.data(), rhsData.data(), 15, nRHS);

    ASSERT_NEAR((Eigen::Map<Eigen::MatrixXd>(rhsVerif.data(), 15, nRHS) -
                 Eigen::Map<Eigen::MatrixXd>(rhsData.data(), 15, nRHS))
                    .norm(),
                0, 1e-5);
}

TEST(Solve, SolveL_Blas) { testSolveL(blasOps(), 5); }

TEST(Solve, SolveL_Ref) { testSolveL(simpleOps(), 5); }

void testSolveLt(OpsPtr&& ops, int nRHS = 1) {
    vector<set<int64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
    SparseStructure ss =
        columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
    vector<int64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
    vector<int64_t> lumpToSpan{0, 2, 4, 6};
    SparseStructure groupedSs =
        columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
    CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, groupedSs.ptrs,
                                  groupedSs.inds);

    vector<double> data(skel.dataSize());
    iota(data.begin(), data.end(), 13);
    skel.damp(data, 5, 50);

    vector<double> rhsData = randomData(15 * nRHS, -1.0, 1.0, 37);
    vector<double> rhsVerif(15 * nRHS);
    Eigen::MatrixXd verifyMat = skel.densify(data);
    Eigen::Map<Eigen::MatrixXd>(rhsVerif.data(), 15, nRHS) =
        verifyMat.triangularView<Eigen::Lower>().adjoint().solve(
            Eigen::Map<Eigen::MatrixXd>(rhsData.data(), 15, nRHS));

    Solver solver(std::move(skel), {}, {}, std::move(ops));
    solver.solveLt(data.data(), rhsData.data(), 15, nRHS);

    ASSERT_NEAR((Eigen::Map<Eigen::MatrixXd>(rhsVerif.data(), 15, nRHS) -
                 Eigen::Map<Eigen::MatrixXd>(rhsData.data(), 15, nRHS))
                    .norm(),
                0, 1e-5);
}

TEST(Solve, SolveLt_Blas) { testSolveLt(blasOps(), 5); }

TEST(Solve, SolveLt_Ref) { testSolveLt(simpleOps(), 5); }