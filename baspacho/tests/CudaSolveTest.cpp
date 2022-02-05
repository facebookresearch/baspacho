#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include "baspacho/CoalescedBlockMatrix.h"
#include "baspacho/CudaDefs.h"
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
    int64_t order = skel.order();

    vector<double> data(skel.dataSize());
    iota(data.begin(), data.end(), 13);
    skel.damp(data, 5, 50);

    vector<double> rhsData = randomData(order * nRHS, -1.0, 1.0, 37);
    vector<double> rhsVerif(order * nRHS);
    Eigen::MatrixXd verifyMat = skel.densify(data);
    Eigen::Map<Eigen::MatrixXd>(rhsVerif.data(), order, nRHS) =
        verifyMat.triangularView<Eigen::Lower>().solve(
            Eigen::Map<Eigen::MatrixXd>(rhsData.data(), order, nRHS));

    Solver solver(std::move(skel), {}, {}, std::move(ops));

    // call solve on gpu data
    {
        DevMirror<double> dataGpu(data), rhsDataGpu(rhsData);
        solver.solveL(dataGpu.ptr, rhsDataGpu.ptr, order, nRHS);
        dataGpu.get(data);
        rhsDataGpu.get(rhsData);
    }

    ASSERT_NEAR((Eigen::Map<Eigen::MatrixXd>(rhsVerif.data(), order, nRHS) -
                 Eigen::Map<Eigen::MatrixXd>(rhsData.data(), order, nRHS))
                    .norm(),
                0, 1e-5);
}

TEST(CudaSolve, SolveL) { testSolveL(cudaOps(), 5); }

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
    int64_t order = skel.order();

    vector<double> data(skel.dataSize());
    iota(data.begin(), data.end(), 13);
    skel.damp(data, 5, 50);

    vector<double> rhsData = randomData(order * nRHS, -1.0, 1.0, 37);
    vector<double> rhsVerif(order * nRHS);
    Eigen::MatrixXd verifyMat = skel.densify(data);
    Eigen::Map<Eigen::MatrixXd>(rhsVerif.data(), order, nRHS) =
        verifyMat.triangularView<Eigen::Lower>().adjoint().solve(
            Eigen::Map<Eigen::MatrixXd>(rhsData.data(), order, nRHS));

    Solver solver(std::move(skel), {}, {}, std::move(ops));

    // call solve on gpu data
    {
        DevMirror<double> dataGpu(data), rhsDataGpu(rhsData);
        solver.solveLt(dataGpu.ptr, rhsDataGpu.ptr, order, nRHS);
        dataGpu.get(data);
        rhsDataGpu.get(rhsData);
    }

    ASSERT_NEAR((Eigen::Map<Eigen::MatrixXd>(rhsVerif.data(), order, nRHS) -
                 Eigen::Map<Eigen::MatrixXd>(rhsData.data(), order, nRHS))
                    .norm(),
                0, 1e-5);
}

TEST(CudaSolve, SolveLt) { testSolveLt(cudaOps(), 5); }