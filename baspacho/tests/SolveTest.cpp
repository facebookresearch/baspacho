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

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
struct Epsilon;
template <>
struct Epsilon<double> {
    static constexpr double value = 1e-10;
    static constexpr double value2 = 1e-8;
};
template <>
struct Epsilon<float> {
    static constexpr float value = 1e-5;
    static constexpr float value2 = 4e-5;
};

template <typename T>
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

    vector<T> data(skel.dataSize());
    iota(data.begin(), data.end(), 13);
    skel.damp(data, T(5), T(50));

    vector<T> rhsData = randomData<T>(order * nRHS, -1.0, 1.0, 37);
    vector<T> rhsVerif(order * nRHS);
    Matrix<T> verifyMat = skel.densify(data);
    Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) =
        verifyMat.template triangularView<Eigen::Lower>().solve(
            Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS));

    Solver solver(std::move(skel), {}, {}, std::move(ops));
    solver.solveL(data.data(), rhsData.data(), order, nRHS);

    ASSERT_NEAR((Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) -
                 Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS))
                    .norm(),
                0, Epsilon<T>::value);
}

TEST(Solve, SolveL_Blas_double) { testSolveL<double>(blasOps(), 5); }

TEST(Solve, SolveL_Ref_double) { testSolveL<double>(simpleOps(), 5); }

TEST(Solve, SolveL_Blas_float) { testSolveL<float>(blasOps(), 5); }

TEST(Solve, SolveL_Ref_float) { testSolveL<float>(simpleOps(), 5); }

template <typename T>
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

    vector<T> data(skel.dataSize());
    iota(data.begin(), data.end(), 13);
    skel.damp(data, T(5), T(50));

    vector<T> rhsData = randomData<T>(order * nRHS, -1.0, 1.0, 37);
    vector<T> rhsVerif(order * nRHS);
    Matrix<T> verifyMat = skel.densify(data);
    Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) =
        verifyMat.template triangularView<Eigen::Lower>().adjoint().solve(
            Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS));

    Solver solver(std::move(skel), {}, {}, std::move(ops));
    solver.solveLt(data.data(), rhsData.data(), order, nRHS);

    ASSERT_NEAR((Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) -
                 Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS))
                    .norm(),
                0, Epsilon<T>::value);
}

TEST(Solve, SolveLt_Blas_double) { testSolveLt<double>(blasOps(), 5); }

TEST(Solve, SolveLt_Ref_double) { testSolveLt<double>(simpleOps(), 5); }

TEST(Solve, SolveLt_Blas_float) { testSolveLt<float>(blasOps(), 5); }

TEST(Solve, SolveLt_Ref_float) { testSolveLt<float>(simpleOps(), 5); }