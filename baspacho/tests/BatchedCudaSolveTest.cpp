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
void testBatchedSolveL(OpsPtr&& ops, int nRHS = 1, int batchSize = 8) {
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

    Solver solver(std::move(skel), {}, {}, std::move(ops));

    // generate a batch of data
    vector<vector<T>> datas(batchSize);
    vector<vector<T>> rhsDatas(batchSize);
    for (int q = 0; q < batchSize; q++) {
        datas[q] =
            randomData<T>(solver.factorSkel.dataSize(), -1.0, 1.0, 37 + q);
        solver.factorSkel.damp(datas[q], T(0),
                               T(solver.factorSkel.order() * 1.3));
        rhsDatas[q] = randomData<T>(order * nRHS, -1.0, 1.0, 137 + q);
    }
    vector<vector<T>> rhsDatasBackup = rhsDatas;

    // call factor on gpu data
    {
        vector<DevMirror<T>> datasGpu(batchSize);
        vector<DevMirror<T>> rhsDatasGpu(batchSize);
        vector<T*> datasPtr(batchSize);
        vector<T*> rhsDatasPtr(batchSize);
        for (int q = 0; q < batchSize; q++) {
            datasGpu[q].load(datas[q]);
            rhsDatasGpu[q].load(rhsDatas[q]);
            datasPtr[q] = datasGpu[q].ptr;
            rhsDatasPtr[q] = rhsDatasGpu[q].ptr;
        }
        solver.solveL(&datasPtr, &rhsDatasPtr, order, nRHS);
        for (int q = 0; q < batchSize; q++) {
            datasGpu[q].get(datas[q]);
            rhsDatasGpu[q].get(rhsDatas[q]);
        }
    }

    for (int q = 0; q < batchSize; q++) {
        vector<T> rhsVerif(order * nRHS);
        Matrix<T> verifyMat = solver.factorSkel.densify(datas[q]);
        Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) =
            verifyMat.template triangularView<Eigen::Lower>().solve(
                Eigen::Map<Matrix<T>>(rhsDatasBackup[q].data(), order, nRHS));

        ASSERT_NEAR((Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) -
                     Eigen::Map<Matrix<T>>(rhsDatas[q].data(), order, nRHS))
                        .norm(),
                    0, Epsilon<T>::value);
    }
}

TEST(BatchedCudaSolve, SolveL_double) {
    testBatchedSolveL<double>(cudaOps(), 5, 8);
}

TEST(BatchedCudaSolve, SolveL_float) {
    testBatchedSolveL<float>(cudaOps(), 5, 8);
}

template <typename T>
void testBatchedSolveLt(OpsPtr&& ops, int nRHS = 1, int batchSize = 8) {
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

    // generate a batch of data
    vector<vector<T>> datas(batchSize);
    vector<vector<T>> rhsDatas(batchSize);
    for (int q = 0; q < batchSize; q++) {
        datas[q] =
            randomData<T>(solver.factorSkel.dataSize(), -1.0, 1.0, 37 + q);
        solver.factorSkel.damp(datas[q], T(0),
                               T(solver.factorSkel.order() * 1.3));
        rhsDatas[q] = randomData<T>(order * nRHS, -1.0, 1.0, 137 + q);
    }
    vector<vector<T>> rhsDatasBackup = rhsDatas;

    // call factor on gpu data
    {
        vector<DevMirror<T>> datasGpu(batchSize);
        vector<DevMirror<T>> rhsDatasGpu(batchSize);
        vector<T*> datasPtr(batchSize);
        vector<T*> rhsDatasPtr(batchSize);
        for (int q = 0; q < batchSize; q++) {
            datasGpu[q].load(datas[q]);
            rhsDatasGpu[q].load(rhsDatas[q]);
            datasPtr[q] = datasGpu[q].ptr;
            rhsDatasPtr[q] = rhsDatasGpu[q].ptr;
        }
        solver.solveLt(&datasPtr, &rhsDatasPtr, order, nRHS);
        for (int q = 0; q < batchSize; q++) {
            datasGpu[q].get(datas[q]);
            rhsDatasGpu[q].get(rhsDatas[q]);
        }
    }

    for (int q = 0; q < batchSize; q++) {
        vector<T> rhsVerif(order * nRHS);
        Matrix<T> verifyMat = solver.factorSkel.densify(datas[q]);
        Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) =
            verifyMat.template triangularView<Eigen::Lower>().adjoint().solve(
                Eigen::Map<Matrix<T>>(rhsDatasBackup[q].data(), order, nRHS));

        ASSERT_NEAR((Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) -
                     Eigen::Map<Matrix<T>>(rhsDatas[q].data(), order, nRHS))
                        .norm(),
                    0, Epsilon<T>::value);
    }
}

TEST(BatchedCudaSolve, SolveLt_double) {
    testBatchedSolveLt<double>(cudaOps(), 5, 8);
}

TEST(BatchedCudaSolve, SolveLt_float) {
    testBatchedSolveLt<float>(cudaOps(), 5, 8);
}