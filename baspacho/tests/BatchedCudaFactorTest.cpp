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
    static constexpr float value2 = 5e-5;
};

template <typename T>
void testBatchedCoalescedFactor(OpsPtr&& ops, int batchSize) {
    vector<set<int64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
    SparseStructure ss =
        columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
    vector<int64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
    vector<int64_t> lumpToSpan{0, 2, 4, 6};
    SparseStructure groupedSs =
        columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
    CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, groupedSs.ptrs,
                                        groupedSs.inds);

    vector<vector<T>> datas(batchSize);
    for (int i = 0; i < batchSize; i++) {
        datas[i] = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 37 + i);
        factorSkel.damp(datas[i], T(0), T(factorSkel.order() * 1.3));
    }
    vector<vector<T>> datasBackup = datas;

    Solver solver(std::move(factorSkel), {}, {}, std::move(ops));

    // call factor on gpu data
    {
        vector<DevMirror<T>> datasGpu(batchSize);
        vector<T*> datasPtr(batchSize);
        for (int i = 0; i < batchSize; i++) {
            datasGpu[i].load(datas[i]);
            datasPtr[i] = datasGpu[i].ptr;
        }
        solver.factor(&datasPtr);
        for (int i = 0; i < batchSize; i++) {
            datasGpu[i].get(datas[i]);
        }
    }

    for (int i = 0; i < batchSize; i++) {
        Matrix<T> verifyMat = solver.factorSkel.densify(datasBackup[i]);
        // cout << "Orign:\n" << verifyMat << endl;
        { Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat); }

        Matrix<T> computedMat = solver.factorSkel.densify(datas[i]);
        /*cout << "Verif:\n" << verifyMat << endl;
        cout << "Cmptd:\n" << computedMat << endl;*/

        ASSERT_NEAR(Matrix<T>((verifyMat - computedMat)
                                  .template triangularView<Eigen::Lower>())
                        .norm(),
                    0, Epsilon<T>::value);
    }
}

TEST(BatchedCudaFactor, CoalescedFactor_double) {
    testBatchedCoalescedFactor<double>(cudaOps(), 8);
}

#if 0
TEST(BatchedCudaFactor, CoalescedFactor_float) {
    testBatchedCoalescedFactor<float>(cudaOps());
}

template <typename T>
void testCoalescedFactor_Many(const std::function<OpsPtr()>& genOps) {
    for (int i = 0; i < 20; i++) {
        auto colBlocks = randomCols(115, 0.037, 57 + i);
        SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

        vector<int64_t> permutation = ss.fillReducingPermutation();
        vector<int64_t> invPerm = inversePermutation(permutation);
        SparseStructure sortedSs = ss.symmetricPermutation(invPerm, false);

        vector<int64_t> paramSize =
            randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47);
        EliminationTree et(paramSize, sortedSs);
        et.buildTree();
        et.computeMerges(/* compute sparse elim ranges = */ false);
        et.computeAggregateStruct();

        CoalescedBlockMatrixSkel factorSkel(
            et.computeSpanStart(), et.lumpToSpan, et.colStart, et.rowParam);

        vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
        factorSkel.damp(data, T(0), T(factorSkel.order() * 1.5));

        Matrix<T> verifyMat = factorSkel.densify(data);
        Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

        Solver solver(std::move(factorSkel), {}, {}, genOps());

        // call factor on gpu data
        {
            DevMirror<T> dataGpu(data);
            solver.factor(dataGpu.ptr);
            dataGpu.get(data);
        }

        Matrix<T> computedMat = solver.factorSkel.densify(data);

        ASSERT_NEAR(Matrix<T>((verifyMat - computedMat)
                                  .template triangularView<Eigen::Lower>())
                        .norm(),
                    0, Epsilon<T>::value2);
    }
}

TEST(BatchedCudaFactor, CoalescedFactor_Many_double) {
    testCoalescedFactor_Many<double>([] { return cudaOps(); });
}

TEST(BatchedCudaFactor, CoalescedFactor_Many_float) {
    testCoalescedFactor_Many<float>([] { return cudaOps(); });
}

template <typename T>
void testSparseElim_Many(const std::function<OpsPtr()>& genOps) {
    for (int i = 0; i < 20; i++) {
        auto colBlocks = randomCols(115, 0.03, 57 + i);
        colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
        SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

        vector<int64_t> permutation = ss.fillReducingPermutation();
        vector<int64_t> invPerm = inversePermutation(permutation);
        SparseStructure sortedSs = ss;

        vector<int64_t> paramSize =
            randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47);
        EliminationTree et(paramSize, sortedSs);
        et.buildTree();
        et.computeMerges(/* compute sparse elim ranges = */ true);
        et.computeAggregateStruct();

        CoalescedBlockMatrixSkel factorSkel(
            et.computeSpanStart(), et.lumpToSpan, et.colStart, et.rowParam);

        vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
        factorSkel.damp(data, T(0), T(factorSkel.order() * 1.5));

        Matrix<T> verifyMat = factorSkel.densify(data);
        Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

        ASSERT_GE(et.sparseElimRanges.size(), 2);
        int64_t largestIndep = et.sparseElimRanges[1];
        Solver solver(move(factorSkel), move(et.sparseElimRanges), {},
                      genOps());

        NumericCtxPtr<T> numCtx = solver.symCtx->createNumericCtx<T>(0);

        // call doElimination with data on device
        {
            DevMirror<T> dataGpu(data);
            numCtx->doElimination(*solver.elimCtxs[0], dataGpu.ptr, 0,
                                  largestIndep);
            dataGpu.get(data);
        }

        Matrix<T> computedMat = solver.factorSkel.densify(data);

        ASSERT_NEAR(Matrix<T>((verifyMat - computedMat)
                                  .template triangularView<Eigen::Lower>())
                        .leftCols(largestIndep)
                        .norm(),
                    0, Epsilon<T>::value);
    }
}

TEST(BatchedCudaFactor, SparseElim_Many_double) {
    testSparseElim_Many<double>([] { return cudaOps(); });
}

TEST(BatchedCudaFactor, SparseElim_Many_float) {
    testSparseElim_Many<float>([] { return cudaOps(); });
}

template <typename T>
void testSparseElimAndFactor_Many(const std::function<OpsPtr()>& genOps) {
    for (int i = 0; i < 20; i++) {
        auto colBlocks = randomCols(115, 0.03, 57 + i);
        colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
        SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

        vector<int64_t> permutation = ss.fillReducingPermutation();
        vector<int64_t> invPerm = inversePermutation(permutation);
        SparseStructure sortedSs = ss;

        vector<int64_t> paramSize =
            randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47);
        EliminationTree et(paramSize, sortedSs);
        et.buildTree();
        et.computeMerges(/* compute sparse elim ranges = */ true);
        et.computeAggregateStruct();

        CoalescedBlockMatrixSkel factorSkel(
            et.computeSpanStart(), et.lumpToSpan, et.colStart, et.rowParam);

        vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
        factorSkel.damp(data, T(0), T(factorSkel.order() * 1.5));

        Matrix<T> verifyMat = factorSkel.densify(data);
        Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

        ASSERT_GE(et.sparseElimRanges.size(), 2);
        int64_t largestIndep = et.sparseElimRanges[1];
        Solver solver(move(factorSkel), move(et.sparseElimRanges), {},
                      genOps());

        // call factor on gpu data
        {
            DevMirror<T> dataGpu(data);
            solver.factor(dataGpu.ptr);
            dataGpu.get(data);
        }

        Matrix<T> computedMat = solver.factorSkel.densify(data);
        ASSERT_NEAR(Matrix<T>((verifyMat - computedMat)
                                  .template triangularView<Eigen::Lower>())
                        .norm(),
                    0, Epsilon<T>::value2);
    }
}

TEST(BatchedCudaFactor, SparseElimAndFactor_Many_double) {
    testSparseElimAndFactor_Many<double>([] { return cudaOps(); });
}

TEST(BatchedCudaFactor, SparseElimAndFactor_Many_float) {
    testSparseElimAndFactor_Many<float>([] { return cudaOps(); });
}
#endif