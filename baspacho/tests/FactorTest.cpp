#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/EliminationTree.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/baspacho/Utils.h"
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
void testCoalescedFactor(OpsPtr&& ops) {
  vector<set<int64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
  SparseStructure ss =
      columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
  vector<int64_t> lumpToSpan{0, 2, 4, 6};
  SparseStructure groupedSs =
      columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, groupedSs.ptrs,
                                      groupedSs.inds);

  vector<T> data(factorSkel.dataSize());
  iota(data.begin(), data.end(), 13);
  factorSkel.damp(data, T(5), T(50));

  Matrix<T> verifyMat = factorSkel.densify(data);
  Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

  Solver solver(std::move(factorSkel), {}, {}, std::move(ops));
  solver.factor(data.data());
  Matrix<T> computedMat = solver.skel().densify(data);

  ASSERT_NEAR(
      Matrix<T>(
          (verifyMat - computedMat).template triangularView<Eigen::Lower>())
          .norm(),
      0, Epsilon<T>::value);
}

TEST(Factor, CoalescedFactor_Blas_double) {
  testCoalescedFactor<double>(blasOps());
}

TEST(Factor, CoalescedFactor_Ref_double) {
  testCoalescedFactor<double>(simpleOps());
}

TEST(Factor, CoalescedFactor_Blas_float) {
  testCoalescedFactor<float>(blasOps());
}

TEST(Factor, CoalescedFactor_Ref_float) {
  testCoalescedFactor<float>(simpleOps());
}

template <typename T>
void testCoalescedFactor_Many(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.037, 57 + i);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss.symmetricPermutation(invPerm, false);

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.computeMerges(/* compute sparse elim ranges = */ false);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan,
                                        et.colStart, et.rowParam);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 1.5));

    Matrix<T> verifyMat = factorSkel.densify(data);
    Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

    Solver solver(std::move(factorSkel), {}, {}, genOps());
    solver.factor(data.data());
    Matrix<T> computedMat = solver.skel().densify(data);

    ASSERT_NEAR(
        Matrix<T>(
            (verifyMat - computedMat).template triangularView<Eigen::Lower>())
            .norm(),
        0, Epsilon<T>::value2);
  }
}

TEST(Factor, CoalescedFactor_Many_Blas_double) {
  testCoalescedFactor_Many<double>([] { return blasOps(); });
}

TEST(Factor, CoalescedFactor_Many_Ref_double) {
  testCoalescedFactor_Many<double>([] { return simpleOps(); });
}

TEST(Factor, CoalescedFactor_Many_Blas_float) {
  testCoalescedFactor_Many<float>([] { return blasOps(); });
}

TEST(Factor, CoalescedFactor_Many_Ref_float) {
  testCoalescedFactor_Many<float>([] { return simpleOps(); });
}

// this value must match the value in EliminationTree.cpp - so that the no-cross
// barriers are placed without preventing the sparse elimination from happening
static constexpr int64_t minNumSparseElimNodes = 50;

template <typename T>
void testSparseElim_Many(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
    SparseStructure sortedSs = columnsToCscStruct(colBlocks).transpose();

    // test no-cross barrier - make sure the elim set is still present
    int64_t nocross =
        (7 * i) % (110 - minNumSparseElimNodes) + minNumSparseElimNodes + 1;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.computeMerges(/* compute sparse elim ranges = */ true, {nocross});
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan,
                                        et.colStart, et.rowParam);
    ASSERT_EQ(factorSkel.spanOffsetInLump[nocross], 0);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 1.5));

    Matrix<T> verifyMat = factorSkel.densify(data);
    Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    int64_t largestIndep = et.sparseElimRanges[1];
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());
    NumericCtxPtr<T> numCtx = solver.symCtx->createNumericCtx<T>(0, nullptr);
    numCtx->doElimination(*solver.elimCtxs[0], data.data(), 0, largestIndep);
    Matrix<T> computedMat = solver.skel().densify(data);

    ASSERT_NEAR(
        Matrix<T>(
            (verifyMat - computedMat).template triangularView<Eigen::Lower>())
            .leftCols(largestIndep)
            .norm(),
        0, Epsilon<T>::value);
  }
}

TEST(Factor, SparseElim_Many_Blas_double) {
  testSparseElim_Many<double>([] { return blasOps(); });
}

TEST(Factor, SparseElim_Many_Ref_double) {
  testSparseElim_Many<double>([] { return simpleOps(); });
}

TEST(Factor, SparseElim_Many_Blas_float) {
  testSparseElim_Many<float>([] { return blasOps(); });
}

TEST(Factor, SparseElim_Many_Ref_float) {
  testSparseElim_Many<float>([] { return simpleOps(); });
}

template <typename T>
void testSparseElimAndFactor_Many(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
    SparseStructure sortedSs = columnsToCscStruct(colBlocks).transpose();

    // test no-cross barrier - make sure the elim set is still present
    int64_t nocross =
        (7 * i) % (110 - minNumSparseElimNodes) + minNumSparseElimNodes + 1;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.computeMerges(/* compute sparse elim ranges = */ true, {nocross});
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan,
                                        et.colStart, et.rowParam);
    ASSERT_EQ(factorSkel.spanOffsetInLump[nocross], 0);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 1.5));

    Matrix<T> verifyMat = factorSkel.densify(data);
    Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    int64_t largestIndep = et.sparseElimRanges[1];
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());
    solver.factor(data.data());
    Matrix<T> computedMat = solver.skel().densify(data);

    ASSERT_NEAR(
        Matrix<T>(
            (verifyMat - computedMat).template triangularView<Eigen::Lower>())
            .norm(),
        0, Epsilon<T>::value2);
  }
}

TEST(Factor, SparseElimAndFactor_Many_Blas_double) {
  testSparseElimAndFactor_Many<double>([] { return blasOps(); });
}

TEST(Factor, SparseElimAndFactor_Many_Ref_double) {
  testSparseElimAndFactor_Many<double>([] { return simpleOps(); });
}

TEST(Factor, SparseElimAndFactor_Many_Blas_float) {
  testSparseElimAndFactor_Many<float>([] { return blasOps(); });
}

TEST(Factor, SparseElimAndFactor_Many_Ref_float) {
  testSparseElimAndFactor_Many<float>([] { return simpleOps(); });
}