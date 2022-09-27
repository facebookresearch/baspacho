/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
  static constexpr double value2 = 1e-9;
};
template <>
struct Epsilon<float> {
  static constexpr float value = 1e-7;
  static constexpr float value2 = 1e-6;
};

// this value must match the value in EliminationTree.cpp - so that the no-cross
// barriers are placed without preventing the sparse elimination from happening
static constexpr int64_t minNumSparseElimNodes = 50;

template <typename T>
void testPartialFactor_Many(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 20; i++) {
    int numParams = 215;
    auto colBlocks = randomCols(numParams, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 150);
    SparseStructure sortedSs = columnsToCscStruct(colBlocks).transpose();

    // test no-cross barrier - make sure the elim set is still present
    int64_t nocross = (7 * i) % (210 - minNumSparseElimNodes) + minNumSparseElimNodes + 1;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 3, 47);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true, {nocross});
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);
    ASSERT_EQ(factorSkel.spanOffsetInLump[nocross], 0);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 2.0));

    Matrix<T> verifyMat = factorSkel.densify(data);
    Matrix<T> origMat = verifyMat;
    Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

    int barrierAt = factorSkel.spanStart[nocross];
    int afterBar = factorSkel.order() - barrierAt;

    Matrix<T> decBl = verifyMat.bottomLeftCorner(afterBar, barrierAt);
    Matrix<T> origBr = origMat.bottomRightCorner(afterBar, afterBar);
    Matrix<T> marginalBr = origBr - decBl * decBl.transpose();
    verifyMat.bottomRightCorner(afterBar, afterBar) = marginalBr;

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());
    solver.factorUpTo(data.data(), nocross);
    Matrix<T> computedMat = solver.skel().densify(data);

    ASSERT_NEAR(
        Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>()).norm() /
            Matrix<T>(verifyMat.template triangularView<Eigen::Lower>()).norm(),
        0, Epsilon<T>::value2);
  }
}

TEST(Partial, PartialFactor_Ref_double) {
  testPartialFactor_Many<double>([] { return simpleOps(); });
}

TEST(Partial, PartialFactor_Ref_float) {
  testPartialFactor_Many<float>([] { return simpleOps(); });
}

template <typename T>
void testSplitFactor_Many(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 20; i++) {
    int numParams = 215;
    auto colBlocks = randomCols(numParams, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 150);
    SparseStructure sortedSs = columnsToCscStruct(colBlocks).transpose();

    // test no-cross barrier - make sure the elim set is still present
    int64_t nocross = (7 * i) % (210 - minNumSparseElimNodes) + minNumSparseElimNodes + 1;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 3, 47);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true, {nocross});
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);
    ASSERT_EQ(factorSkel.spanOffsetInLump[nocross], 0);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 2.0));

    Matrix<T> verifyMat = factorSkel.densify(data);
    Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());
    solver.factorUpTo(data.data(), nocross);
    solver.factorFrom(data.data(), nocross);
    Matrix<T> computedMat = solver.skel().densify(data);

    ASSERT_NEAR(
        Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>()).norm() /
            Matrix<T>(verifyMat.template triangularView<Eigen::Lower>()).norm(),
        0, Epsilon<T>::value2);
  }
}

TEST(Partial, SplitFactor_Ref_double) {
  testSplitFactor_Many<double>([] { return simpleOps(); });
}

TEST(Partial, SplitFactor_Ref_float) {
  testSplitFactor_Many<float>([] { return simpleOps(); });
}

/*
    PA = C
    QA + RB = D
 therefore:
    A = P^-1*C
    B = R^-1*(D - QA)
*/
template <typename T>
void testPartialSolveL_Many(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 20; i++) {
    int numParams = 215;
    auto colBlocks = randomCols(numParams, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 150);
    SparseStructure sortedSs = columnsToCscStruct(colBlocks).transpose();

    // test no-cross barrier - make sure the elim set is still present
    int64_t nocross = (7 * i) % (210 - minNumSparseElimNodes) + minNumSparseElimNodes + 1;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 3, 47);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true, {nocross});
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);
    ASSERT_EQ(factorSkel.spanOffsetInLump[nocross], 0);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 2.0));

    Matrix<T> mat = factorSkel.densify(data);
    int order = factorSkel.order();
    int barrierAt = factorSkel.spanStart[nocross];
    int afterBar = order - barrierAt;

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    for (int j = 0; j < 5; j++) {
      int nRHS = 3;
      vector<T> vecData = randomData<T>(order * nRHS, -1.0, 1.0, 49 + j + i);
      Matrix<T> vec = Eigen::Map<Matrix<T>>(vecData.data(), order, nRHS);
      Matrix<T> vecRef = vec;
      vecRef.topRows(barrierAt) = mat.topLeftCorner(barrierAt, barrierAt)
                                      .template triangularView<Eigen::Lower>()
                                      .solve(vec.topRows(barrierAt));
      vecRef.bottomRows(afterBar) -=
          mat.bottomLeftCorner(afterBar, barrierAt) * vecRef.topRows(barrierAt);
      solver.solveLUpTo(data.data(), nocross, vec.data(), order, nRHS);

      ASSERT_NEAR((vec - vecRef).norm() / vecRef.norm(), 0, Epsilon<T>::value2);
    }
  }
}

TEST(Partial, PartialSolveL_Ref_double) {
  testPartialSolveL_Many<double>([] { return simpleOps(); });
}

TEST(Partial, PartialSolveL_Ref_float) {
  testPartialSolveL_Many<float>([] { return simpleOps(); });
}

template <typename T>
void testPartialSolveLt_Many(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 20; i++) {
    int numParams = 215;
    auto colBlocks = randomCols(numParams, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 150);
    SparseStructure sortedSs = columnsToCscStruct(colBlocks).transpose();

    // test no-cross barrier - make sure the elim set is still present
    int64_t nocross = (7 * i) % (210 - minNumSparseElimNodes) + minNumSparseElimNodes + 1;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 3, 47);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true, {nocross});
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);
    ASSERT_EQ(factorSkel.spanOffsetInLump[nocross], 0);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 2.0));

    Matrix<T> mat = factorSkel.densify(data);
    int order = factorSkel.order();
    int barrierAt = factorSkel.spanStart[nocross];
    int afterBar = order - barrierAt;

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    for (int j = 0; j < 5; j++) {
      int nRHS = 3;
      vector<T> vecData = randomData<T>(order * nRHS, -1.0, 1.0, 49 + j + i);
      Matrix<T> vec = Eigen::Map<Matrix<T>>(vecData.data(), order, nRHS);
      Matrix<T> vecRef = vec;
      vecRef.topRows(barrierAt) = mat.topLeftCorner(barrierAt, barrierAt)
                                      .template triangularView<Eigen::Lower>()
                                      .solve(vec.topRows(barrierAt));
      vecRef.bottomRows(afterBar) -=
          mat.bottomLeftCorner(afterBar, barrierAt) * vecRef.topRows(barrierAt);
      solver.solveLUpTo(data.data(), nocross, vec.data(), order, nRHS);

      ASSERT_NEAR((vec - vecRef).norm() / vecRef.norm(), 0, Epsilon<T>::value2);
    }
  }
}

TEST(Partial, PartialSolveLt_Ref_double) {
  testPartialSolveLt_Many<double>([] { return simpleOps(); });
}

TEST(Partial, PartialSolveLt_Ref_float) {
  testPartialSolveLt_Many<float>([] { return simpleOps(); });
}

template <typename T>
void testPartialAddMv_Many(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 20; i++) {
    int numParams = 215;
    auto colBlocks = randomCols(numParams, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 150);
    SparseStructure sortedSs = columnsToCscStruct(colBlocks).transpose();

    // test no-cross barrier - make sure the elim set is still present
    int64_t nocross = (7 * i) % (210 - minNumSparseElimNodes) + minNumSparseElimNodes + 1;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 3, 47);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true, {nocross});
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);
    ASSERT_EQ(factorSkel.spanOffsetInLump[nocross], 0);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 2.0));

    Matrix<T> mat = factorSkel.densify(data);
    int order = factorSkel.order();
    int barrierAt = factorSkel.spanStart[nocross];
    int afterBar = order - barrierAt;

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    for (int j = 0; j < 5; j++) {
      int nRHS = 3;
      vector<T> vecInData = randomData<T>(order * nRHS, -1.0, 1.0, 49 + j + i);
      vector<T> vecOutData = randomData<T>(order * nRHS, -1.0, 1.0, 49 + j + i);
      Matrix<T> vecIn = Eigen::Map<Matrix<T>>(vecInData.data(), order, nRHS);
      Matrix<T> vecOut = Eigen::Map<Matrix<T>>(vecOutData.data(), order, nRHS);
      Matrix<T> vecRef = vecOut;
      vecRef.bottomRows(afterBar) +=
          mat.bottomRightCorner(afterBar, afterBar).template triangularView<Eigen::Lower>() *
          vecIn.bottomRows(afterBar);
      vecRef.bottomRows(afterBar) += mat.bottomRightCorner(afterBar, afterBar)
                                         .template triangularView<Eigen::StrictlyLower>()
                                         .transpose() *
                                     vecIn.bottomRows(afterBar);
      solver.addMvFrom(data.data(), nocross, vecIn.data(), order, vecOut.data(), order, nRHS);

      ASSERT_NEAR((vecOut - vecRef).norm() / vecRef.norm(), 0, Epsilon<T>::value2);
    }
  }
}

TEST(Partial, PartialAddMv_Ref_double) {
  testPartialAddMv_Many<double>([] { return simpleOps(); });
}

TEST(Partial, PartialAddMv_Ref_float) {
  testPartialAddMv_Many<float>([] { return simpleOps(); });
}

#ifdef BASPACHO_USE_BLAS
TEST(Partial, PartialAddMv_Blas_double) {
  testPartialAddMv_Many<double>([] { return blasOps(); });
}

TEST(Partial, PartialAddMv_Blas_float) {
  testPartialAddMv_Many<float>([] { return blasOps(); });
}
#endif

template <typename T>
void testPseudoFactor_Many(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 20; i++) {
    int numParams = 215;
    auto colBlocks = randomCols(numParams, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 150);
    SparseStructure sortedSs = columnsToCscStruct(colBlocks).transpose();

    // test no-cross barrier - make sure the elim set is still present
    int64_t nocross = (7 * i) % (210 - minNumSparseElimNodes) + minNumSparseElimNodes + 1;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 3, 47);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true, {nocross});
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);
    ASSERT_EQ(factorSkel.spanOffsetInLump[nocross], 0);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 2.0));

    Matrix<T> verifyMat = factorSkel.densify(data);
    int64_t order = factorSkel.order();
    for (int64_t j = 0; j < factorSkel.numSpans(); j++) {
      int64_t start = factorSkel.spanStart[j];
      int64_t end = factorSkel.spanStart[j + 1];
      int64_t size = end - start;

      auto diagBlock = verifyMat.block(start, start, size, size);
      auto belowDiagBlock = verifyMat.block(end, start, order - end, size);

      { Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(diagBlock); }

      diagBlock.template triangularView<Eigen::Lower>()
          .transpose()
          .template solveInPlace<Eigen::OnTheRight>(belowDiagBlock);
    }

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());
    solver.pseudoFactorFrom(data.data(), 0);
    Matrix<T> computedMat = solver.skel().densify(data);

    ASSERT_NEAR(
        Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>()).norm() /
            Matrix<T>(verifyMat.template triangularView<Eigen::Lower>()).norm(),
        0, Epsilon<T>::value2);
  }
}

TEST(Partial, testPseudoFactor_Ref_double) {
  testPseudoFactor_Many<double>([] { return simpleOps(); });
}

TEST(Partial, testPseudoFactor_Ref_float) {
  testPseudoFactor_Many<float>([] { return simpleOps(); });
}

#ifdef BASPACHO_USE_BLAS
TEST(Partial, testPseudoFactor_Blas_double) {
  testPseudoFactor_Many<double>([] { return blasOps(); });
}

TEST(Partial, testPseudoFactor_Blas_float) {
  testPseudoFactor_Many<float>([] { return blasOps(); });
}
#endif

template <typename T>
void testPartialSolveLFrom_Many(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 20; i++) {
    int numParams = 215;
    auto colBlocks = randomCols(numParams, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 150);
    SparseStructure sortedSs = columnsToCscStruct(colBlocks).transpose();

    // test no-cross barrier - make sure the elim set is still present
    int64_t nocross = (7 * i) % (210 - minNumSparseElimNodes) + minNumSparseElimNodes + 1;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 3, 47);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true, {nocross});
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);
    ASSERT_EQ(factorSkel.spanOffsetInLump[nocross], 0);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 2.0));

    Matrix<T> mat = factorSkel.densify(data);
    int order = factorSkel.order();
    int barrierAt = factorSkel.spanStart[nocross];
    int afterBar = order - barrierAt;

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    for (int j = 0; j < 5; j++) {
      int nRHS = 3;
      vector<T> vecData = randomData<T>(order * nRHS, -1.0, 1.0, 49 + j + i);
      Matrix<T> vec = Eigen::Map<Matrix<T>>(vecData.data(), order, nRHS);
      Matrix<T> vecRef = vec;
      vecRef.bottomRows(afterBar) = mat.bottomRightCorner(afterBar, afterBar)
                                        .template triangularView<Eigen::Lower>()
                                        .solve(vec.bottomRows(afterBar));
      solver.solveLFrom(data.data(), nocross, vec.data(), order, nRHS);

      ASSERT_NEAR((vec - vecRef).norm() / vecRef.norm(), 0, Epsilon<T>::value2);
    }
  }
}

TEST(Partial, PartialSolveLFrom_Ref_double) {
  testPartialSolveLFrom_Many<double>([] { return simpleOps(); });
}

TEST(Partial, PartialSolveLFrom_Ref_float) {
  testPartialSolveLFrom_Many<float>([] { return simpleOps(); });
}

template <typename T>
void testPartialSolveLtFrom_Many(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 20; i++) {
    int numParams = 215;
    auto colBlocks = randomCols(numParams, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 150);
    SparseStructure sortedSs = columnsToCscStruct(colBlocks).transpose();

    // test no-cross barrier - make sure the elim set is still present
    int64_t nocross = (7 * i) % (210 - minNumSparseElimNodes) + minNumSparseElimNodes + 1;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 3, 47);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true, {nocross});
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);
    ASSERT_EQ(factorSkel.spanOffsetInLump[nocross], 0);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 2.0));

    Matrix<T> mat = factorSkel.densify(data);
    int order = factorSkel.order();
    int barrierAt = factorSkel.spanStart[nocross];
    int afterBar = order - barrierAt;

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    for (int j = 0; j < 5; j++) {
      int nRHS = 3;
      vector<T> vecData = randomData<T>(order * nRHS, -1.0, 1.0, 49 + j + i);
      Matrix<T> vec = Eigen::Map<Matrix<T>>(vecData.data(), order, nRHS);
      Matrix<T> vecRef = vec;
      vecRef.bottomRows(afterBar) = mat.bottomRightCorner(afterBar, afterBar)
                                        .template triangularView<Eigen::Lower>()
                                        .transpose()
                                        .solve(vec.bottomRows(afterBar));
      solver.solveLtFrom(data.data(), nocross, vec.data(), order, nRHS);

      ASSERT_NEAR((vec - vecRef).norm() / vecRef.norm(), 0, Epsilon<T>::value2);
    }
  }
}

TEST(Partial, PartialSolveLtFrom_Ref_double) {
  testPartialSolveLtFrom_Many<double>([] { return simpleOps(); });
}

TEST(Partial, PartialSolveLtFrom_Ref_float) {
  testPartialSolveLtFrom_Many<float>([] { return simpleOps(); });
}

template <typename T>
void testPartialFragmentedAddMv_Many(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 20; i++) {
    int numParams = 215;
    auto colBlocks = randomCols(numParams, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 150);
    SparseStructure sortedSs = columnsToCscStruct(colBlocks);  //.transpose();

    // test no-cross barrier - make sure the elim set is still present
    int64_t nocross = (7 * i) % (210 - minNumSparseElimNodes) + minNumSparseElimNodes + 1;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 3, 47);

    vector<int64_t> spanStart = paramSize;
    spanStart.push_back(0);
    cumSumVec(spanStart);
    vector<int64_t> lumpToSpan(sortedSs.ptrs.size());
    std::iota(lumpToSpan.begin(), lumpToSpan.end(), 0);
    CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, sortedSs.ptrs, sortedSs.inds);
    ASSERT_EQ(factorSkel.spanOffsetInLump[nocross], 0);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(5.0));

    Matrix<T> mat = factorSkel.densify(data);
    int order = factorSkel.order();
    int barrierAt = factorSkel.spanStart[nocross];
    int afterBar = order - barrierAt;

    Solver solver(move(factorSkel), {}, {}, genOps());

    for (int j = 0; j < 5; j++) {
      int nRHS = 1;
      vector<T> vecInData = randomData<T>(order * nRHS, -1.0, 1.0, 49 + j + i);
      vector<T> vecOutData = randomData<T>(order * nRHS, -1.0, 1.0, 49 + j + i);
      Matrix<T> vecIn = Eigen::Map<Matrix<T>>(vecInData.data(), order, nRHS);
      Matrix<T> vecOut = Eigen::Map<Matrix<T>>(vecOutData.data(), order, nRHS);
      Matrix<T> vecRef = vecOut;
      vecRef.bottomRows(afterBar) +=
          mat.bottomRightCorner(afterBar, afterBar).template triangularView<Eigen::Lower>() *
          vecIn.bottomRows(afterBar);
      vecRef.bottomRows(afterBar) += mat.bottomRightCorner(afterBar, afterBar)
                                         .template triangularView<Eigen::StrictlyLower>()
                                         .transpose() *
                                     vecIn.bottomRows(afterBar);
      solver.addMvFrom(data.data(), nocross, vecIn.data(), order, vecOut.data(), order, nRHS);

      ASSERT_NEAR((vecOut - vecRef).norm() / vecRef.norm(), 0, Epsilon<T>::value2);
    }
  }
}

TEST(Partial, PartialFragmentedAddMv_Ref_double) {
  testPartialFragmentedAddMv_Many<double>([] { return simpleOps(); });
}

TEST(Partial, PartialFragmentedAddMv_Ref_float) {
  testPartialFragmentedAddMv_Many<float>([] { return simpleOps(); });
}

#ifdef BASPACHO_USE_BLAS
TEST(Partial, PartialFragmentedAddMv_Blas_double) {
  testPartialFragmentedAddMv_Many<double>([] { return blasOps(); });
}

TEST(Partial, PartialFragmentedAddMv_Blas_float) {
  testPartialFragmentedAddMv_Many<float>([] { return blasOps(); });
}
#endif