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
#include "baspacho/baspacho/CudaDefs.h"
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
void testSolveL(OpsPtr&& ops, int nRHS = 1) {
  vector<set<int64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
  vector<int64_t> lumpToSpan{0, 2, 4, 6};
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
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

  // call solve on gpu data
  {
    DevMirror<T> dataGpu(data), rhsDataGpu(rhsData);
    solver.solveL(dataGpu.ptr, rhsDataGpu.ptr, order, nRHS);
    rhsDataGpu.get(rhsData);
  }

  ASSERT_NEAR((Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) -
               Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS))
                  .norm(),
              0, Epsilon<T>::value);
}

TEST(CudaSolve, SolveL_double) { testSolveL<double>(cudaOps(), 5); }

TEST(CudaSolve, SolveL_float) { testSolveL<float>(cudaOps(), 5); }

template <typename T>
void testSolveLt(OpsPtr&& ops, int nRHS = 1) {
  vector<set<int64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
  vector<int64_t> lumpToSpan{0, 2, 4, 6};
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
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

  // call solve on gpu data
  {
    DevMirror<T> dataGpu(data), rhsDataGpu(rhsData);
    solver.solveLt(dataGpu.ptr, rhsDataGpu.ptr, order, nRHS);
    rhsDataGpu.get(rhsData);
  }

  ASSERT_NEAR((Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) -
               Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS))
                  .norm(),
              0, Epsilon<T>::value);
}

TEST(CudaSolve, SolveLt_double) { testSolveLt<double>(cudaOps(), 5); }

TEST(CudaSolve, SolveLt_float) { testSolveLt<float>(cudaOps(), 5); }

template <typename T>
void testSolveLt_SparseElimAndFactor_Many(const std::function<OpsPtr()>& genOps, int nRHS) {
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47 + i);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 1.5));

    int64_t order = factorSkel.order();
    vector<T> rhsData = randomData<T>(order * nRHS, -1.0, 1.0, 37 + i);
    vector<T> rhsVerif(order * nRHS);
    Matrix<T> verifyMat = factorSkel.densify(data);
    Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) =
        verifyMat.template triangularView<Eigen::Lower>().adjoint().solve(
            Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS));

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    int64_t largestIndep = et.sparseElimRanges[1];
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    // call solve on gpu data
    {
      DevMirror<T> dataGpu(data), rhsDataGpu(rhsData);
      solver.solveLt(dataGpu.ptr, rhsDataGpu.ptr, order, nRHS);
      rhsDataGpu.get(rhsData);
    }

    ASSERT_NEAR((Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) -
                 Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS))
                    .norm(),
                0, Epsilon<T>::value);
  }
}

TEST(CudaSolve, SolveLt_SparseElimAndFactor_Many_Blas_double) {
  testSolveLt_SparseElimAndFactor_Many<double>([] { return cudaOps(); }, 5);
}

TEST(CudaSolve, SolveLt_SparseElimAndFactor_Many_Blas_float) {
  testSolveLt_SparseElimAndFactor_Many<float>([] { return cudaOps(); }, 5);
}

template <typename T>
void testSolveL_SparseElimAndFactor_Many(const std::function<OpsPtr()>& genOps, int nRHS) {
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47 + i);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 1.5));

    int64_t order = factorSkel.order();
    vector<T> rhsData = randomData<T>(order * nRHS, -1.0, 1.0, 37 + i);
    vector<T> rhsVerif(order * nRHS);
    Matrix<T> verifyMat = factorSkel.densify(data);
    Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) =
        verifyMat.template triangularView<Eigen::Lower>().solve(
            Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS));

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    int64_t largestIndep = et.sparseElimRanges[1];
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    // call solve on gpu data
    {
      DevMirror<T> dataGpu(data), rhsDataGpu(rhsData);
      solver.solveL(dataGpu.ptr, rhsDataGpu.ptr, order, nRHS);
      rhsDataGpu.get(rhsData);
    }

    ASSERT_NEAR((Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) -
                 Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS))
                    .norm(),
                0, Epsilon<T>::value);
  }
}

TEST(CudaSolve, SolveL_SparseElimAndFactor_Many_double) {
  testSolveL_SparseElimAndFactor_Many<double>([] { return cudaOps(); }, 5);
}

TEST(CudaSolve, SolveL_SparseElimAndFactor_Many_float) {
  testSolveL_SparseElimAndFactor_Many<float>([] { return cudaOps(); }, 5);
}
