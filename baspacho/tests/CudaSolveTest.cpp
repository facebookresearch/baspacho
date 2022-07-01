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
void testSolveLt_SparseElimAndFactor_Many(const std::function<OpsPtr()>& genOps,
                                          int nRHS) {
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss;

    vector<int64_t> paramSize =
        randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47 + i);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.computeMerges(/* compute sparse elim ranges = */ true);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan,
                                        et.colStart, et.rowParam);

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
void testSolveL_SparseElimAndFactor_Many(const std::function<OpsPtr()>& genOps,
                                         int nRHS) {
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss;

    vector<int64_t> paramSize =
        randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47 + i);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.computeMerges(/* compute sparse elim ranges = */ true);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan,
                                        et.colStart, et.rowParam);

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

// this value must match the value in EliminationTree.cpp - so that the no-cross
// barriers are placed without preventing the sparse elimination from happening
static constexpr int64_t minNumSparseElimNodes = 50;

template <typename T>
void testPartialAddMv_Many(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 20; i++) {
    int numParams = 215;
    auto colBlocks = randomCols(numParams, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 150);
    SparseStructure sortedSs = columnsToCscStruct(colBlocks).transpose();

    // test no-cross barrier - make sure the elim set is still present
    int64_t nocross =
        (7 * i) % (210 - minNumSparseElimNodes) + minNumSparseElimNodes + 1;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 3, 47);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.computeMerges(/* compute sparse elim ranges = */ true, {nocross});
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan,
                                        et.colStart, et.rowParam);
    ASSERT_EQ(factorSkel.spanOffsetInLump[nocross], 0);

    // test from 0
    nocross = 0;

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 2.0));

    Matrix<T> mat = factorSkel.densify(data);
    int order = factorSkel.order();
    int barrierAt = factorSkel.spanStart[nocross];
    int afterBar = order - barrierAt;

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    int64_t largestIndep = et.sparseElimRanges[1];
    auto sparseElimRangesCp = et.sparseElimRanges;
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());
    CoalescedBlockMatrixSkel factorSkel2(et.computeSpanStart(), et.lumpToSpan,
                                         et.colStart, et.rowParam);
    Solver solver2(move(factorSkel2), move(sparseElimRangesCp), {},
                   simpleOps());

    for (int j = 0; j < 5; j++) {
      int nRHS = 3;
      vector<T> vecInData = randomData<T>(order * nRHS, -1.0, 1.0, 49 + j + i);
      vector<T> vecOutData = randomData<T>(order * nRHS, -1.0, 1.0, 49 + j + i);
      Matrix<T> vecIn = Eigen::Map<Matrix<T>>(vecInData.data(), order, nRHS);
      Matrix<T> vecOut = Eigen::Map<Matrix<T>>(vecOutData.data(), order, nRHS);
      Matrix<T> vecOut2 = vecOut;
      Matrix<T> vecRef = vecOut;
      vecRef.bottomRows(afterBar) +=
          mat.bottomRightCorner(afterBar, afterBar)
              .template triangularView<Eigen::Lower>() *
          vecIn.bottomRows(afterBar);
      vecRef.bottomRows(afterBar) +=
          mat.bottomRightCorner(afterBar, afterBar)
              .template triangularView<Eigen::StrictlyLower>()
              .transpose() *
          vecIn.bottomRows(afterBar);

      // call solve on gpu data
      {
        DevMirror<T> dataGpu(data), vecInGpu(vecInData), vecOutGpu(vecOutData);
        solver.addMvFrom(dataGpu.ptr, nocross, vecInGpu.ptr, order,
                         vecOutGpu.ptr, order, nRHS);
        vecOutGpu.get(vecOutData);
        vecOut = Eigen::Map<Matrix<T>>(vecOutData.data(), order, nRHS);
      }

      // ref
      solver2.addMvFrom(data.data(), nocross, vecIn.data(), order,
                        vecOut2.data(), order, nRHS);

      cout << "dist2: " << (vecOut - vecOut2).norm() << " / " << vecOut2.norm()
           << endl;
      cout << "dist: " << (vecOut - vecRef).norm() << " / " << vecRef.norm()
           << endl;
      ASSERT_NEAR((vecOut - vecRef).norm() / vecRef.norm(), 0,
                  Epsilon<T>::value2);
    }
  }
}

TEST(CudaMv, PartialAddMv_double) {
  testPartialAddMv_Many<double>([] { return cudaOps(); });
}

TEST(CudaMv, PartialAddMv_float) {
  testPartialAddMv_Many<float>([] { return cudaOps(); });
}