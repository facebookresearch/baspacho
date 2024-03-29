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
using namespace ::BaSpaCho::testing_utils;
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
    int64_t nocross = (7 * i) % (210 - minNumSparseElimNodes) + minNumSparseElimNodes + 1;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 3, 47);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true, {nocross});
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);
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
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    for (int j = 0; j < 5; j++) {
      int nRHS = 3;
      vector<T> vecInData = randomData<T>(order * nRHS, -1.0, 1.0, 49 + j + i);
      vector<T> vecOutData = randomData<T>(order * nRHS, -1.0, 1.0, 49 + j + i);
      Matrix<T> vecIn = Eigen::Map<Matrix<T>>(vecInData.data(), order, nRHS);
      Matrix<T> vecOut = Eigen::Map<Matrix<T>>(vecOutData.data(), order, nRHS);
      Matrix<T> vecOut2 = vecOut;
      Matrix<T> vecRef = vecOut;
      vecRef.bottomRows(afterBar) +=
          mat.bottomRightCorner(afterBar, afterBar).template triangularView<Eigen::Lower>() *
          vecIn.bottomRows(afterBar);
      vecRef.bottomRows(afterBar) += mat.bottomRightCorner(afterBar, afterBar)
                                         .template triangularView<Eigen::StrictlyLower>()
                                         .transpose() *
                                     vecIn.bottomRows(afterBar);

      // call solve on gpu data
      {
        DevMirror<T> dataGpu(data), vecInGpu(vecInData), vecOutGpu(vecOutData);
        solver.addMvFrom(dataGpu.ptr, nocross, vecInGpu.ptr, order, vecOutGpu.ptr, order, nRHS);
        vecOutGpu.get(vecOutData);
        vecOut = Eigen::Map<Matrix<T>>(vecOutData.data(), order, nRHS);
      }

      ASSERT_NEAR((vecOut - vecRef).norm() / vecRef.norm(), 0, Epsilon<T>::value2);
    }
  }
}

TEST(CudaPartial, PartialAddMv_double) {
  testPartialAddMv_Many<double>([] { return cudaOps(); });
}

TEST(CudaPartial, PartialAddMv_float) {
  testPartialAddMv_Many<float>([] { return cudaOps(); });
}

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
    for (int64_t i = 0; i < factorSkel.numSpans(); i++) {
      int64_t start = factorSkel.spanStart[i];
      int64_t end = factorSkel.spanStart[i + 1];
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
    {
      DevMirror<T> dataGpu(data);
      solver.pseudoFactorFrom(dataGpu.ptr, 0);
      dataGpu.get(data);
    }
    Matrix<T> computedMat = solver.skel().densify(data);

    ASSERT_NEAR(Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>())
                        .leftCols(151)
                        .norm() /
                    Matrix<T>(verifyMat.template triangularView<Eigen::Lower>()).norm(),
                0, Epsilon<T>::value2);
  }
}

TEST(CudaPartial, testPseudoFactor_double) {
  testPseudoFactor_Many<double>([] { return cudaOps(); });
}

TEST(CudaPartial, testPseudoFactor_float) {
  testPseudoFactor_Many<float>([] { return cudaOps(); });
}
