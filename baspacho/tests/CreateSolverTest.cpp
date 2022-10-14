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
  static constexpr double value2 = 1e-9;
};
template <>
struct Epsilon<float> {
  static constexpr float value = 1e-7;
  static constexpr float value2 = 1e-6;
};

template <typename T>
void check(Solver& solver, int seed, const std::unordered_set<int64_t>& elimLastIds) {
  vector<T> data = randomData<T>(solver.dataSize(), -1.0, 1.0, 9 + seed);
  solver.skel().damp(data, T(0.0), T(solver.order() * 2.0));
  Matrix<T> verifyMat = solver.skel().densify(data);
  Matrix<T> origMat = verifyMat;
  Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

  int factorUpTo = solver.canFactorUpToSpan();
  int barrierAt = solver.skel().spanStart[factorUpTo];
  int afterBar = solver.skel().order() - barrierAt;
  Matrix<T> decBl = verifyMat.bottomLeftCorner(afterBar, barrierAt);
  Matrix<T> origBr = origMat.bottomRightCorner(afterBar, afterBar);
  Matrix<T> marginalBr = origBr - decBl * decBl.transpose();
  verifyMat.bottomRightCorner(afterBar, afterBar) = marginalBr;

  solver.factorUpTo(data.data(), factorUpTo);
  Matrix<T> computedMat = solver.skel().densify(data);

  ASSERT_NEAR(Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>()).norm() /
                  Matrix<T>(verifyMat.template triangularView<Eigen::Lower>()).norm(),
              0, Epsilon<T>::value2);

  if (!elimLastIds.empty()) {
    int64_t s = elimLastIds.size();
    ASSERT_EQ(solver.skel().spanOffsetInLump[solver.skel().numSpans() - s], 0);
    for (int64_t e : elimLastIds) {
      ASSERT_GE(solver.paramToSpan()[e], solver.skel().numSpans() - s);
    }
  }
}

template <typename T>
void testCreateSolver_Many(bool elimSet, bool lastIds) {
  for (int i = 0; i < 20; i++) {
    int numParams = 215;
    auto colBlocks = randomCols(numParams, 0.03, 57 + i);
    std::vector<int64_t> sparseElimRanges;
    if (elimSet) {
      colBlocks = makeIndependentElimSet(colBlocks, 0, 150);
      sparseElimRanges = {0, 90};
    } else {
      colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
    }

    std::unordered_set<int64_t> elimLastIds;
    if (lastIds) {
      elimLastIds = {105, 123, 165, 194, 209, 214};
      if (!elimSet) {
        elimLastIds.insert(0);
        elimLastIds.insert(30);
        elimLastIds.insert(49);
        elimLastIds.insert(87);
      }
    }

    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> paramSize = randomVec(ss.ptrs.size() - 1, 2, 3, 47);

    {
      auto solver = createSolver({.backend = BackendRef, .addFillPolicy = AddFillComplete},
                                 paramSize, ss, sparseElimRanges, elimLastIds);
      ASSERT_EQ(solver->canFactorUpToSpan(), numParams);
      check<T>(*solver, 4 * i + 0, elimLastIds);
    }

    if (!lastIds) {
      {
        auto solver = createSolver({.backend = BackendRef, .addFillPolicy = AddFillForAutoElims},
                                   paramSize, ss, sparseElimRanges);
        std::cout << "Selims: " << printVec(solver->sparseEliminationRanges()) << std::endl;
        if (elimSet) {  // test auto-discovering of elim sets
          ASSERT_GE(solver->canFactorUpToSpan(), 145);
        } else {
          ASSERT_GE(solver->canFactorUpToSpan(), 55);
        }
        check<T>(*solver, 4 * i + 1, {});
      }

      {
        auto solver = createSolver({.backend = BackendRef, .addFillPolicy = AddFillForGivenElims},
                                   paramSize, ss, sparseElimRanges);
        if (elimSet) {
          ASSERT_EQ(solver->canFactorUpToSpan(), 90);
        }
        check<T>(*solver, 4 * i + 2, {});
      }

      {
        auto solver = createSolver({.backend = BackendRef, .addFillPolicy = AddFillNone}, paramSize,
                                   ss, sparseElimRanges);
        ASSERT_EQ(solver->canFactorUpToSpan(), 0);
        check<T>(*solver, 4 * i + 3, {});
      }
    }
  }
}

TEST(CreateSolver, testCreateSolver_Plain_double) { testCreateSolver_Many<double>(false, false); }

TEST(CreateSolver, testCreateSolver_Plain_float) { testCreateSolver_Many<float>(false, false); }

TEST(CreateSolver, testCreateSolver_Elim_double) { testCreateSolver_Many<double>(true, false); }

TEST(CreateSolver, testCreateSolver_Elim_float) { testCreateSolver_Many<float>(true, false); }

TEST(CreateSolver, testCreateSolver_Last_double) { testCreateSolver_Many<double>(false, true); }

TEST(CreateSolver, testCreateSolver_Last_float) { testCreateSolver_Many<float>(false, true); }

TEST(CreateSolver, testCreateSolver_ElimLast_double) { testCreateSolver_Many<double>(true, true); }

TEST(CreateSolver, testCreateSolver_ElimLast_float) { testCreateSolver_Many<float>(true, true); }
