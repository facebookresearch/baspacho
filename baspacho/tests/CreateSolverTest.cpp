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
void check(Solver& solver, int seed) {
  cout << "elims: " << printVec(solver.elimLumpRanges) << endl;
  cout << "maxf: " << solver.maxFactorParam() << endl;
  cout << "nnz: " << solver.dataSize() << endl;

  vector<T> data = randomData<T>(solver.dataSize(), -1.0, 1.0, 9 + seed);
  solver.skel().damp(data, T(0.0), T(solver.order() * 2.0));
  Matrix<T> verifyMat = solver.skel().densify(data);
  Matrix<T> origMat = verifyMat;
  Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

  int factorUpTo = solver.maxFactorParam();
  int barrierAt = solver.skel().spanStart[factorUpTo];
  int afterBar = solver.skel().order() - barrierAt;
  Matrix<T> decBl = verifyMat.bottomLeftCorner(afterBar, barrierAt);
  Matrix<T> origBr = origMat.bottomRightCorner(afterBar, afterBar);
  Matrix<T> marginalBr = origBr - decBl * decBl.transpose();
  verifyMat.bottomRightCorner(afterBar, afterBar) = marginalBr;

  solver.factorUpTo(data.data(), factorUpTo);
  Matrix<T> computedMat = solver.skel().densify(data);

  ASSERT_NEAR(
      Matrix<T>(
          (verifyMat - computedMat).template triangularView<Eigen::Lower>())
              .norm() /
          Matrix<T>(verifyMat.template triangularView<Eigen::Lower>()).norm(),
      0, Epsilon<T>::value2);
}

template <typename T>
void testCreateSolver_Many() {
  for (int i = 0; i < 5; i++) {
    int numParams = 215;
    auto colBlocks = randomCols(numParams, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 150);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    // test no-cross barrier - make sure the elim set is still present
    int64_t nocross =
        (7 * i) % (210 - minNumSparseElimNodes) + minNumSparseElimNodes + 1;

    vector<int64_t> paramSize = randomVec(ss.ptrs.size() - 1, 2, 3, 47);

    {
      auto solver = createSolver(
          {.backend = BackendRef, .addFillPolicy = AddFillComplete}, paramSize,
          ss, {0, 100});
      ASSERT_EQ(solver->maxFactorParam(), numParams);
      check<T>(*solver, 4 * i + 0);
    }

    {
      auto solver = createSolver(
          {.backend = BackendRef, .addFillPolicy = AddFillForAutoElims},
          paramSize, ss, {0, 100});
      ASSERT_GE(solver->maxFactorParam(), 150);
      check<T>(*solver, 4 * i + 1);
    }

    {
      auto solver = createSolver(
          {.backend = BackendRef, .addFillPolicy = AddFillForGivenElims},
          paramSize, ss, {0, 100});
      ASSERT_EQ(solver->maxFactorParam(), 100);
      check<T>(*solver, 4 * i + 2);
    }

    {
      auto solver =
          createSolver({.backend = BackendRef, .addFillPolicy = AddFillNone},
                       paramSize, ss, {0, 100});
      ASSERT_EQ(solver->maxFactorParam(), 0);
      check<T>(*solver, 4 * i + 3);
    }
  }
}

TEST(CreateSolver, testCreateSolver_Many_double) {
  testCreateSolver_Many<double>();
}

TEST(CreateSolver, testCreateSolver_Many_float) {
  testCreateSolver_Many<float>();
}