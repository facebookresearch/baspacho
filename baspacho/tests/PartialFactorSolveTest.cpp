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

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 2.0));

    Matrix<T> verifyMat = factorSkel.densify(data);
    Matrix<T> origMat = verifyMat;
    Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

    int barrierAt = factorSkel.spanStart[nocross];
    int afterBar = factorSkel.order() - barrierAt;
    Matrix<T> decBr = verifyMat.bottomRightCorner(afterBar, afterBar)
                          .template triangularView<Eigen::Lower>();
    Matrix<T> marginalBr0 = decBr * decBr.transpose();

    Matrix<T> decBl = verifyMat.bottomLeftCorner(afterBar, barrierAt);
    Matrix<T> origBr = origMat.bottomRightCorner(afterBar, afterBar);
    Matrix<T> marginalBr = origBr - decBl * decBl.transpose();
    double v1 =
        Matrix<T>(marginalBr0.template triangularView<Eigen::Lower>()).norm();
    double v2 =
        Matrix<T>(marginalBr.template triangularView<Eigen::Lower>()).norm();
    double v3 =
        Matrix<T>(
            (marginalBr - marginalBr0).template triangularView<Eigen::Lower>())
            .norm();

    verifyMat.bottomRightCorner(afterBar, afterBar) = marginalBr;

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    int64_t largestIndep = et.sparseElimRanges[1];
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());
    solver.factorUpTo(data.data(), nocross);
    Matrix<T> computedMat = solver.factorSkel.densify(data);

    ASSERT_NEAR(
        Matrix<T>(
            (verifyMat - computedMat).template triangularView<Eigen::Lower>())
                .norm() /
            Matrix<T>(verifyMat.template triangularView<Eigen::Lower>()).norm(),
        0, Epsilon<T>::value2);
  }
}

TEST(Factor, PartialFactor_Ref_double) {
  testPartialFactor_Many<double>([] { return simpleOps(); });
}

TEST(Factor, PartialFactor_Ref_float) {
  testPartialFactor_Many<float>([] { return simpleOps(); });
}