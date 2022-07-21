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
  static constexpr float value2 = 5e-5;
};

template <typename T>
void testBatchedCoalescedFactor(OpsPtr&& ops, int batchSize) {
  vector<set<int64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
  vector<int64_t> lumpToSpan{0, 2, 4, 6};
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);

  Solver solver(std::move(factorSkel), {}, {}, std::move(ops));

  // generate a batch of data
  vector<vector<T>> datas(batchSize);
  for (int q = 0; q < batchSize; q++) {
    datas[q] = randomData<T>(solver.skel().dataSize(), -1.0, 1.0, 37 + q);
    solver.skel().damp(datas[q], T(0), T(solver.skel().order() * 1.3));
  }
  vector<vector<T>> datasBackup = datas;

  // call factor on gpu data
  {
    vector<DevMirror<T>> datasGpu(batchSize);
    vector<T*> datasPtr(batchSize);
    for (int q = 0; q < batchSize; q++) {
      datasGpu[q].load(datas[q]);
      datasPtr[q] = datasGpu[q].ptr;
    }
    solver.factor(&datasPtr);
    for (int q = 0; q < batchSize; q++) {
      datasGpu[q].get(datas[q]);
    }
  }

  for (int q = 0; q < batchSize; q++) {
    Matrix<T> verifyMat = solver.skel().densify(datasBackup[q]);
    { Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat); }

    Matrix<T> computedMat = solver.skel().densify(datas[q]);

    ASSERT_NEAR(Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>()).norm(),
                0, Epsilon<T>::value);
  }
}

TEST(BatchedCudaFactor, CoalescedFactor_double) {
  testBatchedCoalescedFactor<double>(cudaOps(), 8);
}

TEST(BatchedCudaFactor, CoalescedFactor_float) { testBatchedCoalescedFactor<float>(cudaOps(), 8); }

template <typename T>
void testBatchedCoalescedFactor_Many(const std::function<OpsPtr()>& genOps) {
  vector<int64_t> batchSizes = randomVec(20, 3, 31, 37);
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.037, 57 + i);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss.symmetricPermutation(invPerm, false);

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47 + i);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ false);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);
    Solver solver(std::move(factorSkel), {}, {}, genOps());

    // generate a batch of data
    int batchSize = batchSizes[i];
    vector<vector<T>> datas(batchSize);
    for (int q = 0; q < batchSize; q++) {
      datas[q] = randomData<T>(solver.skel().dataSize(), -1.0, 1.0, 37 + q);
      solver.skel().damp(datas[q], T(0), T(solver.skel().order() * 1.3));
    }
    vector<vector<T>> datasBackup = datas;

    // call factor on gpu data
    {
      vector<DevMirror<T>> datasGpu(batchSize);
      vector<T*> datasPtr(batchSize);
      for (int q = 0; q < batchSize; q++) {
        datasGpu[q].load(datas[q]);
        datasPtr[q] = datasGpu[q].ptr;
      }
      solver.factor(&datasPtr);

      for (int q = 0; q < batchSize; q++) {
        datasGpu[q].get(datas[q]);
      }
    }

    for (int q = 0; q < batchSize; q++) {
      Matrix<T> verifyMat = solver.skel().densify(datasBackup[q]);
      { Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat); }

      Matrix<T> computedMat = solver.skel().densify(datas[q]);

      ASSERT_NEAR(
          Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>()).norm(), 0,
          Epsilon<T>::value2);
    }
  }
}

TEST(BatchedCudaFactor, CoalescedFactor_Many_double) {
  testBatchedCoalescedFactor_Many<double>([] { return cudaOps(); });
}

TEST(BatchedCudaFactor, CoalescedFactor_Many_float) {
  testBatchedCoalescedFactor_Many<float>([] { return cudaOps(); });
}

template <typename T>
void testBatchedSparseElim_Many(const std::function<OpsPtr()>& genOps) {
  vector<int64_t> batchSizes = randomVec(20, 3, 31, 137);
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    int64_t largestIndep = et.sparseElimRanges[1];
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    // generate a batch of data
    int batchSize = batchSizes[i];
    vector<vector<T>> datas(batchSize);
    for (int q = 0; q < batchSize; q++) {
      datas[q] = randomData<T>(solver.skel().dataSize(), -1.0, 1.0, 37 + q);
      solver.skel().damp(datas[q], T(0), T(solver.skel().order() * 1.3));
    }
    vector<vector<T>> datasBackup = datas;

    // call factor on gpu data
    {
      vector<DevMirror<T>> datasGpu(batchSize);
      vector<T*> datasPtr(batchSize);
      for (int q = 0; q < batchSize; q++) {
        datasGpu[q].load(datas[q]);
        datasPtr[q] = datasGpu[q].ptr;
      }
      NumericCtxPtr<vector<T*>> numCtx = solver.symCtx->createNumericCtx<vector<T*>>(0, &datasPtr);
      numCtx->doElimination(solver.internalGetElimCtx(0), &datasPtr, 0, largestIndep);

      for (int q = 0; q < batchSize; q++) {
        datasGpu[q].get(datas[q]);
      }
    }

    for (int q = 0; q < batchSize; q++) {
      Matrix<T> verifyMat = solver.skel().densify(datasBackup[q]);
      { Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat); }

      Matrix<T> computedMat = solver.skel().densify(datas[q]);

      ASSERT_NEAR(Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>())
                      .leftCols(largestIndep)
                      .norm(),
                  0, Epsilon<T>::value);
    }
  }
}

TEST(BatchedCudaFactor, SparseElim_Many_double) {
  testBatchedSparseElim_Many<double>([] { return cudaOps(); });
}

TEST(BatchedCudaFactor, SparseElim_Many_float) {
  testBatchedSparseElim_Many<float>([] { return cudaOps(); });
}

template <typename T>
void testBatchedSparseElimAndFactor_Many(const std::function<OpsPtr()>& genOps) {
  vector<int64_t> batchSizes = randomVec(20, 3, 31, 1037);
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    int64_t largestIndep = et.sparseElimRanges[1];
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    // generate a batch of data
    int batchSize = batchSizes[i];
    vector<vector<T>> datas(batchSize);
    for (int q = 0; q < batchSize; q++) {
      datas[q] = randomData<T>(solver.skel().dataSize(), -1.0, 1.0, 37 + q);
      solver.skel().damp(datas[q], T(0), T(solver.skel().order() * 1.3));
    }
    vector<vector<T>> datasBackup = datas;

    // call factor on gpu data
    {
      vector<DevMirror<T>> datasGpu(batchSize);
      vector<T*> datasPtr(batchSize);
      for (int q = 0; q < batchSize; q++) {
        datasGpu[q].load(datas[q]);
        datasPtr[q] = datasGpu[q].ptr;
      }
      solver.factor(&datasPtr);

      for (int q = 0; q < batchSize; q++) {
        datasGpu[q].get(datas[q]);
      }
    }

    for (int q = 0; q < batchSize; q++) {
      Matrix<T> verifyMat = solver.skel().densify(datasBackup[q]);
      { Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat); }

      Matrix<T> computedMat = solver.skel().densify(datas[q]);

      ASSERT_NEAR(
          Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>()).norm(), 0,
          Epsilon<T>::value2);
    }
  }
}

TEST(BatchedCudaFactor, SparseElimAndFactor_Many_double) {
  testBatchedSparseElimAndFactor_Many<double>([] { return cudaOps(); });
}

TEST(BatchedCudaFactor, SparseElimAndFactor_Many_float) {
  testBatchedSparseElimAndFactor_Many<float>([] { return cudaOps(); });
}
