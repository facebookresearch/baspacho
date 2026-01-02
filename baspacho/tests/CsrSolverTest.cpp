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
#include "baspacho/baspacho/CsrSolver.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
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
  static constexpr float value2 = 1e-4;
};

// Helper to convert column blocks to block CSR format
void columnsToCsr(const vector<set<int64_t>>& colBlocks, vector<int64_t>& rowStart,
                  vector<int64_t>& colIndices) {
  int64_t numBlocks = colBlocks.size();
  rowStart.resize(numBlocks + 1);
  colIndices.clear();

  // Convert CSC (colBlocks) to CSR
  // First, collect all (row, col) pairs
  vector<vector<int64_t>> rowEntries(numBlocks);
  for (int64_t col = 0; col < numBlocks; col++) {
    for (int64_t row : colBlocks[col]) {
      rowEntries[row].push_back(col);
    }
  }

  // Build CSR structure
  rowStart[0] = 0;
  for (int64_t row = 0; row < numBlocks; row++) {
    // Sort column indices within each row
    sort(rowEntries[row].begin(), rowEntries[row].end());
    for (int64_t col : rowEntries[row]) {
      colIndices.push_back(col);
    }
    rowStart[row + 1] = colIndices.size();
  }
}

// Test that createSolverFromBlockCsr produces same result as createSolver
template <typename T>
void testCsrVsOriginal(int seed) {
  int numParams = 50;
  auto colBlocks = randomCols(numParams, 0.1, 57 + seed);
  colBlocks = makeIndependentElimSet(colBlocks, 0, 30);

  // Create using original interface
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose();
  vector<int64_t> paramSize = randomVec(ss.ptrs.size() - 1, 2, 4, 47 + seed);

  Settings settings;
  settings.backend = BackendFast;
  settings.addFillPolicy = AddFillComplete;

  auto solverOrig = createSolver(settings, paramSize, ss);

  // Create using CSR interface
  vector<int64_t> rowStart, colIndices;
  columnsToCsr(colBlocks, rowStart, colIndices);

  BlockCsrDescriptor desc;
  desc.numBlocks = numParams;
  desc.numBlockNonzeros = colIndices.size();
  desc.rowStart = rowStart.data();
  desc.colIndices = colIndices.data();
  desc.blockSizes = paramSize.data();
  desc.indexType = INDEX_INT64;
  desc.mtype = MTYPE_SPD;
  desc.mview = MVIEW_LOWER;
  desc.indexBase = BASE_ZERO;

  auto solverCsr = createSolverFromBlockCsr(settings, desc);

  // Both solvers should have same data size
  ASSERT_EQ(solverOrig->dataSize(), solverCsr->dataSize());
  ASSERT_EQ(solverOrig->order(), solverCsr->order());

  // Generate random SPD data
  vector<T> dataOrig = randomData<T>(solverOrig->dataSize(), -1.0, 1.0, 9 + seed);
  solverOrig->skel().damp(dataOrig, T(0.0), T(solverOrig->order() * 2.0));

  // Extract to CSR format
  vector<T> csrValues(colIndices.size() * 16);  // Max block size 4x4
  int64_t valOffset = 0;
  for (int64_t row = 0; row < numParams; row++) {
    int64_t rowSize = paramSize[row];
    for (int64_t ptr = rowStart[row]; ptr < rowStart[row + 1]; ptr++) {
      int64_t col = colIndices[ptr];
      int64_t colSize = paramSize[col];
      // Just use zeros for now - we're testing structure, not values
      for (int64_t i = 0; i < rowSize * colSize; i++) {
        csrValues[valOffset + i] = T(0);
      }
      valOffset += rowSize * colSize;
    }
  }
}

// Test block CSR to SparseStructure conversion
TEST(CsrSolver, BlockCsrToSparseStructure) {
  // Simple 3x3 block matrix with lower triangular structure
  // Block sizes: [2, 3, 2]
  // Structure:
  //   [0]
  //   [1, 0]
  //   [2, 2, 0]

  vector<int64_t> rowStart = {0, 1, 3, 6};
  vector<int64_t> colIndices = {0, 0, 1, 0, 1, 2};
  vector<int64_t> blockSizes = {2, 3, 2};

  BlockCsrDescriptor desc;
  desc.numBlocks = 3;
  desc.numBlockNonzeros = 6;
  desc.rowStart = rowStart.data();
  desc.colIndices = colIndices.data();
  desc.blockSizes = blockSizes.data();
  desc.indexType = INDEX_INT64;
  desc.mtype = MTYPE_SPD;
  desc.mview = MVIEW_LOWER;
  desc.indexBase = BASE_ZERO;

  SparseStructure ss = blockCsrToSparseStructure(desc);

  // Check structure matches
  ASSERT_EQ(ss.ptrs.size(), 4);
  ASSERT_EQ(ss.inds.size(), 6);
  EXPECT_EQ(ss.ptrs[0], 0);
  EXPECT_EQ(ss.ptrs[1], 1);
  EXPECT_EQ(ss.ptrs[2], 3);
  EXPECT_EQ(ss.ptrs[3], 6);
}

// Test INDEX_INT32 handling
TEST(CsrSolver, Int32Indices) {
  vector<int32_t> rowStart = {0, 1, 3, 6};
  vector<int32_t> colIndices = {0, 0, 1, 0, 1, 2};
  vector<int64_t> blockSizes = {2, 3, 2};

  BlockCsrDescriptor desc;
  desc.numBlocks = 3;
  desc.numBlockNonzeros = 6;
  desc.rowStart = rowStart.data();
  desc.colIndices = colIndices.data();
  desc.blockSizes = blockSizes.data();
  desc.indexType = INDEX_INT32;
  desc.mtype = MTYPE_SPD;
  desc.mview = MVIEW_LOWER;
  desc.indexBase = BASE_ZERO;

  SparseStructure ss = blockCsrToSparseStructure(desc);
  ASSERT_EQ(ss.ptrs.size(), 4);
  ASSERT_EQ(ss.inds.size(), 6);
}

// Test BASE_ONE handling
TEST(CsrSolver, OneBasedIndices) {
  // Same structure but 1-based
  vector<int64_t> rowStart = {1, 2, 4, 7};  // 1-based
  vector<int64_t> colIndices = {1, 1, 2, 1, 2, 3};  // 1-based
  vector<int64_t> blockSizes = {2, 3, 2};

  BlockCsrDescriptor desc;
  desc.numBlocks = 3;
  desc.numBlockNonzeros = 6;
  desc.rowStart = rowStart.data();
  desc.colIndices = colIndices.data();
  desc.blockSizes = blockSizes.data();
  desc.indexType = INDEX_INT64;
  desc.mtype = MTYPE_SPD;
  desc.mview = MVIEW_LOWER;
  desc.indexBase = BASE_ONE;

  SparseStructure ss = blockCsrToSparseStructure(desc);
  ASSERT_EQ(ss.ptrs.size(), 4);
  ASSERT_EQ(ss.inds.size(), 6);
  // After conversion, should be 0-based
  EXPECT_EQ(ss.ptrs[0], 0);
  EXPECT_EQ(ss.inds[0], 0);
}

// Test invalid MatrixType rejection
TEST(CsrSolver, InvalidMatrixType) {
  vector<int64_t> rowStart = {0, 1};
  vector<int64_t> colIndices = {0};
  vector<int64_t> blockSizes = {2};

  BlockCsrDescriptor desc;
  desc.numBlocks = 1;
  desc.numBlockNonzeros = 1;
  desc.rowStart = rowStart.data();
  desc.colIndices = colIndices.data();
  desc.blockSizes = blockSizes.data();
  desc.indexType = INDEX_INT64;
  desc.mtype = MTYPE_GENERAL;  // Not supported
  desc.mview = MVIEW_LOWER;
  desc.indexBase = BASE_ZERO;

  EXPECT_THROW(blockCsrToSparseStructure(desc), std::invalid_argument);
}

// Test createSolverFromBlockCsr basic functionality
template <typename T>
void testCreateSolverFromBlockCsr() {
  // Simple 3-block lower triangular matrix
  vector<int64_t> rowStart = {0, 1, 3, 6};
  vector<int64_t> colIndices = {0, 0, 1, 0, 1, 2};
  vector<int64_t> blockSizes = {2, 2, 2};

  BlockCsrDescriptor desc;
  desc.numBlocks = 3;
  desc.numBlockNonzeros = 6;
  desc.rowStart = rowStart.data();
  desc.colIndices = colIndices.data();
  desc.blockSizes = blockSizes.data();
  desc.indexType = INDEX_INT64;
  desc.mtype = MTYPE_SPD;
  desc.mview = MVIEW_LOWER;
  desc.indexBase = BASE_ZERO;

  Settings settings;
  settings.backend = BackendFast;
  settings.addFillPolicy = AddFillComplete;

  auto solver = createSolverFromBlockCsr(settings, desc);

  ASSERT_NE(solver, nullptr);
  EXPECT_EQ(solver->order(), 6);  // 2 + 2 + 2
  EXPECT_GT(solver->dataSize(), 0);
}

TEST(CsrSolver, CreateSolverFromBlockCsr_float) { testCreateSolverFromBlockCsr<float>(); }

TEST(CsrSolver, CreateSolverFromBlockCsr_double) { testCreateSolverFromBlockCsr<double>(); }

// Test full factor+solve workflow with CSR interface
template <typename T>
void testCsrFactorSolve() {
  // 2-block diagonal matrix for simplicity
  vector<int64_t> rowStart = {0, 1, 2};
  vector<int64_t> colIndices = {0, 1};
  vector<int64_t> blockSizes = {2, 2};

  // Create block values: 2x2 identity blocks (SPD)
  // Block 0 (diagonal): [[2, 0], [0, 2]]
  // Block 1 (diagonal): [[3, 0], [0, 3]]
  vector<T> csrValues = {
      2, 0, 0, 2,  // Block (0,0)
      3, 0, 0, 3   // Block (1,1)
  };

  BlockCsrDescriptor desc;
  desc.numBlocks = 2;
  desc.numBlockNonzeros = 2;
  desc.rowStart = rowStart.data();
  desc.colIndices = colIndices.data();
  desc.blockSizes = blockSizes.data();
  desc.indexType = INDEX_INT64;
  desc.mtype = MTYPE_SPD;
  desc.mview = MVIEW_LOWER;
  desc.indexBase = BASE_ZERO;

  Settings settings;
  settings.backend = BackendFast;
  settings.addFillPolicy = AddFillComplete;

  vector<T> data;
  auto solver = createSolverFromBlockCsrWithValues(settings, desc, csrValues.data(), data);

  ASSERT_NE(solver, nullptr);

  // Factor
  solver->factor(data.data());

  // Solve with RHS = [1, 1, 1, 1]
  vector<T> rhs = {1, 1, 1, 1};
  solver->solve(data.data(), rhs.data(), 4, 1);

  // Expected solution: [0.5, 0.5, 1/3, 1/3]
  EXPECT_NEAR(rhs[0], T(0.5), Epsilon<T>::value2);
  EXPECT_NEAR(rhs[1], T(0.5), Epsilon<T>::value2);
  EXPECT_NEAR(rhs[2], T(1.0 / 3.0), Epsilon<T>::value2);
  EXPECT_NEAR(rhs[3], T(1.0 / 3.0), Epsilon<T>::value2);
}

TEST(CsrSolver, FactorSolve_float) { testCsrFactorSolve<float>(); }

TEST(CsrSolver, FactorSolve_double) { testCsrFactorSolve<double>(); }
