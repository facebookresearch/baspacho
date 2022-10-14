/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <Eigen/Eigenvalues>
#include "baspacho/baspacho/MathUtils.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing_utils;
using namespace std;

template <typename T>
using MatRMaj = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

TEST(MathUtils, Cholesky) {
  int n = 10;
  vector<double> data = randomData(n * n, -1.0, 1.0, 37);

  Eigen::Map<MatRMaj<double>>(data.data(), n, n).diagonal().array() += n * 1.3;

  MatRMaj<double> verifyMat = Eigen::Map<MatRMaj<double>>(data.data(), n, n);

  { Eigen::LLT<Eigen::Ref<MatRMaj<double>>> llt(verifyMat); }

  cholesky(data.data(), n, n);
  MatRMaj<double> computedMat = Eigen::Map<MatRMaj<double>>(data.data(), n, n);

  ASSERT_NEAR(Eigen::MatrixXd((verifyMat - computedMat).triangularView<Eigen::Lower>()).norm(), 0,
              1e-7);
}

TEST(MathUtils, Solve) {
  int n = 10, k = 1;
  vector<double> data = randomData(n * n, -1.0, 1.0, 37);
  Eigen::Map<MatRMaj<double>>(data.data(), n, n).diagonal().array() += n * 0.3;
  Eigen::Map<MatRMaj<double>> verifyMat(data.data(), n, n);

  vector<double> vecData = randomData(n * k, -1.0, 1.0, 39);

  MatRMaj<double> verifyVec = Eigen::Map<MatRMaj<double>>(vecData.data(), k, n);

  verifyMat.template triangularView<Eigen::Lower>()
      .transpose()
      .template solveInPlace<Eigen::OnTheRight>(verifyVec);

  solveUpperT(data.data(), n, n, vecData.data());
  MatRMaj<double> computeVec = Eigen::Map<MatRMaj<double>>(vecData.data(), k, n);

  ASSERT_NEAR((verifyVec - computeVec).norm(), 0, 1e-7);
}

TEST(MathUtils, SolveT) {
  int n = 10, k = 1;
  vector<double> data = randomData(n * n, -1.0, 1.0, 37);
  Eigen::Map<MatRMaj<double>>(data.data(), n, n).diagonal().array() += n * 0.3;
  Eigen::Map<MatRMaj<double>> verifyMat(data.data(), n, n);

  vector<double> vecData = randomData(n * k, -1.0, 1.0, 39);

  MatRMaj<double> verifyVec = Eigen::Map<MatRMaj<double>>(vecData.data(), k, n);

  verifyMat.template triangularView<Eigen::Lower>().template solveInPlace<Eigen::OnTheRight>(
      verifyVec);

  solveUpper(data.data(), n, n, vecData.data());
  MatRMaj<double> computeVec = Eigen::Map<MatRMaj<double>>(vecData.data(), k, n);

  ASSERT_NEAR((verifyVec - computeVec).norm(), 0, 1e-7);
}
