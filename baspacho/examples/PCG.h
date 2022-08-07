/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Core>
#include <functional>

// Simple implementation of preconditioned conjugate gradient
class PCG {
 public:
  using TransFunc = std::function<void(Eigen::VectorXd&, const Eigen::VectorXd&)>;
  using NumFunc = std::function<double(double)>;

  PCG(TransFunc&& applyInvM, TransFunc&& applyA, double wantedResidual, int maxSteps,
      bool trace = false)
      : applyInvM(std::move(applyInvM)),
        applyA(std::move(applyA)),
        wantedResidual(wantedResidual),
        maxSteps(maxSteps),
        trace(trace) {}

  // returns pair (# iterations done, relative residual)
  std::pair<int, double> solve(Eigen::VectorXd& x, const Eigen::VectorXd& b,
                               int overrideMaxSteps = -1);

  size_t memUsage();

  TransFunc applyInvM;
  TransFunc applyA;
  const double wantedResidual;
  const int maxSteps;
  const bool trace;

  Eigen::VectorXd xk, rk, zk, pk;
  Eigen::VectorXd xk1, rk1, zk1, pk1;
  Eigen::VectorXd Apk;
};
