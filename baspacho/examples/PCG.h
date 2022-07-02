#pragma once

#include <Eigen/Core>
#include <functional>

namespace zero {

// Preconditioned conjugate gradient
class PCG {
 public:
  using TransFunc = std::function<void(Eigen::VectorXd&, const Eigen::VectorXd&)>;
  using NumFunc = std::function<double(double)>;

  PCG(const TransFunc& applyInvM,
      const TransFunc& applyA,
      double wantedResidual,
      int maxSteps,
      bool trace = false,
      int nodeId = -1)
      : applyInvM(applyInvM),
        applyA(applyA),
        wantedResidual(wantedResidual),
        maxSteps(maxSteps),
        trace(trace),
        nodeId(nodeId) {}

  // returns pair (# iterations done, relative residual)
  std::pair<int, double>
  solve(Eigen::VectorXd& x, const Eigen::VectorXd& b, int overrideMaxSteps = -1);

  size_t memUsage();

  TransFunc applyInvM;
  TransFunc applyA;
  const double wantedResidual;
  const int maxSteps;
  const bool trace;
  const int nodeId;

  Eigen::VectorXd xk, rk, zk, pk;
  Eigen::VectorXd xk1, rk1, zk1, pk1;
  Eigen::VectorXd Apk;
};
