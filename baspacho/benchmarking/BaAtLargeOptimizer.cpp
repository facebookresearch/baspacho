/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <iomanip>
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/Utils.h"
#include "baspacho/benchmarking/BaAtLarge.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace testing_utils;
using namespace ba_at_large;
using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

struct Optimizer {
  Optimizer(Data& data) : data(data) {
    numPts = data.points.size();
    numCams = data.cameras.size();
    totNumParams = numPts + numCams;

    badObs.resize(data.observations.size());
    ptsBackup.resize(numPts);
    camsBackup.resize(numCams);

    // create sparse structure
    vector<int64_t> paramSize(totNumParams);
    vector<set<int64_t>> colBlocks(totNumParams);
    for (int64_t i = 0; i < numPts; i++) {  // points go first
      paramSize[i] = 3;
      colBlocks[i].insert(i);
    }
    for (int64_t i = numPts; i < totNumParams; i++) {  // then cams
      paramSize[i] = 6;
      colBlocks[i].insert(i);
    }
    for (auto& obs : data.observations) {
      colBlocks[obs.ptIdx].insert(numPts + obs.camIdx);
    }
    SparseStructure blockStructure = columnsToCscStruct(colBlocks).transpose();

    // create solver
    solver = createSolver({}, paramSize, blockStructure, {0, numPts});

    hess.resize(solver->dataSize());
    grad.resize(solver->order());
    step.resize(solver->order());
  };

  double computeCost() {
    double totCost = 0;
    for (auto& obs : data.observations) {
      auto& cam = data.cameras[obs.camIdx];
      auto& pt = data.points[obs.ptIdx];
      Vec2 err = Cost::compute_residual(obs.imgPos, pt, cam.T_W_C, cam.f_k1_k2);
      totCost += 0.5 * err.squaredNorm();
    }
    return totCost;
  }

  // for verification
  Eigen::MatrixXd denseHessian(double lambda) {
    Eigen::MatrixXd retv(solver->order(), solver->order());
    retv.setZero();

    auto accessor = solver->accessor();
    for (auto& obs : data.observations) {
      auto& cam = data.cameras[obs.camIdx];
      auto& pt = data.points[obs.ptIdx];
      Eigen::Matrix<double, 2, 3> ptJac;
      Eigen::Matrix<double, 2, 6> camJac;
      Vec2 err =
          Cost::compute_residual(obs.imgPos, pt, cam.T_W_C, cam.f_k1_k2, &ptJac, &camJac, nullptr);

      int64_t ptId = obs.ptIdx;
      int64_t camId = numPts + obs.camIdx;
      int64_t ptStart = accessor.paramStart(ptId);
      int64_t camStart = accessor.paramStart(camId);

      retv.block<3, 3>(ptStart, ptStart) += ptJac.transpose() * ptJac;
      retv.block<3, 6>(ptStart, camStart) += ptJac.transpose() * camJac;
      retv.block<6, 3>(camStart, ptStart) += camJac.transpose() * ptJac;
      retv.block<6, 6>(camStart, camStart) += camJac.transpose() * camJac;
    }

    retv.diagonal() *= 1.0 + lambda;
    retv.diagonal().array() += lambda * 1e-3;

    return retv;
  }

  double computeStep(double lambda) {
    // create hessian/gradient
    auto startGradHess = hrc::now();
    badObs.assign(data.observations.size(), false);
    hess.setZero();
    grad.setZero();
    double totCost = 0;
    double* d = hess.data();
    auto accessor = solver->accessor();

    for (auto& obs : data.observations) {
      auto& cam = data.cameras[obs.camIdx];
      auto& pt = data.points[obs.ptIdx];
      Eigen::Matrix<double, 2, 3> ptJac;
      Eigen::Matrix<double, 2, 6> camJac;
      Vec2 err =
          Cost::compute_residual(obs.imgPos, pt, cam.T_W_C, cam.f_k1_k2, &ptJac, &camJac, nullptr);
      totCost += 0.5 * err.squaredNorm();

      int64_t ptId = obs.ptIdx;
      int64_t camId = numPts + obs.camIdx;
      accessor.diagBlock(d, ptId) += ptJac.transpose() * ptJac;
      accessor.diagBlock(d, camId) += camJac.transpose() * camJac;
      accessor.block(d, camId, ptId) += camJac.transpose() * ptJac;
      grad.segment<3>(accessor.paramStart(ptId)) += ptJac.transpose() * err;
      grad.segment<6>(accessor.paramStart(camId)) += camJac.transpose() * err;
    }

    // apply damping
    for (int64_t i = 0; i < totNumParams; i++) {
      accessor.diagBlock(d, i).diagonal() *= 1.0 + lambda;
      accessor.diagBlock(d, i).diagonal().array() += lambda * 1e-3;
    }

    auto startFactor = hrc::now();
    solver->resetStats();
    solver->factor(d);

    auto startSolve = hrc::now();
    step = grad;
    solver->solve(d, step.data(), step.size(), 1);

    auto now = hrc::now();
    double factorTime = tdelta(startSolve - startFactor).count();
    double schurTime =  // "Schur" is first sparse elim range
        solver->internalGetElimCtx(0).elimStat.totTime;
    cout << fixed << setprecision(3)
         << "  <timings - costs: " << tdelta(startFactor - startGradHess).count()
         << "s, factor: " << factorTime << "s (points elim: " << schurTime
         << "s, cam-cam: " << (factorTime - schurTime)
         << "s), solve: " << tdelta(now - startSolve).count() << "s>" << endl;

    return totCost;
  }

  void saveParams() {
    auto accessor = solver->accessor();
    for (int64_t i = 0; i < numPts; i++) {
      ptsBackup[i] = data.points[i];
    }
    for (int64_t i = 0; i < numCams; i++) {
      camsBackup[i] = data.cameras[i].T_W_C;
    }
  }

  void restoreParams() {
    auto accessor = solver->accessor();
    for (int64_t i = 0; i < numPts; i++) {
      data.points[i] = ptsBackup[i];
    }
    for (int64_t i = 0; i < numCams; i++) {
      data.cameras[i].T_W_C = camsBackup[i];
    }
  }

  void applyStep() {
    auto accessor = solver->accessor();
    for (int64_t i = 0; i < numPts; i++) {
      BASPACHO_CHECK_LE(accessor.paramStart(i) + 3, step.size());
      data.points[i] -= step.segment<3>(accessor.paramStart(i));
    }
    for (int64_t i = numPts; i < totNumParams; i++) {
      BASPACHO_CHECK_LE(accessor.paramStart(i) + 6, step.size());
      data.cameras[i - numPts].T_W_C = Sophus::SE3d::exp(-step.segment<6>(accessor.paramStart(i))) *
                                       data.cameras[i - numPts].T_W_C;
    }
  }

  void optimize() {
    double lambda = 1e-5;

    for (int i = 0, lastFailed = 0, lastOk = 0, lastGood = 0; i < 50; i++) {
      saveParams();
      double cost = computeStep(lambda);
      applyStep();

      double modelRed = 0.5 * grad.dot(step);
      double newCost = computeCost();

      bool bad = newCost > cost;
      bool good = newCost < cost * 0.999;
      cout << "[" << i << "] cost: " << scientific << setprecision(3) << cost << " -> " << newCost
           << (good  ? "!"
               : bad ? "?"
                     : "~");

      cout << setprecision(1) << " | lambda: " << lambda;
      if (bad) {
        lastFailed = i;
        if (lambda > 1e8 || (i > lastOk + 15)) {
          cout << " (failed, giving up)" << endl;
          break;
        }
        lambda *= 3;
        cout << " -> " << lambda << " (failed, retry...)" << endl;
        restoreParams();
        continue;
      }
      lastOk = i;
      if (good) {
        lastGood = i;
      }

      double relRed = (cost - newCost) / modelRed;
      if (relRed > 0.6) {
        lambda *= 0.6;
      } else if (relRed < 0.4) {
        lambda *= 1.3;
      }
      cout << " -> " << lambda << ", model-red: " << setprecision(2) << modelRed << fixed
           << ", rel-red: " << relRed << endl;
      if (i >= lastGood + 3 && i >= lastFailed + 3) {
        cout << "converged!" << endl;
        break;
      }
    }
  }

  Data& data;
  int64_t numPts;
  int64_t numCams;
  int64_t totNumParams;
  SolverPtr solver;
  Eigen::VectorXd hess;
  Eigen::VectorXd grad;
  Eigen::VectorXd step;
  vector<bool> badObs;
  vector<Eigen::Vector3d> ptsBackup;
  vector<Sophus::SE3d> camsBackup;
};

int main(int argc, char* argv[]) {
  if (argc <= 1) {
    cout << "Usage: prog bal_file.txt" << endl;
    return 1;
  }

  cout << "Loading data..." << endl;
  Data data;
  data.load(argv[1], false);
  data.removeBadObservations(data.points.size());

  Optimizer opt(data);
  opt.optimize();

  return 0;
}