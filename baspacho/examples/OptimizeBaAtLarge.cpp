/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <iomanip>
#include "Optimizer.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/Utils.h"
#include "baspacho/benchmarking/BaAtLarge.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace testing;
using namespace ba_at_large;
using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

int main(int argc, char* argv[]) {
  if (argc <= 1) {
    cout << "Usage: prog bal_file.txt" << endl;
    return 1;
  }

  cout << "Loading data..." << endl;
  Data data;
  data.load(argv[1], false);
  data.removeBadObservations(data.points.size());

  Optimizer opt;
  vector<Variable<Eigen::Vector3d>> pointVars(data.points.size());
  for (size_t i = 0; i < data.points.size(); i++) {
    pointVars[i].value = data.points[i];
    opt.registerVariable(pointVars[i]);
  }
  opt.registeredVariablesToEliminationRange();  // eliminate points
  vector<Variable<Sophus::SE3d>> cameraVars(data.cameras.size());
  for (size_t i = 0; i < data.cameras.size(); i++) {
    cameraVars[i].value = data.cameras[i].T_W_C;
  }

  for (size_t i = 0; i < data.observations.size(); i++) {
    auto& obs = data.observations[i];
    opt.addFactor(
        [&imgPos = obs.imgPos, &calib = data.cameras[obs.camIdx].f_k1_k2](
            const Eigen::Vector<double, 3>& worldPt, const Sophus::SE3d& T_W_C,
            Eigen::Matrix<double, 2, 3>* point_jacobian,
            Eigen::Matrix<double, 2, 6>* camera_jacobian) -> Eigen::Vector<double, 2> {
          return Cost::compute_residual(imgPos, worldPt, T_W_C, calib, point_jacobian,
                                        camera_jacobian, nullptr);
        },
        pointVars[obs.ptIdx], cameraVars[obs.camIdx]);
  }

  opt.optimize({.solverType = data.cameras.size() > 100 ? Optimizer::Solver_PCG_GaussSeidel
                                                        : Optimizer::Solver_Direct});

  return 0;
}