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
using namespace testing_utils;
using namespace ba_at_large;
using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

void solveCameraPoses(Data& data, bool directSolver, int numThreads) {
  Optimizer opt;

  // register points first, and hint them as elimination set
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

  opt.optimize(
      {.numThreads = (unsigned int)numThreads,
       .solverType = directSolver ? Optimizer::Solver_Direct : Optimizer::Solver_PCG_GaussSeidel});
}

struct CameraParam {
  double dummy;
  Eigen::Vector3d calib;
  Sophus::SE3d pose;
};

template <>
struct VarUtil<CameraParam> {
  static constexpr int DataDim = 10;
  static constexpr int TangentDim = 9;

  template <typename VecType>
  static void tangentStep(const VecType& step, CameraParam& value) {
    value.pose = Sophus::SE3d::exp(step.template head<6>()) * value.pose;
    value.calib += step.template tail<3>();
  }

  static double* dataPtr(CameraParam& value) { return value.calib.data(); }

  static const double* dataPtr(const CameraParam& value) { return value.calib.data(); }
};

void solveCameraPosesAndCalibrations(Data& data, bool directSolver, int numThreads) {
  Optimizer opt;

  CameraParam c;
  BASPACHO_CHECK_EQ(c.calib.data() + 3, c.pose.data());

  // register points first, and hint them as elimination set
  vector<Variable<Eigen::Vector3d>> pointVars(data.points.size());
  for (size_t i = 0; i < data.points.size(); i++) {
    pointVars[i].value = data.points[i];
    opt.registerVariable(pointVars[i]);
  }
  opt.registeredVariablesToEliminationRange();  // eliminate points

  vector<Variable<CameraParam>> cameraParams(data.cameras.size());
  for (size_t i = 0; i < data.cameras.size(); i++) {
    cameraParams[i].value.calib = data.cameras[i].f_k1_k2;
    cameraParams[i].value.pose = data.cameras[i].T_W_C;
  }

  for (size_t i = 0; i < data.observations.size(); i++) {
    auto& obs = data.observations[i];
    opt.addFactor(
        [&imgPos = obs.imgPos](
            const Eigen::Vector<double, 3>& worldPt,  //
            const CameraParam& camera,                //
            Eigen::Matrix<double, 2, 3>* point_jacobian,
            Eigen::Matrix<double, 2, 9>* camera_jacobian) -> Eigen::Vector<double, 2> {
          if (camera_jacobian) {
            Eigen::Matrix<double, 2, 6> pose_jacobian;
            Eigen::Matrix<double, 2, 3> calib_jacobian;
            auto retv = Cost::compute_residual(imgPos, worldPt, camera.pose, camera.calib,
                                               point_jacobian, &pose_jacobian, &calib_jacobian);
            *camera_jacobian << pose_jacobian, calib_jacobian;
            return retv;
          } else {
            return Cost::compute_residual(imgPos, worldPt, camera.pose, camera.calib,
                                          point_jacobian, nullptr, nullptr);
          }
        },
        pointVars[obs.ptIdx], cameraParams[obs.camIdx]);
  }

  opt.optimize(
      {.numThreads = (unsigned int)numThreads,
       .solverType = directSolver ? Optimizer::Solver_Direct : Optimizer::Solver_PCG_GaussSeidel});
}

void help() {
  cout << "opt_ba_at_large [-c|-d|-t] -i file.txt\n"
       << " -c    [c]alibrate cameras (cameras have therefore parameters\n"
       << "       with 9dof = 6dof pose + 3dof Snavely-style calibration,\n"
       << "       instead of just 6dof pose)\n"
       << " -t n  set num of [t]hreads\n"
       << " -d    force [d]irect solver\n"
       << " -p    force iterative solver ([p]cg)" << std::endl;
}

int main(int argc, char* argv[]) {
  bool calibrate = false;
  bool forceDirect = false;
  bool forceIterative = false;
  int numThreads = 8;
  std::string input;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-h")) {
      help();
      return 0;
    }
    if (!strcmp(argv[i], "-c")) {
      calibrate = true;
    } else if (!strcmp(argv[i], "-d")) {
      forceDirect = true;
    } else if (!strcmp(argv[i], "-p")) {
      forceIterative = true;
    } else if (!strcmp(argv[i], "-i") && i < argc - 1) {
      input = argv[++i];
    } else if (!strcmp(argv[i], "-t") && i < argc - 1) {
      numThreads = std::stoi(argv[++i]);
    } else {
      std::cout << "Error: unknown argument: '" << argv[i] << "'" << std::endl;
      help();
      return 0;
    }
  }
  if (input.empty()) {
    std::cout << "Error: missing input file (-i file.txt)" << std::endl;
    help();
    return 0;
  }

  cout << "Loading data..." << endl;
  Data data;
  data.load(input, false);
  data.removeBadObservations(data.points.size());

  bool directSolver = forceDirect ? true : forceIterative ? false : (data.cameras.size() < 1000);

  if (calibrate) {
    solveCameraPosesAndCalibrations(data, directSolver, numThreads);
  } else {
    solveCameraPoses(data, directSolver, numThreads);
  }

  return 0;
}
