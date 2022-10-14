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
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace testing_utils;
using namespace std;
using Vec1 = Eigen::Vector<double, 1>;
using Mat11 = Eigen::Matrix<double, 1, 1>;

int main(int argc, char* argv[]) {
  vector<Variable<Vec1>> pointVars = {{-2}, {-1}, {0}, {0.5}, {1.5}, {2.5}};

  Optimizer opt;
  for (size_t i = 0; i < pointVars.size() - 1; i++) {
    static constexpr double springLen = 1.0;
    opt.addFactor(
        [=](const Vec1& x, const Vec1& y, Mat11* dx, Mat11* dy) -> Vec1 {
          if (dx) {
            (*dx)(0, 0) = -1;
          }
          if (dy) {
            (*dy)(0, 0) = 1;
          }
          return Vec1(y[0] - x[0] - springLen);
        },
        pointVars[i], pointVars[i + 1]);
  }

  opt.optimize();

  return 0;
}
