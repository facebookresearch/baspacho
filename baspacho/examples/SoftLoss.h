/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>
#include <tuple>

struct Loss {};

struct TrivialLoss : Loss {
  explicit TrivialLoss(double /* a */ = 0) {}

  void setSize(double /* a */) {}

  inline double val(double s) const { return s; }

  inline double der(double s) const { return 1.0; }

  inline std::pair<double, double> jet2(double s) const { return std::make_pair(s, 1.0); }

  inline std::tuple<double, double, double> jet3(double s) const {
    return std::make_tuple(s, 1.0, 0.0);
  }
};

struct HuberLoss : Loss {
  double a, b;
  explicit HuberLoss(double a) : a(a), b(a * a) {}

  void setSize(double _a) { a = _a; }

  inline double val(double s) const {
    if (s > b) {
      const double r = sqrt(s);
      return 2.0 * a * r - b;
    } else {
      return s;
    }
  }

  inline double der(double s) const {
    if (s > b) {
      const double r = sqrt(s);
      return a / r;
    } else {
      return 1.0;
    }
  }

  inline std::pair<double, double> jet2(double s) const {
    if (s > b) {
      const double r = sqrt(s);
      const double d = a / r;
      return std::make_pair(2.0 * a * r - b, d);
    } else {
      return std::make_pair(s, 1.0);
    }
  }

  inline std::tuple<double, double, double> jet3(double s) const {
    if (s > b) {
      const double r = sqrt(s);
      const double d = a / r;
      return std::make_tuple(2.0 * a * r - b, d, -d / (2.0 * s));
    } else {
      return std::make_tuple(s, 1.0, 0.0);
    }
  }
};

struct CauchyLoss : Loss {
  double b, c;
  explicit CauchyLoss(double a) : b(a * a), c(1 / b) {}

  void setSize(double a) {
    b = a * a;
    c = 1 / b;
  }

  inline double val(double s) const {
    double sum = 1.0 + s * c;
    return b * std::log(sum);
  }

  inline double der(double s) const {
    double sum = 1.0 + s * c;
    return 1.0 / sum;
  }

  inline std::pair<double, double> jet2(double s) const {
    double sum = 1.0 + s * c;
    double inv = 1.0 / sum;
    return std::make_pair(b * std::log(sum), inv);
  }

  inline std::tuple<double, double, double> jet3(double s) const {
    double sum = 1.0 + s * c;
    double inv = 1.0 / sum;
    return std::make_tuple(b * std::log(sum), inv, -c * (inv * inv));
  }
};