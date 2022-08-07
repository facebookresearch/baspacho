/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <sophus/se3.hpp>
#include <Eigen/Geometry>
#include <iostream>
#include <string>
#include <vector>

namespace ba_at_large {

using Quat = Eigen::Quaternion<double>;
using Vec3 = Eigen::Matrix<double, 3, 1>;
using Vec2 = Eigen::Matrix<double, 2, 1>;

struct Data {
  struct Obs {
    int camIdx;
    int ptIdx;
    Vec2 imgPos;

    Obs(int _camIdx, int _ptIdx, const Vec2& _imgPos)
        : camIdx(_camIdx), ptIdx(_ptIdx), imgPos(_imgPos) {}

    Obs(int _camIdx, int _ptIdx, double x, double y)
        : camIdx(_camIdx), ptIdx(_ptIdx), imgPos(x, y) {}
  };

  struct Cam {
    Sophus::SE3d T_W_C;
    Vec3 f_k1_k2;

    Cam(const Quat& _rot, const Vec3& _tr, double _f, double _k1, double _k2)
        : T_W_C(_rot, _tr), f_k1_k2(_f, _k1, _k2) {}
  };

  std::vector<Obs> observations;
  std::vector<Cam> cameras;
  std::vector<Vec3> points;

  void removeBadObservations(int64_t maxNumPts = -1);

  void load(const std::string& path, bool verbose = false);

  void save(const std::string& path, bool verbose = false);

  void compute_fill_stats() const;
};

struct Cost {
  static Vec2 compute_residual(const Vec2& imagePt, const Vec3& worldPt, const Sophus::SE3d& T_W_C,
                               const Vec3& calib) {
    Vec3 camPt = T_W_C * worldPt;
    if (camPt.z() > 0.01) {
      return Vec2(25.0, 0.0);
    }

    Vec2 projPt(-camPt.x() / camPt.z(), -camPt.y() / camPt.z());
    double f = calib[0];
    double k1 = calib[1];
    double k2 = calib[2];
    double projPtSqN = projPt.squaredNorm();
    double r = 1.0 + (k1 + k2 * projPtSqN) * projPtSqN;
    Vec2 calibPt = (f * r) * projPt;

    return calibPt - imagePt;
  }

  static Vec2 compute_residual(const Vec2& imagePt, const Vec3& worldPt, const Sophus::SE3d& T_W_C,
                               const Vec3& calib, Eigen::Matrix<double, 2, 3>* point_jacobian,
                               Eigen::Matrix<double, 2, 6>* camera_jacobian,
                               Eigen::Matrix<double, 2, 3>* calib_jacobian) {
    Vec3 camPt = T_W_C * worldPt;
    if (camPt.z() > 0.01) {
      if (point_jacobian) {
        point_jacobian->setZero();
      }
      if (camera_jacobian) {
        camera_jacobian->setZero();
      }
      if (calib_jacobian) {
        calib_jacobian->setZero();
      }
      return Vec2(25.0, 0.0);
    }

    Vec2 projPt(-camPt.x() / camPt.z(), -camPt.y() / camPt.z());
    double f = calib[0];
    double k1 = calib[1];
    double k2 = calib[2];
    double projPtSqN = projPt.squaredNorm();
    double r = 1.0 + (k1 + k2 * projPtSqN) * projPtSqN;
    Vec2 calibPt = (f * r) * projPt;

    if (point_jacobian) {
      Eigen::Matrix<double, 2, 3>& J = *point_jacobian;

      const Eigen::Matrix<double, 3, 3>& R = T_W_C.so3().matrix();
      double denum = -1.0 / (camPt[2] * camPt[2]);

      // the final values of the non-calibrated case
      Eigen::Matrix<double, 2, 3> DprojPt;
      DprojPt.row(0) = (camPt[2] * R.row(0) - camPt[0] * R.row(2)) * denum;
      DprojPt.row(1) = (camPt[2] * R.row(1) - camPt[1] * R.row(2)) * denum;

      Eigen::Matrix<double, 1, 3> DpPtSqN = 2.0 * projPt.transpose() * DprojPt;
      J = (f * r) * DprojPt + projPt * DpPtSqN * (f * (k1 + k2 * 2.0 * projPtSqN));
    }

    if (camera_jacobian) {
      Eigen::Matrix<double, 2, 6>& J = *camera_jacobian;

      double dz = 1 / camPt[2];
      double xdz = camPt[0] * dz;
      double ydz = camPt[1] * dz;
      double xydz2 = xdz * ydz;

      // the final values of the non-calibrated case
      Eigen::Matrix<double, 2, 6> DprojPt;
      DprojPt(0, 0) = -dz;
      DprojPt(0, 1) = 0;
      DprojPt(0, 2) = xdz * dz;
      DprojPt(0, 3) = xydz2;
      DprojPt(0, 4) = -1 - xdz * xdz;
      DprojPt(0, 5) = ydz;

      DprojPt(1, 0) = 0;
      DprojPt(1, 1) = -dz;
      DprojPt(1, 2) = ydz * dz;
      DprojPt(1, 3) = 1 + ydz * ydz;
      DprojPt(1, 4) = -xydz2;
      DprojPt(1, 5) = -xdz;

      Eigen::Matrix<double, 1, 6> DpPtSqN = 2.0 * projPt.transpose() * DprojPt;
      J = (f * r) * DprojPt + projPt * DpPtSqN * (f * (k1 + k2 * 2.0 * projPtSqN));
    }

    if (calib_jacobian) {
      Eigen::Matrix<double, 2, 3>& J = *calib_jacobian;

      J.col(0) = r * projPt;                            // d(calibPt) / d(f)
      J.col(1) = (f * projPtSqN) * projPt;              // d(calibPt) / d(k1)
      J.col(2) = (f * projPtSqN * projPtSqN) * projPt;  // d(calibPt) / d(k2)
    }

    return calibPt - imagePt;
  }
};

}  // end namespace ba_at_large