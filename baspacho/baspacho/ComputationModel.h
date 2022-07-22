#pragma once

#include <Eigen/Geometry>

namespace BaSpaCho {

struct ComputationModel {
  double potrfEst(double n) const { return potrfModel(potrfParams, n); }

  double trsmEst(double n, double k) const { return trsmModel(trsmParams, n, k); }

  double sygeEst(double m, double n, double k) const { return sygeModel(sygeParams, m, n, k); }

  double asmblEst(double br, double bc) const { return asmblModel(asmblParams, br, bc); }

  Eigen::Vector2d sygeLinEst(double m, double n) const { return sygeLinModel(sygeParams, m, n); }

  Eigen::Vector2d asmblLinEst(double br) const { return asmblLinModel(asmblParams, br); }

  Eigen::Vector<double, 4> potrfParams;
  Eigen::Vector<double, 6> trsmParams;
  Eigen::Vector<double, 6> sygeParams;
  Eigen::Vector<double, 4> asmblParams;

  // t ~= a + b*n + c*n^2 + d*n^3
  static double potrfModel(const Eigen::Vector<double, 4>& p, double n) {
    return p[0] + n * (p[1] + n * (p[2] + n * p[3]));
  }

  static Eigen::Matrix<double, 1, 4> dPotrfModel(double n) {
    return {1.0, n, n * n, n * n * n};  //
  }

  // t ~= a + b*n + c*n^2 + (d + e*n + f*n^2)*k
  static double trsmModel(const Eigen::Vector<double, 6>& p, double n, double k) {
    return p[0] + n * (p[1] + n * p[2]) + k * (p[3] + n * (p[4] + n * p[5]));
  }

  static Eigen::Matrix<double, 1, 6> dTrsmModel(double n, double k) {
    return {1.0, n, n * n, k, k * n, k * n * n};
  }

  // plain model is:
  //   t ~= a + b*m + c*n + d*k + e*m*n + f*m*k + g*n*k + h*n*m*k
  // symmetrized in m,n it becomes (putting u=m+n, v=mn the basis of sym functions):
  //   t ~= a + b*u + c*v + d*k + e*u*k + f*v*k
  static double sygeModel(const Eigen::Vector<double, 6>& p, double m, double n, double k) {
    return p[0] + (m + n) * p[1] + (m * n) * p[2] +  //
           k * (p[3] + (m + n) * p[4] + (m * n) * p[5]);
  }

  static Eigen::Matrix<double, 1, 6> dSygeModel(double m, double n, double k) {
    return {1.0, (m + n), (m * n), k, k * (m + n), k * (m * n)};
  }

  // linear function of k
  static Eigen::Vector2d sygeLinModel(const Eigen::Vector<double, 6>& p, double m, double n) {
    return {p[0] + (m + n) * p[1] + (m * n) * p[2],  //
            p[3] + (m + n) * p[4] + (m * n) * p[5]};
  }

  // t ~= a + b*br + c*bc + d*br*bc
  static double asmblModel(const Eigen::Vector<double, 4>& p, double br, double bc) {
    return p[0] + br * p[1] + bc * p[2] + br * bc * p[3];
  }

  static Eigen::Matrix<double, 1, 4> dAsmblModel(double br, double bc) {
    return {1.0, br, bc, br * bc};
  }

  // linear function of bc
  static Eigen::Vector2d asmblLinModel(const Eigen::Vector<double, 4>& p, double br) {
    return {p[0] + br * p[1], p[2] + br * p[3]};
  }

  // pre-built model collection
  static ComputationModel model_OpenBlas_i7_1185g7;
};

}  // namespace BaSpaCho