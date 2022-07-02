
#include <iostream>
#include "PCG.h"

using namespace std;

std::pair<int, double>
PCG::solve(Eigen::VectorXd& x, const Eigen::VectorXd& b, int overrideMaxSteps) {
  int mxs = overrideMaxSteps >= 0 ? overrideMaxSteps : maxSteps;

  /*
    0-th step (k=0):
      x_0 = 0
      r_0 = b - A*x_0 (=b)
      z_0 = M^{-1}*r_0
      p_0 = z_0
  */
  xk.resize(b.size());
  xk.setZero();
  rk = b;

  applyInvM(zk, rk);
  pk = zk;
  double rk0norm = rk.norm();
  double zk_rk = zk.dot(rk);
  double residual = 0.0;

  if (trace) {
	  std::cout << "|Rk[0]|_2 = " << rk0norm
	            << ", size=" << b.size() << std::endl;
  }

  /*
    k-th step:
      alpha_k = r_k*z_k / (p_k*A*p_k)
      x_k1 = x_k + alpha_k*p_k
      r_k1 = r_k - alpha_k*A*p_k
      (if r_k1 suff small exit loop)
      z_k1 = M^{-1}*r_k1
      beta_k = (z_k1*r_k1) / (z_k*r_k)
      p_k1 = z_k1 + beta_k*p_k
      k += 1
  */
  int k = 0;
  while (1) {
    // alpha_k = r_k*z_k / (p_k*A*p_k)
    applyA(Apk, pk);
    double pkApk = pk.dot(Apk);
    double alpha_k = zk_rk / pkApk;

    // x_k1 = x_k + alpha_k*p_k
    xk1 = xk + pk * alpha_k;

    // r_k1 = r_k - alpha_k*A*p_k
    rk1 = rk - Apk * alpha_k;

    double rk1norm = rk1.norm();
    residual = rk1norm / rk0norm;
    if (trace) {
	    std::cout << "|Rk[" << k + 1 << "]|_2 = " << rk1norm
	              << " (res = " << residual << ")" << std::endl;
    }
    if (residual < wantedResidual) {
      break;
    }
     if (k + 1 >= mxs) {
      if (trace) {
	      std::cout << "max conjugate gradient steps, giving up..."
	                << std::endl;
      }
      x = xk1;
      return std::make_pair(k + 1, residual); // failed
    }

    // z_k1 = M^{-1}*r_k1
    applyInvM(zk1, rk1);

    // beta_k = (z_k1*r_k1) / (z_k*r_k)
    double zk1_rk1 = zk1.dot(rk1);
    double beta_k = zk1_rk1 / zk_rk;

    // p_k1 = z_k1 + beta_k*p_k
    pk1 = zk1 + pk * beta_k;

    swap(xk, xk1);
    swap(rk, rk1);
    swap(zk, zk1);
    swap(pk, pk1);
    zk_rk = zk1_rk1;

    // k += 1
    k++;
  }

  x = xk1;
  return std::make_pair(k + 1, residual);
}

size_t PCG::memUsage() {
	return (xk.size() + rk.size() + zk.size() + pk.size() +
		xk1.size() + rk1.size() + zk1.size() + pk1.size() +
	        Apk.size()) * sizeof(double);
}
