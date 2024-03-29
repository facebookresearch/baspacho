/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iostream>
#include "baspacho/baspacho/Solver.h"

/* abstract preconditioner class */
template <typename T>
class Preconditioner {
 public:
  using Vec = Eigen::Vector<T, Eigen::Dynamic>;

  virtual ~Preconditioner() {}

  virtual void init(T* data) = 0;

  virtual void operator()(T* outVec, const T* inVec) = 0;
};

/* dummy identity preconditioner */
template <typename T>
class IdentityPrecond : public Preconditioner<T> {
 public:
  IdentityPrecond(const BaSpaCho::Solver& solver, int64_t paramStart)
      : solver(solver), vecSize(solver.order() - solver.spanVectorOffset(paramStart)) {}

  virtual ~IdentityPrecond() override {}

  virtual void init(T* /* data */) override {}

  virtual void operator()(T* outVec, const T* inVec) override {
    Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(outVec, vecSize) =
        Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>(inVec, vecSize);
  }

 private:
  const BaSpaCho::Solver& solver;
  int64_t vecSize;
};

/* Jacobi preconditioner, param-sized blocks are inverted */
template <typename T>
class BlockJacobiPrecond : public Preconditioner<T> {
 public:
  using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  BlockJacobiPrecond(const BaSpaCho::Solver& solver, int64_t paramStart)
      : solver(solver),
        paramStart(paramStart),
        vecSize(solver.order() - solver.spanVectorOffset(paramStart)) {
    int64_t numParams = solver.skel().spanStart.size() - 1;
    int64_t offset = 0;
    for (int64_t p = paramStart; p < numParams; p++) {
      diagBlockOffset.push_back(offset);
      int64_t pSize = solver.spanVectorOffset(p + 1) - solver.spanVectorOffset(p);
      offset += pSize * pSize;
    }
    diagBlockData.resize(offset);
  }

  virtual ~BlockJacobiPrecond() override {}

  virtual void init(T* data) override {
    auto acc = solver.accessor();
    for (size_t i = 0; i < diagBlockOffset.size(); i++) {
      int64_t p = i + paramStart;
      int64_t pSize = solver.spanVectorOffset(p + 1) - solver.spanVectorOffset(p);
      Eigen::Map<Mat> mat(diagBlockData.data() + diagBlockOffset[i], pSize, pSize);
      mat = acc.plainAcc.diagBlock(data, p);
      Eigen::LLT<Eigen::Ref<Mat>> llt(mat);
    }
  }

  virtual void operator()(T* outVec, const T* inVec) override {
    Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> out(outVec, vecSize);
    Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> in(inVec, vecSize);
    out = in;
    int64_t secStart = solver.spanVectorOffset(paramStart);
    for (size_t i = 0; i < diagBlockOffset.size(); i++) {
      int64_t p = i + paramStart;
      int64_t pStart = solver.spanVectorOffset(p);
      int64_t pSize = solver.spanVectorOffset(p + 1) - pStart;
      int64_t pStartInSec = pStart - secStart;
      Eigen::Map<Mat> mat(diagBlockData.data() + diagBlockOffset[i], pSize, pSize);
      mat.template triangularView<Eigen::Lower>().template solveInPlace<Eigen::OnTheLeft>(
          out.segment(pStartInSec, pSize));
      mat.template triangularView<Eigen::Lower>()
          .transpose()
          .template solveInPlace<Eigen::OnTheLeft>(out.segment(pStartInSec, pSize));
    }
  }

 private:
  const BaSpaCho::Solver& solver;
  int64_t paramStart;
  int64_t vecSize;
  std::vector<int64_t> diagBlockOffset;
  std::vector<T> diagBlockData;
};

/* Gauss-Seidel preconditioner, param-sized blocks are inverted */
template <typename T>
class BlockGaussSeidelPrecond : public Preconditioner<T> {
 public:
  BlockGaussSeidelPrecond(const BaSpaCho::Solver& solver, int64_t paramStart)
      : solver(solver),
        paramStart(paramStart),
        vecSize(solver.order() - solver.spanVectorOffset(paramStart)) {}

  virtual ~BlockGaussSeidelPrecond() override {}

  virtual void init(T* data) override {
    matData.assign(data + solver.spanMatrixOffset(paramStart), data + solver.dataSize());
    solver.pseudoFactorFrom(matData.data() - solver.spanMatrixOffset(paramStart), paramStart);
  }

  virtual void operator()(T* outVec, const T* inVec) override {
    Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(outVec, vecSize) =
        Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>(inVec, vecSize);

    solver.solveLFrom(  //
        matData.data() - solver.spanMatrixOffset(paramStart), paramStart,
        outVec - solver.spanVectorOffset(paramStart), vecSize, 1);
    solver.solveLtFrom(matData.data() - solver.spanMatrixOffset(paramStart), paramStart,
                       outVec - solver.spanVectorOffset(paramStart), vecSize, 1);
  }

 private:
  const BaSpaCho::Solver& solver;
  int64_t paramStart;
  int64_t vecSize;
  std::vector<T> matData;
};

/* lower precision-solve preconditioner */
template <typename T>
class LowerPrecSolvePrecond;

template <>
class LowerPrecSolvePrecond<double> : public Preconditioner<double> {
 public:
  using T = double;
  LowerPrecSolvePrecond(const BaSpaCho::Solver& solver, int64_t paramStart)
      : solver(solver),
        vecSize(solver.order() - solver.spanVectorOffset(paramStart)),
        paramStart(paramStart) {}

  virtual ~LowerPrecSolvePrecond() override {}

  virtual void init(double* data) override {
    int64_t offset = solver.spanMatrixOffset(paramStart);
    int64_t size = solver.dataSize() - offset;
    matData.resize(size);

    float epsilon = 0.0;
    while (true) {
      Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>(matData.data(), size) =
          Eigen::Map<const Eigen::Vector<double, Eigen::Dynamic>>(data + offset, size)
              .cast<float>();
      std::cout << "Trying with epsilon = " << epsilon << std::endl;
      if (epsilon > 0) {
        auto acc = solver.accessor();
        for (int64_t i = paramStart; i < solver.skel().numSpans(); i++) {
          auto diag = acc.plainAcc.diagBlock(matData.data() - offset, i).diagonal();
          diag *= 1.0 + epsilon;
          diag.array() += epsilon;
        }
        epsilon *= 3.0;
      } else {
        epsilon = 1e-8;
      }
      solver.factorFrom(matData.data() - offset, paramStart);
      if (std::isfinite(
              Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>(matData.data(), size).sum())) {
        std::cout << "Success!" << std::endl;
        break;
      }
    }
  }

  virtual void operator()(double* outVec, const double* inVec) override {
    Eigen::Vector<float, Eigen::Dynamic> temp =
        Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>(inVec, vecSize).cast<float>();

    solver.solveLFrom(  //
        matData.data() - solver.spanMatrixOffset(paramStart), paramStart,
        temp.data() - solver.spanVectorOffset(paramStart), vecSize, 1);
    solver.solveLtFrom(  //
        matData.data() - solver.spanMatrixOffset(paramStart), paramStart,
        temp.data() - solver.spanVectorOffset(paramStart), vecSize, 1);

    Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(outVec, vecSize) = temp.cast<double>();
  }

 private:
  const BaSpaCho::Solver& solver;
  int64_t vecSize;
  int64_t paramStart;
  std::vector<float> matData;
};
