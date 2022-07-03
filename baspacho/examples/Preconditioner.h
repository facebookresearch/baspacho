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
  IdentityPrecond(BaSpaCho::Solver& solver, int64_t paramStart)
      : solver(solver),
        vecSize(solver.order() - solver.paramVecDataStart(paramStart)) {}

  virtual ~IdentityPrecond() {}

  virtual void init(T* /* data */) override {}

  virtual void operator()(T* outVec, const T* inVec) override {
    Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(outVec, vecSize) =
        Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>(inVec, vecSize);
  }

 private:
  BaSpaCho::Solver& solver;
  int64_t vecSize;
};

/* Jacobi preconditioner, param-sized blocks are inverted */
template <typename T>
class BlockJacobiPrecond : public Preconditioner<T> {
 public:
  using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  BlockJacobiPrecond(BaSpaCho::Solver& solver, int64_t paramStart)
      : solver(solver),
        paramStart(paramStart),
        vecSize(solver.order() - solver.paramVecDataStart(paramStart)) {
    int64_t numParams = solver.factorSkel.spanStart.size() - 1;
    int64_t offset = 0;
    for (int64_t p = paramStart; p < numParams; p++) {
      diagBlockOffset.push_back(offset);
      int64_t pSize =
          solver.paramVecDataStart(p + 1) - solver.paramVecDataStart(p);
      offset += pSize * pSize;
    }
    diagBlockData.resize(offset);
  }

  virtual ~BlockJacobiPrecond() {}

  virtual void init(T* data) override {
    auto acc = solver.accessor();
    for (size_t i = 0; i < diagBlockOffset.size(); i++) {
      int64_t p = i + paramStart;
      int64_t pSize =
          solver.paramVecDataStart(p + 1) - solver.paramVecDataStart(p);
      Eigen::Map<Mat> mat(diagBlockData.data() + diagBlockOffset[i], pSize,
                          pSize);
      mat = acc.plainAcc.diagBlock(data, p);
      Eigen::LLT<Eigen::Ref<Mat>> llt(mat);
    }
  }

  virtual void operator()(T* outVec, const T* inVec) override {
    Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> out(outVec, vecSize);
    Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> in(inVec, vecSize);
    out = in;
    int64_t secStart = solver.paramVecDataStart(paramStart);
    for (size_t i = 0; i < diagBlockOffset.size(); i++) {
      int64_t p = i + paramStart;
      int64_t pStart = solver.paramVecDataStart(p);
      int64_t pSize = solver.paramVecDataStart(p + 1) - pStart;
      int64_t pStartInSec = pStart - secStart;
      Eigen::Map<Mat> mat(diagBlockData.data() + diagBlockOffset[i], pSize,
                          pSize);
      mat.template triangularView<Eigen::Lower>()
          .template solveInPlace<Eigen::OnTheLeft>(
              out.segment(pStartInSec, pSize));
      mat.template triangularView<Eigen::Lower>()
          .transpose()
          .template solveInPlace<Eigen::OnTheLeft>(
              out.segment(pStartInSec, pSize));
    }
  }

 private:
  BaSpaCho::Solver& solver;
  int64_t paramStart;
  int64_t vecSize;
  std::vector<int64_t> diagBlockOffset;
  std::vector<T> diagBlockData;
};

/* Gauss-Seidel preconditioner, param-sized blocks are inverted */
template <typename T>
class BlockGaussSeidelPrecond : public Preconditioner<T> {
 public:
  BlockGaussSeidelPrecond(BaSpaCho::Solver& solver, int64_t paramStart)
      : solver(solver),
        paramStart(paramStart),
        vecSize(solver.order() - solver.paramVecDataStart(paramStart)) {}

  virtual ~BlockGaussSeidelPrecond() {}

  virtual void init(T* data) override {
    matData.assign(data + solver.paramMatDataStart(paramStart),
                   data + solver.dataSize());
    solver.pseudoFactorFrom(
        matData.data() - solver.paramMatDataStart(paramStart), paramStart);
  }

  virtual void operator()(T* outVec, const T* inVec) override {
    Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(outVec, vecSize) =
        Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>(inVec, vecSize);

    solver.solveLFrom(  //
        matData.data() - solver.paramMatDataStart(paramStart), paramStart,
        outVec - solver.paramVecDataStart(paramStart), vecSize, 1);
    solver.solveLtFrom(
        matData.data() - solver.paramMatDataStart(paramStart), paramStart,
        outVec - solver.paramVecDataStart(paramStart), vecSize, 1);
  }

 private:
  BaSpaCho::Solver& solver;
  int64_t paramStart;
  int64_t vecSize;
  std::vector<T> matData;
};