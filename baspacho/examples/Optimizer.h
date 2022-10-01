/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <dispenso/parallel_for.h>
#include <sophus/se3.hpp>
#include <Eigen/Geometry>
#include <iostream>
#include <memory>
#include <unordered_set>
#include "baspacho/baspacho/Solver.h"
#include "baspacho/examples/AtomicOps.h"
#include "baspacho/examples/PCG.h"
#include "baspacho/examples/Preconditioner.h"
#include "baspacho/examples/SoftLoss.h"
#include "baspacho/examples/Utils.h"

// C++ utils for variadic templates
template <int i>
struct IntWrap {
  static constexpr int value = i;
};

template <int i, int j, typename F>
void forEach(F&& f) {
  if constexpr (i < j) {
    f(IntWrap<i>());
    forEach<i + 1, j>(f);
  }
}

template <int i, int j, typename F, typename... Args>
decltype(auto) withTuple(F&& f, Args&&... args) {
  if constexpr (i < j) {
    return withTuple<i + 1, j>(f, args..., IntWrap<i>());
  } else {
    return f(args...);
  }
}

// typed storage
template <typename Base>
struct TypedStore {
  template <typename Derived>
  Derived& get() {
    static const std::type_index ti(typeid(Derived));
    std::unique_ptr<Base>& pStoreT = stores[ti];
    if (!pStoreT) {
      pStoreT.reset(new Derived);
      std::cout << "New [" << prettyTypeName<Base>() << "]: " << prettyTypeName<Derived>()
                << std::endl;
    }
    return dynamic_cast<Derived&>(*pStoreT);
  }

  std::unordered_map<std::type_index, std::unique_ptr<Base>> stores;
};

// Utils for variable types
template <typename T>
struct VarUtil;

template <int N>
struct VarUtil<Eigen::Vector<double, N>> {
  static constexpr int DataDim = N;
  static constexpr int TangentDim = N;

  template <typename VecType>
  static void tangentStep(const VecType& step, Eigen::Vector<double, N>& value) {
    value += step;
  }

  static double* dataPtr(Eigen::Vector<double, N>& value) { return value.data(); }

  static const double* dataPtr(const Eigen::Vector<double, N>& value) { return value.data(); }
};

template <>
struct VarUtil<Sophus::SE3d> {
  static constexpr int DataDim = 7;
  static constexpr int TangentDim = 6;

  template <typename VecType>
  static void tangentStep(const VecType& step, Sophus::SE3d& value) {
    value = Sophus::SE3d::exp(step) * value;
  }

  static double* dataPtr(Sophus::SE3d& value) { return value.data(); }

  static const double* dataPtr(const Sophus::SE3d& value) { return value.data(); }
};

// Generic variable class
static constexpr uint64_t kUnsetIndex = -1;
static constexpr uint64_t kConstantVar = -2;

class VarBase {};

template <typename DType>
class Variable : public VarBase {
 public:
  using DataType = DType;
  static constexpr int DataDim = VarUtil<DataType>::DataDim;
  static constexpr int TangentDim = VarUtil<DataType>::TangentDim;

  void setConstant(bool constant) { index = constant ? kConstantVar : kUnsetIndex; }

  bool isOptimized() const { return index >= 0; }

  template <typename... Args>
  Variable(Args&&... args) : value(std::forward<Args>(args)...) {}

  DataType value;
  uint64_t index = kUnsetIndex;
};

// var store
class VariableStoreBase {
 public:
  virtual ~VariableStoreBase() {}

  virtual void applyStep(const Eigen::VectorXd& step,
                         const BaSpaCho::PermutedCoalescedAccessor& acc) = 0;

  virtual int64_t totalSize() const = 0;

  virtual double* backup(double* data) const = 0;

  virtual const double* restore(const double* data) = 0;
};

template <typename Variable>
class VariableStore : public VariableStoreBase {
  static_assert(std::is_base_of_v<VarBase, Variable>);

 public:
  virtual ~VariableStore() override {}

  virtual int64_t totalSize() const override { return Variable::DataDim * variables.size(); }

  virtual double* backup(double* data) const override {
    for (auto var : variables) {
      Eigen::Map<Eigen::Vector<double, Variable::DataDim>> dst(data);
      Eigen::Map<const Eigen::Vector<double, Variable::DataDim>> src(
          VarUtil<typename Variable::DataType>::dataPtr(var->value));
      dst = src;
      data += Variable::DataDim;
    }
    return data;
  }

  virtual const double* restore(const double* data) override {
    for (auto var : variables) {
      Eigen::Map<Eigen::Vector<double, Variable::DataDim>> dst(
          VarUtil<typename Variable::DataType>::dataPtr(var->value));
      Eigen::Map<const Eigen::Vector<double, Variable::DataDim>> src(data);
      dst = src;
      data += Variable::DataDim;
    }
    return data;
  }

  virtual void applyStep(const Eigen::VectorXd& step,
                         const BaSpaCho::PermutedCoalescedAccessor& acc) override {
    for (auto var : variables) {
      VarUtil<typename Variable::DataType>::tangentStep(
          step.segment<Variable::TangentDim>(acc.paramStart(var->index)), var->value);
    }
  }

  std::vector<Variable*> variables;
};

struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

// factor store
class FactorStoreBase {
 public:
  virtual ~FactorStoreBase() {}

  virtual bool verifyJacobians(double epsilon, double maxRelativeError) = 0;

  virtual double computeCost(dispenso::ThreadPool* threadPool = nullptr) = 0;

  virtual double computeGradHess(double* gradData, const BaSpaCho::PermutedCoalescedAccessor& acc,
                                 double* hessData, dispenso::ThreadPool* threadPool = nullptr) = 0;

  virtual void registerVariables(std::vector<int64_t>& sizes,
                                 std::unordered_set<std::pair<int64_t, int64_t>, pair_hash>& blocks,
                                 TypedStore<VariableStoreBase>& varStores) = 0;
};

template <typename Factor, bool HasPrecisionMatrix, typename SoftLoss, typename... Variables>
class FactorStore : public FactorStoreBase {
 public:
  using ErrorType = decltype((*(Factor*)nullptr)(Variables().value..., (Variables(), nullptr)...));
  static constexpr int ErrorSize = VarUtil<ErrorType>::TangentDim;
  static constexpr bool HasSoftLoss = !std::is_same<SoftLoss, TrivialLoss>::value;
  using PrecisionMatrix = Eigen::Matrix<double, ErrorSize, ErrorSize>;
  using VarTuple = std::tuple<Variables*...>;
  using TupleType = std::conditional_t<
      HasPrecisionMatrix,
      std::conditional_t<HasSoftLoss,
                         std::tuple<Factor, VarTuple, PrecisionMatrix, const SoftLoss*>,
                         std::tuple<Factor, VarTuple, PrecisionMatrix>>,
      std::conditional_t<HasSoftLoss, std::tuple<Factor, VarTuple, const SoftLoss*>,
                         std::tuple<Factor, VarTuple>>>;

  auto getLoss(int64_t i) const {
    if constexpr (!HasSoftLoss) {
      return TrivialLoss();
    } else {
      return *std::get<(HasPrecisionMatrix ? 3 : 2)>(boundFactors[i]);
    }
  }

  double squaredError(int64_t i, const ErrorType& err) const {
    if constexpr (HasPrecisionMatrix) {
      return err.dot(std::get<2>(boundFactors[i]) * err);
    } else {
      return err.squaredNorm();
    }
  }

  template <int N>
  Eigen::Matrix<double, ErrorSize, N> precisionScaled(
      int64_t i, const Eigen::Matrix<double, ErrorSize, N>& t) const {
    if constexpr (HasPrecisionMatrix) {
      return std::get<2>(boundFactors[i]) * t;
    } else {
      return t;
    }
  }

  virtual ~FactorStore() override {}

  virtual bool verifyJacobians(double epsilon, double maxRelativeError) override {
    std::tuple<Eigen::Vector<double, Variables::TangentDim>...> maxRelErr(
        (Eigen::Vector<double, Variables::TangentDim>::Zero())...);

    const int nCheck = std::min(boundFactors.size(), 100UL);
    bool stop = false;
    for (size_t k = 0; k < nCheck; k++) {
      auto& factor = std::get<0>(boundFactors[k]);
      auto& args = std::get<1>(boundFactors[k]);

      std::tuple<Eigen::Matrix<double, ErrorSize, Variables::TangentDim>...> jacobians;
      ErrorType err = withTuple<0, sizeof...(Variables)>(  // expanding argument pack...
          [&](auto... iWraps) {
            return factor(std::get<decltype(iWraps)::value>(args)->value...,
                          (&std::get<decltype(iWraps)::value>(jacobians))...);
          });

      forEach<0, sizeof...(Variables)>([&](auto iWrap) {
        static constexpr int i = decltype(iWrap)::value;
        using iVarType = std::remove_reference_t<decltype(*std::get<i>(args))>;
        iVarType& iVar = *std::get<i>(args);
        using iVarUtil = VarUtil<typename iVarType::DataType>;
        static constexpr int iTangentDim = iVarType::TangentDim;
        static constexpr int iDataDim = iVarType::DataDim;
        Eigen::Vector<double, iDataDim> iVarBackup(iVarUtil::dataPtr(iVar.value));
        const auto& iJac = std::get<i>(jacobians);
        Eigen::Matrix<double, ErrorSize, iTangentDim> iNumJac;

        double paramMaxRelErrs = 0;
        int maxRelErrCol = -1;
        for (int t = 0; t < iTangentDim; t++) {
          Eigen::Vector<double, iTangentDim> tgStep = Eigen::Vector<double, iTangentDim>::Zero();
          tgStep[t] = epsilon;
          iVarUtil::tangentStep(tgStep, iVar.value);

          ErrorType pErr = std::apply(  // expanding argument pack...
              [&](auto&&... args) { return factor(args->value..., ((void)args, nullptr)...); },
              args);
          iNumJac.col(t) = (pErr - err) / epsilon;
          double relErr = (iNumJac.col(t) - iJac.col(t)).norm() / (iNumJac.col(t).norm() + epsilon);
          std::get<i>(maxRelErr)[t] = std::max(std::get<i>(maxRelErr)[t], relErr);
          if (relErr > paramMaxRelErrs) {
            paramMaxRelErrs = relErr;
            maxRelErrCol = t;
          }

          Eigen::Map<Eigen::Vector<double, iDataDim>>(iVarUtil::dataPtr(iVar.value)) =
              iVarBackup;  // restore
        }
        if (paramMaxRelErrs > maxRelativeError) {
          std::cout << "Factor" << k << ".Jac" << i << ":\n"
                    << iJac << "\nwhile numeric Jacobian is\n"
                    << iNumJac << "\n and has relative error " << paramMaxRelErrs << " > "
                    << maxRelativeError << " in column " << maxRelErrCol << std::endl;
          stop = true;
        }
      });

      if (stop) {
        break;
      }
    }

    std::cout << "Factor " << prettyTypeName<FactorStore>() << ", factors checked: " << nCheck
              << "/" << boundFactors.size() << ", Jacobians check! (rel errors < "
              << maxRelativeError << ")\n";
    forEach<0, sizeof...(Variables)>([&](auto iWrap) {
      static constexpr int i = decltype(iWrap)::value;
      std::cout << "Relative errors in cols of " << i << "-th Jacobian:\n  "
                << std::get<i>(maxRelErr).transpose() << std::endl;
    });

    return !stop;
  }

  double computeSingleCost(int64_t i) {
    auto& factor = std::get<0>(boundFactors[i]);
    auto& args = std::get<1>(boundFactors[i]);
    const auto& loss = getLoss(i);

    ErrorType err = std::apply(  // expanding argument pack...
        [&](auto&&... args) { return factor(args->value..., ((void)args, nullptr)...); }, args);
    return loss.val(squaredError(i, err)) * 0.5;
  }

  template <typename Ops>
  double computeSingleGradHess(int64_t k, double* gradData,
                               const BaSpaCho::PermutedCoalescedAccessor& acc, double* hessData) {
    auto& factor = std::get<0>(boundFactors[k]);
    auto& args = std::get<1>(boundFactors[k]);
    const auto& loss = getLoss(k);

    std::tuple<Eigen::Matrix<double, ErrorSize, Variables::TangentDim>...> jacobians;
    ErrorType err = withTuple<0, sizeof...(Variables)>(  // expanding argument pack...
        [&](auto... iWraps) {
          return factor(std::get<decltype(iWraps)::value>(args)->value...,
                        (std::get<decltype(iWraps)::value>(args)->index == kConstantVar
                             ? nullptr
                             : &std::get<decltype(iWraps)::value>(jacobians))...);
        });
    auto [softErr, dSoftErr_] = loss.jet2(err.squaredNorm());
    auto& dSoftErr = dSoftErr_;

    forEach<0, sizeof...(Variables)>([&](auto iWrap) {
      static constexpr int i = decltype(iWrap)::value;
      int64_t iIndex = std::get<i>(args)->index;
      if (iIndex == kConstantVar) {
        return;
      }

      static constexpr int iTangentDim =
          std::remove_reference_t<decltype(*std::get<i>(args))>::TangentDim;
      const auto& iJac = std::get<i>(jacobians);
      const Eigen::Matrix<double, ErrorSize, iTangentDim> iAdjJac =
          dSoftErr * precisionScaled(k, iJac);

      // gradient contribution
      int64_t paramStart = acc.paramStart(iIndex);
      Eigen::Map<Eigen::Vector<double, iTangentDim>> gradSeg(gradData + paramStart);
      Eigen::Vector<double, iTangentDim> gradSegAdd = err.transpose() * iAdjJac;
      Ops::vectorAdd(gradSeg, gradSegAdd);

      // Hessian diagonal contribution
      auto dBlock = acc.diagBlock(hessData, iIndex);
      Eigen::Matrix<double, iTangentDim, iTangentDim> dBlockAdd = iAdjJac.transpose() * iJac;
      Ops::matrixAdd(dBlock, dBlockAdd);

      forEach<0, i>([&](auto jWrap) {
        static constexpr int j = decltype(jWrap)::value;
        int64_t jIndex = std::get<j>(args)->index;
        if (jIndex == kConstantVar) {
          return;
        }

        // Hessian off-diagonal contribution
        static constexpr int jTangentDim =
            std::remove_reference_t<decltype(*std::get<j>(args))>::TangentDim;
        const auto& jJac = std::get<j>(jacobians);
        auto odBlock = acc.block(hessData, iIndex, jIndex);
        Eigen::Matrix<double, iTangentDim, jTangentDim> odBlockAdd = iAdjJac.transpose() * jJac;
        Ops::matrixAdd(odBlock, odBlockAdd);
      });
    });

    return softErr * 0.5;
  }

  virtual double computeCost(dispenso::ThreadPool* threadPool = nullptr) override {
    if (threadPool) {  // multi-threaded
      std::vector<double> perThreadCost;
      dispenso::TaskSet taskSet(*threadPool);
      dispenso::parallel_for(
          taskSet, perThreadCost, []() -> double { return 0.0; },
          dispenso::makeChunkedRange(0L, boundFactors.size(),
                                     boundFactors.size() / threadPool->numThreads()),
          [this](double& threadCost, int64_t iBegin, int64_t iEnd) {
            for (int64_t i = iBegin; i < iEnd; i++) {
              threadCost += computeSingleCost(i);
            }
          });
      return Eigen::Map<Eigen::VectorXd>(perThreadCost.data(), perThreadCost.size()).sum();
    } else {  // single-threaded
      double retv = 0;
      for (size_t i = 0; i < boundFactors.size(); i++) {
        retv += computeSingleCost(i);
      }
      return retv;
    }
  }

  virtual double computeGradHess(double* gradData, const BaSpaCho::PermutedCoalescedAccessor& acc,
                                 double* hessData,
                                 dispenso::ThreadPool* threadPool = nullptr) override {
    if (threadPool) {  // multi-threaded
      std::vector<double> perThreadCost;
      dispenso::TaskSet taskSet(*threadPool);
      dispenso::parallel_for(
          taskSet, perThreadCost, []() -> double { return 0.0; },
          dispenso::makeChunkedRange(0L, boundFactors.size()),
          [gradData, &acc, hessData, this](double& threadCost, int64_t iBegin, int64_t iEnd) {
            for (int64_t i = iBegin; i < iEnd; i++) {
              threadCost += computeSingleGradHess<LockedSharedOps>(i, gradData, acc, hessData);
            }
          });
      return Eigen::Map<Eigen::VectorXd>(perThreadCost.data(), perThreadCost.size()).sum();
    } else {  // single-threaded
      double retv = 0;
      for (size_t i = 0; i < boundFactors.size(); i++) {
        retv += computeSingleGradHess<PlainOps>(i, gradData, acc, hessData);
      }
      return retv;
    }
  }

  virtual void registerVariables(std::vector<int64_t>& sizes,
                                 std::unordered_set<std::pair<int64_t, int64_t>, pair_hash>& blocks,
                                 TypedStore<VariableStoreBase>& varStores) override {
    forEach<0, sizeof...(Variables)>([&, this](auto iWrap) {
      static constexpr int i = decltype(iWrap)::value;
      using Variable =
          std::remove_reference_t<decltype(*std::get<i>(std::get<1>(boundFactors[0])))>;
      auto& store = varStores.get<VariableStore<Variable>>();

      for (auto& tup : boundFactors) {
        auto& args = std::get<1>(tup);

        // register variable (if needed)
        auto& Vi = *std::get<i>(args);
        if (Vi.index == kUnsetIndex) {
          Vi.index = sizes.size();
          sizes.push_back(Variable::TangentDim);
          store.variables.push_back(&Vi);
        } else if (Vi.index < 0) {
          continue;
        }

        // add off-diagonal block to Hessian structure
        forEach<0, i>([&](auto jWrap) {
          static constexpr int j = decltype(jWrap)::value;
          auto& Vj = *std::get<j>(args);
          if (Vj.index >= 0) {
            int64_t minIndex = std::min(Vi.index, Vj.index);
            int64_t maxIndex = std::max(Vi.index, Vj.index);
            blocks.insert(std::make_pair(maxIndex, minIndex));
          }
        });
      }
    });
  }

  std::vector<TupleType> boundFactors;
};

class Optimizer {
 public:
  using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
  using hrc = std::chrono::high_resolution_clock;

  TypedStore<FactorStoreBase> factorStores;
  TypedStore<VariableStoreBase> variableStores;
  std::vector<int64_t> paramSizes;
  std::vector<int64_t> elimRanges;

  template <typename Factor, typename Derived, typename SoftLoss, typename... Variables,
            std::enable_if_t<std::is_base_of_v<Loss, SoftLoss>, int> q = 0>
  void addFactor(Factor&& f, const Eigen::MatrixBase<Derived>& prec, const SoftLoss& l,
                 Variables&... v) {
    using FactorStoreType = FactorStore<Factor, true, SoftLoss, Variables...>;
    static_assert((std::is_base_of_v<VarBase, Variables> && ...));
    auto& store = factorStores.get<FactorStoreType>();
    store.boundFactors.emplace_back(std::move(f), std::make_tuple(&v...), prec, &l);
  }

  template <typename Factor, typename Derived, typename Variable0, typename... Variables,
            std::enable_if_t<std::is_base_of_v<VarBase, Variable0>, int> q = 0>
  void addFactor(Factor&& f, const Eigen::MatrixBase<Derived>& prec, Variable0& v0,
                 Variables&... v) {
    using FactorStoreType = FactorStore<Factor, true, TrivialLoss, Variable0, Variables...>;
    static_assert((std::is_base_of_v<VarBase, Variables> && ...));
    auto& store = factorStores.get<FactorStoreType>();
    store.boundFactors.emplace_back(std::move(f), std::make_tuple(&v0, &v...), prec);
  }

  template <typename Factor, typename SoftLoss, typename... Variables,
            std::enable_if_t<std::is_base_of_v<Loss, SoftLoss>, int> q = 0>
  void addFactor(Factor&& f, const SoftLoss& l, Variables&... v) {
    using FactorStoreType = FactorStore<Factor, false, SoftLoss, Variables...>;
    static_assert((std::is_base_of_v<VarBase, Variables> && ...));
    auto& store = factorStores.get<FactorStoreType>();
    store.boundFactors.emplace_back(std::move(f), std::make_tuple(&v...), &l);
  }

  template <typename Factor, typename Variable0, typename... Variables,
            std::enable_if_t<std::is_base_of_v<VarBase, Variable0>, int> q = 0>
  void addFactor(Factor&& f, Variable0& v0, Variables&... v) {
    using FactorStoreType = FactorStore<Factor, false, TrivialLoss, Variable0, Variables...>;
    static_assert((std::is_base_of_v<VarBase, Variables> && ...));
    auto& store = factorStores.get<FactorStoreType>();
    store.boundFactors.emplace_back(std::move(f), std::make_tuple(&v0, &v...));
  }

  template <typename Variable>
  void registerVariable(Variable& v) {
    if (v.index == kUnsetIndex) {
      v.index = paramSizes.size();
      paramSizes.push_back(Variable::TangentDim);
      variableStores.get<VariableStore<Variable>>().variables.push_back(&v);
    }
  }

  void registeredVariablesToEliminationRange() {
    if (elimRanges.empty()) {
      elimRanges.push_back(0);
    }
    elimRanges.push_back(paramSizes.size());
  }

  double computeGradHess(double* gradData, const BaSpaCho::PermutedCoalescedAccessor& acc,
                         double* hessData, dispenso::ThreadPool* threadPool = nullptr) {
    double retv = 0.0;
    for (auto& [ti, fStore] : factorStores.stores) {
      retv += fStore->computeGradHess(gradData, acc, hessData, threadPool);
    }

    return retv;
  }

  bool verifyJacobians(double epsilon = 1e-7, double maxRelativeError = 1e-2) {
    for (auto& [ti, fStore] : factorStores.stores) {
      if (!fStore->verifyJacobians(epsilon, maxRelativeError)) {
        return false;
      }
    }
    return true;
  }

  double computeCost(dispenso::ThreadPool* threadPool = nullptr) {
    double retv = 0.0;
    for (auto& [ti, fStore] : factorStores.stores) {
      retv += fStore->computeCost(threadPool);
    }
    return retv;
  }

  int64_t variableBackupSize() {
    int64_t size = 0;
    for (auto& [ti, vStore] : variableStores.stores) {
      size += vStore->totalSize();
    }
    return size;
  }

  void backupVariables(std::vector<double>& data) const {
    double* dataPtr = data.data();
    for (auto& [ti, vStore] : variableStores.stores) {
      dataPtr = vStore->backup(dataPtr);
    }
  }

  void restoreVariables(const std::vector<double>& data) {
    const double* dataPtr = data.data();
    for (auto& [ti, vStore] : variableStores.stores) {
      dataPtr = vStore->restore(dataPtr);
    }
  }

  void applyStep(const Eigen::VectorXd& step, const BaSpaCho::PermutedCoalescedAccessor& acc) {
    for (auto& [ti, vStore] : variableStores.stores) {
      vStore->applyStep(step, acc);
    }
  }

  void addDamping(Eigen::VectorXd& hess, const BaSpaCho::PermutedCoalescedAccessor& acc,
                  int64_t nVars, double lambda) {
    for (int64_t i = 0; i < nVars; i++) {
      auto diag = acc.diagBlock(hess.data(), i).diagonal();
      diag *= (1.0 + lambda);
      diag.array() += lambda;
    }
  }

  enum SolverType {
    Solver_Direct,
    Solver_PCG_Trivial,
    Solver_PCG_Jacobi,
    Solver_PCG_GaussSeidel,
    Solver_PCG_LowerPrecSolve,
  };

  static std::string solverToString(SolverType solverType) {
    switch (solverType) {
      case Solver_Direct:
        return "direct";
      case Solver_PCG_Trivial:
        return "trivial";
      case Solver_PCG_Jacobi:
        return "jacobi";
      case Solver_PCG_GaussSeidel:
        return "gauss-seidel";
      case Solver_PCG_LowerPrecSolve:
        return "lower-prec-solve";
      default:
        return "<unknown>";
    }
  }

  // creates a solver
  BaSpaCho::SolverPtr initSolver(int numThreads, bool fullElim = true) {
    // collect variable sizes and (lower) off-diagonal blocks that need to be set
    std::unordered_set<std::pair<int64_t, int64_t>, pair_hash> blockSet;
    for (auto& [ti, fStore] : factorStores.stores) {
      fStore->registerVariables(paramSizes, blockSet, variableStores);
    }
    std::vector<std::pair<int64_t, int64_t>> blocks(blockSet.begin(), blockSet.end());
    std::sort(blocks.begin(), blocks.end());

    // create a csr structure for the parameter blocks
    std::vector<int64_t> ptrs{0}, inds;
    int64_t curRow = 0;
    for (auto [row, col] : blocks) {
      while (curRow < row) {
        inds.push_back(curRow);  // diagonal
        ptrs.push_back(inds.size());
        curRow++;
      }
      inds.push_back(col);
    }
    while (curRow < paramSizes.size()) {
      inds.push_back(curRow);  // diagonal
      ptrs.push_back(inds.size());
      curRow++;
    }

    // create sparse linear solver
    return createSolver(
        {.numThreads = numThreads,
         .addFillPolicy = (fullElim ? BaSpaCho::AddFillComplete : BaSpaCho::AddFillForGivenElims)},
        paramSizes, BaSpaCho::SparseStructure(std::move(ptrs), std::move(inds)), elimRanges);
  }

  // creates a "solve" function that will either
  // 1. invoke the direct solver
  // 2. apply partial elimination, run PCG, backtrack to a full solution
  std::function<std::string(Eigen::VectorXd&)> solveFunction(BaSpaCho::Solver& solver,
                                                             Eigen::VectorXd& hess,
                                                             SolverType solverType,
                                                             int iterativeStart) {
    if (solverType == Solver_Direct) {
      return [&](Eigen::VectorXd& vec) -> std::string {
        TimePoint start = hrc::now();

        solver.factor(hess.data());
        solver.solve(hess.data(), vec.data(), solver.order(), 1);
        TimePoint end = hrc::now();

        return "direct solver, t=" + timeString(end - start);
      };
    } else {
      std::shared_ptr<Preconditioner<double>> precond;
      if (solverType == Solver_PCG_Trivial) {
        precond = std::make_shared<IdentityPrecond<double>>(solver, iterativeStart);
      } else if (solverType == Solver_PCG_Jacobi) {
        precond = std::make_shared<BlockJacobiPrecond<double>>(solver, iterativeStart);
      } else if (solverType == Solver_PCG_GaussSeidel) {
        precond = std::make_shared<BlockGaussSeidelPrecond<double>>(solver, iterativeStart);
      } else if (solverType == Solver_PCG_LowerPrecSolve) {
        precond = std::make_shared<LowerPrecSolvePrecond<double>>(solver, iterativeStart);
      } else {
        throw std::runtime_error("Unknown preconditioner " + std::to_string((int)solverType));
      }

      int64_t order = solver.order();
      int64_t secStart = solver.spanVectorOffset(iterativeStart);
      int64_t secSize = order - secStart;
      struct PCGStats {
        void reset() {
          nPrecond = nMatVec = 0;
          totPrecondTime = totMatVecTime = 0.0;
        }
        int nPrecond = 0;
        double totPrecondTime = 0.0;
        int nMatVec = 0;
        double totMatVecTime = 0.0;
      };
      auto pcgStats = std::make_shared<PCGStats>();
      auto pcg = std::make_shared<PCG>(
          [=](Eigen::VectorXd& u, const Eigen::VectorXd& v) {
            TimePoint start = hrc::now();
            u.resize(v.size());
            (*precond)(u.data(), v.data());
            pcgStats->nPrecond++;
            pcgStats->totPrecondTime += std::chrono::duration<double>(hrc::now() - start).count();
          },
          [=, &solver, &hess](Eigen::VectorXd& u, const Eigen::VectorXd& v) {
            TimePoint start = hrc::now();
            u.resize(v.size());
            u.setZero();
            solver.addMvFrom(hess.data(), iterativeStart, v.data() - secStart, order,
                             u.data() - secStart, order, 1);
            pcgStats->nMatVec++;
            pcgStats->totMatVecTime += std::chrono::duration<double>(hrc::now() - start).count();
          },
          1e-10, 40, false);

      return [=, &solver, &hess](Eigen::VectorXd& vec) -> std::string {
        TimePoint start = hrc::now();

        solver.factorUpTo(hess.data(), iterativeStart);
        solver.solveLUpTo(hess.data(), iterativeStart, vec.data(), order, 1);
        TimePoint end_elim = hrc::now();

        precond->init(hess.data());
        pcgStats->reset();
        TimePoint end_precond = hrc::now();

        Eigen::VectorXd tmp;
        auto [its, res] = pcg->solve(tmp, vec.segment(secStart, secSize));
        vec.segment(secStart, secSize) = tmp;
        TimePoint end_res_solve = hrc::now();

        solver.solveLtUpTo(hess.data(), iterativeStart, vec.data(), order, 1);
        TimePoint end_backtrack = hrc::now();

        std::stringstream ss;
        ss << "semi-precond(" << solverToString(solverType)
           << "), elim: " << timeString(end_elim - start)
           << ", pcnd: " << timeString(end_precond - end_elim)       //
           << ", iter: " << timeString(end_res_solve - end_precond)  //
           << ", bktr: " << timeString(end_backtrack - end_res_solve)
           << "\n        (precond: " << pcgStats->nPrecond << "x"
           << microsecondsString(pcgStats->totPrecondTime * 1e6 / pcgStats->nPrecond)
           << ", matvec: " << pcgStats->nMatVec << "x"
           << microsecondsString(pcgStats->totMatVecTime * 1e6 / pcgStats->nMatVec)
           << ", iters: " << its << ", residual: " << res << ")";
        return ss.str();
      };
    }
  }

  // settings for optimize function
  struct Settings {
    int maxNumIterations = 50;
    unsigned int numThreads = 8;
    SolverType solverType = Solver_Direct;
  };

  void optimize() { return optimize(Settings()); }

  void optimize(const Settings& settings) {
    // create sparse linear solver
    BaSpaCho::SolverPtr solver = initSolver(
        settings.numThreads,
        (settings.solverType == Solver_Direct || settings.solverType == Solver_PCG_LowerPrecSolve));
    auto accessor = solver->accessor();

    // linear system data
    std::vector<double> variablesBackup(variableBackupSize());
    Eigen::VectorXd grad(solver->order());
    Eigen::VectorXd hess(solver->dataSize()), hessBackup;
    Eigen::VectorXd step(solver->order());
    auto solveFunc = solveFunction(*solver, hess, settings.solverType,
                                   elimRanges.empty() ? 0 : elimRanges.back());
    double damping = 1e-5;

    const double costReductionForImprovement = 0.999;
    const int stopIfNoImprovementFor = 3;
    const int distanceFromTroubledIteration = 3;
    int iterationNum = 1;
    int lastImprovementIteration = 0;  // last iteration we had a significant improvement
    int lastTroubledIteration = 0;
    double initialCost = 0.0, finalCost;

    dispenso::ThreadPool threadPool{settings.numThreads > 1 ? settings.numThreads : 0};

    // iteration loop
    while (true) {
      TimePoint start_it = hrc::now();

      grad.setZero();
      hess.setZero();
      double prevCost = computeGradHess(grad.data(), accessor, hess.data(),
                                        settings.numThreads > 1 ? &threadPool : nullptr);
      TimePoint end_costs = hrc::now();

      finalCost = prevCost;
      if (iterationNum == 1) {
        initialCost = prevCost;
      }

      double modelCostReduction;
      std::string solverReport;
      do {
        hessBackup = hess;
        addDamping(hess, accessor, solver->paramToSpan().size(), damping);

        step = grad;
        solverReport = solveFunc(step);

        // cost reduction that would occur perfectly quadratic
        modelCostReduction = step.dot(grad) * 0.5;

        if (modelCostReduction < 0) {
          std::cout << " ?:# quadratic model failing, retrying..." << std::endl;
          hess = hessBackup;
          damping *= 2.0;
        }
      } while (0);

      step *= -1.0;
      double gradNorm = grad.norm();
      double stepNorm = step.norm();
      if (modelCostReduction < 1e-10) {
        std::cout << " ^.^ converged, cost: " << finalCost
                  << ",  model cost reduction: " << modelCostReduction << std::endl;
        break;
      }

      backupVariables(variablesBackup);
      applyStep(step, accessor);
      double newCost = computeCost(settings.numThreads > 1 ? &threadPool : nullptr);
      double relativeCostReduction = (prevCost - newCost) / modelCostReduction;

      const char* smiley;
      bool wasSignificantImprovement = newCost < costReductionForImprovement * prevCost;
      if (newCost > prevCost) {
        smiley = ":'(";
        damping *= 3;
        restoreVariables(variablesBackup);
        if (damping > 1e8) {
          std::cout << "damping out of range, quadratic model failing?!" << std::endl;
          break;
        }
        lastTroubledIteration = iterationNum;
      } else {
        if (prevCost - newCost > 0.3 * modelCostReduction) {
          smiley = wasSignificantImprovement ? ";-)" : ":-|";
          damping *= 0.7;
        } else {
          smiley = ":-/";
          damping *= 1.5;
        }
        finalCost = newCost;
      }

      TimePoint end = hrc::now();

      std::cout << " " << smiley << " cost: " << prevCost << " -> " << newCost << " ("
                << percentageString(newCost / prevCost - 1.0, 2)
                << "), t: " << timeString(end - start_it) << "\n"  //
                << "     n." << iterationNum << "; g/H: " << timeString(end_costs - start_it)
                << ", " << solverReport << "\n"
                << "     lmbd: " << damping
                << ", relRed: " << percentageString(relativeCostReduction)
                << ", |G|: " << gradNorm  //
                << ", |S|: " << stepNorm << std::endl;
      iterationNum++;
      if (wasSignificantImprovement) {
        lastImprovementIteration = iterationNum;
      }
      if (iterationNum >= lastImprovementIteration + stopIfNoImprovementFor &&
          iterationNum >= lastTroubledIteration + distanceFromTroubledIteration) {
        std::cout << " >_< converged! (no significant improvement for " << stopIfNoImprovementFor
                  << " iterations)" << std::endl;
        break;
      } else if (iterationNum >= settings.maxNumIterations) {
        std::cout << " X-| iteration limit reached! (" << settings.maxNumIterations
                  << " iterations)" << std::endl;
        break;
      }
    }
  }
};
