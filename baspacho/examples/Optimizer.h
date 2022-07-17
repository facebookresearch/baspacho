#pragma once

#include <cxxabi.h>
#include <sophus/se3.hpp>
#include <Eigen/Geometry>
#include <iostream>
#include <memory>
#include <typeindex>
#include <unordered_set>
#include "baspacho/baspacho/Solver.h"
#include "baspacho/examples/AtomicOps.h"
#include "baspacho/examples/PCG.h"
#include "baspacho/examples/Preconditioner.h"
#include "baspacho/examples/SoftLoss.h"
#include "baspacho/testing/TestingUtils.h"  // temporary

// C++ utils for variadic templates
template <typename... Vars>
struct Pack;

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

// introspection util
template <typename T>
std::string prettyTypeName() {
  char* c_str = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr);
  std::string retv(c_str);
  free(c_str);
  return retv;
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
      std::cout << "New type: " << prettyTypeName<Derived>() << std::endl;
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
  virtual ~VariableStore() {}

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

  virtual double computeCost() = 0;

  virtual double computeGradHess(double* gradData, const BaSpaCho::PermutedCoalescedAccessor& acc,
                                 double* hessData) = 0;

  virtual void registerVariables(std::vector<int64_t>& sizes,
                                 std::unordered_set<std::pair<int64_t, int64_t>, pair_hash>& blocks,
                                 TypedStore<VariableStoreBase>& varStores) = 0;
};

template <typename Factor, typename SoftLoss, typename... Variables>
class FactorStore : public FactorStoreBase {
 public:
  using ErrorType = decltype((*(Factor*)nullptr)(Variables().value..., (Variables(), nullptr)...));
  static constexpr int ErrorSize = VarUtil<ErrorType>::TangentDim;
  static constexpr bool HasSoftLoss = !std::is_same<SoftLoss, TrivialLoss>::value;

  virtual ~FactorStore() override {}

  double computeSingleCost(int64_t i) {
    auto [factor, loss, args] = boundFactors[i];
    ErrorType err = std::apply(  // expanding argument pack...
        [&](auto&&... args) { return factor(args->value..., (args, nullptr)...); }, args);
    return loss->val(err.squaredNorm()) * 0.5;
  }

  template <typename Ops>
  double computeSingleGradHess(int64_t i, double* gradData,
                               const BaSpaCho::PermutedCoalescedAccessor& acc, double* hessData) {
    auto [factor, loss, args] = boundFactors[i];

    std::tuple<Eigen::Matrix<double, ErrorSize, Variables::TangentDim>...> jacobians;
    ErrorType err = withTuple<0, sizeof...(Variables)>(  // expanding argument pack...
        [&](auto... iWraps) {
          return factor(std::get<decltype(iWraps)::value>(args)->value...,
                        (std::get<decltype(iWraps)::value>(args)->index == kConstantVar
                             ? nullptr
                             : &std::get<decltype(iWraps)::value>(jacobians))...);
        });
    auto [softErr, dSoftErr] = loss->jet2(err.squaredNorm());

    forEach<0, sizeof...(Variables)>([&, this](auto iWrap) {
      static constexpr int i = decltype(iWrap)::value;
      int64_t iIndex = std::get<i>(args)->index;
      if (iIndex == kConstantVar) {
        return;
      }

      // Hessian diagonal contribution
      auto const& iJac = std::get<i>(jacobians);
      auto dBlock = acc.diagBlock(hessData, iIndex);
      Ops::matrixAdd(dBlock, dSoftErr * (iJac.transpose() * iJac));

      // gradient contribution
      static constexpr int iTangentDim =
          std::remove_reference_t<decltype(*std::get<i>(args))>::TangentDim;
      int64_t paramStart = acc.paramStart(iIndex);
      Eigen::Map<Eigen::Vector<double, iTangentDim>> gradSeg(gradData + paramStart);
      Ops::vectorAdd(gradSeg, dSoftErr * (err.transpose() * iJac));

      forEach<0, i>([&, this](auto jWrap) {
        static constexpr int j = decltype(jWrap)::value;
        int64_t jIndex = std::get<j>(args)->index;
        if (jIndex == kConstantVar) {
          return;
        }

        // Hessian off-diagonal contribution
        auto const& jJac = std::get<j>(jacobians);
        auto odBlock = acc.block(hessData, iIndex, jIndex);
        Ops::matrixAdd(odBlock, dSoftErr * (iJac.transpose() * jJac));
      });
    });

    return softErr * 0.5;
  }

  virtual double computeCost() override {
    double retv = 0;
    for (size_t i = 0; i < boundFactors.size(); i++) {
      retv += computeSingleCost(i);
    }
    return retv;
  }

  virtual double computeGradHess(double* gradData, const BaSpaCho::PermutedCoalescedAccessor& acc,
                                 double* hessData) override {
    double retv = 0;
    for (size_t i = 0; i < boundFactors.size(); i++) {
      retv += computeSingleGradHess<PlainOps>(i, gradData, acc, hessData);
    }
    return retv;
  }

  virtual void registerVariables(std::vector<int64_t>& sizes,
                                 std::unordered_set<std::pair<int64_t, int64_t>, pair_hash>& blocks,
                                 TypedStore<VariableStoreBase>& varStores) {
    forEach<0, sizeof...(Variables)>([&, this](auto iWrap) {
      static constexpr int i = decltype(iWrap)::value;
      using Variable =
          std::remove_reference_t<decltype(*std::get<i>(std::get<2>(boundFactors[0])))>;
      auto& store = varStores.get<VariableStore<Variable>>();

      for (auto& [factor, loss, args] : boundFactors) {
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
        forEach<0, i>([&, this](auto jWrap) {
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

  std::vector<std::tuple<Factor, SoftLoss*, std::tuple<Variables*...>>> boundFactors;
};

class Optimizer {
 public:
  TypedStore<FactorStoreBase> factorStores;
  TypedStore<VariableStoreBase> variableStores;
  std::vector<int64_t> paramSizes;
  std::vector<int64_t> elimRanges;
  TrivialLoss defaultLoss;

  template <typename Factor, typename SoftLoss, typename... Variables>
  std::enable_if_t<std::is_base_of_v<Loss, SoftLoss>> addFactor(Factor&& f, const SoftLoss& l,
                                                                Variables&... v) {
    static_assert((std::is_base_of_v<VarBase, Variables> && ...));
    auto& store = factorStores.get<FactorStore<Factor, SoftLoss, Variables...>>();
    store.boundFactors.emplace_back(std::move(f), &l, std::make_tuple(&v...));
  }

  template <typename Factor, typename Variable0, typename... Variables>
  std::enable_if_t<!std::is_base_of_v<Loss, Variable0>> addFactor(Factor&& f, Variable0& v0,
                                                                  Variables&... v) {
    static_assert(
        (std::is_base_of_v<VarBase, Variable0> && ... && std::is_base_of_v<VarBase, Variables>));
    auto& store = factorStores.get<FactorStore<Factor, TrivialLoss, Variable0, Variables...>>();
    store.boundFactors.emplace_back(std::move(f), &defaultLoss, std::make_tuple(&v0, &v...));
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
                         double* hessData) {
    double retv = 0.0;
    for (auto& [ti, fStore] : factorStores.stores) {
      retv += fStore->computeGradHess(gradData, acc, hessData);
    }
    return retv;
  }

  double computeCost() {
    double retv = 0.0;
    for (auto& [ti, fStore] : factorStores.stores) {
      retv += fStore->computeCost();
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
    }
  }

  // creates a solver
  BaSpaCho::SolverPtr initSolver() {
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
    return createSolver({}, paramSizes, BaSpaCho::SparseStructure(std::move(ptrs), std::move(inds)),
                        elimRanges);
  }

  // creates a preconditioner for solving the
  std::unique_ptr<Preconditioner<double>> initPreconditioner(const BaSpaCho::Solver& solver,
                                                             int paramStart,
                                                             const std::string& type) {
    if (paramStart == solver.skel().numSpans()) {
      return std::unique_ptr<Preconditioner<double>>();
    } else if (type == "none") {
      return std::make_unique<IdentityPrecond<double>>(solver, paramStart);
    } else if (type == "jacobi") {
      return std::make_unique<BlockJacobiPrecond<double>>(solver, paramStart);
    } else if (type == "jacobi") {
      return std::make_unique<BlockGaussSeidelPrecond<double>>(solver, paramStart);
    } else if (type == "lower-prec-solve") {
      return std::make_unique<LowerPrecSolvePrecond<double>>(solver, paramStart);
    } else {
      throw std::runtime_error("Unknown preconditioner '" + type + "'");
    }
  }

  std::function<void(Eigen::VectorXd&)> solveFunction(BaSpaCho::Solver& solver,
                                                      Eigen::VectorXd& hess, int iterativeStart,
                                                      const std::string& precondType) {
    if (iterativeStart == solver.skel().numSpans()) {
      return [&](Eigen::VectorXd& vec) -> void {
        solver.factor(hess.data());
        solver.solve(hess.data(), vec.data(), solver.order(), 1);
      };
    } else {
      std::shared_ptr<Preconditioner<double>> precond;
      if (precondType == "none") {
        precond = std::make_shared<IdentityPrecond<double>>(solver, iterativeStart);
      } else if (precondType == "jacobi") {
        precond = std::make_shared<BlockJacobiPrecond<double>>(solver, iterativeStart);
      } else if (precondType == "gauss-seidel") {
        precond = std::make_shared<BlockGaussSeidelPrecond<double>>(solver, iterativeStart);
      } else if (precondType == "lower-prec-solve") {
        precond = std::make_shared<LowerPrecSolvePrecond<double>>(solver, iterativeStart);
      } else {
        throw std::runtime_error("Unknown preconditioner '" + precondType + "'");
      }

      int64_t order = solver.order();
      int64_t secStart = solver.paramVecDataStart(iterativeStart);
      int64_t secSize = order - secStart;
      auto pcg = std::make_shared<PCG>(
          [precond = precond](Eigen::VectorXd& u, const Eigen::VectorXd& v) {
            u.resize(v.size());
            (*precond)(u.data(), v.data());
          },
          [&, order, secStart](Eigen::VectorXd& u, const Eigen::VectorXd& v) {
            u.resize(v.size());
            u.setZero();
            solver.addMvFrom(hess.data(), iterativeStart, v.data() - secStart, order,
                             u.data() - secStart, order, 1);
          },
          1e-10, 40, true);

      return [&, pcg = pcg, order, secStart, secSize](Eigen::VectorXd& vec) -> void {
        solver.factorUpTo(hess.data(), iterativeStart);
        solver.solveLUpTo(hess.data(), iterativeStart, vec.data(), order, 1);

        Eigen::VectorXd tmp;
        pcg->solve(tmp, vec.segment(secStart, secSize));
        vec.segment(secStart, secSize) = tmp;

        solver.solveLtUpTo(hess.data(), iterativeStart, vec.data(), order, 1);
      };
    }
  }

  void optimize() {
    // create sparse linear solver
    BaSpaCho::SolverPtr solver = initSolver();
    auto accessor = solver->accessor();

    // linear system data
    std::vector<double> variablesBackup(variableBackupSize());
    Eigen::VectorXd grad(solver->order());
    Eigen::VectorXd hess(solver->dataSize());
    Eigen::VectorXd step(solver->order());
    auto solveFunc = solveFunction(*solver, hess, solver->skel().numSpans(), "jacobi");

    // iteration loop
    while (true) {
      grad.setZero();
      hess.setZero();
      double currentCost = computeGradHess(grad.data(), accessor, hess.data());
      addDamping(hess, accessor, solver->permutation.size(), 1e-4);

      std::cout << "grad:" << grad.transpose() << std::endl;
      std::vector<double> hessCp(hess.data(), hess.data() + hess.size());
      Eigen::MatrixXd H = solver->factorSkel.densify(hessCp);
      H.triangularView<Eigen::Upper>() = H.triangularView<Eigen::Lower>().transpose();
      std::cout << "hess:\n" << H << std::endl;
      std::cout << "perm:\n" << BaSpaCho::testing::printVec(solver->permutation) << std::endl;

      solver->factor(hess.data());

      step = grad;
      solver->solve(hess.data(), step.data(), step.size(), 1);

      std::cout << "step:" << step.transpose() << std::endl;
      std::cout << "H*step:\n" << (H * step).transpose() << std::endl;

      // cost reduction that would occur perfectly quadratic
      double modelCostReduction = step.dot(grad) * 0.5;
      step *= -1.0;

      backupVariables(variablesBackup);
      applyStep(step, accessor);

      std::cout << "prev vars: " << BaSpaCho::testing::printVec(variablesBackup) << std::endl;

      {
        std::vector<double> vdata(variableBackupSize());
        backupVariables(vdata);
        std::cout << "new vars: " << BaSpaCho::testing::printVec(vdata) << std::endl;
      }
      // restoreVariables(variablesBackup);

      double newCost = computeCost();
      std::cout << "Cost: " << currentCost << " -> " << newCost << std::endl;

      break;
    }
  }
};