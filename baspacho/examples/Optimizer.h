
#include <cxxabi.h>

#include <sophus/se3.hpp>
#include <Eigen/Geometry>
#include <iostream>
#include <memory>
#include <typeindex>
#include <unordered_set>

#include "baspacho/baspacho/Solver.h"
#include "baspacho/examples/SoftLoss.h"
#include "baspacho/testing/TestingUtils.h" // temporary

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

  static double* dataPtr(Eigen::Vector<double, N>& value) {
    return value.data();
  }

  static const double* dataPtr(const Eigen::Vector<double, N>& value) {
    return value.data();
  }
};

template <>
struct VarUtil<Sophus::SE3d> {
  static constexpr int DataDim = 7;
  static constexpr int TangentDim = 6;

  template <typename VecType>
  static void tangentStep(const VecType& step, Sophus::SE3d& value) {
    value = Sophus::SE3d::exp(step) * value;
  }

  static double* dataPtr(Sophus::SE3d& value) {
    return value.data();
  }

  static const double* dataPtr(const Sophus::SE3d& value) {
    return value.data();
  }
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

  void setConstant(bool constant) {
    index = constant ? kConstantVar : kUnsetIndex;
  }

  bool isOptimized() const {
    return index >= 0;
  }

  DataType value;
  uint64_t index = kUnsetIndex;
};

// var store
class VariableStoreBase {
 public:
  virtual ~VariableStoreBase() {}

  virtual void applyStep(
      const Eigen::VectorXd& step,
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

  virtual int64_t totalSize() const override {
    return Variable::DataDim * variables.size();
  }

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

  virtual void applyStep(
      const Eigen::VectorXd& step,
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

  virtual double computeGradHess(
      double* gradData,
      const BaSpaCho::PermutedCoalescedAccessor& acc,
      double* hessData) = 0;

  virtual void registerVariables(
      std::vector<int64_t>& sizes,
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
    ErrorType err = std::apply( // expanding argument pack...
        [&](auto&&... args) { return factor(args->value..., (args, nullptr)...); },
        args);
    return loss->val(err.squaredNorm());
  }

  double computeSingleGradHess(
      int64_t i,
      double* gradData,
      const BaSpaCho::PermutedCoalescedAccessor& acc,
      double* hessData) {
    auto [factor, loss, args] = boundFactors[i];

    std::tuple<Eigen::Matrix<double, ErrorSize, Variables::TangentDim>...> jacobians;
    ErrorType err = withTuple<0, sizeof...(Variables)>( // expanding argument pack...
        [&](auto... iWraps) {
          return factor(
              std::get<decltype(iWraps)::value>(args)->value...,
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
      acc.diagBlock(hessData, iIndex) += dSoftErr * (iJac.transpose() * iJac);

      // gradient contribution
      static constexpr int iTangentDim =
          std::remove_reference_t<decltype(*std::get<i>(args))>::TangentDim;
      int64_t paramStart = acc.paramStart(iIndex);
      Eigen::Map<Eigen::Vector<double, iTangentDim>>(gradData + paramStart) +=
          dSoftErr * (err.transpose() * iJac);

      forEach<0, i>([&, this](auto jWrap) {
        static constexpr int j = decltype(jWrap)::value;
        int64_t jIndex = std::get<j>(args)->index;
        if (jIndex == kConstantVar) {
          return;
        }

        // Hessian off-diagonal contribution
        auto const& jJac = std::get<j>(jacobians);
        acc.block(hessData, iIndex, jIndex) += dSoftErr * (iJac.transpose() * jJac);
      });
    });

    return softErr;
  }

  virtual double computeCost() override {
    double retv = 0;
    for (size_t i = 0; i < boundFactors.size(); i++) {
      retv += computeSingleCost(i);
    }
    return retv;
  }

  virtual double computeGradHess(
      double* gradData,
      const BaSpaCho::PermutedCoalescedAccessor& acc,
      double* hessData) override {
    double retv = 0;
    for (size_t i = 0; i < boundFactors.size(); i++) {
      retv += computeSingleGradHess(i, gradData, acc, hessData);
    }
    return retv;
  }

  virtual void registerVariables(
      std::vector<int64_t>& sizes,
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
  std::vector<int64_t> paramSize;
  std::vector<int64_t> elimRanges;
  TrivialLoss defaultLoss;

  template <typename Factor, typename SoftLoss, typename... Variables>
  std::enable_if_t<std::is_base_of_v<Loss, SoftLoss>>
  addFactor(Factor&& f, const SoftLoss& l, Variables&... v) {
    static_assert((std::is_base_of_v<VarBase, Variables> && ...));
    auto& store = factorStores.get<FactorStore<Factor, SoftLoss, Variables...>>();
    store.boundFactors.emplace_back(std::move(f), &l, std::make_tuple(&v...));
  }

  template <typename Factor, typename Variable0, typename... Variables>
  std::enable_if_t<!std::is_base_of_v<Loss, Variable0>>
  addFactor(Factor&& f, Variable0& v0, Variables&... v) {
    static_assert(
        (std::is_base_of_v<VarBase, Variable0> && ... && std::is_base_of_v<VarBase, Variables>));
    auto& store = factorStores.get<FactorStore<Factor, TrivialLoss, Variable0, Variables...>>();
    store.boundFactors.emplace_back(std::move(f), &defaultLoss, std::make_tuple(&v0, &v...));
  }

  template <typename Variable>
  void registerVariable(Variable& v) {
    if (v.index == kUnsetIndex) {
      v.index = paramSize.size();
      paramSize.push_back(Variable::TangentDim);
      variableStores.get<VariableStore<Variable>>().variables.push_back(&v);
    }
  }

  void registeredVariablesToEliminationRange() {
    if (elimRanges.empty()) {
      elimRanges.push_back(0);
    }
    elimRanges.push_back(paramSize.size());
  }

  double computeGradHess(
      double* gradData,
      const BaSpaCho::PermutedCoalescedAccessor& acc,
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

  void addDamping(
      Eigen::VectorXd& hess,
      const BaSpaCho::PermutedCoalescedAccessor& acc,
      int64_t nVars,
      double lambda) {
    for (int64_t i = 0; i < nVars; i++) {
      auto diag = acc.diagBlock(hess.data(), i).diagonal();
      diag *= (1.0 + lambda);
    }
  }

  void optimize() {
    // collect variable sizes and lower off-diagonal blocks
    std::unordered_set<std::pair<int64_t, int64_t>, pair_hash> blockSet;
    for (auto& [ti, fStore] : factorStores.stores) {
      fStore->registerVariables(paramSize, blockSet, variableStores);
    }
    std::vector<std::pair<int64_t, int64_t>> blocks(blockSet.begin(), blockSet.end());
    std::sort(blocks.begin(), blocks.end());

    // create a csr block structure
    std::vector<int64_t> ptrs{0}, inds;
    int64_t curRow = 0;
    for (auto [row, col] : blocks) {
      while (curRow < row) {
        inds.push_back(curRow); // diagonal
        ptrs.push_back(inds.size());
        curRow++;
      }
      inds.push_back(col);
    }
    while (curRow < paramSize.size()) {
      inds.push_back(curRow); // diagonal
      ptrs.push_back(inds.size());
      curRow++;
    }

    // create sparse linear solver
    BaSpaCho::SolverPtr solver = createSolver(
        {}, paramSize, BaSpaCho::SparseStructure(std::move(ptrs), std::move(inds)), elimRanges);
    auto accessor = solver->accessor();

    // linear system data
    std::vector<double> variablesBackup(variableBackupSize());
    Eigen::VectorXd grad(solver->order());
    Eigen::VectorXd hess(solver->dataSize());
    Eigen::VectorXd step(solver->order());

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