
#include <cxxabi.h>

#include <Eigen/Geometry>
#include <iostream>
#include <memory>
#include <sophus/se3.hpp>
#include <typeindex>
#include <unordered_set>

#include "baspacho/baspacho/Solver.h"

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
void withTuple(F&& f, Args&&... args) {
    if constexpr (i < j) {
        withTuple<i+1, j>(f, args..., IntWrap<i>());
    } else {
        f(args...);
    }
}

// introspection util
template <typename T>
std::string prettyTypeName() {
    char* c_str =
        abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr);
    std::string retv(c_str);
    free(c_str);
    return retv;
}

// typed storage
template<typename Base>
struct TypedStore {
    template<typename Derived>
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

    static void tangentStep(const Eigen::Map<const Eigen::Vector<double, N>>& step,
                            Eigen::Vector<double, N>& value) {
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

    static void tangentStep(const Eigen::Map<const Eigen::Vector<double, TangentDim>>& step,
                            Sophus::SE3d& value) {
        value = Sophus::SE3d::exp(step) * value;
    }

    static double* dataPtr(Sophus::SE3d& value) { return value.data(); }

    static const double* dataPtr(const Sophus::SE3d& value) { return value.data(); }
};

// Generic variable class
static constexpr uint64_t kUnsetIndex = -1;
static constexpr uint64_t kConstantVar = -2;
template <typename DataType>
class VarComp;

template <typename DType>
class Variable {
   public:
    using DataType = DType;
    static constexpr int DataDim = VarUtil<DataType>::DataDim;
    static constexpr int TangentDim = VarUtil<DataType>::TangentDim;

    void setConstant(bool constant) {
        index = constant ? kConstantVar : kUnsetIndex;
    }

    bool isOptimized() const { return index >= 0; }

    DataType value;
    uint64_t index = kUnsetIndex;
};

// var store
class VariableStoreBase {
   public:
    virtual ~VariableStoreBase() {}

    virtual const double* applyStep(const double* step) = 0;

    virtual int64_t totalSize() const = 0;

    virtual double* backup(double* data) const = 0;

    virtual const double* restore(const double* data) = 0;
};

template <typename Variable>
class VariableStore : public VariableStoreBase {
   public:
    virtual ~VariableStore() {}

    virtual const double* applyStep(const double* step) override {
        for(auto var: variables) {
            VarUtil<typename Variable::DataType>::tangentStep(
                Eigen::Map<const Eigen::Vector<double, Variable::TangentDim>>(step), var->value);
            step += Variable::TangentDim;
        }
        return step;
    }

    virtual int64_t totalSize() const override {
        return Variable::DataDim * variables.size();
    }

    virtual double* backup(double* data) const override {
        for(auto var: variables) {
            Eigen::Map<Eigen::Vector<double, Variable::DataDim>> dst(data); 
            Eigen::Map<const Eigen::Vector<double, Variable::DataDim>> src(
                VarUtil<typename Variable::DataType>::dataPtr(var->value));
            dst = src;
            data += Variable::DataDim;
        }
        return data;
    }

    virtual const double* restore(const double* data) override {
        for(auto var: variables) {
            Eigen::Map<Eigen::Vector<double, Variable::DataDim>> dst(
                VarUtil<typename Variable::DataType>::dataPtr(var->value));
            Eigen::Map<const Eigen::Vector<double, Variable::DataDim>> src(data);
            dst = src;
            data += Variable::DataDim;
        }
        return data;
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

    virtual double computeGradHess(double* gradData, const BaSpaCho::PermutedCoalescedAccessor& acc, double* hessData) = 0;

    virtual void registerVariables(
        std::vector<int64_t>& sizes,
        std::unordered_set<std::pair<int64_t, int64_t>, pair_hash>& blocks,
        TypedStore<VariableStoreBase>& varStores) = 0;
};

template <typename Factor, typename... Variables>
class FactorStore : public FactorStoreBase {
   public:
    using ErrorType = decltype((*(Factor*)nullptr)(Variables().value...,
                                                   (Variables(), nullptr)...));
    static constexpr int ErrorSize = VarUtil<ErrorType>::TangentDim;

    virtual ~FactorStore() override {}

    virtual double computeCost() override {
        double retv = 0;
        for (auto [factor, args] : boundFactors) {
            std::apply(  // expanding argument pack...
                [&](auto&&... args) {
                    ErrorType err = factor(args->value..., (args, nullptr)...);
                    retv += err.squaredNorm();
                },
                args);
        }
        return retv;
    }

    virtual double computeGradHess(double* gradData,
                                   const BaSpaCho::PermutedCoalescedAccessor& acc,
                                   double* hessData) override {
        double retv = 0;
        // TODO: make parallel (w locking where necessary)
        for (auto [factor, args] : boundFactors) {
            ErrorType err;
            std::tuple<Eigen::Matrix<double, ErrorSize, Variables::TangentDim>...> jacobians;
            withTuple<0, sizeof...(Variables)>(  // expanding argument pack...
                [&](auto... iWraps) {
                    err = factor(
                        std::get<decltype(iWraps)::value>(args)->value...,
                        (std::get<decltype(iWraps)::value>(args)->index == kConstantVar
                        ? nullptr : &std::get<decltype(iWraps)::value>(jacobians))...
                    );
                    retv += err.squaredNorm();
                });

            forEach<0, sizeof...(Variables)>([&, this](auto iWrap) {
                static constexpr int i = decltype(iWrap)::value;
                int64_t iIndex = std::get<i>(args)->index;
                if(iIndex == kConstantVar) {
                    return;
                }

                // Hessian diagonal contribution
                auto const& iJac = std::get<i>(jacobians);
                acc.diagBlock(hessData, iIndex) += iJac.transpose() * iJac;

                // gradient contribution
                static constexpr int iTangentDim = std::remove_reference_t<
                    decltype(*std::get<i>(args))>::TangentDim;
                int64_t paramStart = acc.paramStart(iIndex);
                Eigen::Map<Eigen::Vector<double, iTangentDim>>(gradData + paramStart)
                    += err.transpose() * iJac;

                forEach<0, i>([&, this](auto jWrap) {
                    static constexpr int j = decltype(jWrap)::value;
                    int64_t jIndex = std::get<j>(args)->index;
                    if(jIndex == kConstantVar) {
                        return;
                    }

                    // Hessian off-diagonal contribution
                    auto const& jJac = std::get<j>(jacobians);
                    acc.block(hessData, iIndex, jIndex) += iJac.transpose() * jJac;
                });
            });
        }
        return retv;
    }

    virtual void registerVariables(
        std::vector<int64_t>& sizes,
        std::unordered_set<std::pair<int64_t, int64_t>, pair_hash>& blocks,
        TypedStore<VariableStoreBase>& varStores) {
        forEach<0, sizeof...(Variables)>([&, this](auto iWrap) {
            static constexpr int i = decltype(iWrap)::value;
            using Variable = std::remove_reference_t<decltype(*std::get<i>(
                boundFactors[0].second))>;
            auto& store = varStores.get<VariableStore<Variable>>();

            for (auto [factor, args] : boundFactors) {
                auto& Vi = *std::get<i>(args);
                if (Vi.index == kUnsetIndex) {
                    Vi.index = sizes.size();
                    sizes.push_back(Variable::TangentDim);
                    store.variables.push_back(&Vi);
                } else if (Vi.index < 0) {
                    continue;
                }
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

    std::vector<std::pair<Factor, std::tuple<Variables*...>>> boundFactors;
};

class Optimizer {
   public:
    TypedStore<FactorStoreBase> factorStores;
    TypedStore<VariableStoreBase> variableStores;
    std::vector<int64_t> paramSize;
    std::vector<int64_t> elimRanges;

    template <typename Factor, typename... Variables>
    void addFactor(Factor&& f, Variables&... v) {
        auto& store = factorStores.get<FactorStore<Factor, Variables...>>();
        store.boundFactors.emplace_back(std::move(f), std::make_tuple(&v...));
    }

    template<typename Variable>
    void registerVariable(Variable& v) {
        if (v.index == kUnsetIndex) {
            v.index = paramSize.size();
            paramSize.push_back(Variable::TangentDim);
            variableStores.get<VariableStore<Variable>>().variables.push_back(&v);
        }
    }

    void registeredVariablesToEliminationRange() {
        if(elimRanges.empty()) {
            elimRanges.push_back(0);
        }
        elimRanges.push_back(paramSize.size());
    }

    double computeGradHess(double* gradData,
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

    void initVariableBackup(std::vector<double>& data) {
        int64_t size = 0;
        for (auto& [ti, vStore] : variableStores.stores) {
            size += vStore->totalSize();
        }
        data.resize(size);
    }

    void backupVariables(std::vector<double>& data) {
        double* dataPtr = data.data();
        for (auto& [ti, vStore] : variableStores.stores) {
            dataPtr = vStore->backup(dataPtr);
        }
    }

    void optimize() {
        // collect variable sizes and lower off-diagonal blocks
        std::unordered_set<std::pair<int64_t, int64_t>, pair_hash> blockSet;
        for (auto& [ti, fStore] : factorStores.stores) {
            fStore->registerVariables(paramSize, blockSet, variableStores);
        }
        std::vector<std::pair<int64_t, int64_t>> blocks(blockSet.begin(),
                                                        blockSet.end());
        std::sort(blocks.begin(), blocks.end());

        // create a csr block structure
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
        while (curRow < paramSize.size()) {
            inds.push_back(curRow);  // diagonal
            ptrs.push_back(inds.size());
            curRow++;
        }

        // create solver
        BaSpaCho::SolverPtr solver = createSolverSchur(
            {}, paramSize,
            BaSpaCho::SparseStructure(std::move(ptrs), std::move(inds)),
            elimRanges);
        auto accessor = solver->accessor();

        std::vector<double> grad(solver->order());
        std::vector<double> hess(solver->dataSize());

        while(true) {
            computeGradHess(grad.data(), accessor, hess.data());
            break;
        }
    }
};