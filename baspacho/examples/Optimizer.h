
#include <cxxabi.h>

#include <Eigen/Geometry>
#include <iostream>
#include <memory>
#include <sophus/se3.hpp>
#include <typeindex>
#include <unordered_set>

#include "baspacho/baspacho/Solver.h"

// Utils
template <typename T>
struct VarUtil;

template <int N>
struct VarUtil<Eigen::Vector<double, N>> {
    static constexpr int DataDim = N;
    static constexpr int TangentDim = N;

    static void tangentStep(const Eigen::Vector<double, N>& step,
                            Eigen::Vector<double, N>& value) {
        value += step;
    }

    static double* dataPtr(Eigen::Vector<double, N>& value) {
        return value.data();
    }
};

template <>
struct VarUtil<Sophus::SE3d> {
    static constexpr int DataDim = 7;
    static constexpr int TangentDim = 6;

    static void tangentStep(const Eigen::Vector<double, TangentDim>& step,
                            Sophus::SE3d& value) {
        value = Sophus::SE3d::exp(step) * value;
    }

    static double* dataPtr(Sophus::SE3d& value) { return value.data(); }
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
};

template <typename Variable>
class VariableStore : public VariableStoreBase {
   public:
    virtual ~VariableStore() {}

    std::vector<Variable*> variables;
};

// some utils
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
        std::unordered_map<std::type_index, std::unique_ptr<VariableStoreBase>>&
            varStores) = 0;
};

// introspection shortcuts
template <typename T>
std::string prettyTypeName() {
    char* c_str =
        abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr);
    std::string retv(c_str);
    free(c_str);
    return retv;
}

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

    virtual double computeGradHess(double* gradData, const BaSpaCho::PermutedCoalescedAccessor& acc, double* hessData) override {
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
        std::unordered_map<std::type_index, std::unique_ptr<VariableStoreBase>>&
            varStores) {
        forEach<0, sizeof...(Variables)>([&, this](auto iWrap) {
            static constexpr int i = decltype(iWrap)::value;
            using Variable = std::remove_reference_t<decltype(*std::get<i>(
                boundFactors[0].second))>;

            using VariableStoreT = VariableStore<Variable>;
            static const std::type_index ti(typeid(VariableStoreT));
            std::unique_ptr<VariableStoreBase>& pStoreT = varStores[ti];
            if (!pStoreT) {
                pStoreT.reset(new VariableStoreT);
                std::cout << "New var: " << prettyTypeName<VariableStoreT>()
                          << std::endl;
            }
            VariableStoreT& store = dynamic_cast<VariableStoreT&>(*pStoreT);

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
    std::unordered_map<std::type_index, std::unique_ptr<FactorStoreBase>>
        factorStores;
    std::unordered_map<std::type_index, std::unique_ptr<VariableStoreBase>>
        variableStores;

    template <typename Factor, typename... Variables>
    void addFactor(Factor&& f, Variables&... v) {
        using FactorStoreT = FactorStore<Factor, Variables...>;
        static const std::type_index ti(typeid(FactorStoreT));
        std::unique_ptr<FactorStoreBase>& pStoreT = factorStores[ti];
        if (!pStoreT) {
            pStoreT.reset(new FactorStoreT);
            std::cout << "New fac: " << prettyTypeName<FactorStoreT>()
                      << std::endl;
        }
        FactorStoreT& store = dynamic_cast<FactorStoreT&>(*pStoreT);
        store.boundFactors.emplace_back(std::move(f), std::make_tuple(&v...));
    }

    void optimize() {
        // collect variable sizes and lower off-diagonal blocks
        std::vector<int64_t> paramSize;
        std::unordered_set<std::pair<int64_t, int64_t>, pair_hash> blockSet;
        for (auto& [ti, fStore] : factorStores) {
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
            BaSpaCho::SparseStructure(std::move(ptrs), std::move(inds)), {});
    }
};