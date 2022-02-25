
#include <cxxabi.h>

#include <Eigen/Geometry>
#include <iostream>
#include <memory>
#include <sophus/se3.hpp>
#include <typeindex>
#include <unordered_set>

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

    void registerInto(std::vector<int64_t>& sizes) {
        if (index == kUnsetIndex) {
            index = sizes.size();
            sizes.push_back(TangentDim);
        }
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
                        blocks.insert(std::make_pair(minIndex, maxIndex));
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
        std::vector<int64_t> sizes;
        std::unordered_set<std::pair<int64_t, int64_t>, pair_hash> blockSet;
        for (auto& [ti, fStore] : factorStores) {
            fStore->registerVariables(sizes, blockSet, variableStores);
        }

        std::vector<std::pair<int64_t, int64_t>> blocks(blockSet.begin(),
                                                        blockSet.end());
        std::sort(blocks.begin(), blocks.end());
    }
};