
#include <memory>

#include "baspacho/CoalescedBlockMatrix.h"
#include "baspacho/MatOps.h"
#include "baspacho/SparseStructure.h"

namespace BaSpaCho {

struct Solver {
    Solver(CoalescedBlockMatrixSkel&& factorSkel,
           std::vector<int64_t>&& elimLumpRanges,
           std::vector<int64_t>&& permutation, OpsPtr&& ops);

    PermutedCoalescedAccessor accessor() const {
        return PermutedCoalescedAccessor(factorSkel.accessor(),
                                         permutation.data());
    }

    void printStats() const;

    void resetStats();

    template <typename T>
    void factor(T* data, bool verbose = false) const;

    template <typename T>
    void solve(const T* matData, T* vecData, int64_t stride, int nRHS) const;

    template <typename T>
    void solveL(const T* matData, T* vecData, int64_t stride, int nRHS) const;

    template <typename T>
    void solveLt(const T* matData, T* vecData, int64_t stride, int nRHS) const;

    int64_t order() const { return factorSkel.order(); }

   private:
    void initElimination();

    int64_t boardElimTempSize(int64_t lump, int64_t boardIndexInSN) const;

    template <typename T>
    void factorLump(NumericCtx<T>& numCtx, T* data, int64_t lump) const;

    template <typename T>
    void eliminateBoard(NumericCtx<T>& numCtx, T* data, int64_t ptr) const;

    template <typename T>
    void internalSolveL(SolveCtx<T>& slvCtx, const T* matData, T* vecData,
                        int64_t stride) const;

    template <typename T>
    void internalSolveLt(SolveCtx<T>& slvCtx, const T* matData, T* vecData,
                         int64_t stride) const;

   public:
    CoalescedBlockMatrixSkel factorSkel;
    std::vector<int64_t> elimLumpRanges;
    std::vector<int64_t> permutation;  // *on indices*: v'[p[i]] = v[i];

    OpsPtr ops;
    SymbolicCtxPtr symCtx;
    std::vector<SymElimCtxPtr> elimCtxs;
    std::vector<int64_t> startElimRowPtr;
    int64_t maxElimTempSize;
};

using SolverPtr = std::unique_ptr<Solver>;

enum BackendType {
    BackendRef,
    BackendBlas,
    BackendCuda,
};

struct Settings {
    bool findSparseEliminationRanges = true;
    int numThreads = 16;
    BackendType backend = BackendBlas;
};

SolverPtr createSolver(const Settings& settings,
                       const std::vector<int64_t>& paramSize,
                       const SparseStructure& ss);

SolverPtr createSolverSchur(const Settings& settings,
                            const std::vector<int64_t>& paramSize,
                            const SparseStructure& ss,
                            const std::vector<int64_t>& elimLumpRanges);

}  // end namespace BaSpaCho