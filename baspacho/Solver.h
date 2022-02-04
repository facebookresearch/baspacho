
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

    void solveL(const double* matData, double* vecData, int64_t stride,
                int nRHS) const;

    void solveLt(const double* matData, double* vecData, int64_t stride,
                 int nRHS) const;

    void factor(double* data, bool verbose = false) const;

    void factorXp(double* data, bool verbose = false) const;

    void factorXp2(double* data, bool verbose = false) const;

    void initElimination();

    int64_t boardElimTempSize(int64_t lump, int64_t boardIndexInSN) const;

    void factorLump(NumericCtx<double>& numCtx, double* data,
                    int64_t lump) const;

    void eliminateBoard(NumericCtx<double>& numCtx, double* data,
                        int64_t ptr) const;

    void eliminateBoardBatch(NumericCtx<double>& numCtx, double* data,
                             int64_t ptr, int64_t batchSize) const;

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