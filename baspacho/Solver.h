
#include <memory>

#include "CoalescedBlockMatrix.h"
#include "MatOps.h"
#include "SparseStructure.h"

struct Solver {
    Solver(CoalescedBlockMatrixSkel&& factorSkel,
           std::vector<uint64_t>&& elimLumpRanges,
           std::vector<uint64_t>&& permutation, OpsPtr&& ops);

    PermutedCoalescedAccessor accessor() const {
        return PermutedCoalescedAccessor(factorSkel.accessor(),
                                         permutation.data());
    }

    void solveL(const double* matData, double* vecData, int stride,
                int nRHS) const;

    void solveLt(const double* matData, double* vecData, int stride,
                 int nRHS) const;

    void factor(double* data, bool verbose = false) const;

    void factorXp(double* data, bool verbose = false) const;

    void factorXp2(double* data, bool verbose = false) const;

    void initElimination();

    uint64_t boardElimTempSize(uint64_t lump, uint64_t boardIndexInSN) const;

    void factorLump(NumericCtx<double>& numCtx, double* data,
                    uint64_t lump) const;

    void eliminateBoard(NumericCtx<double>& numCtx, double* data,
                        uint64_t ptr) const;

    void eliminateBoardBatch(NumericCtx<double>& numCtx, double* data,
                             uint64_t ptr, uint64_t batchSize) const;

    CoalescedBlockMatrixSkel factorSkel;
    std::vector<uint64_t> elimLumpRanges;
    std::vector<uint64_t> permutation;  // *on indices*: v'[p[i]] = v[i];

    OpsPtr ops;
    SymbolicCtxPtr symCtx;
    std::vector<SymElimCtxPtr> elimCtxs;
    std::vector<uint64_t> startElimRowPtr;
    uint64_t maxElimTempSize;
};

using SolverPtr = std::unique_ptr<Solver>;

struct Settings {
    bool findSparseEliminationRanges = true;
    int numThreads = 16;
};

SolverPtr createSolver(const Settings& settings,
                       const std::vector<uint64_t>& paramSize,
                       const SparseStructure& ss, bool verbose = false);

SolverPtr createSolverSchur(const Settings& settings,
                            const std::vector<uint64_t>& paramSize,
                            const SparseStructure& ss,
                            const std::vector<uint64_t>& elimLumpRanges,
                            bool verbose = false);