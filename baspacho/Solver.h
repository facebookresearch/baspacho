
#include <memory>

#include "CoalescedBlockMatrix.h"
#include "MatOps.h"
#include "SparseStructure.h"

struct Solver {
    Solver(CoalescedBlockMatrixSkel&& factorSkel,
           std::vector<uint64_t>&& elimLumpRanges, OpsPtr&& ops);

    void solveL(const double* matData, double* vecData, int stride,
                int nRHS) const;

    void solveLt(const double* matData, double* vecData, int stride,
                 int nRHS) const;

    void factor(double* data, bool verbose = false) const;

    void factorXp(double* data, bool verbose = false) const;

    void factorXp2(double* data, bool verbose = false) const;

    void initElimination();

    uint64_t boardElimTempSize(uint64_t lump, uint64_t boardIndexInSN) const;

    void factorLump(double* data, uint64_t lump) const;

    void eliminateBoard(double* data, uint64_t ptr, OpaqueData& ax) const;

    void eliminateBoardBatch(double* data, uint64_t ptr, uint64_t batchSize,
                             OpaqueData& ax) const;

    CoalescedBlockMatrixSkel factorSkel;
    std::vector<uint64_t> elimLumpRanges;

    OpsPtr ops;
    OpaqueDataPtr opMatrixSkel;
    std::vector<OpaqueDataPtr> opElimination;
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