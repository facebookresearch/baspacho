
#include <memory>

#include "BlockMatrix.h"
#include "MatOps.h"
#include "SparseStructure.h"

struct Settings {
    bool findSparseEliminationRanges = true;
};

struct Solver {
    Solver(BlockMatrixSkel&& skel, std::vector<uint64_t>&& elimLumpRanges,
           OpsPtr&& ops);

    void initElimination();

    uint64_t boardElimTempSize(uint64_t lump, uint64_t boardIndexInSN) const;

    void solveL(const double* matData, double* vecData, int stride,
                int nRHS) const;

    void solveLt(const double* matData, double* vecData, int stride,
                 int nRHS) const;

    void factor(double* data, bool verbose = false) const;

    void factorXp(double* data, bool verbose = false) const;

    void factorXp2(double* data, bool verbose = false) const;

    void factorLump(double* data, uint64_t lump) const;

    void eliminateBoard(double* data, uint64_t ptr, OpaqueData& ax) const;

    void eliminateBoardBatch(double* data, uint64_t ptr, uint64_t batchSize,
                             OpaqueData& ax) const;

    BlockMatrixSkel skel;
    std::vector<uint64_t> elimLumpRanges;
    OpsPtr ops;

    OpaqueDataPtr opMatrixSkel;
    std::vector<OpaqueDataPtr> opElimination;

    std::vector<uint64_t> startRowElimPtr;
    uint64_t maxElimTempSize;
};

using SolverPtr = std::unique_ptr<Solver>;

SolverPtr createSolver(const Settings& settings,
                       const std::vector<uint64_t>& paramSize,
                       const SparseStructure& ss, bool verbose = false);

SolverPtr createSolverSchur(const Settings& settings,
                            const std::vector<uint64_t>& paramSize,
                            const SparseStructure& ss,
                            const std::vector<uint64_t>& elimLumpRanges,
                            bool verbose = false);