
#include <memory>

#include "BlockMatrix.h"
#include "MatOps.h"
#include "SparseStructure.h"

struct Solver {
    Solver(BlockMatrixSkel&& skel, std::vector<uint64_t>&& elimLumpRanges,
           OpsPtr&& ops);

    void factor(double* data, bool verbose = false) const;

    void factorLump(double* data, uint64_t lump) const;

    void eliminateBoard(double* data, uint64_t lump, uint64_t boardIndexInCol,
                        OpaqueData& ax) const;

    BlockMatrixSkel skel;
    std::vector<uint64_t> elimLumpRanges;
    OpsPtr ops;

    OpaqueDataPtr opMatrixSkel;
    std::vector<OpaqueDataPtr> opElimination;
};

using SolverPtr = std::unique_ptr<Solver>;

SolverPtr createSolver(const std::vector<uint64_t>& paramSize,
                       const SparseStructure& ss, bool verbose = false);

SolverPtr createSolverSchur(const std::vector<uint64_t>& paramSize,
                            const SparseStructure& ss,
                            const std::vector<uint64_t>& elimLumpRanges,
                            bool verbose = false);