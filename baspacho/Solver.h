
#include <memory>

#include "BlockMatrix.h"
#include "MatOps.h"
#include "SparseStructure.h"

struct Solver {
    Solver(BlockMatrixSkel&& skel, std::vector<uint64_t>&& elimLumps,
           OpsPtr&& ops);

    void factorLump(double* data, uint64_t lump) const;

    void factor(double* data, bool verbose = false) const;

    void eliminateBoard(double* data, uint64_t lump, uint64_t boardIndexInCol,
                        OpaqueData& ax) const;

    void assemble(double* data, uint64_t lump, uint64_t boardIndexInCol,
                  OpaqueData& ax) const;

    BlockMatrixSkel skel;
    std::vector<uint64_t> elimLumps;
    OpsPtr ops;

    OpaqueDataPtr opMatrixSkel;
    std::vector<OpaqueDataPtr> opElimination;

    mutable uint64_t assembleCalls = 0;
    mutable double assembleTotTime = 0.0;
    mutable double assembleLastCallTime;
    mutable double assembleMaxCallTime = 0.0;
};

using SolverPtr = std::unique_ptr<Solver>;

SolverPtr createSolver(const std::vector<uint64_t>& paramSize,
                       const SparseStructure& ss, bool verbose = false);

SolverPtr createSolverSchur(const std::vector<uint64_t>& paramSize,
                            const SparseStructure& ss,
                            const std::vector<uint64_t>& elimLumps,
                            bool verbose = false);