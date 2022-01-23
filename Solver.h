
#include <memory>

#include "BlockMatrix.h"
#include "MatOps.h"
#include "SparseStructure.h"

struct Solver {
    Solver(BlockMatrixSkel&& skel, std::vector<uint64_t>&& elimLumps,
           OpsPtr&& ops);

    void factorLump(double* data, uint64_t lump) const;

    void factor(double* data) const;

    struct SolverContext {
        std::vector<uint64_t> paramToChainOffset;
        uint64_t stride;
        std::vector<double> tempBuffer;
    };

    void prepareContextForTargetAggreg(uint64_t targetAggreg,
                                       SolverContext& ctx) const;

    void eliminateBoard(double* data, uint64_t lump, uint64_t boardIndexInCol,
                        SolverContext& ctx) const;

    void assemble(double* data, uint64_t lump, uint64_t boardIndexInCol,
                  SolverContext& ctx) const;

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
                       const SparseStructure& ss);