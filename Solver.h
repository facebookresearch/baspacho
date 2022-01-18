
#include <memory>

#include "BlockMatrix.h"
#include "MatOps.h"
#include "SparseStructure.h"

struct Solver {
    Solver(BlockMatrixSkel&& skel, std::vector<uint64_t>&& elimRanges,
           OpsPtr ops);

    void factorAggreg(double* data, uint64_t aggreg) const;

    void factor(double* data) const;

    struct SolverContext {
        std::vector<uint64_t> paramToSliceOffset;
        uint64_t stride;
        std::vector<double> tempBuffer;
    };

    void prepareContextForTargetAggreg(uint64_t targetAggreg,
                                       SolverContext& ctx) const;

    void eliminateAggregItem(double* data, uint64_t aggreg,
                             uint64_t slabIndexInCol, SolverContext& ctx) const;

    void assemble(double* data, uint64_t aggreg, uint64_t slabIndexInCol,
                  SolverContext& ctx) const;

    BlockMatrixSkel skel;
    std::vector<uint64_t> elimRanges;
    OpsPtr ops;

    OpaqueDataPtr opMatrixSkel;
    std::vector<OpaqueDataPtr> opElimination;
};

using SolverPtr = std::unique_ptr<Solver>;

SolverPtr createSolver(const std::vector<uint64_t>& paramSize,
                       const SparseStructure& ss);