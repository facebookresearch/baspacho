
#include <memory>

#include "BlockMatrix.h"
#include "SparseStructure.h"

struct OpaqueData {
    virtual ~OpaqueData() {}
};

using OpaqueDataPtr = std::unique_ptr<OpaqueData>;

struct Ops {
    // (optionally) allows creation of op-specific global data (eg GPU copies)
    virtual OpaqueDataPtr prepareMatrixSkel(const BlockMatrixSkel& skel) = 0;

    virtual OpaqueDataPtr prepareElimination(const BlockMatrixSkel& skel,
                                             uint64_t aggrStart,
                                             uint64_t aggrEnd) = 0;

    virtual void doElimination(const OpaqueData& skel, double* data,
                               uint64_t aggrStart, uint64_t aggrEnd,
                               const OpaqueData& elimData) = 0;

    virtual void potrf(uint64_t n, double* A) = 0;

    virtual void trsm(uint64_t n, uint64_t k, const double* A, double* B) = 0;

    // C = A * B'
    virtual void gemm(uint64_t m, uint64_t n, uint64_t k, const double* A,
                      const double* B, double* C) = 0;

    // TODO
    // virtual void assemble();
};

using OpsPtr = std::unique_ptr<Ops>;

OpsPtr simpleOps();

struct Solver {
    Solver(BlockMatrixSkel&& skel, std::vector<uint64_t>&& elimRanges,
           OpsPtr ops);

    void factorAggreg(double* data, uint64_t aggreg);

    void factor(double* data);

    BlockMatrixSkel skel;
    std::vector<uint64_t> elimRanges;
    OpsPtr ops;

    OpaqueDataPtr opMatrixSkel;
    std::vector<OpaqueDataPtr> opElimination;
};

using SolverPtr = std::unique_ptr<Solver>;

SolverPtr createSolver(const std::vector<uint64_t>& paramSize,
                       const SparseStructure& ss);