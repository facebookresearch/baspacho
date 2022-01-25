#include <memory>

#include "BlockMatrix.h"

struct OpaqueData {
    virtual ~OpaqueData() {}
};

using OpaqueDataPtr = std::unique_ptr<OpaqueData>;

struct Ops {
    virtual void printStats() const = 0;

    // (optionally) allows creation of op-specific global data (eg GPU copies)
    virtual OpaqueDataPtr prepareMatrixSkel(const BlockMatrixSkel& skel) = 0;

    // prepares data for a parallel elimination op
    virtual OpaqueDataPtr prepareElimination(const BlockMatrixSkel& skel,
                                             uint64_t lumpsBegin,
                                             uint64_t lumpsEnd) = 0;

    // does (possibly parallel) elimination on a lump of aggregs
    virtual void doElimination(const OpaqueData& skel, double* data,
                               uint64_t lumpsBegin, uint64_t lumpsEnd,
                               const OpaqueData& elimData) = 0;

    // dense Cholesky on dense row-major matrix A (in place)
    virtual void potrf(uint64_t n, double* A) = 0;

    // solve: X * A.lowerHalf().transpose() = B (in place, B becomes X)
    virtual void trsm(uint64_t n, uint64_t k, const double* A, double* B) = 0;

    // C = B * A'
    virtual void gemm(uint64_t m, uint64_t n, uint64_t k, const double* A,
                      const double* B, double* C) = 0;

    virtual void gemmToTemp(OpaqueData& assCtx, uint64_t m, uint64_t n,
                            uint64_t k, const double* A, const double* B) = 0;

    virtual OpaqueDataPtr createAssembleContext(const OpaqueData& skel) = 0;

    virtual void prepareAssembleContext(const OpaqueData& skel,
                                        OpaqueData& assCtx,
                                        uint64_t targetLump) = 0;

    virtual void assemble(const OpaqueData& skel, const OpaqueData& assCtx,
                          double* data, uint64_t rectRowBegin,
                          uint64_t dstStride,  //
                          uint64_t srcColDataOffset, uint64_t numBlockRows,
                          uint64_t numBlockCols) = 0;
};

using OpsPtr = std::unique_ptr<Ops>;

OpsPtr simpleOps();

OpsPtr blasOps();