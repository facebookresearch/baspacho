#include <memory>

#include "CoalescedBlockMatrix.h"

struct Ops;
struct SymbolicCtx;
struct SymElimCtx;
template <typename T>
struct NumericCtx;
using OpsPtr = std::unique_ptr<Ops>;
using SymbolicCtxPtr = std::unique_ptr<SymbolicCtx>;
using SymElimCtxPtr = std::unique_ptr<SymElimCtx>;
template <typename T>
using NumericCtxPtr = std::unique_ptr<NumericCtx<T>>;

struct Ops {
    virtual ~Ops() {}

    // (optionally) allows creation of op-specific global data (eg GPU copies)
    virtual SymbolicCtxPtr initSymbolicInfo(
        const CoalescedBlockMatrixSkel& skel) = 0;
};

struct SymbolicCtx {
    virtual ~SymbolicCtx() {}

    // prepares data for a parallel elimination op
    virtual SymElimCtxPtr prepareElimination(uint64_t lumpsBegin,
                                             uint64_t lumpsEnd) = 0;

    /*virtual NumericCtxPtr<float> createFloatContext(uint64_t tempBufSize,
                                                    int maxBatchSize = 1) = 0;*/

    virtual NumericCtxPtr<double> createDoubleContext(uint64_t tempBufSize,
                                                      int maxBatchSize = 1) = 0;
};

struct SymElimCtx {
    virtual ~SymElimCtx() {}
};

// ops and contexts depending on the float/double type
template <typename T>
struct NumericCtx {
    virtual ~NumericCtx() {}

    virtual void printStats() const = 0;

    // does (possibly parallel) elimination on a lump of aggregs
    virtual void doElimination(const SymElimCtx& elimData, T* data,
                               uint64_t lumpsBegin, uint64_t lumpsEnd) = 0;

    // dense Cholesky on dense row-major matrix A (in place)
    virtual void potrf(uint64_t n, T* A) = 0;

    // solve: X * A.lowerHalf().transpose() = B (in place, B becomes X)
    virtual void trsm(uint64_t n, uint64_t k, const T* A, T* B) = 0;

    virtual void saveSyrkGemm(uint64_t m, uint64_t n, uint64_t k, const T* data,
                              uint64_t offset) = 0;

    // computes (A|B) * A', upper diag part doesn't matter
    virtual void saveSyrkGemmBatched(uint64_t* ms, uint64_t* ns, uint64_t* ks,
                                     const T* data, uint64_t* offsets,
                                     int batchSize) = 0;

    virtual void prepareAssemble(uint64_t targetLump) = 0;

    virtual void assemble(T* data, uint64_t rectRowBegin, uint64_t dstStride,
                          uint64_t srcColDataOffset, uint64_t srcRectWidth,
                          uint64_t numBlockRows, uint64_t numBlockCols,
                          int numBatch = -1) = 0;

    virtual void solveL(const T* data, uint64_t offset, uint64_t n, T* C,
                        uint64_t offC, uint64_t ldc, uint64_t nRHS) = 0;

    virtual void gemv(const T* data, uint64_t offset, uint64_t nRows,
                      uint64_t nCols, const T* A, uint64_t offA, uint64_t lda,
                      T* C, uint64_t nRHS) = 0;

    virtual void assembleVec(const T* A, uint64_t chainColPtr,
                             uint64_t numColItems, T* C, uint64_t ldc,
                             uint64_t nRHS) = 0;

    virtual void solveLt(const T* data, uint64_t offset, uint64_t n, T* C,
                         uint64_t offC, uint64_t ldc, uint64_t nRHS) = 0;

    virtual void gemvT(const T* data, uint64_t offset, uint64_t nRows,
                       uint64_t nCols, const T* C, uint64_t nRHS, T* A,
                       uint64_t offA, uint64_t lda) = 0;

    virtual void assembleVecT(const T* C, uint64_t ldc, uint64_t nRHS, T* A,
                              uint64_t chainColPtr, uint64_t numColItems) = 0;
};

using OpsPtr = std::unique_ptr<Ops>;

OpsPtr simpleOps();

OpsPtr blasOps();