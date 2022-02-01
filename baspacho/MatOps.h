#include <memory>

#include "baspacho/CoalescedBlockMatrix.h"

namespace BaSpaCho {

struct Ops;
struct SymbolicCtx;
struct SymElimCtx;
template <typename T>
struct NumericCtx;
template <typename T>
struct SolveCtx;
using OpsPtr = std::unique_ptr<Ops>;
using SymbolicCtxPtr = std::unique_ptr<SymbolicCtx>;
using SymElimCtxPtr = std::unique_ptr<SymElimCtx>;
template <typename T>
using NumericCtxPtr = std::unique_ptr<NumericCtx<T>>;
template <typename T>
using SolveCtxPtr = std::unique_ptr<SolveCtx<T>>;

struct Ops {
    virtual ~Ops() {}

    // (optionally) allows creation of op-specific global data (eg GPU copies)
    virtual SymbolicCtxPtr initSymbolicInfo(
        const CoalescedBlockMatrixSkel& skel) = 0;
};

struct SymbolicCtx {
    virtual ~SymbolicCtx() {}

    // prepares data for a parallel elimination op
    virtual SymElimCtxPtr prepareElimination(int64_t lumpsBegin,
                                             int64_t lumpsEnd) = 0;

    /*virtual NumericCtxPtr<float> createFloatContext(int64_t tempBufSize,
                                                    int maxBatchSize = 1) = 0;*/

    virtual NumericCtxPtr<double> createDoubleContext(int64_t tempBufSize,
                                                      int maxBatchSize = 1) = 0;

    virtual SolveCtxPtr<double> createDoubleSolveContext() = 0;
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
                               int64_t lumpsBegin, int64_t lumpsEnd) = 0;

    // dense Cholesky on dense row-major matrix A (in place)
    virtual void potrf(int64_t n, T* A) = 0;

    // solve: X * A.lowerHalf().transpose() = B (in place, B becomes X)
    virtual void trsm(int64_t n, int64_t k, const T* A, T* B) = 0;

    virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const T* data,
                              int64_t offset) = 0;

    // computes (A|B) * A', upper diag part doesn't matter
    virtual void saveSyrkGemmBatched(int64_t* ms, int64_t* ns, int64_t* ks,
                                     const T* data, int64_t* offsets,
                                     int batchSize) = 0;

    virtual void prepareAssemble(int64_t targetLump) = 0;

    virtual void assemble(T* data, int64_t rectRowBegin, int64_t dstStride,
                          int64_t srcColDataOffset, int64_t srcRectWidth,
                          int64_t numBlockRows, int64_t numBlockCols,
                          int numBatch = -1) = 0;
};

template <typename T>
struct SolveCtx {
    virtual ~SolveCtx() {}
    virtual void solveL(const T* data, int64_t offset, int64_t n, T* C,
                        int64_t offC, int64_t ldc, int64_t nRHS) = 0;

    virtual void gemv(const T* data, int64_t offset, int64_t nRows,
                      int64_t nCols, const T* A, int64_t offA, int64_t lda,
                      T* C, int64_t nRHS) = 0;

    virtual void assembleVec(const T* A, int64_t chainColPtr,
                             int64_t numColItems, T* C, int64_t ldc,
                             int64_t nRHS) = 0;

    virtual void solveLt(const T* data, int64_t offset, int64_t n, T* C,
                         int64_t offC, int64_t ldc, int64_t nRHS) = 0;

    virtual void gemvT(const T* data, int64_t offset, int64_t nRows,
                       int64_t nCols, const T* C, int64_t nRHS, T* A,
                       int64_t offA, int64_t lda) = 0;

    virtual void assembleVecT(const T* C, int64_t ldc, int64_t nRHS, T* A,
                              int64_t chainColPtr, int64_t numColItems) = 0;
};

using OpsPtr = std::unique_ptr<Ops>;

OpsPtr simpleOps();

OpsPtr blasOps();

}  // end namespace BaSpaCho