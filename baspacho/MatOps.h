#include <memory>
#include <typeindex>

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

// generator class for operation contexts
struct Ops {
    virtual ~Ops() {}

    // creates a symbolic context from a matrix structure
    virtual SymbolicCtxPtr createSymbolicCtx(
        const CoalescedBlockMatrixSkel& skel) = 0;
};

struct NumericCtxBase {
    virtual ~NumericCtxBase() {}
};

struct SolveCtxBase {
    virtual ~SolveCtxBase() {}
};

// (symbolic) context for factorization, constant indices (and GPU copies)
struct SymbolicCtx {
    virtual ~SymbolicCtx() {}

    // prepares data for a parallel elimination op
    virtual SymElimCtxPtr prepareElimination(int64_t lumpsBegin,
                                             int64_t lumpsEnd) = 0;

    virtual NumericCtxBase* createNumericCtxForType(std::type_index tIdx,
                                                    int64_t tempBufSize,
                                                    int maxBatchSize = 1) = 0;

    virtual SolveCtxBase* createSolveCtxForType(std::type_index tIdx) = 0;

    template <typename T>
    NumericCtxPtr<T> createNumericCtx(int64_t tempBufSize,
                                      int maxBatchSize = 1);

    template <typename T>
    SolveCtxPtr<T> createSolveCtx();
};

// (symbolic) context for sparse elimination of a range of parameters
struct SymElimCtx {
    virtual ~SymElimCtx() {}
};

// ops and contexts depending on the float/double type
template <typename T>
struct NumericCtx : NumericCtxBase {
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

// methods (and possibly context) for solve operations
template <typename T>
struct SolveCtx : SolveCtxBase {
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

template <typename T>
NumericCtxPtr<T> SymbolicCtx::createNumericCtx(int64_t tempBufSize,
                                               int maxBatchSize) {
    static const std::type_index T_tIdx(typeid(T));
    NumericCtxBase* ctx =
        createNumericCtxForType(T_tIdx, tempBufSize, maxBatchSize);
    NumericCtx<T>* typedCtx = dynamic_cast<NumericCtx<T>*>(ctx);
    BASPACHO_CHECK_NOTNULL(typedCtx);
    return NumericCtxPtr<T>(typedCtx);
}

template <typename T>
SolveCtxPtr<T> SymbolicCtx::createSolveCtx() {
    static const std::type_index T_tIdx(typeid(T));
    SolveCtxBase* ctx = createSolveCtxForType(T_tIdx);
    SolveCtx<T>* typedCtx = dynamic_cast<SolveCtx<T>*>(ctx);
    BASPACHO_CHECK_NOTNULL(typedCtx);
    return SolveCtxPtr<T>(typedCtx);
}

OpsPtr simpleOps();

OpsPtr blasOps();

OpsPtr cudaOps();

}  // end namespace BaSpaCho