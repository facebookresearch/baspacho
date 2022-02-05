#include <alloca.h>
#include <dispenso/parallel_for.h>

#include <chrono>

#include "baspacho/DebugMacros.h"
#include "baspacho/MatOpsCpuBase.h"
#include "baspacho/Utils.h"

#ifdef BASPACHO_USE_MKL

#include "mkl.h"
#define BLAS_INT MKL_INT
#define BASPACHO_USE_TRSM_WORAROUND 0

#else

#include "baspacho/BlasDefs.h"
#define BASPACHO_USE_TRSM_WORAROUND 1

#endif

namespace BaSpaCho {

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

struct BlasSymbolicCtx : CpuBaseSymbolicCtx {
    BlasSymbolicCtx(const CoalescedBlockMatrixSkel& skel, int numThreads)
        : CpuBaseSymbolicCtx(skel),
          useThreads(numThreads > 1),
          threadPool(useThreads ? numThreads : 0) {}

    virtual NumericCtxBase* createNumericCtxForType(
        std::type_index tIdx, int64_t tempBufSize) override;

    virtual SolveCtxBase* createSolveCtxForType(std::type_index tIdx,
                                                int nRHS) override;

    bool useThreads;
    mutable dispenso::ThreadPool threadPool;
};

// Blas ops aiming at high performance using BLAS/LAPACK
struct BlasOps : Ops {
    virtual SymbolicCtxPtr createSymbolicCtx(
        const CoalescedBlockMatrixSkel& skel) override {
        // todo: use settings
        return SymbolicCtxPtr(new BlasSymbolicCtx(skel, 16));
    }
};

template <typename T>
struct BlasNumericCtx : CpuBaseNumericCtx<T> {
    BlasNumericCtx(const BlasSymbolicCtx& sym, int64_t bufSize,
                   int64_t numSpans)
        : CpuBaseNumericCtx<T>(bufSize, numSpans), sym(sym) {}

    virtual void doElimination(const SymElimCtx& elimData, T* data,
                               int64_t lumpsBegin, int64_t lumpsEnd) override {
        const CpuBaseSymElimCtx* pElim =
            dynamic_cast<const CpuBaseSymElimCtx*>(&elimData);
        BASPACHO_CHECK_NOTNULL(pElim);
        const CpuBaseSymElimCtx& elim = *pElim;
        const CoalescedBlockMatrixSkel& skel = sym.skel;
        OpInstance timer(elim.elimStat);

        if (!sym.useThreads) {
            for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
                factorLump(skel, data, l);
            }
        } else {
            dispenso::TaskSet taskSet(sym.threadPool);
            dispenso::parallel_for(
                taskSet, dispenso::makeChunkedRange(lumpsBegin, lumpsEnd, 5UL),
                [&](int64_t lBegin, int64_t lEnd) {
                    for (int64_t l = lBegin; l < lEnd; l++) {
                        factorLump(skel, data, l);
                    }
                });
        }

        int64_t numElimRows = elim.rowPtr.size() - 1;
        if (elim.colLump.size() > 3 * (elim.rowPtr.size() - 1)) {
            int64_t numSpans = skel.spanStart.size() - 1;
            if (!sym.useThreads) {
                std::vector<T> tempBuffer(elim.maxBufferSize);
                std::vector<int64_t> spanToChainOffset(numSpans);
                for (int64_t sRel = 0UL; sRel < numElimRows; sRel++) {
                    eliminateRowChain(elim, skel, data, sRel, spanToChainOffset,
                                      tempBuffer);
                }
            } else {
                struct ElimContext {
                    std::vector<T> tempBuffer;
                    std::vector<int64_t> spanToChainOffset;
                    ElimContext(int64_t bufSize, int64_t numSpans)
                        : tempBuffer(bufSize), spanToChainOffset(numSpans) {}
                };
                vector<ElimContext> contexts;
                dispenso::TaskSet taskSet(sym.threadPool);
                dispenso::parallel_for(
                    taskSet, contexts,
                    [=]() -> ElimContext {
                        return ElimContext(elim.maxBufferSize, numSpans);
                    },
                    dispenso::makeChunkedRange(0L, numElimRows, 5L),
                    [&, this](ElimContext& ctx, int64_t sBegin, int64_t sEnd) {
                        for (int64_t sRel = sBegin; sRel < sEnd; sRel++) {
                            eliminateRowChain(elim, skel, data, sRel,
                                              ctx.spanToChainOffset,
                                              ctx.tempBuffer);
                        }
                    });
            }
        } else {
            if (!sym.useThreads) {
                for (int64_t sRel = 0UL; sRel < numElimRows; sRel++) {
                    eliminateVerySparseRowChain(elim, skel, data, sRel);
                }
            } else {
                dispenso::TaskSet taskSet(sym.threadPool);
                dispenso::parallel_for(
                    taskSet, dispenso::makeChunkedRange(0L, numElimRows, 5L),
                    [&, this](int64_t sBegin, int64_t sEnd) {
                        for (int64_t sRel = sBegin; sRel < sEnd; sRel++) {
                            eliminateVerySparseRowChain(elim, skel, data, sRel);
                        }
                    });
            }
        }
    }

    virtual void potrf(int64_t n, T* A) override;

    virtual void trsm(int64_t n, int64_t k, const T* A, T* B) override;

    virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const T* data,
                              int64_t offset) override;

    virtual void prepareAssemble(int64_t targetLump) override {
        const CoalescedBlockMatrixSkel& skel = sym.skel;

        for (int64_t i = skel.chainColPtr[targetLump],
                     iEnd = skel.chainColPtr[targetLump + 1];
             i < iEnd; i++) {
            spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
        }
    }

    virtual void assemble(T* data, int64_t rectRowBegin,
                          int64_t dstStride,  //
                          int64_t srcColDataOffset, int64_t srcRectWidth,
                          int64_t numBlockRows, int64_t numBlockCols) override {
        OpInstance timer(sym.asmblStat);
        const CoalescedBlockMatrixSkel& skel = sym.skel;
        const int64_t* chainRowsTillEnd =
            skel.chainRowsTillEnd.data() + srcColDataOffset;
        const int64_t* pToSpan = skel.chainRowSpan.data() + srcColDataOffset;
        const int64_t* pSpanToChainOffset = spanToChainOffset.data();
        const int64_t* pSpanOffsetInLump = skel.spanOffsetInLump.data();
        const T* matRectPtr = tempBuffer.data();

        if (!sym.useThreads) {
            // non-threaded reference implementation:
            for (int64_t r = 0; r < numBlockRows; r++) {
                int64_t rBegin = chainRowsTillEnd[r - 1] - rectRowBegin;
                int64_t rSize = chainRowsTillEnd[r] - rBegin - rectRowBegin;
                int64_t rParam = pToSpan[r];
                int64_t rOffset = pSpanToChainOffset[rParam];
                const T* matRowPtr = matRectPtr + rBegin * srcRectWidth;

                int64_t cEnd = std::min(numBlockCols, r + 1);
                for (int64_t c = 0; c < cEnd; c++) {
                    int64_t cStart = chainRowsTillEnd[c - 1] - rectRowBegin;
                    int64_t cSize = chainRowsTillEnd[c] - cStart - rectRowBegin;
                    int64_t offset = rOffset + pSpanOffsetInLump[pToSpan[c]];

                    T* dst = data + offset;
                    const T* src = matRowPtr + cStart;
                    stridedMatSub(dst, dstStride, src, srcRectWidth, rSize,
                                  cSize);
                }
            }
        } else {
            dispenso::TaskSet taskSet(sym.threadPool);
            dispenso::parallel_for(
                taskSet, dispenso::makeChunkedRange(0, numBlockRows, 3UL),
                [&](int64_t rFrom, int64_t rTo) {
                    for (int64_t r = rFrom; r < rTo; r++) {
                        int64_t rBegin = chainRowsTillEnd[r - 1] - rectRowBegin;
                        int64_t rSize =
                            chainRowsTillEnd[r] - rBegin - rectRowBegin;
                        int64_t rParam = pToSpan[r];
                        int64_t rOffset = pSpanToChainOffset[rParam];
                        const T* matRowPtr = matRectPtr + rBegin * srcRectWidth;

                        int64_t cEnd = std::min(numBlockCols, r + 1);
                        int64_t nextCStart =
                            chainRowsTillEnd[-1] - rectRowBegin;
                        for (int64_t c = 0; c < cEnd; c++) {
                            int64_t cStart = nextCStart;
                            nextCStart = chainRowsTillEnd[c] - rectRowBegin;
                            int64_t cSize = nextCStart - cStart;
                            int64_t offset =
                                rOffset + pSpanOffsetInLump[pToSpan[c]];

                            T* dst = data + offset;
                            const T* src = matRowPtr + cStart;
                            stridedMatSub(dst, dstStride, src, srcRectWidth,
                                          rSize, cSize);
                        }
                    }
                });
        }
    }

    using CpuBaseNumericCtx<T>::factorLump;
    using CpuBaseNumericCtx<T>::eliminateRowChain;
    using CpuBaseNumericCtx<T>::eliminateVerySparseRowChain;
    using CpuBaseNumericCtx<T>::stridedMatSub;

    using CpuBaseNumericCtx<T>::tempBuffer;
    using CpuBaseNumericCtx<T>::spanToChainOffset;

    const BlasSymbolicCtx& sym;
};

template <>
void BlasNumericCtx<double>::potrf(int64_t n, double* A) {
    OpInstance timer(sym.potrfStat);
    sym.potrfBiggestN = std::max(sym.potrfBiggestN, n);

    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', n, A, n);
}

template <>
void BlasNumericCtx<float>::potrf(int64_t n, float* A) {
    OpInstance timer(sym.potrfStat);
    sym.potrfBiggestN = std::max(sym.potrfBiggestN, n);

    LAPACKE_spotrf(LAPACK_COL_MAJOR, 'U', n, A, n);
}

template <>
void BlasNumericCtx<double>::trsm(int64_t n, int64_t k, const double* A,
                                  double* B) {
    OpInstance timer(sym.trsmStat);

    // TSRM should be fast but appears very slow in OpenBLAS
    static constexpr bool slowTrsmWorkaround = BASPACHO_USE_TRSM_WORAROUND;
    if (slowTrsmWorkaround) {
        using MatCMajD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::ColMajor>;

        // col-major's upper = (row-major's lower).transpose()
        Eigen::Map<const MatCMajD> matA(A, n, n);
        dispenso::TaskSet taskSet(sym.threadPool);
        dispenso::parallel_for(
            taskSet, dispenso::makeChunkedRange(0, k, 16UL),
            [&](int64_t k1, int64_t k2) {
                Eigen::Map<MatRMaj<double>> matB(B + n * k1, k2 - k1, n);
                matA.template triangularView<Eigen::Upper>()
                    .template solveInPlace<Eigen::OnTheRight>(matB);
            });
        return;
    }

    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                CblasNonUnit, n, k, 1.0, A, n, B, n);
}

template <>
void BlasNumericCtx<float>::trsm(int64_t n, int64_t k, const float* A,
                                 float* B) {
    OpInstance timer(sym.trsmStat);

    // TSRM should be fast but appears very slow in OpenBLAS
    static constexpr bool slowTrsmWorkaround = BASPACHO_USE_TRSM_WORAROUND;
    if (slowTrsmWorkaround) {
        using MatCMajD = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::ColMajor>;

        // col-major's upper = (row-major's lower).transpose()
        Eigen::Map<const MatCMajD> matA(A, n, n);
        dispenso::TaskSet taskSet(sym.threadPool);
        dispenso::parallel_for(
            taskSet, dispenso::makeChunkedRange(0, k, 16UL),
            [&](int64_t k1, int64_t k2) {
                Eigen::Map<MatRMaj<float>> matB(B + n * k1, k2 - k1, n);
                matA.template triangularView<Eigen::Upper>()
                    .template solveInPlace<Eigen::OnTheRight>(matB);
            });
        return;
    }

    cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                CblasNonUnit, n, k, 1.0, A, n, B, n);
}

template <>
void BlasNumericCtx<double>::saveSyrkGemm(int64_t m, int64_t n, int64_t k,
                                          const double* data, int64_t offset) {
    OpInstance timer(sym.sygeStat);
    BASPACHO_CHECK_LE(m * n, (int64_t)tempBuffer.size());

    // in some cases it could be faster with syrk+gemm
    // as it saves some computation, not the case in practice
    bool doSyrk = (m == n) || (m + n + k > 150);
    bool doGemm = !(doSyrk && m == n);

    if (doSyrk) {
        cblas_dsyrk(CblasColMajor, CblasUpper, CblasConjTrans, m, k, 1.0,
                    data + offset, k, 0.0, tempBuffer.data(), m);
        sym.syrkCalls++;
    }

    if (doGemm) {
        int64_t gemmStart = doSyrk ? m : 0;
        int64_t gemmInOffset = doSyrk ? m * k : 0;
        int64_t gemmOutOffset = doSyrk ? m * m : 0;
        cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, m,
                    n - gemmStart, k, 1.0, data + offset, k,
                    data + offset + gemmInOffset, k, 0.0,
                    tempBuffer.data() + gemmOutOffset, m);
        sym.gemmCalls++;
    }
}

template <>
void BlasNumericCtx<float>::saveSyrkGemm(int64_t m, int64_t n, int64_t k,
                                         const float* data, int64_t offset) {
    OpInstance timer(sym.sygeStat);
    BASPACHO_CHECK_LE(m * n, (int64_t)tempBuffer.size());

    // in some cases it could be faster with syrk+gemm
    // as it saves some computation, not the case in practice
    bool doSyrk = (m == n) || (m + n + k > 150);
    bool doGemm = !(doSyrk && m == n);

    if (doSyrk) {
        cblas_ssyrk(CblasColMajor, CblasUpper, CblasConjTrans, m, k, 1.0,
                    data + offset, k, 0.0, tempBuffer.data(), m);
        sym.syrkCalls++;
    }

    if (doGemm) {
        int64_t gemmStart = doSyrk ? m : 0;
        int64_t gemmInOffset = doSyrk ? m * k : 0;
        int64_t gemmOutOffset = doSyrk ? m * m : 0;
        cblas_sgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, m,
                    n - gemmStart, k, 1.0, data + offset, k,
                    data + offset + gemmInOffset, k, 0.0,
                    tempBuffer.data() + gemmOutOffset, m);
        sym.gemmCalls++;
    }
}

template <typename T>
struct BlasSolveCtx : SolveCtx<T> {
    BlasSolveCtx(const BlasSymbolicCtx& sym, int nRHS)
        : sym(sym), nRHS(nRHS), tmpBuf(sym.skel.order() * nRHS) {}
    virtual ~BlasSolveCtx() override {}

    virtual void solveL(const T* data, int64_t offM, int64_t n, T* C,
                        int64_t offC, int64_t ldc) override;

    virtual void gemv(const T* data, int64_t offM, int64_t nRows, int64_t nCols,
                      const T* A, int64_t offA, int64_t lda) override;

    static inline void stridedTransSub(T* dst, int64_t dstStride, const T* src,
                                       int64_t srcStride, int64_t rSize,
                                       int64_t cSize) {
        for (uint j = 0; j < rSize; j++) {
            T* pDst = dst + j;
            for (uint i = 0; i < cSize; i++) {
                *pDst -= src[i];
                pDst += dstStride;
            }
            src += srcStride;
        }
    }

    virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, T* C,
                             int64_t ldc) override {
        const T* A = tmpBuf.data();
        const CoalescedBlockMatrixSkel& skel = sym.skel;
        const int64_t* chainRowsTillEnd =
            skel.chainRowsTillEnd.data() + chainColPtr;
        const int64_t* toSpan = skel.chainRowSpan.data() + chainColPtr;
        int64_t startRow = chainRowsTillEnd[-1];
        for (int64_t i = 0; i < numColItems; i++) {
            int64_t rowOffset = chainRowsTillEnd[i - 1] - startRow;
            int64_t span = toSpan[i];
            int64_t spanStart = skel.spanStart[span];
            int64_t spanSize = skel.spanStart[span + 1] - spanStart;

            stridedTransSub(C + spanStart, ldc, A + rowOffset * nRHS, nRHS,
                            spanSize, nRHS);
        }
    }

    virtual void solveLt(const T* data, int64_t offM, int64_t n, T* C,
                         int64_t offC, int64_t ldc) override;

    virtual void gemvT(const T* data, int64_t offM, int64_t nRows,
                       int64_t nCols, T* A, int64_t offA, int64_t lda) override;

    static inline void stridedTransSet(T* dst, int64_t dstStride, const T* src,
                                       int64_t srcStride, int64_t rSize,
                                       int64_t cSize) {
        for (uint j = 0; j < rSize; j++) {
            T* pDst = dst + j;
            for (uint i = 0; i < cSize; i++) {
                *pDst = src[i];
                pDst += dstStride;
            }
            src += srcStride;
        }
    }

    virtual void assembleVecT(const T* C, int64_t ldc, /* int64_t nRHS, T* A, */
                              int64_t chainColPtr,
                              int64_t numColItems) override {
        T* A = tmpBuf.data();
        const CoalescedBlockMatrixSkel& skel = sym.skel;
        const int64_t* chainRowsTillEnd =
            skel.chainRowsTillEnd.data() + chainColPtr;
        const int64_t* toSpan = skel.chainRowSpan.data() + chainColPtr;
        int64_t startRow = chainRowsTillEnd[-1];
        for (int64_t i = 0; i < numColItems; i++) {
            int64_t rowOffset = chainRowsTillEnd[i - 1] - startRow;
            int64_t span = toSpan[i];
            int64_t spanStart = skel.spanStart[span];
            int64_t spanSize = skel.spanStart[span + 1] - spanStart;

            stridedTransSet(A + rowOffset * nRHS, nRHS, C + spanStart, ldc,
                            nRHS, spanSize);
        }
    }

    const BlasSymbolicCtx& sym;
    int64_t nRHS;
    vector<T> tmpBuf;
};

template <>
void BlasSolveCtx<double>::solveL(const double* data, int64_t offM, int64_t n,
                                  double* C, int64_t offC, int64_t ldc) {
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                CblasNonUnit, n, nRHS, 1.0, data + offM, n, C + offC, ldc);
}

template <>
void BlasSolveCtx<float>::solveL(const float* data, int64_t offM, int64_t n,
                                 float* C, int64_t offC, int64_t ldc) {
    cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                CblasNonUnit, n, nRHS, 1.0, data + offM, n, C + offC, ldc);
}

template <>
void BlasSolveCtx<double>::gemv(const double* data, int64_t offM, int64_t nRows,
                                int64_t nCols, const double* A, int64_t offA,
                                int64_t lda) {
    cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nRHS, nRows, nCols,
                1.0, A + offA, lda, data + offM, nCols, 0.0, tmpBuf.data(),
                nRHS);
}

template <>
void BlasSolveCtx<float>::gemv(const float* data, int64_t offM, int64_t nRows,
                               int64_t nCols, const float* A, int64_t offA,
                               int64_t lda) {
    cblas_sgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nRHS, nRows, nCols,
                1.0, A + offA, lda, data + offM, nCols, 0.0, tmpBuf.data(),
                nRHS);
}

template <>
void BlasSolveCtx<double>::solveLt(const double* data, int64_t offM, int64_t n,
                                   double* C, int64_t offC, int64_t ldc) {
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                CblasNonUnit, n, nRHS, 1.0, data + offM, n, C + offC, ldc);
}

template <>
void BlasSolveCtx<float>::solveLt(const float* data, int64_t offM, int64_t n,
                                  float* C, int64_t offC, int64_t ldc) {
    cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                CblasNonUnit, n, nRHS, 1.0, data + offM, n, C + offC, ldc);
}

template <>
void BlasSolveCtx<double>::gemvT(const double* data, int64_t offM,
                                 int64_t nRows, int64_t nCols, double* A,
                                 int64_t offA, int64_t lda) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, nCols, nRHS, nRows,
                -1.0, data + offM, nCols, tmpBuf.data(), nRHS, 1.0, A + offA,
                lda);
}

template <>
void BlasSolveCtx<float>::gemvT(const float* data, int64_t offM, int64_t nRows,
                                int64_t nCols, float* A, int64_t offA,
                                int64_t lda) {
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, nCols, nRHS, nRows,
                -1.0, data + offM, nCols, tmpBuf.data(), nRHS, 1.0, A + offA,
                lda);
}

NumericCtxBase* BlasSymbolicCtx::createNumericCtxForType(std::type_index tIdx,
                                                         int64_t tempBufSize) {
    if (tIdx == std::type_index(typeid(double))) {
        return new BlasNumericCtx<double>(*this, tempBufSize,
                                          skel.spanStart.size() - 1);
    } else if (tIdx == std::type_index(typeid(float))) {
        return new BlasNumericCtx<float>(*this, tempBufSize,
                                         skel.spanStart.size() - 1);
    } else {
        return nullptr;
    }
}

SolveCtxBase* BlasSymbolicCtx::createSolveCtxForType(std::type_index tIdx,
                                                     int nRHS) {
    if (tIdx == std::type_index(typeid(double))) {
        return new BlasSolveCtx<double>(*this, nRHS);
    } else if (tIdx == std::type_index(typeid(float))) {
        return new BlasSolveCtx<float>(*this, nRHS);
    } else {
        return nullptr;
    }
}

OpsPtr blasOps() { return OpsPtr(new BlasOps); }

}  // end namespace BaSpaCho