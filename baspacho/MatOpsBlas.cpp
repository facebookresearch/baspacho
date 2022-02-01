#include <alloca.h>
#include <dispenso/parallel_for.h>

#include <chrono>

#include "DebugMacros.h"
#include "MatOpsCpuBase.h"
#include "Utils.h"

#ifdef BASPACHO_USE_MKL

#include "mkl.h"
#define BLAS_INT MKL_INT
#define BASPACHO_HAVE_GEMM_BATCH

#else

// BLAS/LAPACK famously go without headers.
extern "C" {

// TODO: detect in build if blas type is long or int
#if 0
#define BLAS_INT long
#else
#define BLAS_INT int
#endif

void dpotrf_(const char* uplo, BLAS_INT* n, double* A, BLAS_INT* lda,
             BLAS_INT* info);

void dtrsm_(const char* side, const char* uplo, const char* transa, char* diag,
            BLAS_INT* m, BLAS_INT* n, const double* alpha, const double* A,
            BLAS_INT* lda, double* B, BLAS_INT* ldb);

void dgemm_(const char* transa, const char* transb, BLAS_INT* m, BLAS_INT* n,
            BLAS_INT* k, const double* alpha, const double* A, BLAS_INT* lda,
            const double* B, BLAS_INT* ldb, const double* beta, double* C,
            BLAS_INT* ldc);

void dsyrk_(const char* uplo, const char* transa, const BLAS_INT* n,
            const BLAS_INT* k, const double* alpha, const double* A,
            const BLAS_INT* lda, const double* beta, double* C,
            const BLAS_INT* ldc);
}

#endif

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

#include <chrono>
#include <iostream>

#include "DebugMacros.h"
#include "MatOpsCpuBase.h"
#include "Utils.h"

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

using OuterStride = Eigen::OuterStride<>;
template <typename T>
using OuterStridedMatM = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0,
    OuterStride>;
template <typename T>
using OuterStridedCMajMatM = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
    OuterStride>;
template <typename T>
using OuterStridedCMajMatK = Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
    OuterStride>;

struct BlasSymbolicCtx : CpuBaseSymbolicCtx {
    BlasSymbolicCtx(const CoalescedBlockMatrixSkel& skel, int numThreads)
        : CpuBaseSymbolicCtx(skel),
          useThreads(numThreads > 1),
          threadPool(useThreads ? numThreads : 0) {}

    virtual NumericCtxPtr<double> createDoubleContext(
        uint64_t tempBufSize, int maxBatchSize = 1) override;

    virtual SolveCtxPtr<double> createDoubleSolveContext() override;

    bool useThreads;
    mutable dispenso::ThreadPool threadPool;
};

// Blas ops aiming at high performance using BLAS/LAPACK
struct BlasOps : Ops {
    virtual SymbolicCtxPtr initSymbolicInfo(
        const CoalescedBlockMatrixSkel& skel) override {
        // todo: use settings
        return SymbolicCtxPtr(new BlasSymbolicCtx(skel, 16));
    }
};

template <typename T>
struct BlasNumericCtx : CpuBaseNumericCtx<T> {
    BlasNumericCtx(const BlasSymbolicCtx& sym, uint64_t bufSize,
                   uint64_t numSpans, int maxBatchSize)
        : CpuBaseNumericCtx<T>(bufSize * maxBatchSize, numSpans),
          sym(sym)
#ifdef BASPACHO_HAVE_GEMM_BATCH
          ,
          tempCtxSize(bufSize),
          maxBatchSize(maxBatchSize)
#endif
    {
#ifndef BASPACHO_HAVE_GEMM_BATCH
        BASPACHO_CHECK_EQ(maxBatchSize, 1);
#endif
    }

    virtual void doElimination(const SymElimCtx& elimData, double* data,
                               uint64_t lumpsBegin,
                               uint64_t lumpsEnd) override {
        OpInstance timer(elimStat);
        const CpuBaseSymElimCtx* pElim =
            dynamic_cast<const CpuBaseSymElimCtx*>(&elimData);
        BASPACHO_CHECK_NOTNULL(pElim);
        const CpuBaseSymElimCtx& elim = *pElim;
        const CoalescedBlockMatrixSkel& skel = sym.skel;

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

        uint64_t numElimRows = elim.rowPtr.size() - 1;
        if (elim.colLump.size() > 3 * (elim.rowPtr.size() - 1)) {
            uint64_t numSpans = skel.spanStart.size() - 1;
            if (!sym.useThreads) {
                std::vector<double> tempBuffer(elim.maxBufferSize);
                std::vector<uint64_t> spanToChainOffset(numSpans);
                for (uint64_t sRel = 0UL; sRel < numElimRows; sRel++) {
                    eliminateRowChain(elim, skel, data, sRel, spanToChainOffset,
                                      tempBuffer);
                }
            } else {
                struct ElimContext {
                    std::vector<double> tempBuffer;
                    std::vector<uint64_t> spanToChainOffset;
                    ElimContext(uint64_t bufSize, uint64_t numSpans)
                        : tempBuffer(bufSize), spanToChainOffset(numSpans) {}
                };
                vector<ElimContext> contexts;
                dispenso::TaskSet taskSet(sym.threadPool);
                dispenso::parallel_for(
                    taskSet, contexts,
                    [=]() -> ElimContext {
                        return ElimContext(elim.maxBufferSize, numSpans);
                    },
                    dispenso::makeChunkedRange(0UL, numElimRows, 5UL),
                    [&, this](ElimContext& ctx, size_t sBegin, size_t sEnd) {
                        for (uint64_t sRel = sBegin; sRel < sEnd; sRel++) {
                            eliminateRowChain(elim, skel, data, sRel,
                                              ctx.spanToChainOffset,
                                              ctx.tempBuffer);
                        }
                    });
            }
        } else {
            if (!sym.useThreads) {
                for (uint64_t sRel = 0UL; sRel < numElimRows; sRel++) {
                    eliminateVerySparseRowChain(elim, skel, data, sRel);
                }
            } else {
                dispenso::TaskSet taskSet(sym.threadPool);
                dispenso::parallel_for(
                    taskSet, dispenso::makeChunkedRange(0UL, numElimRows, 5UL),
                    [&, this](size_t sBegin, size_t sEnd) {
                        for (uint64_t sRel = sBegin; sRel < sEnd; sRel++) {
                            eliminateVerySparseRowChain(elim, skel, data, sRel);
                        }
                    });
            }
        }
    }

    virtual void potrf(uint64_t n, T* A) override {
        OpInstance timer(potrfStat);
        char argUpLo = 'U';
        BLAS_INT argN = n;
        BLAS_INT argLdA = n;
#ifdef BASPACHO_USE_MKL
        LAPACKE_dpotrf(LAPACK_COL_MAJOR, argUpLo, argN, A, argLdA);
#else
        BLAS_INT info;
        dpotrf_(&argUpLo, &argN, A, &argLdA, &info);
#endif
    }

    virtual void trsm(uint64_t n, uint64_t k, const T* A, T* B) override {
        OpInstance timer(trsmStat);

        // TSRM should be fast but appears very slow in OpenBLAS
        static constexpr bool slowTrsmWorkaround = true;
        if (slowTrsmWorkaround) {
            using MatCMajD = Eigen::Matrix<double, Eigen::Dynamic,
                                           Eigen::Dynamic, Eigen::ColMajor>;

            // col-major's upper = (row-major's lower).transpose()
            Eigen::Map<const MatCMajD> matA(A, n, n);
            dispenso::TaskSet taskSet(dispenso::globalThreadPool());
            dispenso::parallel_for(
                taskSet,                                 //
                dispenso::makeChunkedRange(0, k, 16UL),  //
                [&](int64_t k1, int64_t k2) {
                    Eigen::Map<MatRMaj<double>> matB(B + n * k1, k2 - k1, n);
                    matA.triangularView<Eigen::Upper>()
                        .solveInPlace<Eigen::OnTheRight>(matB);
                });
            return;
        }

        BLAS_INT argM = n;
        BLAS_INT argN = k;
        double argAlpha = 1.0;
        BLAS_INT argLdA = n;
        BLAS_INT argLdB = n;
#ifdef BASPACHO_USE_MKL
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                    CblasNonUnit, argM, argN, argAlpha, (double*)A, argLdA, B,
                    argLdB);
#else
        char argSide = 'L';
        char argUpLo = 'U';
        char argTransA = 'C';
        char argDiag = 'N';
        dtrsm_(&argSide, &argUpLo, &argTransA, &argDiag, &argM, &argN,
               &argAlpha, (double*)A, &argLdA, B, &argLdB);
#endif
    }

    virtual void saveSyrkGemm(uint64_t m, uint64_t n, uint64_t k, const T* data,
                              uint64_t offset) override {
        OpInstance timer(sygeStat);
        BASPACHO_CHECK_LE(m * n, tempBuffer.size());

        // in some cases it could be faster with syrk+gemm
        // as it saves some computation, not the case in practice
        bool doSyrk = (m == n) || (m + n + k > 150);
        bool doGemm = !(doSyrk && m == n);

        if (doSyrk) {
            BLAS_INT argN = m;
            BLAS_INT argK = k;
            double argAlpha = 1.0;
            double* argA = (double*)data + offset;
            BLAS_INT argLdA = k;
            double argBeta = 0.0;
            double* argC = tempBuffer.data();
            BLAS_INT argLdC = m;
#ifdef BASPACHO_USE_MKL
            cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, argN, argK,
                        argAlpha, argA, argLdA, argBeta, argC, argLdC);
#else
            char argUpLo = 'U';
            char argTransA = 'C';
            dsyrk_(&argUpLo, &argTransA, &argN, &argK, &argAlpha, argA, &argLdA,
                   &argBeta, argC, &argLdC);
#endif
            syrkCalls++;
        }

        if (doGemm) {
            uint64_t gemmStart = doSyrk ? m : 0;
            uint64_t gemmInOffset = doSyrk ? m * k : 0;
            uint64_t gemmOutOffset = doSyrk ? m * m : 0;
            BLAS_INT argM = m;
            BLAS_INT argN = n - gemmStart;
            BLAS_INT argK = k;
            double argAlpha = 1.0;
            double* argA = (double*)data + offset;
            BLAS_INT argLdA = k;
            double* argB = (double*)data + offset + gemmInOffset;
            BLAS_INT argLdB = k;
            double argBeta = 0.0;
            double* argC = tempBuffer.data() + gemmOutOffset;
            BLAS_INT argLdC = m;
#ifdef BASPACHO_USE_MKL
            cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, argM, argN,
                        argK, argAlpha, argA, argLdA, argB, argLdB, argBeta,
                        argC, argLdC);
#else
            char argTransA = 'C';
            char argTransB = 'N';
            dgemm_(&argTransA, &argTransB, &argM, &argN, &argK, &argAlpha, argA,
                   &argLdA, argB, &argLdB, &argBeta, argC, &argLdC);
#endif
            gemmCalls++;
        }
    }

    virtual void saveSyrkGemmBatched(uint64_t* ms, uint64_t* ns, uint64_t* ks,
                                     const T* data, uint64_t* offsets,
                                     int batchSize) {
#ifndef BASPACHO_HAVE_GEMM_BATCH
        BASPACHO_CHECK(!"Batching not supported");
#else
        OpInstance timer(sygeStat);
        BASPACHO_CHECK_LE(batchSize, maxBatchSize);

        // for testing: serial execution of the batch
        /*tempBufPtrs.clear();
        double* ptr = tempBuffer.data();
        for (int i = 0; i < batchSize; i++) {
            BASPACHO_CHECK_LE(ms[i] * ns[i], tempCtxSize);
            this->gemm(ms[i], ns[i], ks[i], data + offsets[i],
                       data + offsets[i], ptr);
            tempBufPtrs.push_back(ptr);
            ptr += ms[i] * ns[i];
        }*/
        CBLAS_TRANSPOSE* argTransAs =
            (CBLAS_TRANSPOSE*)alloca(batchSize * sizeof(CBLAS_TRANSPOSE));
        CBLAS_TRANSPOSE* argTransBs =
            (CBLAS_TRANSPOSE*)alloca(batchSize * sizeof(CBLAS_TRANSPOSE));
        BLAS_INT* argMs = (BLAS_INT*)alloca(batchSize * sizeof(BLAS_INT));
        BLAS_INT* argNs = (BLAS_INT*)alloca(batchSize * sizeof(BLAS_INT));
        BLAS_INT* argKs = (BLAS_INT*)alloca(batchSize * sizeof(BLAS_INT));
        double* argAlphas = (double*)alloca(batchSize * sizeof(double));
        double* argBetas = (double*)alloca(batchSize * sizeof(double));
        const double** argAs =
            (const double**)alloca(batchSize * sizeof(const double*));
        double** argCs = (double**)alloca(batchSize * sizeof(double*));
        BLAS_INT* argGroupSize =
            (BLAS_INT*)alloca(batchSize * sizeof(BLAS_INT));

        tempBufPtrs.clear();
        double* ptr = tempBuffer.data();
        for (int i = 0; i < batchSize; i++) {
            argTransAs[i] = CblasConjTrans;
            argTransBs[i] = CblasNoTrans;
            argMs[i] = ms[i];
            argNs[i] = ns[i];
            argKs[i] = ks[i];
            argAlphas[i] = 1.0;
            argBetas[i] = 0.0;
            argAs[i] = data + offsets[i];
            argCs[i] = ptr;
            argGroupSize[i] = 1;

            BASPACHO_CHECK_LE(ms[i] * ns[i], tempCtxSize);
            tempBufPtrs.push_back(ptr);
            ptr += ms[i] * ns[i];
        }
        const double** argBs = argAs;
        BLAS_INT* argLdAs = argKs;
        BLAS_INT* argLdBs = argKs;
        BLAS_INT* argLdCs = argMs;
        BLAS_INT argNumGroups = batchSize;
        cblas_dgemm_batch(CblasColMajor, argTransAs, argTransBs, argMs, argNs,
                          argKs, argAlphas, argAs, argLdAs, argAs, argLdBs,
                          argBetas, argCs, argLdCs, argNumGroups, argGroupSize);
        gemmCalls++;
#endif
    }

    virtual void prepareAssemble(uint64_t targetLump) override {
        const CoalescedBlockMatrixSkel& skel = sym.skel;

        for (uint64_t i = skel.chainColPtr[targetLump],
                      iEnd = skel.chainColPtr[targetLump + 1];
             i < iEnd; i++) {
            spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
        }
    }

    virtual void assemble(T* data, uint64_t rectRowBegin,
                          uint64_t dstStride,  //
                          uint64_t srcColDataOffset, uint64_t srcRectWidth,
                          uint64_t numBlockRows, uint64_t numBlockCols,
                          int numBatch = -1) override {
        OpInstance timer(asmblStat);
        const CoalescedBlockMatrixSkel& skel = sym.skel;
        const uint64_t* chainRowsTillEnd =
            skel.chainRowsTillEnd.data() + srcColDataOffset;
        const uint64_t* pToSpan = skel.chainRowSpan.data() + srcColDataOffset;
        const uint64_t* pSpanToChainOffset = spanToChainOffset.data();
        const uint64_t* pSpanOffsetInLump = skel.spanOffsetInLump.data();

#ifdef BASPACHO_HAVE_GEMM_BATCH
        BASPACHO_CHECK_LT(numBatch, tempBuffer.size());
        const double* matRectPtr =
            numBatch == -1 ? tempBuffer.data() : tempBufPtrs[numBatch];
#else
        BASPACHO_CHECK_EQ(numBatch, -1);  // Batching not supported
        const double* matRectPtr = tempBuffer.data();
#endif

        if (!sym.useThreads) {
            // non-threaded reference implementation:
            for (uint64_t r = 0; r < numBlockRows; r++) {
                uint64_t rBegin = chainRowsTillEnd[r - 1] - rectRowBegin;
                uint64_t rSize = chainRowsTillEnd[r] - rBegin - rectRowBegin;
                uint64_t rParam = pToSpan[r];
                uint64_t rOffset = pSpanToChainOffset[rParam];
                const double* matRowPtr = matRectPtr + rBegin * srcRectWidth;

                uint64_t cEnd = std::min(numBlockCols, r + 1);
                for (uint64_t c = 0; c < cEnd; c++) {
                    uint64_t cStart = chainRowsTillEnd[c - 1] - rectRowBegin;
                    uint64_t cSize =
                        chainRowsTillEnd[c] - cStart - rectRowBegin;
                    uint64_t offset = rOffset + pSpanOffsetInLump[pToSpan[c]];

                    double* dst = data + offset;
                    const double* src = matRowPtr + cStart;
                    stridedMatSub(dst, dstStride, src, srcRectWidth, rSize,
                                  cSize);
                }
            }
        } else {
            dispenso::TaskSet taskSet(sym.threadPool);
            dispenso::parallel_for(
                taskSet, dispenso::makeChunkedRange(0, numBlockRows, 3UL),
                [&](int64_t rFrom, int64_t rTo) {
                    for (uint64_t r = rFrom; r < rTo; r++) {
                        uint64_t rBegin =
                            chainRowsTillEnd[r - 1] - rectRowBegin;
                        uint64_t rSize =
                            chainRowsTillEnd[r] - rBegin - rectRowBegin;
                        uint64_t rParam = pToSpan[r];
                        uint64_t rOffset = pSpanToChainOffset[rParam];
                        const double* matRowPtr =
                            matRectPtr + rBegin * srcRectWidth;

                        uint64_t cEnd = std::min(numBlockCols, r + 1);
                        uint64_t nextCStart =
                            chainRowsTillEnd[-1] - rectRowBegin;
                        for (uint64_t c = 0; c < cEnd; c++) {
                            uint64_t cStart = nextCStart;
                            nextCStart = chainRowsTillEnd[c] - rectRowBegin;
                            uint64_t cSize = nextCStart - cStart;
                            uint64_t offset =
                                rOffset + pSpanOffsetInLump[pToSpan[c]];

                            double* dst = data + offset;
                            const double* src = matRowPtr + cStart;
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

    using CpuBaseNumericCtx<T>::elimStat;
    using CpuBaseNumericCtx<T>::potrfStat;
    using CpuBaseNumericCtx<T>::potrfBiggestN;
    using CpuBaseNumericCtx<T>::trsmStat;
    using CpuBaseNumericCtx<T>::sygeStat;
    using CpuBaseNumericCtx<T>::gemmCalls;
    using CpuBaseNumericCtx<T>::syrkCalls;
    using CpuBaseNumericCtx<T>::asmblStat;

    const BlasSymbolicCtx& sym;
#ifdef BASPACHO_HAVE_GEMM_BATCH
    std::vector<double*> tempBufPtrs;
    uint64_t tempCtxSize;
    int maxBatchSize;
#endif
};

template <typename T>
struct BlasSolveCtx : SolveCtx<T> {
    BlasSolveCtx(const BlasSymbolicCtx& sym) : sym(sym) {}
    virtual ~BlasSolveCtx() override {}

    virtual void solveL(const T* data, uint64_t offM, uint64_t n, T* C,
                        uint64_t offC, uint64_t ldc, uint64_t nRHS) override {
        BLAS_INT argM = n;
        BLAS_INT argN = nRHS;
        double argAlpha = 1.0;
        const double* argA = data + offM;
        BLAS_INT argLdA = n;
        double* argB = C + offC;
        BLAS_INT argLdB = ldc;
#ifdef BASPACHO_USE_MKL
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                    CblasNonUnit, argM, argN, argAlpha, argA, argLdA, argB,
                    argLdB);
#else
        char argSide = 'L';
        char argUpLo = 'U';
        char argTransA = 'C';
        char argDiag = 'N';
        dtrsm_(&argSide, &argUpLo, &argTransA, &argDiag, &argM, &argN,
               &argAlpha, argA, &argLdA, argB, &argLdB);
#endif
    }

    virtual void gemv(const T* data, uint64_t offM, uint64_t nRows,
                      uint64_t nCols, const T* A, uint64_t offA, uint64_t lda,
                      T* C, uint64_t nRHS) override {
        BLAS_INT argM = nRHS;
        BLAS_INT argN = nRows;
        BLAS_INT argK = nCols;
        double argAlpha = 1.0;
        const double* argA = A + offA;
        BLAS_INT argLdA = lda;
        const double* argB = data + offM;
        BLAS_INT argLdB = nCols;
        double argBeta = 0.0;
        double* argC = C;
        BLAS_INT argLdC = nRHS;
#ifdef BASPACHO_USE_MKL
        cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, argM, argN,
                    argK, argAlpha, argA, argLdA, argB, argLdB, argBeta, argC,
                    argLdC);
#else
        char argTransA = 'C';
        char argTransB = 'N';
        dgemm_(&argTransA, &argTransB, &argM, &argN, &argK, &argAlpha, argA,
               &argLdA, argB, &argLdB, &argBeta, argC, &argLdC);
#endif
    }

    static inline void stridedTransSub(double* dst, uint64_t dstStride,
                                       const double* src, uint64_t srcStride,
                                       uint64_t rSize, uint64_t cSize) {
        for (uint j = 0; j < rSize; j++) {
            double* pDst = dst + j;
            for (uint i = 0; i < cSize; i++) {
                *pDst -= src[i];
                pDst += dstStride;
            }
            src += srcStride;
        }
    }

    virtual void assembleVec(const T* A, uint64_t chainColPtr,
                             uint64_t numColItems, T* C, uint64_t ldc,
                             uint64_t nRHS) override {
        const CoalescedBlockMatrixSkel& skel = sym.skel;
        const uint64_t* chainRowsTillEnd =
            skel.chainRowsTillEnd.data() + chainColPtr;
        const uint64_t* toSpan = skel.chainRowSpan.data() + chainColPtr;
        uint64_t startRow = chainRowsTillEnd[-1];
        for (uint64_t i = 0; i < numColItems; i++) {
            uint64_t rowOffset = chainRowsTillEnd[i - 1] - startRow;
            uint64_t span = toSpan[i];
            uint64_t spanStart = skel.spanStart[span];
            uint64_t spanSize = skel.spanStart[span + 1] - spanStart;

            stridedTransSub(C + spanStart, ldc, A + rowOffset * nRHS, nRHS,
                            spanSize, nRHS);
        }
    }

    virtual void solveLt(const T* data, uint64_t offM, uint64_t n, T* C,
                         uint64_t offC, uint64_t ldc, uint64_t nRHS) override {
        BLAS_INT argM = n;
        BLAS_INT argN = nRHS;
        double argAlpha = 1.0;
        const double* argA = data + offM;
        BLAS_INT argLdA = n;
        double* argB = C + offC;
        BLAS_INT argLdB = ldc;
#ifdef BASPACHO_USE_MKL
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                    CblasNonUnit, argM, argN, argAlpha, argA, argLdA, argB,
                    argLdB);
#else
        char argSide = 'L';
        char argUpLo = 'U';
        char argTransA = 'N';
        char argDiag = 'N';
        dtrsm_(&argSide, &argUpLo, &argTransA, &argDiag, &argM, &argN,
               &argAlpha, argA, &argLdA, argB, &argLdB);
#endif
    }

    virtual void gemvT(const T* data, uint64_t offM, uint64_t nRows,
                       uint64_t nCols, const T* C, uint64_t nRHS, T* A,
                       uint64_t offA, uint64_t lda) override {
        BLAS_INT argM = nCols;
        BLAS_INT argN = nRHS;
        BLAS_INT argK = nRows;
        double argAlpha = -1.0;
        const double* argA = data + offM;
        BLAS_INT argLdA = nCols;
        const double* argB = C;
        BLAS_INT argLdB = nRHS;
        double argBeta = 1.0;
        double* argC = A + offA;
        BLAS_INT argLdC = lda;
#ifdef BASPACHO_USE_MKL
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, argM, argN,
                    argK, argAlpha, argA, argLdA, argB, argLdB, argBeta, argC,
                    argLdC);
#else
        char argTransA = 'N';
        char argTransB = 'C';
        dgemm_(&argTransA, &argTransB, &argM, &argN, &argK, &argAlpha, argA,
               &argLdA, argB, &argLdB, &argBeta, argC, &argLdC);
#endif
    }

    static inline void stridedTransSet(double* dst, uint64_t dstStride,
                                       const double* src, uint64_t srcStride,
                                       uint64_t rSize, uint64_t cSize) {
        for (uint j = 0; j < rSize; j++) {
            double* pDst = dst + j;
            for (uint i = 0; i < cSize; i++) {
                *pDst = src[i];
                pDst += dstStride;
            }
            src += srcStride;
        }
    }

    virtual void assembleVecT(const T* C, uint64_t ldc, uint64_t nRHS, T* A,
                              uint64_t chainColPtr,
                              uint64_t numColItems) override {
        const CoalescedBlockMatrixSkel& skel = sym.skel;
        const uint64_t* chainRowsTillEnd =
            skel.chainRowsTillEnd.data() + chainColPtr;
        const uint64_t* toSpan = skel.chainRowSpan.data() + chainColPtr;
        uint64_t startRow = chainRowsTillEnd[-1];
        for (uint64_t i = 0; i < numColItems; i++) {
            uint64_t rowOffset = chainRowsTillEnd[i - 1] - startRow;
            uint64_t span = toSpan[i];
            uint64_t spanStart = skel.spanStart[span];
            uint64_t spanSize = skel.spanStart[span + 1] - spanStart;

            stridedTransSet(A + rowOffset * nRHS, nRHS, C + spanStart, ldc,
                            nRHS, spanSize);
        }
    }

    const BlasSymbolicCtx& sym;
};

NumericCtxPtr<double> BlasSymbolicCtx::createDoubleContext(uint64_t tempBufSize,
                                                           int maxBatchSize) {
    return NumericCtxPtr<double>(new BlasNumericCtx<double>(
        *this, tempBufSize, skel.spanStart.size() - 1, maxBatchSize));
}

SolveCtxPtr<double> BlasSymbolicCtx::createDoubleSolveContext() {
    return SolveCtxPtr<double>(new BlasSolveCtx<double>(*this));
}

OpsPtr blasOps() { return OpsPtr(new BlasOps); }
