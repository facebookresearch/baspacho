#if 0
#include <alloca.h>
#include <dispenso/parallel_for.h>

#include <chrono>

#include "DebugMacros.h"
#include "MatOpsCpuBase.h"
#include "Utils.h"

#ifdef BASPACHO_USE_MKL

#include "mkl.h"
#define BLAS_INT MKL_INT

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

struct BlasOps : CpuBaseOps {
    // will just contain a reference to the skel
    struct BlasSymbolicInfo : OpaqueData {
        BlasSymbolicInfo(const CoalescedBlockMatrixSkel& skel,
                         int numThreads = 16)
            : skel(skel),
              useThreads(numThreads > 1),
              threadPool(useThreads ? numThreads : 0) {}
        virtual ~BlasSymbolicInfo() {}
        const CoalescedBlockMatrixSkel& skel;
        bool useThreads;
        mutable dispenso::ThreadPool threadPool;
    };

    virtual OpaqueDataPtr initSymbolicInfo(
        const CoalescedBlockMatrixSkel& skel) override {
        return OpaqueDataPtr(new BlasSymbolicInfo(skel));
    }

    virtual void doElimination(const OpaqueData& info, double* data,
                               uint64_t lumpsBegin, uint64_t lumpsEnd,
                               const OpaqueData& elimData) override {
        OpInstance timer(elimStat);
        const BlasSymbolicInfo* pInfo =
            dynamic_cast<const BlasSymbolicInfo*>(&info);
        const SparseEliminationInfo* pElim =
            dynamic_cast<const SparseEliminationInfo*>(&elimData);
        BASPACHO_CHECK_NOTNULL(pInfo);
        BASPACHO_CHECK_NOTNULL(pElim);
        const CoalescedBlockMatrixSkel& skel = pInfo->skel;
        const SparseEliminationInfo& elim = *pElim;

        if (!pInfo->useThreads) {
            for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
                factorLump(skel, data, l);
            }
        } else {
            dispenso::TaskSet taskSet(pInfo->threadPool);
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
            if (!pInfo->useThreads) {
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
                dispenso::TaskSet taskSet(pInfo->threadPool);
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
            if (!pInfo->useThreads) {
                for (uint64_t sRel = 0UL; sRel < numElimRows; sRel++) {
                    eliminateVerySparseRowChain(elim, skel, data, sRel);
                }
            } else {
                dispenso::TaskSet taskSet(pInfo->threadPool);
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

    virtual void potrf(uint64_t n, double* A) override {
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

    virtual void trsm(uint64_t n, uint64_t k, const double* A,
                      double* B) override {
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

    // C = A * B'
    /*static void gemm(uint64_t m, uint64_t n, uint64_t k, const double* A,
                     const double* B, double* C) {
        BLAS_INT argM = m;
        BLAS_INT argN = n;
        BLAS_INT argK = k;
        double argAlpha = 1.0;
        BLAS_INT argLdA = k;
        BLAS_INT argLdB = k;
        double argBeta = 0.0;
        BLAS_INT argLdC = m;
#ifdef BASPACHO_USE_MKL
        cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, argM, argN,
                    argK, argAlpha, (double*)A, argLdA, (double*)B, argLdB,
                    argBeta, C, argLdC);
#else
        char argTransA = 'C';
        char argTransB = 'N';
        dgemm_(&argTransA, &argTransB, &argM, &argN, &argK, &argAlpha,
               (double*)A, &argLdA, (double*)B, &argLdB, &argBeta, C, &argLdC);
#endif
    }*/

    struct AssembleContext : OpaqueData {
        std::vector<uint64_t> spanToChainOffset;
        std::vector<double> tempBuffer;
#ifdef BASPACHO_USE_MKL
        std::vector<double*> tempBufPtrs;
        uint64_t tempCtxSize;
        int maxBatchSize;
#endif
    };

    virtual void saveSyrkGemm(OpaqueData& assCtx, uint64_t m, uint64_t n,
                              uint64_t k, const double* data,
                              uint64_t offset) override {
        OpInstance timer(sygeStat);
        AssembleContext* pAx = dynamic_cast<AssembleContext*>(&assCtx);
        BASPACHO_CHECK_NOTNULL(pAx);
        AssembleContext& ax = *pAx;
        BASPACHO_CHECK_LE(m * n, ax.tempBuffer.size());

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
            double* argC = ax.tempBuffer.data();
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
            double* argC = ax.tempBuffer.data() + gemmOutOffset;
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

    // computes (A|B) * A', upper diag part doesn't matter
    virtual void saveSyrkGemmBatched(OpaqueData& assCtx, uint64_t* ms,
                                     uint64_t* ns, uint64_t* ks,
                                     const double* data, uint64_t* offsets,
                                     int batchSize) {
#ifndef BASPACHO_USE_MKL
        BASPACHO_CHECK(!"Batching not supported");
#else
        OpInstance timer(sygeStat);
        AssembleContext* pAx = dynamic_cast<AssembleContext*>(&assCtx);
        BASPACHO_CHECK_NOTNULL(pAx);
        AssembleContext& ax = *pAx;
        BASPACHO_CHECK_LE(batchSize, ax.maxBatchSize);

        // for testing: serial execution of the batch
        /*ax.tempBufPtrs.clear();
        double* ptr = ax.tempBuffer.data();
        for (int i = 0; i < batchSize; i++) {
            BASPACHO_CHECK_LE(ms[i] * ns[i], ax.tempCtxSize);
            this->gemm(ms[i], ns[i], ks[i], data + offsets[i],
                       data + offsets[i], ptr);
            ax.tempBufPtrs.push_back(ptr);
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

        ax.tempBufPtrs.clear();
        double* ptr = ax.tempBuffer.data();
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

            BASPACHO_CHECK_LE(ms[i] * ns[i], ax.tempCtxSize);
            ax.tempBufPtrs.push_back(ptr);
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

    virtual OpaqueDataPtr createAssembleContext(const OpaqueData& info,
                                                uint64_t tempBufSize,
                                                int maxBatchSize = 1) override {
#ifndef BASPACHO_USE_MKL
        BASPACHO_CHECK_EQ(maxBatchSize, 1);  // Batching not supported
#endif
        const BlasSymbolicInfo* pInfo =
            dynamic_cast<const BlasSymbolicInfo*>(&info);
        BASPACHO_CHECK_NOTNULL(pInfo);
        const CoalescedBlockMatrixSkel& skel = pInfo->skel;
        AssembleContext* ax = new AssembleContext;
        ax->spanToChainOffset.resize(skel.spanStart.size() - 1);
        ax->tempBuffer.resize(tempBufSize * maxBatchSize);
#ifdef BASPACHO_USE_MKL
        ax->tempCtxSize = tempBufSize;
        ax->maxBatchSize = maxBatchSize;
#endif
        return OpaqueDataPtr(ax);
    }

    virtual void prepareAssembleContext(const OpaqueData& info,
                                        OpaqueData& assCtx,
                                        uint64_t targetLump) override {
        const BlasSymbolicInfo* pInfo =
            dynamic_cast<const BlasSymbolicInfo*>(&info);
        AssembleContext* pAx = dynamic_cast<AssembleContext*>(&assCtx);
        BASPACHO_CHECK_NOTNULL(pInfo);
        BASPACHO_CHECK_NOTNULL(pAx);
        const CoalescedBlockMatrixSkel& skel = pInfo->skel;
        AssembleContext& ax = *pAx;

        for (uint64_t i = skel.chainColPtr[targetLump],
                      iEnd = skel.chainColPtr[targetLump + 1];
             i < iEnd; i++) {
            ax.spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
        }
    }

    virtual void assemble(const OpaqueData& info, const OpaqueData& assCtx,
                          double* data, uint64_t rectRowBegin,
                          uint64_t dstStride,  //
                          uint64_t srcColDataOffset, uint64_t srcRectWidth,
                          uint64_t numBlockRows, uint64_t numBlockCols,
                          int numBatch = -1) override {
        OpInstance timer(asmblStat);
        const BlasSymbolicInfo* pInfo =
            dynamic_cast<const BlasSymbolicInfo*>(&info);
        const AssembleContext* pAx =
            dynamic_cast<const AssembleContext*>(&assCtx);
        BASPACHO_CHECK_NOTNULL(pInfo);
        BASPACHO_CHECK_NOTNULL(pAx);
        const CoalescedBlockMatrixSkel& skel = pInfo->skel;
        const AssembleContext& ax = *pAx;
        const uint64_t* chainRowsTillEnd =
            skel.chainRowsTillEnd.data() + srcColDataOffset;
        const uint64_t* toSpan = skel.chainRowSpan.data() + srcColDataOffset;
        const uint64_t* spanToChainOffset = ax.spanToChainOffset.data();
        const uint64_t* spanOffsetInLump = skel.spanOffsetInLump.data();

#ifdef BASPACHO_USE_MKL
        const double* matRectPtr =
            numBatch == -1 ? ax.tempBuffer.data() : ax.tempBufPtrs[numBatch];
#else
        BASPACHO_CHECK_EQ(numBatch, -1);  // Batching not supported
        const double* matRectPtr = ax.tempBuffer.data();
#endif

        if (!pInfo->useThreads) {
            // non-threaded reference implementation:
            for (uint64_t r = 0; r < numBlockRows; r++) {
                uint64_t rBegin = chainRowsTillEnd[r - 1] - rectRowBegin;
                uint64_t rSize = chainRowsTillEnd[r] - rBegin - rectRowBegin;
                uint64_t rParam = toSpan[r];
                uint64_t rOffset = spanToChainOffset[rParam];
                const double* matRowPtr = matRectPtr + rBegin * srcRectWidth;

                uint64_t cEnd = std::min(numBlockCols, r + 1);
                for (uint64_t c = 0; c < cEnd; c++) {
                    uint64_t cStart = chainRowsTillEnd[c - 1] - rectRowBegin;
                    uint64_t cSize =
                        chainRowsTillEnd[c] - cStart - rectRowBegin;
                    uint64_t offset = rOffset + spanOffsetInLump[toSpan[c]];

                    double* dst = data + offset;
                    const double* src = matRowPtr + cStart;
                    stridedMatSub(dst, dstStride, src, srcRectWidth, rSize,
                                  cSize);
                }
            }
        } else {
            dispenso::TaskSet taskSet(pInfo->threadPool);
            dispenso::parallel_for(
                taskSet, dispenso::makeChunkedRange(0, numBlockRows, 3UL),
                [&](int64_t rFrom, int64_t rTo) {
                    for (uint64_t r = rFrom; r < rTo; r++) {
                        uint64_t rBegin =
                            chainRowsTillEnd[r - 1] - rectRowBegin;
                        uint64_t rSize =
                            chainRowsTillEnd[r] - rBegin - rectRowBegin;
                        uint64_t rParam = toSpan[r];
                        uint64_t rOffset = spanToChainOffset[rParam];
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
                                rOffset + spanOffsetInLump[toSpan[c]];

                            double* dst = data + offset;
                            const double* src = matRowPtr + cStart;
                            stridedMatSub(dst, dstStride, src, srcRectWidth,
                                          rSize, cSize);
                        }
                    }
                });
        }
    }

    virtual void solveL(const double* data, uint64_t offM, uint64_t n,
                        double* C, uint64_t offC, uint64_t ldc,
                        uint64_t nRHS) override {
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

    virtual void gemv(const double* data, uint64_t offM, uint64_t nRows,
                      uint64_t nCols, const double* A, uint64_t offA,
                      uint64_t lda, double* C, uint64_t nRHS) override {
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

    virtual void assembleVec(const OpaqueData& info, const double* A,
                             uint64_t chainColPtr, uint64_t numColItems,
                             double* C, uint64_t ldc, uint64_t nRHS) override {
        const BlasSymbolicInfo* pInfo =
            dynamic_cast<const BlasSymbolicInfo*>(&info);
        BASPACHO_CHECK_NOTNULL(pInfo);
        const CoalescedBlockMatrixSkel& skel = pInfo->skel;
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

    virtual void solveLt(const double* data, uint64_t offM, uint64_t n,
                         double* C, uint64_t offC, uint64_t ldc,
                         uint64_t nRHS) override {
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

    virtual void gemvT(const double* data, uint64_t offM, uint64_t nRows,
                       uint64_t nCols, const double* C, uint64_t nRHS,
                       double* A, uint64_t offA, uint64_t lda) override {
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

    virtual void assembleVecT(const OpaqueData& info, const double* C,
                              uint64_t ldc, uint64_t nRHS, double* A,
                              uint64_t chainColPtr,
                              uint64_t numColItems) override {
        const BlasSymbolicInfo* pInfo =
            dynamic_cast<const BlasSymbolicInfo*>(&info);
        BASPACHO_CHECK_NOTNULL(pInfo);
        const CoalescedBlockMatrixSkel& skel = pInfo->skel;
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
};

OpsPtr blasOps() { return OpsPtr(new BlasOps); }
#endif