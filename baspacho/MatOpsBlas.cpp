
#include <alloca.h>
#include <dispenso/parallel_for.h>
#include <glog/logging.h>

#include <chrono>

#include "MatOps.h"
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

void dpotrf_(char* uplo, BLAS_INT* n, double* A, BLAS_INT* lda, BLAS_INT* info);

void dtrsm_(char* side, char* uplo, char* transa, char* diag, BLAS_INT* m,
            BLAS_INT* n, double* alpha, double* A, BLAS_INT* lda, double* B,
            BLAS_INT* ldb);

void dgemm_(char* transa, char* transb, BLAS_INT* m, BLAS_INT* n, BLAS_INT* k,
            double* alpha, double* A, BLAS_INT* lda, double* B, BLAS_INT* ldb,
            double* beta, double* C, BLAS_INT* ldc);
}

#endif

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

// helper for elimination
static void factorLump(const BlockMatrixSkel& skel, double* data,
                       uint64_t lump) {
    uint64_t lumpStart = skel.lumpStart[lump];
    uint64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
    uint64_t colStart = skel.chainColPtr[lump];
    uint64_t dataPtr = skel.chainData[colStart];

    // compute lower diag cholesky dec on diagonal block
    Eigen::Map<MatRMaj<double>> diagBlock(data + dataPtr, lumpSize, lumpSize);
    { Eigen::LLT<Eigen::Ref<MatRMaj<double>>> llt(diagBlock); }

    uint64_t gatheredStart = skel.boardColPtr[lump];
    uint64_t gatheredEnd = skel.boardColPtr[lump + 1];
    uint64_t rowDataStart = skel.boardChainColOrd[gatheredStart + 1];
    uint64_t rowDataEnd = skel.boardChainColOrd[gatheredEnd - 1];
    uint64_t belowDiagStart = skel.chainData[colStart + rowDataStart];
    uint64_t numRows = skel.chainRowsTillEnd[colStart + rowDataEnd - 1] -
                       skel.chainRowsTillEnd[colStart + rowDataStart - 1];

    Eigen::Map<MatRMaj<double>> belowDiagBlock(data + belowDiagStart, numRows,
                                               lumpSize);
    diagBlock.triangularView<Eigen::Lower>()
        .transpose()
        .solveInPlace<Eigen::OnTheRight>(belowDiagBlock);
}

inline void stridedMatSub(double* dst, uint64_t dstStride, const double* src,
                          uint64_t srcStride, uint64_t rSize, uint64_t cSize) {
    for (uint j = 0; j < rSize; j++) {
        for (uint i = 0; i < cSize; i++) {
            dst[i] -= src[i];
        }
        dst += dstStride;
        src += srcStride;
    }
}

struct BlasOps : Ops {
    // will just contain a reference to the skel
    struct OpaqueDataMatrixSkel : OpaqueData {
        OpaqueDataMatrixSkel(const BlockMatrixSkel& skel)
            : skel(skel), threadPool(dispenso::globalThreadPool()) {
            threadPool.resize(16);
        }
        virtual ~OpaqueDataMatrixSkel() {}
        const BlockMatrixSkel& skel;
        dispenso::ThreadPool& threadPool;
    };

    virtual void printStats() const override {
        LOG(INFO) << "matOp stats:"
                  << "\nelim: #=" << elimCalls << ", time=" << elimTotTime
                  << "s, last=" << elimLastCallTime
                  << "s, max=" << elimMaxCallTime << "s"
                  << "\nBiggest dense block: " << potrfBiggestN
                  << "\npotrf: #=" << potrfCalls << ", time=" << potrfTotTime
                  << "s, last=" << potrfLastCallTime
                  << "s, max=" << potrfMaxCallTime << "s"
                  << "\ntrsm: #=" << trsmCalls << ", time=" << trsmTotTime
                  << "s, last=" << trsmLastCallTime
                  << "s, max=" << trsmMaxCallTime << "s"
                  << "\ngemm: #=" << gemmCalls << ", time=" << gemmTotTime
                  << "s, last=" << gemmLastCallTime
                  << "s, max=" << gemmMaxCallTime << "s"
                  << "\nasmbl: #=" << asmblCalls << ", time=" << asmblTotTime
                  << "s, last=" << asmblLastCallTime
                  << "s, max=" << asmblMaxCallTime << "s";
    }

    virtual OpaqueDataPtr prepareMatrixSkel(
        const BlockMatrixSkel& skel) override {
        return OpaqueDataPtr(new OpaqueDataMatrixSkel(skel));
    }

    struct OpaqueDataElimData : OpaqueData {
        OpaqueDataElimData() {}
        virtual ~OpaqueDataElimData() {}

        // per-row pointers to chains in a rectagle:
        // * span-rows from lumpToSpan[lumpsEnd],
        // * board cols in interval lumpsBegin:lumpsEnd
        uint64_t spanRowBegin;
        uint64_t maxBufferSize;
        vector<uint64_t> rowPtr;       // row data pointer
        vector<uint64_t> colLump;      // col-lump
        vector<uint64_t> chainColOrd;  // order in col chain elements
    };

    static uint64_t computeMaxBufSize(const OpaqueDataElimData& elim,
                                      const BlockMatrixSkel& skel,
                                      uint64_t sRel) {
        uint64_t maxBufferSize = 0;

        // iterate over chains present in this row
        for (uint64_t i = elim.rowPtr[sRel], iEnd = elim.rowPtr[sRel + 1];
             i < iEnd; i++) {
            uint64_t lump = elim.colLump[i];
            uint64_t chainColOrd = elim.chainColOrd[i];
            CHECK_GE(chainColOrd, 1);  // there must be a diagonal block

            uint64_t ptrStart = skel.chainColPtr[lump] + chainColOrd;
            uint64_t ptrEnd = skel.chainColPtr[lump + 1];

            uint64_t nRowsAbove = skel.chainRowsTillEnd[ptrStart - 1];
            uint64_t nRowsChain = skel.chainRowsTillEnd[ptrStart] - nRowsAbove;
            uint64_t nRowsOnward = skel.chainRowsTillEnd[ptrEnd - 1];

            maxBufferSize = max(maxBufferSize, nRowsOnward * nRowsChain);
        }

        return maxBufferSize;
    }

    virtual OpaqueDataPtr prepareElimination(const BlockMatrixSkel& skel,
                                             uint64_t lumpsBegin,
                                             uint64_t lumpsEnd) override {
        OpaqueDataElimData* elim = new OpaqueDataElimData;

        uint64_t spanRowBegin = skel.lumpToSpan[lumpsEnd];
        uint64_t numSpanRows = skel.spanStart.size() - 1 - spanRowBegin;
        elim->rowPtr.assign(numSpanRows + 1, 0);
        for (uint64_t l = lumpsBegin; l < lumpsEnd; l++) {
            for (uint64_t i = skel.chainColPtr[l],
                          iEnd = skel.chainColPtr[l + 1];
                 i < iEnd; i++) {
                uint64_t s = skel.chainRowSpan[i];
                if (s < spanRowBegin) {
                    continue;
                }
                uint64_t sRel = s - spanRowBegin;
                elim->rowPtr[sRel]++;
            }
        }
        /*size_t empty = 0;
        for (int64_t q = 0; q < numSpanRows; q++) {
            empty += !elim->rowPtr[q];
        }*/
        uint64_t totNumChains = cumSumVec(elim->rowPtr);
        // LOG(INFO) << "Empty rows: " << empty << " / " << numSpanRows
        //          << ", to elim " << totNumChains << " blocks";
        elim->colLump.resize(totNumChains);
        elim->chainColOrd.resize(totNumChains);
        for (uint64_t l = lumpsBegin; l < lumpsEnd; l++) {
            for (uint64_t iBegin = skel.chainColPtr[l],
                          iEnd = skel.chainColPtr[l + 1], i = iBegin;
                 i < iEnd; i++) {
                uint64_t s = skel.chainRowSpan[i];
                if (s < spanRowBegin) {
                    continue;
                }
                uint64_t sRel = s - spanRowBegin;
                elim->colLump[elim->rowPtr[sRel]] = l;
                elim->chainColOrd[elim->rowPtr[sRel]] = i - iBegin;
                elim->rowPtr[sRel]++;
            }
        }
        rewindVec(elim->rowPtr);
        elim->spanRowBegin = spanRowBegin;

        elim->maxBufferSize = 0;
        for (uint64_t r = 0; r < elim->rowPtr.size() - 1; r++) {
            elim->maxBufferSize =
                max(elim->maxBufferSize, computeMaxBufSize(*elim, skel, r));
        }
        return OpaqueDataPtr(elim);
    }

    struct ElimContext {
        std::vector<double> tempBuffer;
        std::vector<uint64_t> spanToChainOffset;
        ElimContext(uint64_t bufSize, uint64_t numSpans)
            : tempBuffer(bufSize), spanToChainOffset(numSpans) {}
    };

    static void prepareContextForTargetLump(
        const BlockMatrixSkel& skel, uint64_t targetLump,
        vector<uint64_t>& spanToChainOffset) {
        // incomment below if check is needed
        // spanToChainOffset.assign(skel.spanStart.size() - 1, kInvalid);
        for (uint64_t i = skel.chainColPtr[targetLump],
                      iEnd = skel.chainColPtr[targetLump + 1];
             i < iEnd; i++) {
            spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
        }
    }

    static uint64_t bisect(const uint64_t* array, uint64_t size,
                           uint64_t needle) {
        uint64_t a = 0, b = size;
        while (b - a > 1) {
            uint64_t m = (a + b) / 2;
            if (needle >= array[m]) {
                a = m;
            } else {
                b = m;
            }
        }
        return a;
    }

    static void eliminateRowChain(const OpaqueDataElimData& elim,
                                  const BlockMatrixSkel& skel, double* data,
                                  uint64_t sRel, ElimContext& ctx) {
        uint64_t s = sRel + elim.spanRowBegin;
        if (elim.rowPtr[sRel] == elim.rowPtr[sRel + 1]) {
            return;
        }
        uint64_t targetLump = skel.spanToLump[s];
        uint64_t targetLumpSize =
            skel.lumpStart[targetLump + 1] - skel.lumpStart[targetLump];
        uint64_t spanOffsetInLump =
            skel.spanStart[s] - skel.lumpStart[targetLump];
        prepareContextForTargetLump(skel, targetLump, ctx.spanToChainOffset);

        // iterate over chains present in this row
        for (uint64_t i = elim.rowPtr[sRel], iEnd = elim.rowPtr[sRel + 1];
             i < iEnd; i++) {
            uint64_t lump = elim.colLump[i];
            uint64_t chainColOrd = elim.chainColOrd[i];
            CHECK_GE(chainColOrd, 1);  // there must be a diagonal block

            uint64_t ptrStart = skel.chainColPtr[lump] + chainColOrd;
            uint64_t ptrEnd = skel.chainColPtr[lump + 1];
            CHECK_EQ(skel.chainRowSpan[ptrStart], s);

            uint64_t nRowsAbove = skel.chainRowsTillEnd[ptrStart - 1];
            uint64_t nRowsChain = skel.chainRowsTillEnd[ptrStart] - nRowsAbove;
            uint64_t nRowsOnward = skel.chainRowsTillEnd[ptrEnd - 1];
            uint64_t dataOffset = skel.chainData[ptrStart];
            CHECK_EQ(nRowsChain, skel.spanStart[s + 1] - skel.spanStart[s]);
            uint64_t lumpSize = skel.lumpStart[lump + 1] - skel.lumpStart[lump];

            Eigen::Map<MatRMaj<double>> chainSubMat(data + dataOffset,
                                                    nRowsChain, lumpSize);
            Eigen::Map<MatRMaj<double>> chainOnwardSubMat(
                data + dataOffset, nRowsOnward, lumpSize);

            CHECK_GE(ctx.tempBuffer.size(), nRowsOnward * nRowsChain);
            Eigen::Map<MatRMaj<double>> prod(ctx.tempBuffer.data(), nRowsOnward,
                                             nRowsChain);
            prod = chainOnwardSubMat * chainSubMat.transpose();

            // assemble blocks, iterating on chain and below chains
            for (uint64_t ptr = ptrStart; ptr < ptrEnd; ptr++) {
                uint64_t s2 = skel.chainRowSpan[ptr];
                uint64_t relRow = skel.chainRowsTillEnd[ptr - 1] - nRowsAbove;
                uint64_t s2_size =
                    skel.chainRowsTillEnd[ptr] - nRowsAbove - relRow;

                // incomment below if check is needed
                // CHECK(ctx.spanToChainOffset[s2] != kInvalid);
                double* targetData =
                    data + spanOffsetInLump + ctx.spanToChainOffset[s2];

                stridedMatSub(targetData, targetLumpSize,
                              ctx.tempBuffer.data() + nRowsChain * relRow,
                              nRowsChain, s2_size, nRowsChain);
            }
        }
    }

    static void eliminateVerySparseRowChain(const OpaqueDataElimData& elim,
                                            const BlockMatrixSkel& skel,
                                            double* data, uint64_t sRel) {
        uint64_t s = sRel + elim.spanRowBegin;
        if (elim.rowPtr[sRel] == elim.rowPtr[sRel + 1]) {
            return;
        }
        uint64_t targetLump = skel.spanToLump[s];
        uint64_t targetLumpSize =
            skel.lumpStart[targetLump + 1] - skel.lumpStart[targetLump];
        uint64_t spanOffsetInLump =
            skel.spanStart[s] - skel.lumpStart[targetLump];
        uint64_t bisectStart = skel.chainColPtr[targetLump];
        uint64_t bisectEnd = skel.chainColPtr[targetLump + 1];

        // iterate over chains present in this row
        for (uint64_t i = elim.rowPtr[sRel], iEnd = elim.rowPtr[sRel + 1];
             i < iEnd; i++) {
            uint64_t lump = elim.colLump[i];
            uint64_t chainColOrd = elim.chainColOrd[i];
            CHECK_GE(chainColOrd, 1);  // there must be a diagonal block

            uint64_t ptrStart = skel.chainColPtr[lump] + chainColOrd;
            uint64_t ptrEnd = skel.chainColPtr[lump + 1];
            CHECK_EQ(skel.chainRowSpan[ptrStart], s);

            uint64_t nRowsAbove = skel.chainRowsTillEnd[ptrStart - 1];
            uint64_t nRowsChain = skel.chainRowsTillEnd[ptrStart] - nRowsAbove;
            uint64_t nRowsOnward = skel.chainRowsTillEnd[ptrEnd - 1];
            uint64_t dataOffset = skel.chainData[ptrStart];
            CHECK_EQ(nRowsChain, skel.spanStart[s + 1] - skel.spanStart[s]);
            uint64_t lumpSize = skel.lumpStart[lump + 1] - skel.lumpStart[lump];

            Eigen::Map<MatRMaj<double>> chainSubMat(data + dataOffset,
                                                    nRowsChain, lumpSize);
            Eigen::Map<MatRMaj<double>> chainOnwardSubMat(
                data + dataOffset, nRowsOnward, lumpSize);

            double* tempBuffer =
                (double*)alloca(sizeof(double) * nRowsOnward * nRowsChain);
            Eigen::Map<MatRMaj<double>> prod(tempBuffer, nRowsOnward,
                                             nRowsChain);
            prod = chainOnwardSubMat * chainSubMat.transpose();

            // assemble blocks, iterating on chain and below chains
            for (uint64_t ptr = ptrStart; ptr < ptrEnd; ptr++) {
                uint64_t s2 = skel.chainRowSpan[ptr];
                uint64_t relRow = skel.chainRowsTillEnd[ptr - 1] - nRowsAbove;
                uint64_t s2_size =
                    skel.chainRowsTillEnd[ptr] - nRowsAbove - relRow;

                uint64_t pos = bisect(skel.chainRowSpan.data() + bisectStart,
                                      bisectEnd - bisectStart, s2);
                uint64_t chainOffset = skel.chainData[bisectStart + pos];
                double* targetData = data + spanOffsetInLump + chainOffset;

                stridedMatSub(targetData, targetLumpSize,
                              tempBuffer + nRowsChain * relRow, nRowsChain,
                              s2_size, nRowsChain);
            }
        }
    }

    virtual void doElimination(const OpaqueData& ref, double* data,
                               uint64_t lumpsBegin, uint64_t lumpsEnd,
                               const OpaqueData& elimData) override {
        auto start = hrc::now();
        const OpaqueDataMatrixSkel* pSkel =
            dynamic_cast<const OpaqueDataMatrixSkel*>(&ref);
        const OpaqueDataElimData* pElim =
            dynamic_cast<const OpaqueDataElimData*>(&elimData);
        CHECK_NOTNULL(pSkel);
        CHECK_NOTNULL(pElim);
        const BlockMatrixSkel& skel = pSkel->skel;
        const OpaqueDataElimData& elim = *pElim;

        dispenso::TaskSet taskSet(pSkel->threadPool);
        dispenso::parallel_for(
            taskSet, dispenso::makeChunkedRange(lumpsBegin, lumpsEnd, 5UL),
            [&](int64_t lBegin, int64_t lEnd) {
                for (int64_t l = lBegin; l < lEnd; l++) {
                    factorLump(skel, data, l);
                }
            });

        if (elim.colLump.size() > 3 * (elim.rowPtr.size() - 1)) {
            vector<ElimContext> contexts;
            dispenso::TaskSet taskSet1(pSkel->threadPool);
            uint64_t numSpans = skel.spanStart.size() - 1;
            dispenso::parallel_for(
                taskSet1, contexts,
                [=]() -> ElimContext {
                    return ElimContext(elim.maxBufferSize, numSpans);
                },
                dispenso::makeChunkedRange(0UL, elim.rowPtr.size() - 1, 5UL),
                [&, this](ElimContext& ctx, size_t sBegin, size_t sEnd) {
                    for (uint64_t sRel = sBegin; sRel < sEnd; sRel++) {
                        eliminateRowChain(elim, skel, data, sRel, ctx);
                    }
                });
        } else {
            dispenso::TaskSet taskSet1(pSkel->threadPool);
            dispenso::parallel_for(
                taskSet1,
                dispenso::makeChunkedRange(0UL, elim.rowPtr.size() - 1, 5UL),
                [&, this](size_t sBegin, size_t sEnd) {
                    for (uint64_t sRel = sBegin; sRel < sEnd; sRel++) {
                        eliminateVerySparseRowChain(elim, skel, data, sRel);
                    }
                });
        }

        elimLastCallTime = tdelta(hrc::now() - start).count();
        elimCalls++;
        elimTotTime += elimLastCallTime;
        elimMaxCallTime = std::max(elimMaxCallTime, elimLastCallTime);
    }

    virtual void potrf(uint64_t n, double* A) override {
        auto start = hrc::now();

        char argUpLo = 'U';
        BLAS_INT argN = n;
        BLAS_INT argLdA = n;
#ifdef BASPACHO_USE_MKL
        LAPACKE_dpotrf(LAPACK_COL_MAJOR, argUpLo, argN, A, argLdA);
#else
        BLAS_INT info;
        dpotrf_(&argUpLo, &argN, A, &argLdA, &info);
#endif

        potrfLastCallTime = tdelta(hrc::now() - start).count();
        potrfCalls++;
        potrfTotTime += potrfLastCallTime;
        potrfMaxCallTime = std::max(potrfMaxCallTime, potrfLastCallTime);
        potrfBiggestN = std::max(potrfBiggestN, n);
    }

    virtual void trsm(uint64_t n, uint64_t k, const double* A,
                      double* B) override {
        auto start = hrc::now();

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

        trsmLastCallTime = tdelta(hrc::now() - start).count();
        trsmCalls++;
        trsmTotTime += trsmLastCallTime;
        trsmMaxCallTime = std::max(trsmMaxCallTime, trsmLastCallTime);
    }

    // C = A * B'
    virtual void gemm(uint64_t m, uint64_t n, uint64_t k, const double* A,
                      const double* B, double* C) override {
        auto start = hrc::now();

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

        gemmLastCallTime = tdelta(hrc::now() - start).count();
        gemmCalls++;
        gemmTotTime += gemmLastCallTime;
        gemmMaxCallTime = std::max(gemmMaxCallTime, gemmLastCallTime);
    }

    struct AssembleContext : OpaqueData {
        std::vector<uint64_t> paramToChainOffset;
        uint64_t stride;
        std::vector<double> tempBuffer;
#ifdef BASPACHO_USE_MKL
        std::vector<double*> tempBufPtrs;
        uint64_t tempCtxSize;
        int maxBatchSize;
#endif
    };

    virtual void gemmToTemp(OpaqueData& assCtx, uint64_t m, uint64_t n,
                            uint64_t k, const double* A,
                            const double* B) override {
        AssembleContext* pAx = dynamic_cast<AssembleContext*>(&assCtx);
        CHECK_NOTNULL(pAx);
        AssembleContext& ax = *pAx;

        ax.stride = m;
        CHECK_LE(m * n, ax.tempBuffer.size());
        this->gemm(m, n, k, A, B, ax.tempBuffer.data());
    }

    // computes (A|B) * A', upper diag part doesn't matter
    virtual void saveSyrkGemmBatched(OpaqueData& assCtx, uint64_t* ms,
                                     uint64_t* ns, uint64_t* ks,
                                     const double* data, uint64_t* offsets,
                                     int batchSize) {
#ifndef BASPACHO_USE_MKL
        LOG(FATAL) << "Batching not supported";
#else
        auto start = hrc::now();
        AssembleContext* pAx = dynamic_cast<AssembleContext*>(&assCtx);
        CHECK_NOTNULL(pAx);
        AssembleContext& ax = *pAx;
        CHECK_LE(batchSize, ax.maxBatchSize);

        // for testing: serial execution of the batch
        /*ax.tempBufPtrs.clear();
        double* ptr = ax.tempBuffer.data();
        for (int i = 0; i < batchSize; i++) {
            CHECK_LE(ms[i] * ns[i], ax.tempCtxSize);
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

            CHECK_LE(ms[i] * ns[i], ax.tempCtxSize);
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

        gemmLastCallTime = tdelta(hrc::now() - start).count();
        gemmCalls++;
        gemmTotTime += gemmLastCallTime;
        gemmMaxCallTime = std::max(gemmMaxCallTime, gemmLastCallTime);
#endif
    }

    virtual OpaqueDataPtr createAssembleContext(const OpaqueData& ref,
                                                uint64_t tempBufSize,
                                                int maxBatchSize = 1) override {
#ifndef BASPACHO_USE_MKL
        CHECK_EQ(maxBatchSize, 1) << "Batching not supported";
#endif
        const OpaqueDataMatrixSkel* pSkel =
            dynamic_cast<const OpaqueDataMatrixSkel*>(&ref);
        CHECK_NOTNULL(pSkel);
        const BlockMatrixSkel& skel = pSkel->skel;
        AssembleContext* ax = new AssembleContext;
        ax->paramToChainOffset.resize(skel.spanStart.size() - 1);
        ax->tempBuffer.resize(tempBufSize * maxBatchSize);
#ifdef BASPACHO_USE_MKL
        ax->tempCtxSize = tempBufSize;
        ax->maxBatchSize = maxBatchSize;
#endif
        return OpaqueDataPtr(ax);
    }

    virtual void prepareAssembleContext(const OpaqueData& ref,
                                        OpaqueData& assCtx,
                                        uint64_t targetLump) override {
        const OpaqueDataMatrixSkel* pSkel =
            dynamic_cast<const OpaqueDataMatrixSkel*>(&ref);
        AssembleContext* pAx = dynamic_cast<AssembleContext*>(&assCtx);
        CHECK_NOTNULL(pSkel);
        CHECK_NOTNULL(pAx);
        const BlockMatrixSkel& skel = pSkel->skel;
        AssembleContext& ax = *pAx;

        for (uint64_t i = skel.chainColPtr[targetLump],
                      iEnd = skel.chainColPtr[targetLump + 1];
             i < iEnd; i++) {
            ax.paramToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
        }
    }

    virtual void assemble(const OpaqueData& ref, const OpaqueData& assCtx,
                          double* data, uint64_t rectRowBegin,
                          uint64_t dstStride,  //
                          uint64_t srcColDataOffset, uint64_t srcRectWidth,
                          uint64_t numBlockRows, uint64_t numBlockCols,
                          int numBatch = -1) override {
        auto start = hrc::now();
        const OpaqueDataMatrixSkel* pSkel =
            dynamic_cast<const OpaqueDataMatrixSkel*>(&ref);
        const AssembleContext* pAx =
            dynamic_cast<const AssembleContext*>(&assCtx);
        CHECK_NOTNULL(pSkel);
        CHECK_NOTNULL(pAx);
        const BlockMatrixSkel& skel = pSkel->skel;
        const AssembleContext& ax = *pAx;
        const uint64_t* chainRowsTillEnd =
            skel.chainRowsTillEnd.data() + srcColDataOffset;
        const uint64_t* toSpan = skel.chainRowSpan.data() + srcColDataOffset;
        const uint64_t* paramToChainOffset = ax.paramToChainOffset.data();
        const uint64_t* spanOffsetInLump = skel.spanOffsetInLump.data();

        // uint64_t rectStride = ax.stride;
#ifdef BASPACHO_USE_MKL
        const double* matRectPtr =
            numBatch == -1 ? ax.tempBuffer.data() : ax.tempBufPtrs[numBatch];
#else
        CHECK_EQ(numBatch, -1) << "Batching not supported";
        const double* matRectPtr = ax.tempBuffer.data();
#endif

#if 1
        dispenso::TaskSet taskSet(pSkel->threadPool);
        dispenso::parallel_for(
            taskSet, dispenso::makeChunkedRange(0, numBlockRows, 3UL),
            [&](int64_t rFrom, int64_t rTo) {
                for (uint64_t r = rFrom; r < rTo; r++) {
                    // for (uint64_t r = 0; r < numBlockRows; r++) {
                    uint64_t rBegin = chainRowsTillEnd[r - 1] - rectRowBegin;
                    uint64_t rSize =
                        chainRowsTillEnd[r] - rBegin - rectRowBegin;
                    uint64_t rParam = toSpan[r];
                    uint64_t rOffset = paramToChainOffset[rParam];
                    const double* matRowPtr =
                        matRectPtr + rBegin * srcRectWidth;

                    uint64_t cEnd = std::min(numBlockCols, r + 1);
                    uint64_t nextCStart = chainRowsTillEnd[-1] - rectRowBegin;
                    for (uint64_t c = 0; c < cEnd; c++) {
                        uint64_t cStart = nextCStart;
                        nextCStart = chainRowsTillEnd[c] - rectRowBegin;
                        uint64_t cSize = nextCStart - cStart;
                        uint64_t offset = rOffset + spanOffsetInLump[toSpan[c]];

                        double* dst = data + offset;
                        const double* src = matRowPtr + cStart;
                        stridedMatSub(dst, dstStride, src, srcRectWidth, rSize,
                                      cSize);
                    }
                }
            });
#else
        for (uint64_t r = 0; r < numBlockRows; r++) {
            uint64_t rBegin = chainRowsTillEnd[r - 1] - rectRowBegin;
            uint64_t rSize = chainRowsTillEnd[r] - rBegin - rectRowBegin;
            uint64_t rParam = toSpan[r];
            uint64_t rOffset = paramToChainOffset[rParam];
            const double* matRowPtr = matRectPtr + rBegin * srcRectWidth;

            uint64_t cEnd = std::min(numBlockCols, r + 1);
            for (uint64_t c = 0; c < cEnd; c++) {
                uint64_t cStart = chainRowsTillEnd[c - 1] - rectRowBegin;
                uint64_t cSize = chainRowsTillEnd[c] - cStart - rectRowBegin;
                uint64_t offset = rOffset + spanOffsetInLump[toSpan[c]];

                double* dst = data + offset;
                const double* src = matRowPtr + cStart;
                stridedMatSub(dst, dstStride, src, srcRectWidth, rSize, cSize);
            }
        }
#endif

        asmblLastCallTime = tdelta(hrc::now() - start).count();
        asmblCalls++;
        asmblTotTime += asmblLastCallTime;
        asmblMaxCallTime = std::max(asmblMaxCallTime, asmblLastCallTime);
    }

    uint64_t elimCalls = 0;
    double elimTotTime = 0.0;
    double elimLastCallTime;
    double elimMaxCallTime = 0.0;
    uint64_t potrfBiggestN = 0;
    uint64_t potrfCalls = 0;
    double potrfTotTime = 0.0;
    double potrfLastCallTime;
    double potrfMaxCallTime = 0.0;
    uint64_t trsmCalls = 0;
    double trsmTotTime = 0.0;
    double trsmLastCallTime;
    double trsmMaxCallTime = 0.0;
    uint64_t gemmCalls = 0;
    double gemmTotTime = 0.0;
    double gemmLastCallTime;
    double gemmMaxCallTime = 0.0;
    uint64_t asmblCalls = 0;
    double asmblTotTime = 0.0;
    double asmblLastCallTime;
    double asmblMaxCallTime = 0.0;
};

OpsPtr blasOps() { return OpsPtr(new BlasOps); }