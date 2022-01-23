
#include <dispenso/parallel_for.h>
#include <glog/logging.h>

#include <chrono>

#include "MatOps.h"
#include "Utils.h"

// BLAS/LAPACK famously go without headers.
extern "C" {

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

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

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

static void prepareContextForTargetAggreg(const BlockMatrixSkel& skel,
                                          uint64_t targetLump,
                                          vector<uint64_t>& spanToChainOffset) {
    spanToChainOffset.assign(skel.spanStart.size() - 1, 999999);
    for (uint64_t i = skel.chainColPtr[targetLump],
                  iEnd = skel.chainColPtr[targetLump + 1];
         i < iEnd; i++) {
        spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
    }
}

using OuterStride = Eigen::OuterStride<>;
using OuterStridedMatM = Eigen::Map<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0,
    OuterStride>;

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
                  << "\nBiggest dense block: " << potrfBiggestN
                  << "\npotrf: #=" << potrfCalls << ", time=" << potrfTotTime
                  << "s, last=" << potrfLastCallTime
                  << "s, max=" << potrfMaxCallTime << "s"
                  << "\ntrsm: #=" << trsmCalls << ", time=" << trsmTotTime
                  << "s, last=" << trsmLastCallTime
                  << "s, max=" << trsmMaxCallTime << "s"
                  << "\ngemm: #=" << gemmCalls << ", time=" << gemmTotTime
                  << "s, last=" << gemmLastCallTime
                  << "s, max=" << gemmMaxCallTime << "s";
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
        vector<uint64_t> rowPtr;       // row data pointer
        vector<uint64_t> colLump;      // col-lump
        vector<uint64_t> chainColOrd;  // order in col chain elements
    };

    // TODO: unit test
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
        uint64_t totNumChains = cumSumVec(elim->rowPtr);
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
        return OpaqueDataPtr(elim);
    }

    struct ElimContext {
        std::vector<double> tempBuffer;
        std::vector<uint64_t> spanToChainOffset;
    };

    static void eliminateRowChain(const OpaqueDataElimData& elim,
                                  const BlockMatrixSkel& skel, double* data,
                                  uint64_t sRel, ElimContext& ctx) {
        uint64_t s = sRel + elim.spanRowBegin;
        uint64_t targetLump = skel.spanToLump[s];
        uint64_t targetLumpSize =
            skel.lumpStart[targetLump + 1] - skel.lumpStart[targetLump];
        uint64_t spanOffsetInLump =
            skel.spanStart[s] - skel.lumpStart[targetLump];
        prepareContextForTargetAggreg(skel, targetLump, ctx.spanToChainOffset);

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

            ctx.tempBuffer.resize(nRowsOnward * nRowsChain);
            Eigen::Map<MatRMaj<double>> prod(ctx.tempBuffer.data(), nRowsOnward,
                                             nRowsChain);
            prod = chainOnwardSubMat * chainSubMat.transpose();

            // assemble blocks, iterating on chain and below chains
            for (uint64_t ptr = ptrStart; ptr < ptrEnd; ptr++) {
                uint64_t s2 = skel.chainRowSpan[ptr];
                uint64_t relRow = skel.chainRowsTillEnd[ptr - 1] - nRowsAbove;
                uint64_t s2_size =
                    skel.chainRowsTillEnd[ptr] - nRowsAbove - relRow;
                CHECK_EQ(s2_size, skel.spanStart[s2 + 1] - skel.spanStart[s2]);

                double* targetData =
                    data + spanOffsetInLump + ctx.spanToChainOffset[s2];

                OuterStridedMatM targetBlock(targetData, s2_size, nRowsChain,
                                             OuterStride(targetLumpSize));
                targetBlock -= prod.block(relRow, 0, s2_size, nRowsChain);
            }
        }
    }

    virtual void doElimination(const OpaqueData& ref, double* data,
                               uint64_t lumpsBegin, uint64_t lumpsEnd,
                               const OpaqueData& elimData) override {
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

        // TODO: parallel2
        vector<ElimContext> contexts;

        dispenso::TaskSet taskSet1(pSkel->threadPool);
        dispenso::parallel_for(
            taskSet1, contexts, []() -> ElimContext { return ElimContext(); },
            dispenso::makeChunkedRange(0UL, elim.rowPtr.size() - 1, 5UL),
            [&, this](ElimContext& ctx, size_t sBegin, size_t sEnd) {
                for (uint64_t sRel = sBegin; sRel < sEnd; sRel++) {
                    eliminateRowChain(elim, skel, data, sRel, ctx);
                }
            });
    }

    virtual void potrf(uint64_t n, double* A) override {
        auto start = hrc::now();

        char argUpLo = 'U';
        BLAS_INT argN = n;
        BLAS_INT argLdA = n;
        BLAS_INT info;

        dpotrf_(&argUpLo, &argN, A, &argLdA, &info);

        potrfLastCallTime = tdelta(hrc::now() - start).count();
        potrfCalls++;
        potrfTotTime += potrfLastCallTime;
        potrfMaxCallTime = std::max(potrfMaxCallTime, potrfLastCallTime);
        potrfBiggestN = std::max(potrfBiggestN, n);
    }

    virtual void trsm(uint64_t n, uint64_t k, const double* A,
                      double* B) override {
        auto start = hrc::now();

        char argSide = 'L';
        char argUpLo = 'U';
        char argTransA = 'C';
        char argDiag = 'N';
        BLAS_INT argM = n;
        BLAS_INT argN = k;
        double argAlpha = 1.0;
        BLAS_INT argLdA = n;
        BLAS_INT argLdB = n;

        dtrsm_(&argSide, &argUpLo, &argTransA, &argDiag, &argM, &argN,
               &argAlpha, (double*)A, &argLdA, B, &argLdB);

        trsmLastCallTime = tdelta(hrc::now() - start).count();
        trsmCalls++;
        trsmTotTime += trsmLastCallTime;
        trsmMaxCallTime = std::max(trsmMaxCallTime, trsmLastCallTime);
    }

    // C = A * B'
    virtual void gemm(uint64_t m, uint64_t n, uint64_t k, const double* A,
                      const double* B, double* C) override {
        auto start = hrc::now();

        char argTransA = 'C';
        char argTransB = 'N';
        BLAS_INT argM = m;
        BLAS_INT argN = n;
        BLAS_INT argK = k;
        double argAlpha = 1.0;
        BLAS_INT argLdA = k;
        BLAS_INT argLdB = k;
        double argBeta = 0.0;
        BLAS_INT argLdC = m;

        dgemm_(&argTransA, &argTransB, &argM, &argN, &argK, &argAlpha,
               (double*)A, &argLdA, (double*)B, &argLdB, &argBeta, C, &argLdC);

        gemmLastCallTime = tdelta(hrc::now() - start).count();
        gemmCalls++;
        gemmTotTime += gemmLastCallTime;
        gemmMaxCallTime = std::max(gemmMaxCallTime, gemmLastCallTime);
    }

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
};

OpsPtr blasOps() { return OpsPtr(new BlasOps); }