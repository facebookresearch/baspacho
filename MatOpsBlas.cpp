
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

static void factorAggreg(const BlockMatrixSkel& skel, double* data,
                         uint64_t range) {
    uint64_t rangeStart = skel.rangeStart[range];
    uint64_t rangeSize = skel.rangeStart[range + 1] - rangeStart;
    uint64_t colStart = skel.sliceColPtr[range];
    uint64_t dataPtr = skel.sliceData[colStart];

    // compute lower diag cholesky dec on diagonal block
    Eigen::Map<MatRMaj<double>> diagBlock(data + dataPtr, rangeSize, rangeSize);
    { Eigen::LLT<Eigen::Ref<MatRMaj<double>>> llt(diagBlock); }

    uint64_t gatheredStart = skel.slabColPtr[range];
    uint64_t gatheredEnd = skel.slabColPtr[range + 1];
    uint64_t rowDataStart = skel.slabSliceColOrd[gatheredStart + 1];
    uint64_t rowDataEnd = skel.slabSliceColOrd[gatheredEnd - 1];
    uint64_t belowDiagStart = skel.sliceData[colStart + rowDataStart];
    uint64_t numRows = skel.sliceRowsTillEnd[colStart + rowDataEnd - 1] -
                       skel.sliceRowsTillEnd[colStart + rowDataStart - 1];

    Eigen::Map<MatRMaj<double>> belowDiagBlock(data + belowDiagStart, numRows,
                                               rangeSize);
    diagBlock.triangularView<Eigen::Lower>()
        .transpose()
        .solveInPlace<Eigen::OnTheRight>(belowDiagBlock);
}

static void prepareContextForTargetAggreg(const BlockMatrixSkel& skel,
                                          uint64_t targetRange,
                                          vector<uint64_t>& spanToSliceOffset) {
    spanToSliceOffset.assign(skel.spanStart.size() - 1, 999999);
    for (uint64_t i = skel.sliceColPtr[targetRange],
                  iEnd = skel.sliceColPtr[targetRange + 1];
         i < iEnd; i++) {
        spanToSliceOffset[skel.sliceRowSpan[i]] = skel.sliceData[i];
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

        // per-row pointers to slices in a rectagle:
        // * span-rows from rangeToSpan[rangesEnd],
        // * slab cols in interval rangesBegin:rangesEnd
        uint64_t spanRowBegin;
        vector<uint64_t> rowPtr;       // row data pointer
        vector<uint64_t> colRange;     // col-range
        vector<uint64_t> sliceColOrd;  // order in col slice elements
    };

    // TODO: unit test
    virtual OpaqueDataPtr prepareElimination(const BlockMatrixSkel& skel,
                                             uint64_t rangesBegin,
                                             uint64_t rangesEnd) override {
        OpaqueDataElimData* elim = new OpaqueDataElimData;

        uint64_t spanRowBegin = skel.rangeToSpan[rangesEnd];
        uint64_t numSpanRows = skel.spanStart.size() - 1 - spanRowBegin;
        elim->rowPtr.assign(numSpanRows + 1, 0);
        for (uint64_t r = rangesBegin; r < rangesEnd; r++) {
            for (uint64_t i = skel.sliceColPtr[r],
                          iEnd = skel.sliceColPtr[r + 1];
                 i < iEnd; i++) {
                uint64_t s = skel.sliceRowSpan[i];
                if (s < spanRowBegin) {
                    continue;
                }
                uint64_t sRel = s - spanRowBegin;
                elim->rowPtr[sRel]++;
            }
        }
        uint64_t totNumSlices = cumSum(elim->rowPtr);
        elim->colRange.resize(totNumSlices);
        elim->sliceColOrd.resize(totNumSlices);
        for (uint64_t r = rangesBegin; r < rangesEnd; r++) {
            for (uint64_t iBegin = skel.sliceColPtr[r],
                          iEnd = skel.sliceColPtr[r + 1], i = iBegin;
                 i < iEnd; i++) {
                uint64_t s = skel.sliceRowSpan[i];
                if (s < spanRowBegin) {
                    continue;
                }
                uint64_t sRel = s - spanRowBegin;
                elim->colRange[elim->rowPtr[sRel]] = r;
                elim->sliceColOrd[elim->rowPtr[sRel]] = i - iBegin;
                elim->rowPtr[sRel]++;
            }
        }
        rewind(elim->rowPtr);
        elim->spanRowBegin = spanRowBegin;
        return OpaqueDataPtr(elim);
    }

    struct ElimContext {
        std::vector<double> tempBuffer;
        std::vector<uint64_t> spanToSliceOffset;
    };

    static void eliminateRowSlice(const OpaqueDataElimData& elim,
                                  const BlockMatrixSkel& skel, double* data,
                                  uint64_t sRel, ElimContext& ctx) {
        uint64_t s = sRel + elim.spanRowBegin;
        uint64_t targetRange = skel.spanToRange[s];
        uint64_t targetRangeSize =
            skel.rangeStart[targetRange + 1] - skel.rangeStart[targetRange];
        uint64_t spanOffsetInRange =
            skel.spanStart[s] - skel.rangeStart[targetRange];
        prepareContextForTargetAggreg(skel, targetRange, ctx.spanToSliceOffset);

        // iterate over slices present in this row
        for (uint64_t i = elim.rowPtr[sRel], iEnd = elim.rowPtr[sRel + 1];
             i < iEnd; i++) {
            uint64_t range = elim.colRange[i];
            uint64_t sliceColOrd = elim.sliceColOrd[i];
            CHECK_GE(sliceColOrd, 1);  // there must be a diagonal block

            uint64_t ptrStart = skel.sliceColPtr[range] + sliceColOrd;
            uint64_t ptrEnd = skel.sliceColPtr[range + 1];
            CHECK_EQ(skel.sliceRowSpan[ptrStart], s);

            uint64_t nRowsAbove = skel.sliceRowsTillEnd[ptrStart - 1];
            uint64_t nRowsSlice = skel.sliceRowsTillEnd[ptrStart] - nRowsAbove;
            uint64_t nRowsOnward = skel.sliceRowsTillEnd[ptrEnd - 1];
            uint64_t dataOffset = skel.sliceData[ptrStart];
            CHECK_EQ(nRowsSlice, skel.spanStart[s + 1] - skel.spanStart[s]);
            uint64_t rangeSize =
                skel.rangeStart[range + 1] - skel.rangeStart[range];

            Eigen::Map<MatRMaj<double>> sliceSubMat(data + dataOffset,
                                                    nRowsSlice, rangeSize);
            Eigen::Map<MatRMaj<double>> sliceOnwardSubMat(
                data + dataOffset, nRowsOnward, rangeSize);

            ctx.tempBuffer.resize(nRowsOnward * nRowsSlice);
            Eigen::Map<MatRMaj<double>> prod(ctx.tempBuffer.data(), nRowsOnward,
                                             nRowsSlice);
            prod = sliceOnwardSubMat * sliceSubMat.transpose();

            // assemble blocks, iterating on slice and below slices
            for (uint64_t ptr = ptrStart; ptr < ptrEnd; ptr++) {
                uint64_t s2 = skel.sliceRowSpan[ptr];
                uint64_t relRow = skel.sliceRowsTillEnd[ptr - 1] - nRowsAbove;
                uint64_t s2_size =
                    skel.sliceRowsTillEnd[ptr] - nRowsAbove - relRow;
                CHECK_EQ(s2_size, skel.spanStart[s2 + 1] - skel.spanStart[s2]);

                double* targetData =
                    data + spanOffsetInRange + ctx.spanToSliceOffset[s2];

                OuterStridedMatM targetBlock(targetData, s2_size, nRowsSlice,
                                             OuterStride(targetRangeSize));
                targetBlock -= prod.block(relRow, 0, s2_size, nRowsSlice);
            }
        }
    }

    virtual void doElimination(const OpaqueData& ref, double* data,
                               uint64_t rangesBegin, uint64_t rangesEnd,
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
            taskSet, dispenso::makeChunkedRange(rangesBegin, rangesEnd, 5UL),
            [&](int64_t rStart, int64_t rEnd) {
                for (int64_t r = rStart; r < rEnd; r++) {
                    factorAggreg(skel, data, r);
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
                    eliminateRowSlice(elim, skel, data, sRel, ctx);
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